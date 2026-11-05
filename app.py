"""ResearchMind AI - Streamlit interface.

This file contains only UI code. All document processing, retrieval and
generation logic lives in ``src/`` and is shared with the tests and the
evaluation script; HTML building blocks live in ``ui/``.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import src  # noqa: F401  (sets the macOS OpenMP workaround before torch/faiss load)
import streamlit as st

from src.config import Settings
from src.embeddings import Embedder
from src.llm_client import LLMError, get_llm_client, ollama_status
from src.query_rewriter import detect_language
from src.rag_pipeline import RAGPipeline
from src.reranker import get_reranker
from src.sample_papers import PAPERS, ensure_downloaded
from src.schemas import RAGAnswer
from ui import benchmarks
from ui.components import (
    LEDE,
    PROVIDER_LABELS,
    answer_html,
    esc,
    hero_html,
    paper_html,
    refusal_html,
    refused_before_llm,
    sources_html,
    trace_html,
)
from ui.style import CSS

st.set_page_config(page_title="ResearchMind AI", page_icon=":material/menu_book:", layout="wide")

EXAMPLE_QUESTIONS = [
    "What are the main findings?",
    "Which methods and data were used?",
    "What are the limitations of the study?",
    "ما حجم العينة في هذه الدراسة؟",
]
HISTORY_TURNS = 2  # earlier turns passed to the pipeline for follow-up questions


# ----------------------------------------------------------------- resources
@st.cache_resource(show_spinner="Loading the embedding model (first run downloads about 90 MB)…")
def load_embedder(model_name: str) -> Embedder:
    return Embedder(model_name)


@st.cache_resource(show_spinner="Loading the reranking models (first run downloads them)…")
def load_reranker(name: str):
    return get_reranker(name)


def get_settings() -> Settings:
    if "settings" not in st.session_state:
        st.session_state.settings = Settings.from_env()
    return st.session_state.settings


def get_pipeline() -> RAGPipeline:
    if "pipeline" not in st.session_state:
        settings = get_settings()
        st.session_state.pipeline = RAGPipeline(
            settings, load_embedder(settings.embedding_model),
            load_reranker(settings.reranker), load_reranker(settings.rerank_gate),
        )
    return st.session_state.pipeline


def init_state() -> None:
    st.session_state.setdefault("history", [])  # list[RAGAnswer | dict(question, error)]
    st.session_state.setdefault("uploader_key", 0)
    st.session_state.setdefault("ingest_log", [])
    st.session_state.setdefault("comparison", None)


def html_block(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


# ------------------------------------------------------------------- actions
def ingest_files(pipeline: RAGPipeline, files: list[tuple[str, bytes]], label: str) -> None:
    log = []
    with st.status(label, expanded=True) as status:
        for name, data in files:
            st.write(f"Reading {name}")
            log.append(pipeline.ingest_pdf(data, name))
        n_ok = sum(r.ok for r in log)
        status.update(label=f"Added {n_ok} of {len(log)} paper{'s' if len(log) != 1 else ''}",
                      state="complete", expanded=False)
    st.session_state.ingest_log = log


def load_sample_papers(pipeline: RAGPipeline) -> None:
    try:
        with st.spinner("Downloading 4 open-access papers from arXiv (first time only)…"):
            paths = ensure_downloaded(pause_s=1.0)
    except Exception as exc:  # network errors, arXiv changes
        st.error(f"Could not download the sample papers: {exc}. Check your internet connection, or add your own "
                 "PDFs in the sidebar.")
        return
    ingest_files(pipeline, [(p.name, p.read_bytes()) for p in paths], "Adding the sample papers…")
    st.rerun()


def run_question(pipeline: RAGPipeline, settings: Settings, opts: dict, question: str) -> None:
    previous = [h for h in st.session_state.history if isinstance(h, RAGAnswer)][-HISTORY_TURNS:]
    history = [(h.question, h.raw_answer or h.answer) for h in previous]
    try:
        llm = get_llm_client(opts["provider"], settings, model=opts["model"])
        with st.spinner("Finding evidence and writing the answer…"):
            answer = pipeline.ask(question, llm, doc_ids=opts["doc_ids"], top_k=opts["top_k"], history=history)
        st.session_state.history.append(answer)
    except (LLMError, ValueError) as exc:
        st.session_state.history.append({"question": question, "error": str(exc)})


# ------------------------------------------------------------------- sidebar
def retrieval_summary(settings: Settings) -> str:
    ranked, gated = settings.reranker != "none", settings.rerank_gate != "none"
    if not ranked and not gated:
        return (f"Passages are ranked by embedding similarity. Questions whose best passage scores below "
                f"{settings.min_similarity:.2f} are not sent to the language model.")
    judge = settings.rerank_gate if gated else settings.reranker
    text = f"The {settings.reranker} reranker orders the passages. " if ranked else ""
    return text + (f"The {judge} model checks the evidence; below a relevance of {settings.rerank_min_score:.2f} "
                   "nothing is sent to the language model.")


def sidebar(pipeline: RAGPipeline, settings: Settings) -> dict:
    with st.sidebar:
        html_block('<div class="rm-wordmark">ResearchMind</div>'
                   '<div class="rm-wordmark-sub">Answers from your papers, with the page for every claim.</div>')

        html_block('<div class="rm-side-h">Your papers</div>')
        files = st.file_uploader(
            "Add PDF files", type=["pdf"], accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}", label_visibility="collapsed",
        )
        if st.button("Add papers", type="primary", disabled=not files, width="stretch"):
            ingest_files(pipeline, [(f.name, f.getvalue()) for f in files], "Adding papers…")
            st.session_state.uploader_key += 1  # reset the uploader
            st.rerun()

        for r in st.session_state.ingest_log:
            if not r.ok:
                st.caption(f":orange[{r.file_name}: {r.message}]")

        selected: set[str] | None = None
        if pipeline.documents:
            with st.container(key="rm-papers", gap="small"):
                for doc_id, info in list(pipeline.documents.items()):
                    html_block(paper_html(info.doc_name, info.page_count, info.chunk_count))
                    for w in info.warnings:
                        st.caption(f":orange[{w}]")
                    if st.button("Remove", key=f"rm_{doc_id}", type="tertiary", help=f"Remove {info.doc_name}"):
                        pipeline.remove_document(doc_id)
                        st.rerun()
            names = {info.doc_name: doc_id for doc_id, info in pipeline.documents.items()}
            if len(names) > 1:
                chosen = st.multiselect("Search in", list(names), placeholder="All papers")
                selected = {names[n] for n in chosen} or None
            if st.button("Remove all papers", width="stretch"):
                pipeline.clear()
                st.session_state.history = []
                st.session_state.ingest_log = []
                st.session_state.comparison = None
                st.rerun()
        else:
            st.caption("Add text-based PDFs, or load the sample papers from the main screen.")

        with st.expander("Model and retrieval"):
            providers = list(PROVIDER_LABELS)
            provider = st.selectbox(
                "Language model", providers, index=providers.index(settings.llm_provider),
                format_func=PROVIDER_LABELS.get,
            )
            model = settings.model_for(provider)
            if provider == "ollama":
                reachable, models = ollama_status(settings.ollama_base_url)
                if reachable and models:
                    index = models.index(model) if model in models else 0
                    model = st.selectbox("Ollama model", models, index=index)
                    st.caption(":green[Ollama is running]")
                elif reachable:
                    st.caption(f":orange[Ollama has no models yet. Run `ollama pull {model}`.]")
                else:
                    st.caption(":red[Ollama isn't running. Start the Ollama app, then reload this page.]")
            elif provider == "gemini":
                model = st.text_input("Gemini model", value=model)
                if not settings.gemini_api_key:
                    st.caption(":red[Add GEMINI_API_KEY to .env to use Gemini.]")
            elif provider == "claude":
                options = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
                if model not in options:
                    options.insert(0, model)
                model = st.selectbox("Claude model", options, index=options.index(model))
                if not settings.anthropic_api_key:
                    st.caption(":red[Add ANTHROPIC_API_KEY to .env to use Claude.]")

            top_k = st.slider("Passages per answer", 1, 10, settings.top_k)
            st.caption(retrieval_summary(settings))

        if st.session_state.history and st.button("Clear answers", width="stretch"):
            st.session_state.history = []
            st.rerun()

    return {"provider": provider, "model": model, "top_k": top_k, "doc_ids": selected}


# ---------------------------------------------------------------------- main
def render_answer(answer: RAGAnswer, latest: bool) -> None:
    if refused_before_llm(answer):
        html_block(refusal_html(answer))
    else:
        html_block(answer_html(answer))
    for warning in answer.warnings:
        html_block(f'<div class="rm-warn">{esc(warning)}</div>')
    html_block(trace_html(answer, open_=latest))
    html_block(sources_html(answer))


def welcome(pipeline: RAGPipeline) -> None:
    html_block(hero_html())
    st.write("")
    loaded = {i.doc_name for i in pipeline.documents.values()}
    cols = st.columns([0.42, 0.58], vertical_alignment="center")
    if cols[0].button("Load 4 open-access papers", type="primary", disabled=set(PAPERS) <= loaded,
                      width="stretch"):
        load_sample_papers(pipeline)
    cols[1].caption("Or add your own PDFs in the sidebar.")
    html_block(f'<p class="rm-lede">{LEDE}</p>')


def ask_tab(pipeline: RAGPipeline, settings: Settings, opts: dict) -> None:
    if not pipeline.documents and not st.session_state.history:
        welcome(pipeline)
    elif not st.session_state.history:
        st.caption("Try one of these, or type your own question below.")
        with st.container(key="rm-examples", horizontal=True):
            for q in EXAMPLE_QUESTIONS:
                if st.button(q, type="secondary"):
                    st.session_state.pending_question = q

    items = st.session_state.history
    for n, item in enumerate(items):
        first = " rm-q-first" if n == 0 else ""
        question = item["question"] if isinstance(item, dict) else item.question
        rtl = ' rm-rtl" dir="rtl' if detect_language(question) == "ar" else ""
        html_block(f'<p class="rm-q{first}{rtl}">{esc(question)}</p>')
        if isinstance(item, dict):
            st.error(item["error"])
        else:
            render_answer(item, latest=n == len(items) - 1)

    question = st.chat_input(
        "Ask about findings, methods, sample sizes or limitations" if pipeline.documents
        else "Add papers to start asking questions",
        disabled=not pipeline.documents,
    )
    question = question or st.session_state.pop("pending_question", None)
    if question:
        run_question(pipeline, settings, opts, question)
        st.rerun()


def compare_tab(pipeline: RAGPipeline, settings: Settings, opts: dict) -> None:
    if len(pipeline.documents) < 2:
        st.caption("Add at least two papers to compare them side by side.")
        return
    st.caption("A table of objective, design, data, methods, findings and limitations for two papers. Every cell "
               "cites its page; anything the papers don't state is marked as not reported.")
    names = {info.doc_name: doc_id for doc_id, info in pipeline.documents.items()}
    labels = list(names)
    col1, col2 = st.columns(2)
    a = col1.selectbox("First paper", labels, index=0)
    b = col2.selectbox("Second paper", labels, index=1)
    if st.button("Compare papers", type="primary", disabled=a == b):
        try:
            llm = get_llm_client(opts["provider"], settings, model=opts["model"])
            with st.spinner("Finding evidence for each aspect and writing the comparison…"):
                st.session_state.comparison = pipeline.compare(names[a], names[b], llm)
        except (LLMError, ValueError) as exc:
            st.error(str(exc))
    if st.session_state.comparison is not None:
        render_answer(st.session_state.comparison, latest=True)


def main() -> None:
    init_state()
    st.html(CSS)
    try:
        settings = get_settings()
    except ValueError as exc:
        st.error(f"Configuration error: {exc}")
        st.stop()
    pipeline = get_pipeline()
    opts = sidebar(pipeline, settings)

    ask, compare, bench = st.tabs(["Ask", "Compare papers", "Benchmarks"])
    with ask:
        ask_tab(pipeline, settings, opts)
    with compare:
        compare_tab(pipeline, settings, opts)
    with bench:
        benchmarks.render()


main()
