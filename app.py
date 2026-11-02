"""ResearchMind AI - Streamlit interface.

This file contains only UI code. All document processing, retrieval and
generation logic lives in ``src/`` and is shared with the tests and the
evaluation script.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import html

import src  # noqa: F401  (sets the macOS OpenMP workaround before torch/faiss load)
import streamlit as st

from src.config import Settings
from src.embeddings import Embedder
from src.llm_client import LLMError, get_llm_client, ollama_status
from src.rag_pipeline import RAGPipeline
from src.schemas import RAGAnswer

st.set_page_config(page_title="ResearchMind AI", page_icon=":material/menu_book:", layout="wide")

PROVIDER_LABELS = {
    "ollama": "Ollama (local)",
    "gemini": "Google Gemini",
    "claude": "Anthropic Claude",
    "extractive": "No LLM (passages only)",
}
EXAMPLE_QUESTIONS = [
    "What are the main findings of this paper?",
    "What methodology and data were used?",
    "What are the limitations of the study?",
    "Which statistical methods were applied?",
]

CSS = """
<style>
h1, h2, h3 { font-family: Georgia, 'Times New Roman', serif; letter-spacing: -0.01em; }
.block-container { padding-top: 2rem; max-width: 1100px; }
.rm-subtitle { color: #5a5a5a; margin-top: -0.6rem; margin-bottom: 1.2rem; }
.rm-chip { display: inline-block; padding: 0.15rem 0.55rem; margin: 0 0.35rem 0.35rem 0;
           border: 1px solid #d5d1c7; border-radius: 999px; font-size: 0.82rem;
           background: #f6f4ef; color: #1f4e79; }
.rm-excerpt { border-left: 3px solid #1f4e79; padding: 0.4rem 0.8rem; background: #f6f4ef;
              font-size: 0.9rem; line-height: 1.5; color: #333; white-space: pre-wrap; }
.rm-meta { color: #777; font-size: 0.8rem; }
.rm-step { padding: 0.9rem 1rem; border: 1px solid #e3e0d8; border-radius: 8px; background: #fff; height: 100%; }
.rm-step b { color: #1f4e79; }
</style>
"""


# ----------------------------------------------------------------- resources
@st.cache_resource(show_spinner="Loading the embedding model (first run downloads ~90 MB)…")
def load_embedder(model_name: str) -> Embedder:
    return Embedder(model_name)


def get_settings() -> Settings:
    if "settings" not in st.session_state:
        st.session_state.settings = Settings.from_env()
    return st.session_state.settings


def get_pipeline() -> RAGPipeline:
    if "pipeline" not in st.session_state:
        settings = get_settings()
        st.session_state.pipeline = RAGPipeline(settings, load_embedder(settings.embedding_model))
    return st.session_state.pipeline


def init_state() -> None:
    st.session_state.setdefault("history", [])  # list[RAGAnswer | dict(error)]
    st.session_state.setdefault("uploader_key", 0)
    st.session_state.setdefault("ingest_log", [])
    st.session_state.setdefault("comparison", None)


# ------------------------------------------------------------------- helpers
def escape_md(text: str) -> str:
    """Stop '$' in scientific text from being rendered as LaTeX."""
    return text.replace("$", r"\$")


def excerpt_html(text: str) -> str:
    return f'<div class="rm-excerpt">{html.escape(text)}</div>'


def render_answer(answer: RAGAnswer) -> None:
    st.markdown(escape_md(answer.answer))
    for warning in answer.warnings:
        st.caption(f":orange[⚠ {warning}]")

    if answer.citations:
        chips = "".join(
            f'<span class="rm-chip">{html.escape(c.label)}</span>' for c in answer.citations
        )
        st.markdown(f"<div><span class='rm-meta'>Sources cited:</span><br>{chips}</div>", unsafe_allow_html=True)
        for c in answer.citations:
            with st.expander(f"{c.source_id} · {c.doc_name}, page {c.page_number} · similarity {c.score:.2f}"):
                st.markdown(excerpt_html(c.excerpt), unsafe_allow_html=True)

    if answer.retrieved:
        with st.expander(f"All retrieved passages ({len(answer.retrieved)})"):
            for i, r in enumerate(answer.retrieved, start=1):
                st.markdown(
                    f"<span class='rm-meta'>S{i} · <b>{html.escape(r.chunk.doc_name)}</b>, page "
                    f"{r.chunk.page_number} · similarity {r.score:.2f}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(excerpt_html(r.chunk.text), unsafe_allow_html=True)

    t = answer.timings
    parts = [f"{PROVIDER_LABELS.get(answer.provider, answer.provider)} · {answer.model}"]
    if "retrieval_ms" in t:
        parts.append(f"retrieval {t['retrieval_ms']:.0f} ms")
    if "generation_ms" in t:
        parts.append(f"generation {t['generation_ms'] / 1000:.1f} s")
    st.markdown(f"<span class='rm-meta'>{' · '.join(parts)}</span>", unsafe_allow_html=True)


# ------------------------------------------------------------------- sidebar
def sidebar(pipeline: RAGPipeline, settings: Settings) -> dict:
    with st.sidebar:
        st.markdown("## ResearchMind AI")
        st.caption("Ask questions about scientific papers and get answers grounded in cited passages.")

        st.markdown("#### 1 · Upload papers")
        files = st.file_uploader(
            "PDF files", type=["pdf"], accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}", label_visibility="collapsed",
        )
        if st.button("Process documents", type="primary", disabled=not files, width="stretch"):
            log = []
            with st.status("Processing documents…", expanded=True) as status:
                for f in files:
                    st.write(f"Reading **{f.name}**…")
                    log.append(pipeline.ingest_pdf(f.getvalue(), f.name))
                n_ok = sum(r.ok for r in log)
                status.update(label=f"Indexed {n_ok} of {len(log)} file(s)", state="complete", expanded=False)
            st.session_state.ingest_log = log
            st.session_state.uploader_key += 1  # reset the uploader
            st.rerun()

        for r in st.session_state.ingest_log:
            icon = {"indexed": "✅", "duplicate": "↺"}.get(r.status, "⚠️")
            st.caption(f"{icon} **{r.file_name}** — {r.message}")

        st.markdown("#### 2 · Your library")
        selected: set[str] | None = None
        if not pipeline.documents:
            st.caption("No documents indexed yet.")
        else:
            for doc_id, info in list(pipeline.documents.items()):
                col1, col2 = st.columns([5, 1])
                col1.markdown(
                    f"**{html.escape(info.doc_name)}**<br><span class='rm-meta'>{info.page_count} pages · "
                    f"{info.chunk_count} chunks</span>", unsafe_allow_html=True,
                )
                if col2.button("✕", key=f"rm_{doc_id}", help=f"Remove {info.doc_name}"):
                    pipeline.remove_document(doc_id)
                    st.rerun()
                for w in info.warnings:
                    col1.caption(f":orange[{w}]")
            names = {info.doc_name: doc_id for doc_id, info in pipeline.documents.items()}
            chosen = st.multiselect("Search in", list(names), placeholder="All documents")
            selected = {names[n] for n in chosen} or None
            if st.button("Clear all documents", width="stretch"):
                pipeline.clear()
                st.session_state.history = []
                st.session_state.ingest_log = []
                st.session_state.comparison = None
                st.rerun()

        st.markdown("#### 3 · Model")
        providers = list(PROVIDER_LABELS)
        provider = st.selectbox(
            "LLM provider", providers, index=providers.index(settings.llm_provider),
            format_func=PROVIDER_LABELS.get,
        )
        model = settings.model_for(provider)
        if provider == "ollama":
            reachable, models = ollama_status(settings.ollama_base_url)
            if reachable and models:
                index = models.index(model) if model in models else 0
                model = st.selectbox("Ollama model", models, index=index)
                st.caption(":green[● Ollama is running]")
            elif reachable:
                st.caption(f":orange[● Ollama is running but has no models. Run `ollama pull {model}`.]")
            else:
                st.caption(":red[● Ollama is not reachable. Start the Ollama app.]")
        elif provider == "gemini":
            model = st.text_input("Gemini model", value=model)
            st.caption(":green[● API key found]" if settings.gemini_api_key else ":red[● GEMINI_API_KEY missing in .env]")
        elif provider == "claude":
            options = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
            if model not in options:
                options.insert(0, model)
            model = st.selectbox("Claude model", options, index=options.index(model))
            st.caption(":green[● API key found]" if settings.anthropic_api_key else ":red[● ANTHROPIC_API_KEY missing in .env]")

        with st.expander("Retrieval settings"):
            top_k = st.slider("Passages to retrieve (top-k)", 1, 10, settings.top_k)
            st.caption(
                f"Chunk size {settings.chunk_size} chars, overlap {settings.chunk_overlap}. "
                f"Questions whose best match scores below {settings.min_similarity:.2f} are answered "
                "with 'insufficient evidence' without calling the LLM."
            )

        if st.session_state.history and st.button("Clear chat", width="stretch"):
            st.session_state.history = []
            st.rerun()

    return {"provider": provider, "model": model, "top_k": top_k, "doc_ids": selected}


# ---------------------------------------------------------------------- main
def welcome() -> None:
    st.markdown("### Get started")
    cols = st.columns(3)
    steps = [
        ("Upload", "Add one or more text-based research papers (PDF) in the sidebar and press <i>Process</i>."),
        ("Ask", "Ask about findings, methods, sample sizes, statistics or limitations — across papers or one at a time."),
        ("Verify", "Every answer cites the document and page it came from, with the supporting passage."),
    ]
    for col, (title, body) in zip(cols, steps):
        col.markdown(f"<div class='rm-step'><b>{title}</b><br>{body}</div>", unsafe_allow_html=True)
    st.markdown("")
    st.info(
        "ResearchMind reduces, but cannot eliminate, the risk of incorrect answers. "
        "Always check important claims against the cited passages.", icon=":material/info:",
    )


def run_question(pipeline: RAGPipeline, settings: Settings, opts: dict, question: str) -> None:
    try:
        llm = get_llm_client(opts["provider"], settings, model=opts["model"])
        with st.spinner("Retrieving evidence and generating an answer…"):
            answer = pipeline.ask(question, llm, doc_ids=opts["doc_ids"], top_k=opts["top_k"])
        st.session_state.history.append(answer)
    except LLMError as exc:
        st.session_state.history.append({"question": question, "error": str(exc)})
    except ValueError as exc:
        st.session_state.history.append({"question": question, "error": str(exc)})


def ask_tab(pipeline: RAGPipeline, settings: Settings, opts: dict) -> None:
    if not st.session_state.history and pipeline.documents:
        st.markdown("<span class='rm-meta'>Try one of these:</span>", unsafe_allow_html=True)
        cols = st.columns(len(EXAMPLE_QUESTIONS))
        for col, q in zip(cols, EXAMPLE_QUESTIONS):
            if col.button(q, width="stretch"):
                st.session_state.pending_question = q

    for item in st.session_state.history:
        if isinstance(item, dict):
            with st.chat_message("user"):
                st.markdown(escape_md(item["question"]))
            with st.chat_message("assistant", avatar=":material/error:"):
                st.error(item["error"])
            continue
        with st.chat_message("user"):
            st.markdown(escape_md(item.question))
        with st.chat_message("assistant", avatar=":material/menu_book:"):
            render_answer(item)

    question = st.chat_input(
        "Ask a research question…" if pipeline.documents else "Upload documents to start asking questions",
        disabled=not pipeline.documents,
    )
    question = question or st.session_state.pop("pending_question", None)
    if question:
        run_question(pipeline, settings, opts, question)
        st.rerun()


def compare_tab(pipeline: RAGPipeline, settings: Settings, opts: dict) -> None:
    if len(pipeline.documents) < 2:
        st.caption("Upload at least two papers to compare them side by side.")
        return
    st.caption(
        "Generates a structured comparison (objective, design, data, methods, findings, limitations). "
        "Each cell cites its evidence; aspects not found in the retrieved text are labelled as such."
    )
    names = {info.doc_name: doc_id for doc_id, info in pipeline.documents.items()}
    labels = list(names)
    col1, col2 = st.columns(2)
    a = col1.selectbox("Paper A", labels, index=0)
    b = col2.selectbox("Paper B", labels, index=1)
    if st.button("Compare papers", type="primary", disabled=a == b):
        try:
            llm = get_llm_client(opts["provider"], settings, model=opts["model"])
            with st.spinner("Retrieving evidence for each aspect and generating the comparison…"):
                st.session_state.comparison = pipeline.compare(names[a], names[b], llm)
        except (LLMError, ValueError) as exc:
            st.error(str(exc))
    if st.session_state.comparison is not None:
        render_answer(st.session_state.comparison)


def main() -> None:
    init_state()
    st.markdown(CSS, unsafe_allow_html=True)
    try:
        settings = get_settings()
    except ValueError as exc:
        st.error(f"Configuration error: {exc}")
        st.stop()
    pipeline = get_pipeline()
    opts = sidebar(pipeline, settings)

    st.title("ResearchMind AI")
    st.markdown(
        "<div class='rm-subtitle'>An evidence-grounded research assistant for scientific literature</div>",
        unsafe_allow_html=True,
    )
    if not pipeline.documents and not st.session_state.history:
        welcome()
        return

    ask, compare = st.tabs(["Ask questions", "Compare papers"])
    with ask:
        ask_tab(pipeline, settings, opts)
    with compare:
        compare_tab(pipeline, settings, opts)


main()
