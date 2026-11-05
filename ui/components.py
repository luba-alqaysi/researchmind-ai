"""HTML building blocks for the Streamlit app. Pure functions returning HTML
strings, so they can be unit-tested without running Streamlit."""

from __future__ import annotations

import html
import re

from src.highlight import best_sentence, claims_by_source
from src.schemas import RAGAnswer, RetrievedChunk, Trace

PROVIDER_LABELS = {
    "ollama": "Ollama (local)",
    "gemini": "Google Gemini",
    "claude": "Anthropic Claude",
    "extractive": "No LLM (passages only)",
}

# Real example from the evaluation set (rnaseq_gene_clusters.pdf, CC BY 4.0, page 2).
HERO_QUOTE = ("The dataset used in this work includes the complete mRNAseq data from ",
              "nc = 105 controls (Minnesota Grading System 1) and ns = 61 subjects",
              " (Minnesota Grading System 4) …")
HERO_QUESTION = "How many controls and AMD subjects were included?"
HERO_ANSWER = "105 controls and 61 subjects with late-stage AMD."
HERO_SOURCE = ("rnaseq_gene_clusters.pdf", 2)

_LABEL = re.compile(r"\[([^\[\]\n]+?), Page (\d+)\]")


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def file_name(name: str) -> str:
    """Escaped file name that may wrap after underscores, hyphens and dots instead of mid-word."""
    return re.sub(r"([_\-.])", r"\1<wbr>", esc(name))


MODEL_NAMES = {"laya": "Laya", "ms-marco": "ms-marco", "bge": "bge"}


def model_name(name: str) -> str:
    return MODEL_NAMES.get(name, name)


def cite_tag(doc_name: str, page: int, show_doc: bool = True) -> str:
    label = f"{esc(doc_name.removesuffix('.pdf'))}, p.&nbsp;{page}" if show_doc else f"p.&nbsp;{page}"
    return f'<span class="rm-cite" title="{esc(doc_name)}, page {page}">{label}</span>'


def is_rtl(answer: RAGAnswer) -> bool:
    return answer.trace is not None and answer.trace.language == "ar"


# ------------------------------------------------------------------- hero
def hero_html() -> str:
    before, marked, after = HERO_QUOTE
    doc, page = HERO_SOURCE
    return f"""
<section class="rm-sweep">
  <h1 class="rm-hero-title">Ask your papers a question.<br>See the sentence that answers it.</h1>
  <figure class="rm-page" style="margin:0 0 1.1rem">
    <figcaption class="rm-folio"><b>p. {page}</b>{file_name(doc)}</figcaption>
    <blockquote class="rm-quote">{esc(before)}<mark class="rm-hl">{esc(marked)}</mark>{esc(after)}</blockquote>
  </figure>
  <p class="rm-q rm-q-first">{esc(HERO_QUESTION)}</p>
  <div class="rm-answer">{esc(HERO_ANSWER)} {cite_tag(doc, page)}</div>
</section>"""


LEDE = ("Runs on your computer. Cites the page for every claim, and says so when your papers don't cover a "
        "question. Ask in English or Arabic.")


# ----------------------------------------------------------------- answer
def answer_html(answer: RAGAnswer) -> str:
    """The answer text, with verified citation labels shown as tags. Markdown is kept
    (the result goes through st.markdown); HTML in the model's output is escaped."""
    body = html.escape(answer.answer.replace("$", r"\$"), quote=False)
    # With a single paper cited, the page alone is enough; the paper is named under "Where the answer comes from".
    show_doc = len({c.doc_name for c in answer.citations}) > 1
    body = _LABEL.sub(lambda m: cite_tag(html.unescape(m.group(1)), int(m.group(2)), show_doc), body)
    cls = "rm-answer rm-rtl" if is_rtl(answer) else "rm-answer"
    direction = ' dir="rtl"' if is_rtl(answer) else ""
    return f'<div class="{cls}"{direction}>\n\n{body}\n\n</div>'


def refused_before_llm(answer: RAGAnswer) -> bool:
    t = answer.trace
    return t is not None and t.evidence is not None and not t.evidence.passed


def refusal_html(answer: RAGAnswer) -> str:
    ev = answer.trace.evidence
    if ev.score is None:
        why = "No passages were found in the selected papers."
    else:
        measure = "similarity" if ev.role == "cosine" else "relevance"
        why = ("None of the passages was relevant enough to answer from "
               f"(best {measure} <span class='rm-num'>{ev.score:.2f}</span>, needed "
               f"<span class='rm-num'>{ev.threshold:.2f}</span>).")
    if is_rtl(answer):
        return ('<div class="rm-refusal rm-rtl" dir="rtl"><h4>الأوراق المختارة لا تجيب عن هذا السؤال</h4>'
                '<p>لم يُرسَل أي شيء إلى النموذج اللغوي. جرّب ذكر اسم الورقة أو مصطلح محدد، أو ابحث في أوراق أخرى.</p></div>')
    return (f'<div class="rm-refusal"><h4>The papers you selected don\'t cover this question</h4>'
            f'<p>{why} Nothing was sent to the language model. Try naming the paper or a specific term, '
            f'or search in other papers.</p></div>')


# ------------------------------------------------------------------ trace
def _ms(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value / 1000:.1f}&nbsp;s" if value >= 1000 else f"{value:.0f}&nbsp;ms"


def _step(name: str, detail: str, state: str = "ok") -> str:
    return f'<li class="{state}"><div class="rm-step-name">{name}</div><div class="rm-step-detail">{detail}</div></li>'


def trace_html(answer: RAGAnswer, open_: bool = True) -> str:
    t: Trace | None = answer.trace
    if t is None:
        return ""
    ms = t.stage_ms
    steps = []

    if t.rewritten:
        lang = "Arabic question; " if t.language == "ar" else ""
        steps.append(_step("Read", f"{lang}searched for <q>{esc(t.search_query)}</q>"))
    else:
        steps.append(_step("Read", "Searched with your question as written" +
                           (f" ({_ms(ms.get('rewrite_ms'))})" if ms.get("rewrite_ms") else "")))

    papers = "paper" if t.papers == 1 else "papers"
    steps.append(_step("Found", f"<span class='rm-num'>{t.candidates}</span> passages in "
                                f"<span class='rm-num'>{t.papers}</span> {papers}<br>{_ms(ms.get('search_ms'))}"))

    if t.reranker:
        steps.append(_step("Ranked", f"Reordered by the {esc(model_name(t.reranker))} reranker<br>"
                                     f"{_ms(ms.get('rerank_ms'))}"))
    else:
        steps.append(_step("Ranked", "Ordered by embedding similarity"))

    ev = t.evidence
    if ev is not None and ev.score is not None:
        measure = "similarity" if ev.role == "cosine" else "relevance"
        judge = "" if ev.role == "cosine" else f"Judged by {esc(model_name(ev.judge))}<br>"
        fill = max(0.0, min(1.0, ev.score)) * 100
        mark = max(0.0, min(1.0, ev.threshold)) * 100
        meter = (f'<div class="rm-meter" role="img" aria-label="{measure} {ev.score:.2f}, needs {ev.threshold:.2f}">'
                 f'<span class="fill" style="width:{fill:.0f}%"></span><span class="mark" style="left:{mark:.0f}%"></span></div>')
        detail = (f"{judge}{measure} <span class='rm-num'>{ev.score:.2f}</span>, needs "
                  f"<span class='rm-num'>{ev.threshold:.2f}</span>{meter}{_ms(ms.get('gate_ms'))}")
        steps.append(_step("Checked evidence" if ev.passed else "Not enough evidence", detail,
                           "pass" if ev.passed else "refuse"))
    else:
        steps.append(_step("Not enough evidence", "No passages found", "refuse"))

    if refused_before_llm(answer):
        steps.append(_step("Wrote", "Not written. Nothing was sent to the language model.", "skip"))
    elif answer.provider == "extractive":
        steps.append(_step("Wrote", "Generation is off; showing the passages"))
    else:
        n, removed = len(answer.citations), len(answer.invalid_source_ids)
        checked = f"{n} citation{'s' if n != 1 else ''} checked"
        if removed:
            checked += f", {removed} removed"
        if answer.insufficient_evidence:
            checked = "The model reported that the passages don't answer it"
        steps.append(_step("Wrote", f"{esc(answer.model)}, {_ms(ms.get('generation_ms'))}<br>{checked}"))

    return (f'<details class="rm-how"{" open" if open_ else ""}><summary>How this answer was made</summary>'
            f'<ol class="rm-trace">{"".join(steps)}</ol></details>')


# ---------------------------------------------------------------- sources
def page_slip_html(chunk_doc: str, page: int, text: str, highlight: str | None, meta: str = "") -> str:
    quote = esc(text)
    if highlight:
        marked = esc(highlight)
        if marked in quote:
            quote = quote.replace(marked, f'<mark class="rm-hl">{marked}</mark>', 1)
    meta_html = f'<div class="rm-slip-meta">{meta}</div>' if meta else ""
    return (f'<figure class="rm-page"><figcaption class="rm-folio"><b>p. {page}</b>{file_name(chunk_doc)}</figcaption>'
            f'<blockquote class="rm-quote">{quote}</blockquote>{meta_html}</figure>')


def _scores(r: RetrievedChunk) -> str:
    parts = []
    if r.gate_score is not None:
        parts.append(f"relevance <span class='rm-num'>{r.gate_score:.2f}</span>")
    parts.append(f"similarity <span class='rm-num'>{r.score:.2f}</span>")
    return ", ".join(parts)


def sources_html(answer: RAGAnswer) -> str:
    """Cited passages as page slips with the closest sentence highlighted, then the rest folded away."""
    if not answer.retrieved:
        return ""
    claims = claims_by_source(answer.raw_answer)
    query = answer.trace.search_query if answer.trace else answer.question
    cited_idx = []
    for c in answer.citations:
        i = int(c.source_id[1:]) - 1
        if 0 <= i < len(answer.retrieved) and i not in cited_idx:
            cited_idx.append(i)

    parts = []
    if cited_idx:
        parts.append('<div class="rm-sources-title">Where the answer comes from, with the closest sentence highlighted</div>')
        for i in cited_idx:
            r = answer.retrieved[i]
            sentence = best_sentence(r.chunk.text, claims.get(i + 1, ""), query)
            parts.append(page_slip_html(r.chunk.doc_name, r.chunk.page_number, r.chunk.text, sentence, _scores(r)))
    rest = [r for i, r in enumerate(answer.retrieved) if i not in cited_idx]
    if rest:
        label = "Other passages it read" if cited_idx else "Passages it read"
        slips = "".join(page_slip_html(r.chunk.doc_name, r.chunk.page_number, r.chunk.text,
                                       best_sentence(r.chunk.text, query), _scores(r)) for r in rest)
        parts.append(f'<details class="rm-more"><summary>{label} ({len(rest)})</summary>{slips}</details>')
    return "".join(parts)


def paper_html(name: str, pages: int, chunks: int) -> str:
    return (f'<div class="rm-paper"><div class="rm-paper-name">{esc(name)}</div>'
            f'<div class="rm-paper-meta"><span class="rm-num">{pages}</span> pages, '
            f'<span class="rm-num">{chunks}</span> passages</div></div>')
