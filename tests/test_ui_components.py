"""HTML building blocks of the app (no Streamlit needed)."""

from src.schemas import Chunk, Citation, EvidenceCheck, RAGAnswer, RetrievedChunk, Trace
from ui.components import answer_html, file_name, page_slip_html, refusal_html, sources_html, trace_html

CHUNK = Chunk("c1", "d1", "paper.pdf", 3, "Intro text. RNAseq data from 60 subjects (n=15 per group) were analysed.", 0)


def make_answer(passed=True, language="en", answer="60 subjects [paper.pdf, Page 3].",
                raw="60 subjects [S1].", provider="ollama"):
    ev = EvidenceCheck(passed=passed, score=0.81 if passed else 0.12, threshold=0.35, judge="laya", role="gate")
    trace = Trace(search_query="sample size of the study", rewritten=True, language=language, candidates=20,
                  papers=4, reranker="ms-marco", gate="laya", evidence=ev,
                  stage_ms={"search_ms": 12, "rerank_ms": 80, "gate_ms": 600, "generation_ms": 9600})
    r = RetrievedChunk(CHUNK, score=0.42, rank=1, rerank_score=0.9, gate_score=0.81)
    cites = [Citation("S1", "paper.pdf", 3, CHUNK.text, 0.42)] if passed else []
    return RAGAnswer(question="what about the sample size?", answer=answer, raw_answer=raw if passed else "",
                     citations=cites, retrieved=[r], insufficient_evidence=not passed, provider=provider,
                     model="llama3.1:8b", trace=trace)


def test_citation_labels_become_tags_and_html_is_escaped():
    out = answer_html(make_answer(answer="<script>x</script> 60 subjects [paper.pdf, Page 3]."))
    assert "&lt;script&gt;" in out and "<script>" not in out
    assert 'class="rm-cite"' in out and "p.&nbsp;3" in out


def test_arabic_answers_are_right_to_left():
    assert 'dir="rtl"' in answer_html(make_answer(language="ar"))
    assert 'dir="rtl"' not in answer_html(make_answer())


def test_trace_shows_each_step_and_the_rewritten_query():
    out = trace_html(make_answer())
    for step in ("Read", "Found", "Ranked", "Checked evidence", "Wrote"):
        assert step in out
    assert "sample size of the study" in out and "ms-marco" in out and "1 citation checked" in out
    assert 'class="pass"' in out and "width:81%" in out and "left:35%" in out


def test_trace_marks_a_refusal_and_the_skipped_llm():
    out = trace_html(make_answer(passed=False))
    assert 'class="refuse"' in out and "Nothing was sent to the language model" in out and 'class="skip"' in out
    assert "needed" in refusal_html(make_answer(passed=False))


def test_sources_highlight_the_closest_sentence():
    out = sources_html(make_answer())
    assert '<mark class="rm-hl">RNAseq data from 60 subjects (n=15 per group) were analysed.</mark>' in out
    assert "relevance" in out and "p. 3" in out


def test_page_slip_escapes_text_and_wraps_file_names():
    out = page_slip_html("a_b.pdf", 1, "x < y. Done.", "x < y.")
    assert "<mark class=\"rm-hl\">x &lt; y.</mark>" in out
    assert file_name("a_b.pdf") == "a_<wbr>b.<wbr>pdf"
