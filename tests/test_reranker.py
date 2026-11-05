"""Reranked retrieval and the reranker-based evidence check (no models are loaded)."""

import pytest

from src.config import Settings
from src.rag_pipeline import RAGPipeline
from src.reranker import get_reranker
from tests.conftest import make_pdf
from tests.test_rag_pipeline import FakeLLM


class KeywordReranker:
    """Scores 0.9 for passages containing the keyword, 0.05 otherwise, and records its calls."""

    name = "keyword"
    model = "fake"

    def __init__(self, keyword: str):
        self.keyword = keyword
        self.calls = []

    def score(self, question, passages):
        self.calls.append((question, len(passages)))
        return [0.9 if self.keyword in p else 0.05 for p in passages]


@pytest.fixture
def docs(fake_embedder):
    settings = Settings(chunk_size=300, chunk_overlap=50, top_k=1, skip_references=False,
                        rerank_candidates=10, rerank_min_score=0.5)
    p = RAGPipeline(settings, fake_embedder)
    p.ingest_pdf(make_pdf([
        "Introduction. Psychiatric disorders affect gene expression in the orbitofrontal cortex.",
        "Methods. RNAseq data from 60 subjects (n=15 per group) were analysed.",
    ]), "paper.pdf")
    p.ingest_pdf(make_pdf(["Gut microbiome samples were collected daily from mice."]), "mice.pdf")
    return p


def test_reranker_reorders_candidates(docs):
    docs.reranker = KeywordReranker("60 subjects")
    paper = {d for d, i in docs.documents.items() if i.doc_name == "paper.pdf"}
    retrieved = docs.retrieve("what about the sample size?", doc_ids=paper)
    assert "60 subjects" in retrieved[0].chunk.text
    assert retrieved[0].rerank_score == 0.9 and retrieved[0].rank == 1
    question, n_candidates = docs.reranker.calls[0]
    assert question == "what about the sample size?" and n_candidates == 2  # all chunks of the paper


def test_reranker_score_drives_the_evidence_check(docs):
    # Cosine similarity of an unrelated question is low, but the check now follows the reranker.
    docs.settings = docs.settings.with_overrides(min_similarity=0.99)
    docs.reranker = KeywordReranker("60 subjects")
    llm = FakeLLM("60 subjects [S1].")
    answer = docs.ask("what about the sample size?", llm)
    assert not answer.insufficient_evidence and llm.calls

    docs.reranker = KeywordReranker("no passage contains this")
    llm = FakeLLM("should not be called")
    answer = docs.ask("what about the sample size?", llm)
    assert answer.insufficient_evidence and llm.calls == []
    assert "keyword reranker" in answer.answer and "threshold 0.50" in answer.answer


def test_reranked_balancing_requires_the_reranker_cutoff(docs):
    docs.reranker = KeywordReranker("60 subjects")  # the mice paper scores 0.05 < 0.5
    retrieved = docs.retrieve("what about the sample size?")
    assert [r.chunk.doc_name for r in retrieved] == ["paper.pdf"]

    docs.reranker = KeywordReranker("e")  # every passage scores 0.9, so each paper gets a slot
    retrieved = docs.retrieve("what about the sample size?")
    assert {r.chunk.doc_name for r in retrieved} == {"paper.pdf", "mice.pdf"}


def test_without_reranker_cosine_check_is_unchanged(docs):
    docs.settings = docs.settings.with_overrides(min_similarity=0.99)
    answer = docs.ask("what about the sample size?", FakeLLM("x"))
    assert answer.insufficient_evidence and "best passage similarity" in answer.answer
    assert all(r.rerank_score is None for r in answer.retrieved)


def test_get_reranker_names():
    assert get_reranker("none") is None
    with pytest.raises(ValueError, match="Unknown reranker"):
        get_reranker("not-a-model")


def test_gate_judges_only_the_final_passages_and_drives_the_check(docs):
    docs.reranker = KeywordReranker("60 subjects")  # ranks the right chunk first
    docs.gate = KeywordReranker("no passage contains this")  # but finds nothing relevant
    paper = {d for d, i in docs.documents.items() if i.doc_name == "paper.pdf"}
    llm = FakeLLM("should not be called")
    answer = docs.ask("what about the sample size?", llm, doc_ids=paper)
    assert answer.insufficient_evidence and llm.calls == []
    assert "keyword gate" in answer.answer
    assert docs.gate.calls[0][1] == 1  # top_k=1: only the final passage is judged, not all candidates
    assert "60 subjects" in answer.retrieved[0].chunk.text  # ranking still follows the reranker

    docs.gate = KeywordReranker("60 subjects")
    answer = docs.ask("what about the sample size?", FakeLLM("60 subjects [S1]."), doc_ids=paper)
    assert not answer.insufficient_evidence and answer.retrieved[0].gate_score == 0.9


def test_gate_without_reranker_keeps_embedding_order(docs):
    docs.gate = KeywordReranker("60 subjects")
    paper = {d for d, i in docs.documents.items() if i.doc_name == "paper.pdf"}
    retrieved = docs.retrieve("what about the sample size?", doc_ids=paper)
    assert retrieved[0].rerank_score is None and retrieved[0].gate_score is not None
    assert docs.gate.calls[0][1] == 1


def test_gate_decides_the_per_document_guarantee(docs):
    docs.reranker = KeywordReranker("e")  # every passage looks relevant to the ranker
    docs.gate = KeywordReranker("60 subjects")  # only the paper's passage is relevant to the gate
    retrieved = docs.retrieve("what about the sample size?")
    assert [r.chunk.doc_name for r in retrieved] == ["paper.pdf"]


def test_trace_records_the_gate_decision_and_stage_times(docs):
    docs.reranker = KeywordReranker("60 subjects")
    docs.gate = KeywordReranker("60 subjects")
    paper = {d for d, i in docs.documents.items() if i.doc_name == "paper.pdf"}
    answer = docs.ask("what about the sample size?", FakeLLM("60 subjects [S1]."), doc_ids=paper)
    t = answer.trace
    assert t.search_query == "what about the sample size?" and not t.rewritten and t.language == "en"
    assert t.candidates == 2 and t.papers == 1 and t.reranker == "keyword" and t.gate == "keyword"
    assert t.evidence.passed and t.evidence.score == 0.9 and t.evidence.role == "gate"
    assert {"search_ms", "rerank_ms", "gate_ms", "generation_ms"} <= set(t.stage_ms)

    docs.gate = KeywordReranker("no passage contains this")
    answer = docs.ask("what about the sample size?", FakeLLM("not called"), doc_ids=paper)
    assert not answer.trace.evidence.passed and "generation_ms" not in answer.trace.stage_ms


def test_trace_without_reranker_uses_cosine(docs):
    answer = docs.ask("what about the sample size?", FakeLLM("60 subjects [S1]."))
    t = answer.trace
    assert t.evidence.role == "cosine" and t.evidence.judge == "cosine similarity"
    assert t.reranker is None and t.gate is None and t.papers == 2
