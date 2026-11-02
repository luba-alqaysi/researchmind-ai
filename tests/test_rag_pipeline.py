import pytest

from src.config import Settings
from src.llm_client import ExtractiveClient, LLMResponse
from src.rag_pipeline import RAGPipeline
from tests.conftest import make_pdf, make_scanned_pdf


class FakeLLM:
    provider = "fake"
    model = "fake-1"

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = []

    def generate(self, system: str, user: str) -> LLMResponse:
        self.calls.append((system, user))
        return LLMResponse(text=self.reply, model=self.model)


@pytest.fixture
def pipeline(fake_embedder):
    settings = Settings(chunk_size=300, chunk_overlap=50, top_k=3, min_similarity=0.15, skip_references=False)
    p = RAGPipeline(settings, fake_embedder)
    p.ingest_pdf(
        make_pdf(["Methods. We performed bulk RNA sequencing on 24 mouse liver samples using Illumina.",
                  "Results. DESeq2 identified 312 differentially expressed genes."]),
        "rna.pdf",
    )
    p.microbiome_bytes = make_pdf(["Gut microbiome samples were collected daily from mice for metagenomic sequencing."])
    p.ingest_pdf(p.microbiome_bytes, "microbiome.pdf")
    return p


def test_ingest_reports_status(pipeline):
    assert len(pipeline.documents) == 2
    names = sorted(d.doc_name for d in pipeline.documents.values())
    assert names == ["microbiome.pdf", "rna.pdf"]


def test_duplicate_upload_is_not_reindexed(pipeline):
    before = len(pipeline.store)
    result = pipeline.ingest_pdf(pipeline.microbiome_bytes, "copy.pdf")  # same bytes, new name
    assert result.status == "duplicate" and len(pipeline.store) == before


def test_same_name_different_content_gets_unique_name(pipeline):
    result = pipeline.ingest_pdf(make_pdf(["A completely different paper about protein folding."]), "rna.pdf")
    assert result.ok and result.document.doc_name == "rna (2).pdf"


def test_scanned_and_corrupt_files(pipeline):
    assert pipeline.ingest_pdf(make_scanned_pdf(), "scan.pdf").status == "scanned"
    assert pipeline.ingest_pdf(b"garbage" * 50, "bad.pdf").status == "failed"
    assert len(pipeline.documents) == 2


def test_ask_produces_verified_citations(pipeline):
    llm = FakeLLM("DESeq2 found 312 genes [S1].")
    answer = pipeline.ask("How many genes did DESeq2 identify?", llm, top_k=2)
    system, user = llm.calls[0]
    assert "untrusted" in system and "DESeq2 identified 312" in user
    top = answer.retrieved[0].chunk
    assert answer.answer == f"DESeq2 found 312 genes [{top.doc_name}, Page {top.page_number}]."
    assert answer.citations[0].doc_name == "rna.pdf" and answer.citations[0].page_number == 2
    assert not answer.insufficient_evidence
    assert "retrieval_ms" in answer.timings and "generation_ms" in answer.timings


def test_document_filter_restricts_evidence(pipeline):
    micro_id = next(d for d, info in pipeline.documents.items() if info.doc_name == "microbiome.pdf")
    llm = FakeLLM("Answer [S1].")
    answer = pipeline.ask("How many genes did DESeq2 identify?", llm, doc_ids={micro_id}, top_k=3)
    assert answer.retrieved and all(r.chunk.doc_id == micro_id for r in answer.retrieved)


def test_cross_document_retrieval_includes_each_relevant_document(pipeline):
    answer = pipeline.ask("Which samples were collected from mice for sequencing?", FakeLLM("x [S1]"), top_k=1)
    assert {r.chunk.doc_name for r in answer.retrieved} == {"rna.pdf", "microbiome.pdf"}


def test_low_similarity_skips_llm(pipeline):
    llm = FakeLLM("should not be called")
    answer = pipeline.ask("Quarterly revenue of the airline industry", llm)
    assert answer.insufficient_evidence and llm.calls == []
    assert "Insufficient evidence" in answer.answer


def test_llm_reported_insufficient_evidence(pipeline):
    answer = pipeline.ask("What funding supported the RNA sequencing?", FakeLLM("INSUFFICIENT_EVIDENCE: no funding info."))
    assert answer.insufficient_evidence and answer.answer.startswith("**Insufficient evidence.**")


def test_uncited_and_invalid_citations_warn(pipeline):
    a = pipeline.ask("How many genes did DESeq2 identify?", FakeLLM("312 genes."))
    assert any("no source citations" in w for w in a.warnings)
    b = pipeline.ask("How many genes did DESeq2 identify?", FakeLLM("312 genes [S42]."))
    assert b.invalid_source_ids == ["S42"] and any("non-existent" in w for w in b.warnings)


def test_extractive_mode_needs_no_llm(pipeline):
    answer = pipeline.ask("How many genes did DESeq2 identify?", ExtractiveClient())
    assert answer.citations and "extractive mode" in answer.answer
    assert "[rna.pdf, Page 2]" in answer.answer


def test_empty_index_and_empty_question(fake_embedder):
    p = RAGPipeline(Settings(), fake_embedder)
    assert p.ask("anything?", FakeLLM("x")).insufficient_evidence
    with pytest.raises(ValueError):
        p.ask("   ", FakeLLM("x"))


def test_remove_and_clear(pipeline):
    doc_id = next(iter(pipeline.documents))
    pipeline.remove_document(doc_id)
    assert doc_id not in pipeline.documents and doc_id not in pipeline.store.doc_ids
    pipeline.clear()
    assert len(pipeline.store) == 0 and pipeline.documents == {}


def test_compare_uses_both_documents(pipeline):
    ids = list(pipeline.documents)
    llm = FakeLLM("| Aspect | A | B |\n|---|---|---|\n| Main findings | 312 genes [S1] | Not reported in retrieved text |")
    answer = pipeline.compare(ids[0], ids[1], llm)
    assert {r.chunk.doc_id for r in answer.retrieved} == set(ids)
    assert "Paper A" in llm.calls[0][1] and answer.citations
    assert all(-1.0 <= r.score <= 1.0 for r in answer.retrieved)
    with pytest.raises(ValueError):
        pipeline.compare(ids[0], ids[0], llm)


def test_document_balancing_can_be_disabled(pipeline):
    pipeline.settings = pipeline.settings.with_overrides(balance_documents=False)
    answer = pipeline.ask("Which samples were collected from mice for sequencing?", FakeLLM("x [S1]"), top_k=1)
    assert len(answer.retrieved) == 1
