"""End-to-end orchestration: ingestion, retrieval, grounded generation, citations.

The pipeline owns the embedder, the vector store and the document registry.
The Streamlit app keeps one instance in ``st.session_state``; the evaluation
script and tests use it directly. Nothing here imports Streamlit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.citations import make_citation, resolve_citations
from src.config import Settings
from src.llm_client import ExtractiveClient, LLMClient
from src.pdf_processor import PDFProcessingError, extract_pdf
from src.prompts import (
    COMPARISON_ASPECTS,
    COMPARISON_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_comparison_prompt,
    build_qa_prompt,
)
from src.reranker import Reranker
from src.schemas import RAGAnswer, RetrievedChunk
from src.text_chunker import chunk_document
from src.vector_store import VectorStore


def _rank_key(r: RetrievedChunk) -> float:
    return r.rerank_score if r.rerank_score is not None else r.score


@dataclass
class DocumentInfo:
    doc_id: str
    doc_name: str
    page_count: int
    chunk_count: int
    warnings: list[str] = field(default_factory=list)


@dataclass
class IngestResult:
    file_name: str
    status: str  # "indexed" | "duplicate" | "scanned" | "no_text" | "failed"
    message: str
    document: DocumentInfo | None = None

    @property
    def ok(self) -> bool:
        return self.status == "indexed"


class RAGPipeline:
    def __init__(self, settings: Settings, embedder, reranker: Reranker | None = None,
                 gate: Reranker | None = None):
        self.settings = settings
        self.embedder = embedder
        self.reranker = reranker  # reorders the candidates
        self.gate = gate  # optional separate model that judges the final passages for the evidence check
        self.store = VectorStore(embedder.dim)
        self.documents: dict[str, DocumentInfo] = {}

    # ------------------------------------------------------------- ingestion
    def ingest_pdf(self, data: bytes, file_name: str) -> IngestResult:
        try:
            doc = extract_pdf(data, file_name)
        except PDFProcessingError as exc:
            return IngestResult(file_name, "failed", str(exc))

        if doc.doc_id in self.documents:
            existing = self.documents[doc.doc_id].doc_name
            return IngestResult(file_name, "duplicate", f"Already indexed as {existing}.", self.documents[doc.doc_id])
        if doc.is_scanned:
            return IngestResult(file_name, "scanned", " ".join(doc.warnings))

        # Two different files with the same name would make citations ambiguous.
        doc.doc_name = self._unique_name(doc.doc_name)
        chunks = chunk_document(
            doc, self.settings.chunk_size, self.settings.chunk_overlap, self.settings.skip_references
        )
        if not chunks:
            return IngestResult(file_name, "no_text", " ".join(doc.warnings) or "No extractable text found.")

        vectors = self.embedder.embed_documents([c.text for c in chunks])
        self.store.add(chunks, vectors)
        info = DocumentInfo(doc.doc_id, doc.doc_name, doc.page_count, len(chunks), doc.warnings)
        self.documents[doc.doc_id] = info
        return IngestResult(file_name, "indexed", f"{doc.page_count} pages, {len(chunks)} chunks.", info)

    def _unique_name(self, name: str) -> str:
        taken = {d.doc_name for d in self.documents.values()}
        if name not in taken:
            return name
        stem, dot, ext = name.rpartition(".")
        stem, ext = (stem, f".{ext}") if dot else (name, "")
        n = 2
        while f"{stem} ({n}){ext}" in taken:
            n += 1
        return f"{stem} ({n}){ext}"

    def remove_document(self, doc_id: str) -> None:
        self.store.remove_document(doc_id)
        self.documents.pop(doc_id, None)

    def clear(self) -> None:
        self.store.clear()
        self.documents.clear()

    # ------------------------------------------------------------- retrieval
    def retrieve(self, question: str, doc_ids: set[str] | None = None, top_k: int | None = None) -> list[RetrievedChunk]:
        """Top-k semantic search. When several documents are in scope, each one
        whose best chunk clears the similarity threshold is guaranteed at least one
        slot, so cross-document questions are not dominated by a single paper."""
        k = top_k or self.settings.top_k
        query = self.embedder.embed_query(question)
        if self.judge is not None:
            return self._retrieve_reranked(question, query, doc_ids, k)
        results = self.store.search(query, k, doc_ids=doc_ids)
        scope = doc_ids or self.store.doc_ids
        if self.settings.balance_documents and len(scope) > 1:
            present = {r.chunk.doc_id for r in results}
            for doc_id in scope - present:
                best = self.store.search(query, 1, doc_ids={doc_id})
                if best and best[0].score >= self.settings.min_similarity:
                    results.append(best[0])
            results.sort(key=lambda r: r.score, reverse=True)
            for rank, r in enumerate(results, start=1):
                r.rank = rank
        return results

    @property
    def judge(self) -> Reranker | None:
        """The model whose score decides whether there is enough evidence: the gate if set, else the reranker."""
        return self.gate or self.reranker

    def _retrieve_reranked(self, question: str, query, doc_ids: set[str] | None, k: int) -> list[RetrievedChunk]:
        """Embedding search, then an optional reranker reorders the candidates and the top k are kept.

        The judge (gate, or the reranker itself) scores the passages that may be shown to the LLM;
        its score drives the evidence check and the per-document guarantee. With a separate gate,
        only the final passages are judged, so a slow but well-calibrated gate stays affordable."""
        n = max(k, self.settings.rerank_candidates) if self.reranker is not None else k
        candidates = self.store.search(query, n, doc_ids=doc_ids)
        scope = doc_ids or self.store.doc_ids
        balance = self.settings.balance_documents and len(scope) > 1
        if balance:  # make sure every document in scope has at least one candidate
            present = {c.chunk.doc_id for c in candidates}
            for doc_id in scope - present:
                candidates += self.store.search(query, 1, doc_ids={doc_id})
        if not candidates:
            return []
        if self.reranker is not None:
            for c, s in zip(candidates, self.reranker.score(question, [c.chunk.text for c in candidates])):
                c.rerank_score = s
        candidates.sort(key=_rank_key, reverse=True)
        results = candidates[:k]
        extra = []  # each missing document's best candidate, admitted only if the judge finds it relevant
        if balance:
            present = {r.chunk.doc_id for r in results}
            for doc_id in scope - present:
                best = next((c for c in candidates if c.chunk.doc_id == doc_id), None)
                if best is not None:
                    extra.append(best)
        judged = results + extra
        if self.gate is not None:
            for c, s in zip(judged, self.gate.score(question, [c.chunk.text for c in judged])):
                c.gate_score = s
        else:
            for c in judged:
                c.gate_score = c.rerank_score
        results += [c for c in extra if c.gate_score >= self.settings.rerank_min_score]
        results.sort(key=_rank_key, reverse=True)
        for rank, r in enumerate(results, start=1):
            r.rank = rank
        return results

    def evidence_check(self, retrieved: list[RetrievedChunk]) -> tuple[bool, str]:
        """Layer 1 of insufficient-evidence handling: (enough evidence?, explanation).
        Uses the judge's relevance score when a reranker or gate is active, otherwise cosine similarity."""
        if not retrieved:
            return False, "no passages were found"
        if self.judge is not None:
            best, threshold = max(r.gate_score for r in retrieved), self.settings.rerank_min_score
            role = "gate" if self.gate is not None else "reranker"
            detail = f"best passage relevance {best:.2f} from the {self.judge.name} {role}, threshold {threshold:.2f}"
        else:
            best, threshold = retrieved[0].score, self.settings.min_similarity
            detail = f"best passage similarity {best:.2f}, threshold {threshold:.2f}"
        return best >= threshold, detail

    def _fit_context(self, retrieved: list[RetrievedChunk]) -> tuple[list[RetrievedChunk], list[str]]:
        kept, total = [], 0
        for r in retrieved:
            if kept and total + len(r.chunk.text) > self.settings.max_context_chars:
                break
            kept.append(r)
            total += len(r.chunk.text)
        warnings = []
        if len(kept) < len(retrieved):
            warnings.append(f"Only the top {len(kept)} of {len(retrieved)} passages fit in the context limit.")
        return kept, warnings

    # ------------------------------------------------------------ answering
    def ask(
        self,
        question: str,
        llm: LLMClient,
        doc_ids: set[str] | None = None,
        top_k: int | None = None,
    ) -> RAGAnswer:
        question = question.strip()
        if not question:
            raise ValueError("Please enter a question.")
        base = dict(question=question, provider=llm.provider, model=llm.model)

        if len(self.store) == 0:
            return RAGAnswer(
                answer="No documents are indexed yet. Upload one or more PDFs first.",
                raw_answer="", citations=[], retrieved=[], insufficient_evidence=True, **base,
            )

        t0 = time.perf_counter()
        retrieved = self.retrieve(question, doc_ids, top_k)
        retrieval_ms = (time.perf_counter() - t0) * 1000
        timings = {"retrieval_ms": retrieval_ms}

        # Layer 1: if nothing is even moderately relevant, do not ask the LLM at all.
        enough, detail = self.evidence_check(retrieved)
        if not enough:
            return RAGAnswer(
                answer=(
                    "**Insufficient evidence.** The uploaded documents do not appear to contain information "
                    f"relevant to this question ({detail}). Try rephrasing or selecting other documents."
                ),
                raw_answer="", citations=[], retrieved=retrieved, insufficient_evidence=True,
                timings=timings, **base,
            )

        retrieved, warnings = self._fit_context(retrieved)

        if isinstance(llm, ExtractiveClient):
            return self._extractive_answer(retrieved, timings, warnings, base)

        t1 = time.perf_counter()
        response = llm.generate(SYSTEM_PROMPT, build_qa_prompt(question, retrieved))
        timings["generation_ms"] = (time.perf_counter() - t1) * 1000
        timings["total_ms"] = retrieval_ms + timings["generation_ms"]

        # Layer 2: the model may itself report insufficient evidence.
        result = resolve_citations(response.text, retrieved)
        if response.truncated:
            warnings.append("The answer was cut off by the output token limit.")
        if result.invalid_ids:
            warnings.append(f"Removed citations to non-existent sources: {', '.join(result.invalid_ids)}.")
        if not result.citations and not result.insufficient:
            warnings.append("The answer contains no source citations; verify it against the passages below.")

        return RAGAnswer(
            answer=result.text, raw_answer=response.text, citations=result.citations, retrieved=retrieved,
            insufficient_evidence=result.insufficient, invalid_source_ids=result.invalid_ids,
            timings=timings, warnings=warnings, **base,
        )

    def _extractive_answer(self, retrieved, timings, warnings, base) -> RAGAnswer:
        top = retrieved[:3]
        lines = ["*Generation is turned off (extractive mode). The most relevant passages are:*", ""]
        citations = []
        for i, r in enumerate(top):
            citation = make_citation(i, r)
            citations.append(citation)
            lines.append(f"- “{citation.excerpt}” {citation.label}")
        timings["total_ms"] = timings["retrieval_ms"]
        return RAGAnswer(
            answer="\n".join(lines), raw_answer="", citations=citations, retrieved=retrieved,
            insufficient_evidence=False, timings=timings, warnings=warnings, **base,
        )

    # ------------------------------------------------------------ comparison
    def _abstract_chunks(self, doc_id: str, query_vectors, max_chunks: int = 2) -> list[RetrievedChunk]:
        """Chunks from the abstract (heuristic: the first early chunk mentioning 'abstract',
        otherwise the first chunks of the document), scored by their best similarity to
        any of the comparison queries."""
        positions = [i for i, c in enumerate(self.store.chunks) if c.doc_id == doc_id]
        start = next(
            (j for j, i in enumerate(positions)
             if self.store.chunks[i].page_number <= 3 and "abstract" in self.store.chunks[i].text.lower()),
            0,
        )
        return [
            RetrievedChunk(
                chunk=self.store.chunks[i],
                score=max(self.store.similarity(i, v) for v in query_vectors),
                rank=0,
            )
            for i in positions[start:start + max_chunks]
        ]

    def compare(self, doc_a: str, doc_b: str, llm: LLMClient, per_aspect_k: int = 2) -> RAGAnswer:
        """Structured comparison of two documents across fixed research aspects."""
        if doc_a == doc_b:
            raise ValueError("Select two different documents to compare.")
        names = {d: self.documents[d].doc_name for d in (doc_a, doc_b)}
        base = dict(question=f"Compare {names[doc_a]} and {names[doc_b]}", provider=llm.provider, model=llm.model)

        t0 = time.perf_counter()
        seen: dict[str, RetrievedChunk] = {}
        # The abstract summarises objective, design, data and findings, so always include it.
        vectors = [self.embedder.embed_query(q) for q in COMPARISON_ASPECTS.values()]
        for doc_id in (doc_a, doc_b):
            for r in self._abstract_chunks(doc_id, vectors):
                seen[r.chunk.chunk_id] = r
        for vector in vectors:
            for doc_id in (doc_a, doc_b):
                for r in self.store.search(vector, per_aspect_k, doc_ids={doc_id}):
                    if r.chunk.chunk_id not in seen or r.score > seen[r.chunk.chunk_id].score:
                        seen[r.chunk.chunk_id] = r
        # Group by paper so the model sees Paper A's evidence, then Paper B's.
        retrieved = sorted(seen.values(), key=lambda r: (r.chunk.doc_id != doc_a, r.chunk.page_number))
        for rank, r in enumerate(retrieved, start=1):
            r.rank = rank
        timings = {"retrieval_ms": (time.perf_counter() - t0) * 1000}
        retrieved, warnings = self._fit_context(retrieved)

        if isinstance(llm, ExtractiveClient):
            return RAGAnswer(
                answer="Paper comparison needs an LLM provider. Select Ollama, Gemini or Claude in the sidebar.",
                raw_answer="", citations=[], retrieved=retrieved, insufficient_evidence=True,
                timings=timings, **base,
            )

        t1 = time.perf_counter()
        response = llm.generate(COMPARISON_SYSTEM_PROMPT, build_comparison_prompt(names[doc_a], names[doc_b], retrieved))
        timings["generation_ms"] = (time.perf_counter() - t1) * 1000
        timings["total_ms"] = timings["retrieval_ms"] + timings["generation_ms"]
        result = resolve_citations(response.text, retrieved)
        if response.truncated:
            warnings.append("The comparison was cut off by the output token limit.")
        if result.invalid_ids:
            warnings.append(f"Removed citations to non-existent sources: {', '.join(result.invalid_ids)}.")
        return RAGAnswer(
            answer=result.text, raw_answer=response.text, citations=result.citations, retrieved=retrieved,
            insufficient_evidence=result.insufficient, invalid_source_ids=result.invalid_ids,
            timings=timings, warnings=warnings, **base,
        )
