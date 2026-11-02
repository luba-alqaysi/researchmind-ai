"""Plain data containers shared across the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class PageText:
    page_number: int  # 1-based, as printed in PDF viewers
    text: str
    has_images: bool = False


@dataclass
class DocumentResult:
    doc_id: str  # SHA-256 of the file bytes (stable across re-uploads)
    doc_name: str
    pages: list[PageText]
    page_count: int
    warnings: list[str] = field(default_factory=list)
    is_scanned: bool = False

    @property
    def text_pages(self) -> list[PageText]:
        return [p for p in self.pages if p.text.strip()]

    @property
    def char_count(self) -> int:
        return sum(len(p.text) for p in self.pages)


@dataclass
class Chunk:
    chunk_id: str  # "<doc_id[:12]>-p<page>-c<n>"
    doc_id: str
    doc_name: str
    page_number: int
    text: str
    chunk_index: int  # position within the page

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float  # cosine similarity in [-1, 1]
    rank: int  # 1-based


@dataclass
class Citation:
    source_id: str  # "S1", "S2", ... as shown to the LLM
    doc_name: str
    page_number: int
    excerpt: str
    score: float

    @property
    def label(self) -> str:
        return f"[{self.doc_name}, Page {self.page_number}]"


@dataclass
class RAGAnswer:
    question: str
    answer: str  # markdown with verified citations rendered
    raw_answer: str  # exactly what the LLM returned
    citations: list[Citation]  # sources the answer actually cites
    retrieved: list[RetrievedChunk]  # everything shown to the LLM
    insufficient_evidence: bool
    provider: str
    model: str
    invalid_source_ids: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)  # milliseconds
    warnings: list[str] = field(default_factory=list)
