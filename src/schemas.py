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
    rerank_score: float | None = None  # reranker relevance in [0, 1], when a reranker is active
    gate_score: float | None = None  # relevance used by the evidence check, when a reranker or gate is active


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
class EvidenceCheck:
    """Layer-1 decision: is there enough evidence to call the LLM?"""

    passed: bool
    score: float | None  # best relevance (gate/reranker) or cosine similarity; None if nothing was retrieved
    threshold: float
    judge: str  # model that produced the score, or "cosine similarity"
    role: str  # "gate" | "reranker" | "cosine"

    @property
    def detail(self) -> str:
        if self.score is None:
            return "no passages were found"
        if self.role == "cosine":
            return f"best passage similarity {self.score:.2f}, threshold {self.threshold:.2f}"
        return (f"best passage relevance {self.score:.2f} from the {self.judge} {self.role}, "
                f"threshold {self.threshold:.2f}")


@dataclass
class Trace:
    """What the pipeline did for one question, for display and debugging."""

    search_query: str  # the text actually searched (the rewritten query when rewriting happened)
    rewritten: bool = False
    language: str = "en"  # language of the question and the answer: "en" | "ar"
    candidates: int = 0  # passages scored before the top-k cut
    papers: int = 0  # documents in scope
    reranker: str | None = None
    gate: str | None = None
    evidence: EvidenceCheck | None = None
    stage_ms: dict[str, float] = field(default_factory=dict)  # rewrite, search, rerank, gate, generation


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
    trace: Trace | None = None
