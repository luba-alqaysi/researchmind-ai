"""Application configuration loaded from environment variables / `.env`.

Keeping configuration in one dataclass means the UI, retrieval, and LLM code
never read environment variables directly, which makes them easy to test.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

from dotenv import load_dotenv

PROVIDERS = ("ollama", "gemini", "claude", "extractive")


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


@dataclass(frozen=True)
class Settings:
    # LLM
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:9b"
    ollama_num_ctx: int = 8192
    # Reasoning ("thinking") models such as Qwen3.5 think step by step before answering. Off by default:
    # the answer is grounded in retrieved passages, and thinking makes each answer several times slower.
    ollama_think: bool = False
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    anthropic_api_key: str = ""
    claude_model: str = "claude-opus-5-5"
    claude_effort: str = "medium"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2000
    llm_timeout_seconds: int = 180

    # Retrieval
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 5
    min_similarity: float = 0.30
    skip_references: bool = True
    # With several documents in scope, give each document whose best chunk clears
    # min_similarity at least one slot (helps cross-document questions).
    balance_documents: bool = True
    max_context_chars: int = 20000  # cap on evidence text sent to the LLM
    # Optional second-stage reranker (see src/reranker.py): "none", "ms-marco", "bge" or "laya".
    reranker: str = "none"
    rerank_candidates: int = 20  # passages fetched by embedding search before reranking
    # Optional separate relevance gate: a second model that only judges the final top-k passages
    # (e.g. reranker="ms-marco", rerank_gate="laya"). "none" means the reranker is also the gate.
    rerank_gate: str = "none"
    rerank_min_score: float = 0.0  # insufficient-evidence cutoff on the gate's (or reranker's) [0, 1] score
    # Rewrite follow-up and non-English questions into a standalone English search query (one extra LLM call).
    query_rewriting: bool = True

    @classmethod
    def from_env(cls, dotenv: bool = True) -> "Settings":
        if dotenv:
            load_dotenv(override=False)
        d = cls()
        provider = _get("LLM_PROVIDER", d.llm_provider).lower()
        if provider not in PROVIDERS:
            raise ValueError(f"LLM_PROVIDER must be one of {PROVIDERS}, got {provider!r}")
        return cls(
            llm_provider=provider,
            ollama_base_url=_get("OLLAMA_BASE_URL", d.ollama_base_url).rstrip("/"),
            ollama_model=_get("OLLAMA_MODEL", d.ollama_model),
            ollama_num_ctx=int(_get("OLLAMA_NUM_CTX", str(d.ollama_num_ctx))),
            ollama_think=_get_bool("OLLAMA_THINK", d.ollama_think),
            gemini_api_key=_get("GEMINI_API_KEY", ""),
            gemini_model=_get("GEMINI_MODEL", d.gemini_model),
            anthropic_api_key=_get("ANTHROPIC_API_KEY", ""),
            claude_model=_get("CLAUDE_MODEL", d.claude_model),
            claude_effort=_get("CLAUDE_EFFORT", d.claude_effort).lower(),
            llm_temperature=float(_get("LLM_TEMPERATURE", str(d.llm_temperature))),
            llm_max_tokens=int(_get("LLM_MAX_TOKENS", str(d.llm_max_tokens))),
            llm_timeout_seconds=int(_get("LLM_TIMEOUT_SECONDS", str(d.llm_timeout_seconds))),
            embedding_model=_get("EMBEDDING_MODEL", d.embedding_model),
            chunk_size=int(_get("CHUNK_SIZE", str(d.chunk_size))),
            chunk_overlap=int(_get("CHUNK_OVERLAP", str(d.chunk_overlap))),
            top_k=int(_get("TOP_K", str(d.top_k))),
            min_similarity=float(_get("MIN_SIMILARITY", str(d.min_similarity))),
            skip_references=_get_bool("SKIP_REFERENCES", d.skip_references),
            balance_documents=_get_bool("BALANCE_DOCUMENTS", d.balance_documents),
            max_context_chars=int(_get("MAX_CONTEXT_CHARS", str(d.max_context_chars))),
            reranker=_get("RERANKER", d.reranker).lower(),
            rerank_candidates=int(_get("RERANK_CANDIDATES", str(d.rerank_candidates))),
            rerank_gate=_get("RERANK_GATE", d.rerank_gate).lower(),
            rerank_min_score=float(_get("RERANK_MIN_SCORE", str(d.rerank_min_score))),
            query_rewriting=_get_bool("QUERY_REWRITING", d.query_rewriting),
        )

    def with_overrides(self, **kwargs) -> "Settings":
        return replace(self, **kwargs)

    def model_for(self, provider: str) -> str:
        return {
            "ollama": self.ollama_model,
            "gemini": self.gemini_model,
            "claude": self.claude_model,
            "extractive": "none",
        }[provider]

    def __repr__(self) -> str:  # never leak API keys into logs or the UI
        hidden = {"gemini_api_key", "anthropic_api_key"}
        fields = ", ".join(
            f"{k}={'***' if k in hidden and v else v!r}" for k, v in self.__dict__.items()
        )
        return f"Settings({fields})"
