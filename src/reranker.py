"""Optional second-stage rerankers.

First-stage search compares one embedding of the question with one embedding
of each passage. That is fast but blunt: a short question such as "what about
the sample size?" scores low against every passage, and a passage that mixes
one useful sentence with table fragments scores low for any question.

A reranker reads the question and one candidate passage together and returns a
relevance score in [0, 1]. With a reranker active, the pipeline fetches
``rerank_candidates`` passages by embedding search, reranks them, keeps the
best ``top_k``, and uses the best reranker score (instead of cosine similarity)
for the insufficient-evidence check.

Available rerankers
-------------------
* ``ms-marco``  cross-encoder/ms-marco-MiniLM-L6-v2 (22M parameters), a
  cross-encoder trained for search relevance on MS MARCO.
* ``bge``       BAAI/bge-reranker-base (278M), a cross-encoder reranker.
* ``laya``      convaiinnovations/laya (421M), a typed-decision model. Each
  passage gets one yes/no ("noul") question and P(true) is the score.

Which one works best here is measured, not assumed: see
``python evaluation/evaluate_rag.py rerank`` and evaluation/RESULTS.md.
"""

from __future__ import annotations

import os
from typing import Protocol

RERANKERS: dict[str, str] = {
    "ms-marco": "cross-encoder/ms-marco-MiniLM-L6-v2",
    "bge": "BAAI/bge-reranker-base",
    "laya": "convaiinnovations/laya",
}

LAYA_QUESTION = {
    "relevant": {
        "type": "noul",
        "instructions": "Does the passage contain information that answers the question?",
    }
}


class Reranker(Protocol):
    name: str
    model: str

    def score(self, question: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, name: str, model: str, batch_size: int = 16):
        import torch
        from sentence_transformers import CrossEncoder

        self.name, self.model, self.batch_size = name, model, batch_size
        self._encoder = CrossEncoder(model)
        self._sigmoid = torch.nn.Sigmoid()

    def score(self, question: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        scores = self._encoder.predict(
            [(question, p) for p in passages], batch_size=self.batch_size,
            activation_fn=self._sigmoid, show_progress_bar=False,
        )
        return [float(s) for s in scores]


class LayaReranker:
    def __init__(self, name: str = "laya", model: str = RERANKERS["laya"], batch_size: int = 8):
        # Laya's docs: transformers probes for TensorFlow at import, which can deadlock.
        os.environ.setdefault("USE_TF", "0")
        import laya

        self.name, self.model, self.batch_size = name, model, batch_size
        self._agent = laya.load(model)

    def score(self, question: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        states = [{"question": question, "passage": p} for p in passages]
        results = self._agent.predict_batch(states, LAYA_QUESTION, batch_size=self.batch_size)
        return [float(r["answers"]["relevant"]["noul"]) for r in results]


def get_reranker(name: str) -> Reranker | None:
    """Return a loaded reranker, or None for ``"none"``."""
    name = (name or "none").lower()
    if name == "none":
        return None
    if name not in RERANKERS:
        raise ValueError(f"Unknown reranker '{name}'. Choose one of: none, {', '.join(RERANKERS)}.")
    if name == "laya":
        return LayaReranker()
    return CrossEncoderReranker(name, RERANKERS[name])
