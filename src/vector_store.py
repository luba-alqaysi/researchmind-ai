"""FAISS vector index plus the chunk metadata needed for citations.

Design notes
------------
* ``IndexFlatIP`` performs exact search. For a handful of papers (thousands of
  chunks) this takes well under a millisecond, so approximate indexes (IVF,
  HNSW) would add complexity without benefit.
* Vector ID ``i`` always corresponds to ``self.chunks[i]``; the mapping is kept
  in one place and rebuilt together with the index, so it cannot drift.
* The embedding matrix is retained so documents can be removed by rebuilding
  the index without re-running the embedding model.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from src.schemas import Chunk, RetrievedChunk


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []
        self._vectors = np.zeros((0, dim), dtype="float32")

    # ------------------------------------------------------------------ write
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if len(chunks) == 0:
            return
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected vectors of shape (n, {self.dim}), got {vectors.shape}")
        self.index.add(vectors)
        self.chunks.extend(chunks)
        self._vectors = np.vstack([self._vectors, vectors])

    def remove_document(self, doc_id: str) -> int:
        keep = [i for i, c in enumerate(self.chunks) if c.doc_id != doc_id]
        removed = len(self.chunks) - len(keep)
        if removed:
            self._rebuild([self.chunks[i] for i in keep], self._vectors[keep])
        return removed

    def clear(self) -> None:
        self._rebuild([], np.zeros((0, self.dim), dtype="float32"))

    def _rebuild(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self.index = faiss.IndexFlatIP(self.dim)
        self.chunks, self._vectors = [], np.zeros((0, self.dim), dtype="float32")
        self.add(chunks, vectors)

    # ------------------------------------------------------------------- read
    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def doc_ids(self) -> set[str]:
        return {c.doc_id for c in self.chunks}

    def search(self, query_vector: np.ndarray, k: int = 5, doc_ids: set[str] | None = None) -> list[RetrievedChunk]:
        """Return the top-k chunks by cosine similarity, optionally within selected documents.

        With a document filter we search every vector and filter afterwards. On a
        flat index this is still exact and cheap, and it guarantees k results from
        the selected documents (a fixed over-fetch could return fewer).
        """
        if len(self) == 0 or k <= 0:
            return []
        query = np.ascontiguousarray(query_vector, dtype="float32").reshape(1, -1)
        fetch = len(self) if doc_ids else min(k, len(self))
        scores, ids = self.index.search(query, fetch)
        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            if doc_ids and chunk.doc_id not in doc_ids:
                continue
            results.append(RetrievedChunk(chunk=chunk, score=float(score), rank=len(results) + 1))
            if len(results) == k:
                break
        return results

    # ------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        np.save(directory / "vectors.npy", self._vectors)
        (directory / "chunks.json").write_text(
            json.dumps({"dim": self.dim, "chunks": [c.to_dict() for c in self.chunks]})
        )

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        directory = Path(directory)
        meta = json.loads((directory / "chunks.json").read_text())
        store = cls(meta["dim"])
        store.index = faiss.read_index(str(directory / "index.faiss"))
        store.chunks = [Chunk(**c) for c in meta["chunks"]]
        store._vectors = np.load(directory / "vectors.npy")
        if store.index.ntotal != len(store.chunks):
            raise ValueError("Corrupt index: vector count does not match chunk metadata")
        return store
