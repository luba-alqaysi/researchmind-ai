"""Sentence-embedding wrapper.

The same model embeds document chunks and user questions, so both live in the
same vector space. Vectors are L2-normalised, which makes the inner product
used by FAISS equal to cosine similarity.

``encode_query`` / ``encode_document`` (sentence-transformers >= 5) apply any
query/document prompts a model defines (e.g. BGE or E5 models), so switching
``EMBEDDING_MODEL`` does not require code changes.
"""

from __future__ import annotations

import numpy as np


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None):
        # Imported lazily: loading torch is slow and unnecessary for most unit tests.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = int(get_dim())

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        vectors = self.model.encode_document(
            texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        vector = self.model.encode_query(
            [text], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return np.asarray(vector[0], dtype="float32")
