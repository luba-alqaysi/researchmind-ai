"""Shared fixtures. All PDFs are generated on the fly, so tests need no external files."""

from __future__ import annotations

import numpy as np
import pymupdf
import pytest


def make_pdf(pages: list[str], fontsize: int = 10) -> bytes:
    """Create a text PDF with one string per page."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        rect = pymupdf.Rect(50, 50, page.rect.width - 50, page.rect.height - 50)
        page.insert_textbox(rect, text, fontsize=fontsize)
    data = doc.tobytes()
    doc.close()
    return data


def make_scanned_pdf(n_pages: int = 2) -> bytes:
    """Create a PDF whose pages contain only an image (no text layer)."""
    doc = pymupdf.open()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 60, 60), False)
    pix.set_rect(pix.irect, (120, 120, 120))
    for _ in range(n_pages):
        page = doc.new_page()
        page.insert_image(pymupdf.Rect(50, 50, 300, 300), pixmap=pix)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def two_page_pdf() -> bytes:
    return make_pdf(
        [
            "Methods. We performed bulk RNA sequencing on 24 mouse liver samples. "
            "Reads were aligned with STAR and counted with featureCounts.",
            "Results. Differential expression analysis with DESeq2 identified 312 genes. "
            "The main limitation is the small sample size.",
        ]
    )


class FakeEmbedder:
    """Deterministic bag-of-words embedder so retrieval tests run without a model download."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.model_name = "fake-bow"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        for token in text.lower().replace(".", " ").replace(",", " ").replace("?", " ").split():
            v[hash(token) % self.dim] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        return np.vstack([self._vec(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()
