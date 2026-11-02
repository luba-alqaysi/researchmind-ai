"""PDF text extraction with page-level metadata (PyMuPDF).

Design notes
------------
* Text is extracted page by page so every chunk can be cited by page number.
* We use PyMuPDF "blocks" in content-stream order rather than sorting by
  position: LaTeX two-column papers store text column by column, while sorting
  by (y, x) would interleave the two columns line by line.
* Scanned (image-only) PDFs have no text layer. OCR is out of scope for the MVP,
  so they are detected and reported instead of silently producing nothing.
"""

from __future__ import annotations

import hashlib
import re

import pymupdf

from src.schemas import DocumentResult, PageText

# A page with fewer characters than this but with images is treated as scanned.
MIN_CHARS_PER_TEXT_PAGE = 25

_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"}
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​﻿]")
# "effi-\ncient" -> "efficient" (only when a lowercase letter continues the word)
_HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*([a-z])")
_ARXIV_STAMP = re.compile(r"^arXiv:\d{4}\.\d{4,5}(v\d+)?\b")


class PDFProcessingError(Exception):
    """Raised when a file cannot be read as a PDF at all."""


def clean_text(text: str) -> str:
    """Normalise raw PDF text while keeping paragraph boundaries ("\\n\\n")."""
    for lig, repl in _LIGATURES.items():
        text = text.replace(lig, repl)
    text = text.replace("­", "")  # soft hyphen
    text = _CONTROL_CHARS.sub("", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    paragraphs = re.split(r"\n\s*\n", text)
    cleaned = []
    for para in paragraphs:
        para = re.sub(r"\s+", " ", para).strip()
        if para:
            cleaned.append(para)
    return "\n\n".join(cleaned)


def _page_text(page: pymupdf.Page) -> str:
    blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
    parts = []
    for block in blocks:
        if block[6] != 0:  # 1 = image block
            continue
        text = block[4].strip()
        if not text or _ARXIV_STAMP.match(text):  # drop the vertical arXiv margin stamp
            continue
        if text.isdigit() and len(text) <= 4:  # bare page number
            continue
        parts.append(text)
    return clean_text("\n\n".join(parts))


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_pdf(data: bytes, doc_name: str) -> DocumentResult:
    """Extract cleaned text for every page of a PDF given as raw bytes.

    Raises PDFProcessingError for corrupt, encrypted, or empty files. Scanned
    PDFs do *not* raise; they are returned with ``is_scanned=True`` and a warning
    so the UI can explain why they were not indexed.
    """
    if not data:
        raise PDFProcessingError(f"{doc_name}: the file is empty.")
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several exception types
        raise PDFProcessingError(f"{doc_name}: not a readable PDF ({exc}).") from exc

    try:
        if doc.needs_pass:
            raise PDFProcessingError(f"{doc_name}: the PDF is password-protected.")
        if doc.page_count == 0:
            raise PDFProcessingError(f"{doc_name}: the PDF has no pages.")

        pages: list[PageText] = []
        for index, page in enumerate(doc):
            try:
                text = _page_text(page)
                has_images = bool(page.get_images(full=False))
            except Exception:  # one damaged page should not sink the document
                text, has_images = "", False
            pages.append(PageText(page_number=index + 1, text=text, has_images=has_images))
    finally:
        doc.close()

    result = DocumentResult(
        doc_id=file_hash(data),
        doc_name=doc_name,
        pages=pages,
        page_count=len(pages),
    )

    image_only = [p for p in pages if len(p.text) < MIN_CHARS_PER_TEXT_PAGE and p.has_images]
    text_pages = [p for p in pages if len(p.text) >= MIN_CHARS_PER_TEXT_PAGE]

    if not text_pages:
        if image_only:
            result.is_scanned = True
            result.warnings.append(
                "This PDF appears to be scanned (images without a text layer). "
                "OCR is not supported in this version, so it cannot be indexed."
            )
        else:
            result.warnings.append("No extractable text was found in this PDF.")
    elif len(image_only) >= max(1, len(pages) // 2):
        result.warnings.append(
            f"{len(image_only)} of {len(pages)} pages look scanned (no text layer); "
            "only pages with real text were indexed."
        )
    elif image_only:
        pages_list = ", ".join(str(p.page_number) for p in image_only[:10])
        result.warnings.append(f"Pages without extractable text were skipped: {pages_list}.")

    return result
