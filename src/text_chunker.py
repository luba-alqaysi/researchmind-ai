"""Split extracted pages into overlapping, citation-safe chunks.

Design notes
------------
* **Page-bounded:** a chunk never spans two pages (or two documents), so the
  page number attached to every chunk is always exact. The trade-off is that a
  paragraph continuing across a page break is split into two chunks.
* **Sentence-aware:** chunks are packed from whole sentences up to
  ``chunk_size`` characters, so retrieved excerpts read naturally. Sentences
  longer than ``chunk_size`` are split on word boundaries.
* **Overlap:** the trailing sentences of a chunk (up to ``overlap`` characters)
  are repeated at the start of the next one, so a fact near a boundary is still
  retrievable with its context.
* **Reference lists:** bibliography entries match many queries lexically but
  rarely answer them. ``skip_references`` drops text after a "References"
  heading until an "Appendix"/"Supplementary" heading appears.
"""

from __future__ import annotations

import re

from src.schemas import Chunk, DocumentResult

MIN_CHUNK_CHARS = 20

_ABBREVIATIONS = {
    "al", "e.g", "i.e", "fig", "figs", "eq", "eqs", "ref", "refs", "sec", "no",
    "vs", "etc", "approx", "cf", "dr", "mr", "ms", "prof", "st", "tab", "vol",
    "pp", "ed", "eds", "resp", "suppl", "ca", "inc", "ltd", "jan", "feb", "mar",
    "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")
_REFERENCES_HEADING = re.compile(
    r"^(\d+\.?\s*)?(references|bibliography|literature cited|works cited)\s*:?$", re.I
)
# Some styles (e.g. REVTeX) have no heading: the list simply starts with "[1] A. Author".
_FIRST_REFERENCE = re.compile(r"^\[1\]\s+[A-Z]")
_RESUME_HEADING = re.compile(
    r"^(([A-Z]\.?\s*)?(appendix|appendices|supplementary|supplemental)\b"
    r"|(table|figure|fig\.)\s*S?\d+)",  # tables/figures placed after the bibliography
    re.I,
)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping common scientific abbreviations intact."""
    sentences: list[str] = []
    for paragraph in text.split("\n\n"):
        pieces = [p.strip() for p in _SENTENCE_BOUNDARY.split(paragraph) if p.strip()]
        for i, piece in enumerate(pieces):
            # "et al. 2020" / "Fig. 3": glue back pieces split after an abbreviation,
            # but never glue across paragraphs.
            if i > 0 and _ends_with_abbreviation(sentences[-1]):
                sentences[-1] = f"{sentences[-1]} {piece}"
            else:
                sentences.append(piece)
    return sentences


def _ends_with_abbreviation(sentence: str) -> bool:
    last_word = sentence.rsplit(" ", 1)[-1].rstrip(".").lower()
    return last_word in _ABBREVIATIONS or (len(last_word) == 1 and last_word.isalpha())


def _split_long(sentence: str, limit: int) -> list[str]:
    words, parts, current = sentence.split(" "), [], ""
    for word in words:
        while len(word) > limit:  # pathological token (e.g. a long URL or sequence)
            if current:
                parts.append(current)
                current = ""
            parts.append(word[:limit])
            word = word[limit:]
        candidate = f"{current} {word}".strip()
        if len(candidate) > limit and current:
            parts.append(current)
            current = word
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def _overlap_tail(sentences: list[str], overlap: int) -> list[str]:
    """Trailing sentences whose combined length fits within ``overlap``."""
    if overlap <= 0:
        return []
    tail: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        added = len(sentence) + (1 if tail else 0)
        if total + added > overlap:
            break
        tail.insert(0, sentence)
        total += added
    if not tail and sentences:
        # Last sentence alone exceeds the overlap: carry its final words instead.
        text = sentences[-1][-overlap:]
        if " " in text:
            text = text.split(" ", 1)[1]
        tail = [text] if text else []
    return tail


def chunk_page_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Chunk the text of a single page."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    units: list[str] = []
    for sentence in split_sentences(text):
        units.extend(_split_long(sentence, chunk_size) if len(sentence) > chunk_size else [sentence])

    chunks: list[str] = []
    current: list[str] = []
    has_new_content = False
    for unit in units:
        candidate_len = len(" ".join(current + [unit]))
        if current and candidate_len > chunk_size:
            if has_new_content:
                chunks.append(" ".join(current))
            current = _overlap_tail(current, overlap)
            # Make sure the carried overlap plus the new unit still fits.
            while current and len(" ".join(current + [unit])) > chunk_size:
                current.pop(0)
            has_new_content = False
        current.append(unit)
        has_new_content = True
    if current and has_new_content:
        chunks.append(" ".join(current))
    return [c for c in chunks if len(c.replace(" ", "")) >= MIN_CHUNK_CHARS]


def _strip_references(doc: DocumentResult) -> dict[int, str]:
    """Return page_number -> text with the bibliography removed (heuristic)."""
    texts = {p.page_number: p.text for p in doc.pages}
    min_page = max(1, int(doc.page_count * 0.4))  # a "References" heading early on is unlikely
    in_refs = False
    for page in doc.pages:
        if page.page_number < min_page and not in_refs:
            continue
        kept = []
        for para in page.text.split("\n\n"):
            stripped = para.strip()
            if not in_refs and _REFERENCES_HEADING.match(stripped):
                in_refs = True
                continue
            if not in_refs and _FIRST_REFERENCE.match(stripped):
                in_refs = True
            if in_refs and _RESUME_HEADING.match(stripped):
                in_refs = False
            if not in_refs:
                kept.append(para)
        texts[page.page_number] = "\n\n".join(kept)
    return texts


def chunk_document(
    doc: DocumentResult,
    chunk_size: int = 800,
    overlap: int = 150,
    skip_references: bool = False,
) -> list[Chunk]:
    """Chunk every page of a document, attaching verified source metadata."""
    texts = _strip_references(doc) if skip_references else {p.page_number: p.text for p in doc.pages}
    chunks: list[Chunk] = []
    for page_number in sorted(texts):
        for index, text in enumerate(chunk_page_text(texts[page_number], chunk_size, overlap)):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id[:12]}-p{page_number}-c{index}",
                    doc_id=doc.doc_id,
                    doc_name=doc.doc_name,
                    page_number=page_number,
                    text=text,
                    chunk_index=index,
                )
            )
    return chunks
