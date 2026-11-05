"""Turn the model's source-ID references into verified citations.

The LLM only ever sees source IDs (S1, S2, ...). This module maps each ID back
to the chunk that was actually retrieved, so every rendered citation such as
``[paper.pdf, Page 4]`` comes from stored metadata, never from model output.
IDs that do not correspond to a retrieved chunk are removed and reported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.prompts import INSUFFICIENT_MARKER, source_id
from src.schemas import Citation, RetrievedChunk

# Matches [S1], [S1, S3], [S1; S2], [S2-S4], [S1, 3] and the same with parentheses.
_GROUP = re.compile(r"[\[(]\s*(S\d+(?:\s*(?:[,;]|and|[-–])\s*S?\d+)*)\s*[\])]")
_ID = re.compile(r"S?(\d+)")
_INSUFFICIENT = re.compile(rf"^\W*{INSUFFICIENT_MARKER}\W*", re.I)

EXCERPT_CHARS = 600


@dataclass
class CitationResult:
    text: str  # answer with verified citation labels
    citations: list[Citation] = field(default_factory=list)  # in order of first use
    invalid_ids: list[str] = field(default_factory=list)
    insufficient: bool = False


# Small models sometimes write "According to S1" or "source S2" instead of "[S1]". Only these
# explicit phrasings are accepted, so IDs such as "Figure S2" or "Table S1" quoted from a paper
# are never mistaken for citations.
_BARE = re.compile(
    r"(\b(?:[Aa]ccording to|[Ff]rom|[Ss]ources?|[Ss]ee)\s+|(?:المصدر|المصادر|حسب|وفقاً ل|وفقا ل)\s*)"
    r"(S\d+(?:\s*(?:,|and)\s*S\d+)*)(?![\w\]])"
)


def _bracket_bare_ids(answer: str) -> str:
    return _BARE.sub(lambda m: f"{m.group(1)}[{m.group(2)}]", answer)


def _expand(group: str) -> list[int]:
    """'S1, S3' -> [1, 3];  'S2-S4' -> [2, 3, 4]."""
    numbers: list[int] = []
    for part in re.split(r"\s*(?:[,;]|and)\s*", group):
        bounds = re.split(r"\s*[-–]\s*", part)
        ids = [int(m.group(1)) for b in bounds if (m := _ID.fullmatch(b.strip()))]
        if len(ids) == 2 and ids[0] <= ids[1] and ids[1] - ids[0] <= 20:
            numbers.extend(range(ids[0], ids[1] + 1))
        else:
            numbers.extend(ids)
    return numbers


def make_citation(index: int, r: RetrievedChunk) -> Citation:
    excerpt = r.chunk.text if len(r.chunk.text) <= EXCERPT_CHARS else r.chunk.text[:EXCERPT_CHARS].rsplit(" ", 1)[0] + " …"
    return Citation(
        source_id=source_id(index),
        doc_name=r.chunk.doc_name,
        page_number=r.chunk.page_number,
        excerpt=excerpt,
        score=r.score,
    )


def resolve_citations(answer: str, retrieved: list[RetrievedChunk]) -> CitationResult:
    """Replace [S#] markers with verified labels and collect the cited sources."""
    answer = (answer or "").strip()
    insufficient = bool(_INSUFFICIENT.match(answer))
    if insufficient:
        answer = "**Insufficient evidence.** " + _INSUFFICIENT.sub("", answer, count=1).strip()

    answer = _bracket_bare_ids(answer)
    cited: dict[int, Citation] = {}
    invalid: list[str] = []

    def replace(match: re.Match) -> str:
        labels: list[str] = []
        for number in _expand(match.group(1)):
            index = number - 1
            if 0 <= index < len(retrieved):
                citation = cited.setdefault(index, make_citation(index, retrieved[index]))
                if citation.label not in labels:
                    labels.append(citation.label)
            elif f"S{number}" not in invalid:
                invalid.append(f"S{number}")
        return " ".join(labels)

    text = _GROUP.sub(replace, answer)
    text = re.sub(r"[ \t]+([.,;:])", r"\1", text)  # tidy spaces left by removed citations
    text = re.sub(r"[ \t]{2,}", " ", text)
    return CitationResult(text=text, citations=list(cited.values()), invalid_ids=invalid, insufficient=insufficient)
