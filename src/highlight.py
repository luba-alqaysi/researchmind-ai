"""Find the sentence of a cited passage that best matches the answer.

The UI highlights that sentence in the passage, so a reader can see at a glance
which part of the page the answer relies on. It is a lexical heuristic, not
verification: the sentence of the passage that shares the most content words
(and numbers) with the answer sentences citing it, plus the search query.
"""

from __future__ import annotations

import math
import re

from src.citations import _GROUP, _expand
from src.text_chunker import split_sentences

_WORD = re.compile(r"[A-Za-z][A-Za-z\-]{2,}|\d+(?:\.\d+)?")
_STOP = frozenset(
    "the and for with that this from were was are has have had been their which these those than then into "
    "using used use also such not but can may its our they them there what when where who whom how about "
    "between within over under more most other some any each both all per study paper sources source".split()
)


def content_words(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text) if w.lower() not in _STOP}


def claims_by_source(raw_answer: str) -> dict[int, str]:
    """Map each cited source number to the answer sentences that cite it."""
    claims: dict[int, list[str]] = {}
    for sentence in re.split(r"(?<=[.!?؟])\s+|\n+", raw_answer):
        for group in _GROUP.finditer(sentence):
            for n in _expand(group.group(1)):
                claims.setdefault(n, []).append(sentence)
    return {n: " ".join(s) for n, s in claims.items()}


def best_sentence(passage: str, *evidence: str, min_overlap: int = 2) -> str | None:
    """The passage sentence sharing the most content words with ``evidence``, or None
    if no sentence shares at least ``min_overlap`` words (one is enough for a number)."""
    target = set().union(*(content_words(e) for e in evidence if e))
    if not target:
        return None
    best, best_score = None, 0.0
    for sentence in split_sentences(passage):
        words = content_words(sentence)
        shared = words & target
        numbers = {w for w in shared if w[0].isdigit()}
        if len(shared) < min_overlap and not numbers:
            continue
        score = (len(shared) + len(numbers)) / math.sqrt(max(len(words), 1))
        if score > best_score:
            best, best_score = sentence, score
    return best
