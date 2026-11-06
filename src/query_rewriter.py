"""Query rewriting for follow-up and Arabic questions.

The retrieval stack (MiniLM embeddings, the ms-marco reranker and the Laya gate)
is English-only and sees one question at a time. Before retrieval, the LLM turns
the new question, plus the last turns of the conversation, into one standalone
English search query. The answer is still written in the user's language.

Rewriting costs one extra LLM call, so it runs only when it can help: when the
question is not in English or when there is earlier conversation. Any failure
falls back to the original question. Some models (seen with Qwen3.5 9B) sometimes
return the rewritten question still in Arabic; that reply is sent back once with a
plain "translate this" request before giving up.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from src.prompts import REWRITE_SYSTEM_PROMPT, TRANSLATE_SYSTEM_PROMPT, build_rewrite_prompt, build_translate_prompt

_ARABIC = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
_JSON = re.compile(r"\{.*?\}", re.S)


def detect_language(text: str) -> str:
    """"ar" when at least 30% of the letters are Arabic script, else "en"."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "en"
    arabic = sum(bool(_ARABIC.match(c)) for c in letters)
    return "ar" if arabic / len(letters) >= 0.3 else "en"


def needs_rewrite(language: str, history: list[tuple[str, str]] | None) -> bool:
    return language != "en" or bool(history)


@dataclass
class Rewrite:
    query: str  # the text to search with
    rewritten: bool  # False when the original question is used
    ms: float = 0.0
    error: str | None = None


def parse_query(text: str) -> str | None:
    """Extract ``search_query`` from the model's reply; None if it is unusable."""
    for candidate in _JSON.findall(text or ""):
        try:
            value = json.loads(candidate).get("search_query")
        except (json.JSONDecodeError, AttributeError):
            continue
        if isinstance(value, str) and 3 <= len(value.strip()) <= 300:
            return " ".join(value.split())
    return None


def rewrite_query(question: str, history: list[tuple[str, str]] | None, llm) -> Rewrite:
    t0 = time.perf_counter()
    try:
        response = llm.generate(REWRITE_SYSTEM_PROMPT, build_rewrite_prompt(question, history or []))
    except Exception as exc:  # the answer step reports LLM errors; here we only fall back
        return Rewrite(question, False, (time.perf_counter() - t0) * 1000, error=str(exc))
    query = parse_query(response.text)
    if query is not None and detect_language(query) != "en":
        # The model resolved the question but did not translate it: ask once for a translation only.
        try:
            retry = llm.generate(TRANSLATE_SYSTEM_PROMPT, build_translate_prompt(query))
            query = parse_query(retry.text)
        except Exception:
            query = None
    ms = (time.perf_counter() - t0) * 1000
    if query is None or detect_language(query) != "en":
        return Rewrite(question, False, ms, error="unusable rewrite")
    return Rewrite(query, query != question.strip(), ms)
