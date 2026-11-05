"""Query rewriting for follow-up and Arabic questions (fake LLM, no network)."""

from src.llm_client import ExtractiveClient, LLMConnectionError, LLMResponse
from src.prompts import build_qa_prompt
from src.query_rewriter import detect_language, needs_rewrite, parse_query, rewrite_query
from tests.test_reranker import KeywordReranker, docs  # noqa: F401  (fixture)


class ScriptedLLM:
    """Replies to the rewrite call and the answer call in order; records both."""

    provider, model = "fake", "fake-1"

    def __init__(self, *replies: str):
        self.replies, self.calls = list(replies), []

    def generate(self, system: str, user: str) -> LLMResponse:
        self.calls.append((system, user))
        return LLMResponse(text=self.replies.pop(0), model=self.model)


def test_detect_language():
    assert detect_language("ما حجم العينة في هذه الدراسة؟") == "ar"
    assert detect_language("What was the sample size?") == "en"
    assert detect_language("ما هو RNAseq؟") == "ar"
    assert detect_language("12345 ?") == "en"


def test_needs_rewrite_only_for_follow_ups_and_other_languages():
    assert not needs_rewrite("en", None) and not needs_rewrite("en", [])
    assert needs_rewrite("ar", None) and needs_rewrite("en", [("q", "a")])


def test_parse_query_tolerates_chatter_and_rejects_bad_output():
    assert parse_query('Sure! {"search_query": "sample size of the  RNAseq study"}') == "sample size of the RNAseq study"
    assert parse_query("no json here") is None
    assert parse_query('{"search_query": ""}') is None
    assert parse_query('{"other": "x"}') is None


def test_rewrite_falls_back_on_errors_and_non_english_output():
    class Failing:
        def generate(self, system, user):
            raise LLMConnectionError("down")

    r = rewrite_query("ما حجم العينة؟", None, Failing())
    assert r.query == "ما حجم العينة؟" and not r.rewritten and r.error
    r = rewrite_query("ما حجم العينة؟", None, ScriptedLLM('{"search_query": "ما حجم العينة"}'))
    assert not r.rewritten and r.error == "unusable rewrite"


def test_arabic_question_is_searched_in_english_and_answered_in_arabic(docs):  # noqa: F811
    docs.reranker = KeywordReranker("60 subjects")
    llm = ScriptedLLM('{"search_query": "What was the sample size of the RNAseq study?"}',
                      "شملت الدراسة 60 مشاركاً [S1].")
    answer = docs.ask("ما حجم العينة في هذه الدراسة؟", llm)
    assert answer.trace.language == "ar" and answer.trace.rewritten
    assert answer.trace.search_query == "What was the sample size of the RNAseq study?"
    assert "60 subjects" in answer.retrieved[0].chunk.text
    assert docs.reranker.calls[0][0] == "What was the sample size of the RNAseq study?"  # reranked in English
    answer_prompt = llm.calls[1][1]
    assert "Write the answer in Arabic" in answer_prompt and "Interpreted as: What was the sample size" in answer_prompt
    assert answer.citations and answer.trace.stage_ms["rewrite_ms"] >= 0


def test_follow_up_uses_history(docs):  # noqa: F811
    llm = ScriptedLLM('{"search_query": "sample size of the psychiatric gene expression study"}', "60 [S1].")
    history = [("What does the paper study?", "Gene expression in psychiatric disorders [S1].")]
    answer = docs.ask("ok what about the sample size?", llm, history=history)
    rewrite_prompt = llm.calls[0][1]
    assert "What does the paper study?" in rewrite_prompt and "ok what about the sample size?" in rewrite_prompt
    assert answer.trace.rewritten and answer.trace.language == "en"


def test_no_rewrite_call_for_standalone_english(docs):  # noqa: F811
    docs.reranker = KeywordReranker("60 subjects")
    llm = ScriptedLLM("60 subjects [S1].")
    answer = docs.ask("What was the sample size?", llm)
    assert len(llm.calls) == 1 and not answer.trace.rewritten and "rewrite_ms" not in answer.trace.stage_ms


def test_rewriting_can_be_turned_off(docs):  # noqa: F811
    docs.settings = docs.settings.with_overrides(query_rewriting=False)
    docs.reranker = KeywordReranker("60 subjects")
    llm = ScriptedLLM("60 [S1].")
    docs.ask("ok what about the sample size?", llm, history=[("q", "a")])
    assert len(llm.calls) == 1


def test_extractive_mode_skips_rewriting_and_warns_for_arabic(docs):  # noqa: F811
    answer = docs.ask("ما حجم العينة في هذه الدراسة؟", ExtractiveClient())
    assert not answer.trace.rewritten and any("Arabic" in w for w in answer.warnings)


def test_english_prompt_has_no_language_line():
    assert "Write the answer in" not in build_qa_prompt("What?", [])
    assert "Interpreted as" not in build_qa_prompt("What?", [], search_query="What?")
