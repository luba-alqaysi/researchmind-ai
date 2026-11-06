from unittest.mock import MagicMock, patch

import pytest
import requests

from src.config import Settings
from src.llm_client import (
    ClaudeClient,
    GeminiClient,
    LLMConfigError,
    LLMConnectionError,
    LLMResponseError,
    OllamaClient,
    get_llm_client,
    ollama_status,
)

SETTINGS = Settings(ollama_model="llama3.1:8b", ollama_num_ctx=8192)


def _response(status=200, payload=None, text=""):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload or {}
    r.text = text
    return r


def test_ollama_success_sends_context_window_and_strips_think_blocks():
    payload = {"message": {"content": "<think>hmm</think>Answer [S1]."}, "done_reason": "stop"}
    with patch("src.llm_client.requests.post", return_value=_response(200, payload)) as post:
        result = OllamaClient(SETTINGS).generate("sys", "user")
    assert result.text == "Answer [S1]." and not result.truncated
    sent = post.call_args.kwargs["json"]
    assert sent["options"]["num_ctx"] == 8192 and sent["stream"] is False
    assert sent["messages"][0] == {"role": "system", "content": "sys"}


def test_ollama_not_running():
    with patch("src.llm_client.requests.post", side_effect=requests.exceptions.ConnectionError()):
        with pytest.raises(LLMConnectionError, match="Start the Ollama app"):
            OllamaClient(SETTINGS).generate("s", "u")


def test_ollama_timeout():
    with patch("src.llm_client.requests.post", side_effect=requests.exceptions.ReadTimeout()):
        with pytest.raises(LLMConnectionError, match="did not respond"):
            OllamaClient(SETTINGS).generate("s", "u")


def test_ollama_model_missing():
    with patch("src.llm_client.requests.post", return_value=_response(404, text="model not found")):
        with pytest.raises(LLMConfigError, match="ollama pull llama3.1:8b"):
            OllamaClient(SETTINGS).generate("s", "u")


def test_ollama_empty_or_malformed():
    with patch("src.llm_client.requests.post", return_value=_response(200, {"message": {"content": "  "}})):
        with pytest.raises(LLMResponseError):
            OllamaClient(SETTINGS).generate("s", "u")
    with patch("src.llm_client.requests.post", return_value=_response(200, {"unexpected": 1})):
        with pytest.raises(LLMResponseError):
            OllamaClient(SETTINGS).generate("s", "u")


def test_ollama_truncation_flag():
    payload = {"message": {"content": "partial"}, "done_reason": "length"}
    with patch("src.llm_client.requests.post", return_value=_response(200, payload)):
        assert OllamaClient(SETTINGS).generate("s", "u").truncated


def test_ollama_status_unreachable():
    with patch("src.llm_client.requests.get", side_effect=requests.exceptions.ConnectionError()):
        assert ollama_status("http://localhost:1") == (False, [])


def test_cloud_clients_require_keys():
    with pytest.raises(LLMConfigError, match="GEMINI_API_KEY"):
        GeminiClient(Settings(gemini_api_key=""))
    with pytest.raises(LLMConfigError, match="ANTHROPIC_API_KEY"):
        ClaudeClient(Settings(anthropic_api_key=""))


def test_unknown_provider():
    with pytest.raises(LLMConfigError, match="Unknown LLM provider"):
        get_llm_client("gpt", SETTINGS)


def test_settings_repr_hides_keys():
    s = Settings(gemini_api_key="secret-gemini", anthropic_api_key="secret-claude")
    assert "secret" not in repr(s) and "***" in repr(s)


def test_claude_request_shape_and_errors():
    import anthropic

    client = ClaudeClient(Settings(anthropic_api_key="test-key", claude_model="claude-opus-5-5"))
    block = MagicMock(type="text", text="Grounded answer [S1].")
    fake = MagicMock(stop_reason="end_turn", content=[block], model="claude-opus-5-5")
    with patch.object(client.client.beta.messages, "create", return_value=fake) as create:
        assert client.generate("sys", "user").text == "Grounded answer [S1]."
    kwargs = create.call_args.kwargs
    assert "temperature" not in kwargs  # rejected by current Claude models
    assert kwargs["output_config"] == {"effort": "medium"} and kwargs["fallbacks"] == "default"

    refusal = MagicMock(stop_reason="refusal", content=[], model="claude-opus-5-5")
    with patch.object(client.client.beta.messages, "create", return_value=refusal):
        with pytest.raises(LLMResponseError, match="declined"):
            client.generate("s", "u")

    err = anthropic.APIConnectionError(request=MagicMock())
    with patch.object(client.client.beta.messages, "create", side_effect=err):
        with pytest.raises(LLMConnectionError):
            client.generate("s", "u")


def test_haiku_has_no_fallback_parameter():
    client = ClaudeClient(Settings(anthropic_api_key="k", claude_model="claude-haiku-5-5"))
    fake = MagicMock(stop_reason="end_turn", content=[MagicMock(type="text", text="ok")], model="claude-haiku-5-5")
    with patch.object(client.client.beta.messages, "create", return_value=fake) as create:
        client.generate("s", "u")
    assert "fallbacks" not in create.call_args.kwargs


def test_ollama_context_grows_with_prompt_so_evidence_is_not_truncated():
    client = OllamaClient(Settings(ollama_num_ctx=8192, llm_max_tokens=2000))
    assert client.context_size("sys", "short question") == 8192
    big = "x" * 24 * 800  # e.g. a comparison prompt with 24 passages
    size = client.context_size("s" * 3000, big)
    assert size >= (3000 + len(big)) // 3 + 2000 and size % 2048 == 0
    assert client.context_size("s", "x" * 10_000_000) == 32768


def test_ollama_turns_thinking_off_by_default():
    payload = {"message": {"content": "Answer [S1]."}, "done_reason": "stop"}
    with patch("src.llm_client.requests.post", return_value=_response(200, payload)) as post:
        OllamaClient(SETTINGS).generate("sys", "user")
    assert post.call_args.kwargs["json"]["think"] is False
    with patch("src.llm_client.requests.post", return_value=_response(200, payload)) as post:
        OllamaClient(SETTINGS.with_overrides(ollama_think=True)).generate("sys", "user")
    assert post.call_args.kwargs["json"]["think"] is True


def test_ollama_retries_without_think_for_models_that_reject_it():
    ok = _response(200, {"message": {"content": "Answer [S1]."}, "done_reason": "stop"})
    rejected = _response(400, text='{"error":"\\"llama3.1:8b\\" does not support thinking"}')
    with patch("src.llm_client.requests.post", side_effect=[rejected, ok]) as post:
        result = OllamaClient(SETTINGS).generate("sys", "user")
    assert result.text == "Answer [S1]." and post.call_count == 2
    assert "think" not in post.call_args_list[1].kwargs["json"]


def test_default_model_is_qwen():
    assert Settings().ollama_model == "qwen3.5:9b" and Settings().ollama_think is False
