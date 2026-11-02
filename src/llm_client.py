"""LLM provider clients behind one small interface.

Every client exposes ``generate(system, user) -> LLMResponse``. The RAG pipeline
depends only on that method, so providers can be swapped from the sidebar or
``.env`` without touching retrieval or UI code.

Providers
---------
* ``ollama``     - local models through the Ollama HTTP API (default; free, private).
* ``gemini``     - Google Gemini through the ``google-genai`` SDK.
* ``claude``     - Anthropic Claude through the ``anthropic`` SDK.
* ``extractive`` - no generation: returns the top passages. Useful offline and as a baseline.

Errors are mapped to a few typed exceptions with messages that tell the user
what to do. API keys are never included in messages or logs.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Protocol

import requests

from src.config import Settings


# --------------------------------------------------------------------- errors
class LLMError(Exception):
    """Base class for all provider errors (safe to show to the user)."""


class LLMConfigError(LLMError):
    """Missing key, unknown model, or provider not set up."""


class LLMConnectionError(LLMError):
    """The provider could not be reached (network, Ollama not running, timeout)."""


class LLMRateLimitError(LLMError):
    """The provider rejected the request because of rate limits or quota."""


class LLMResponseError(LLMError):
    """The provider answered, but with an error or an unusable response."""


@dataclass
class LLMResponse:
    text: str
    model: str
    truncated: bool = False  # output hit the token limit


class LLMClient(Protocol):
    provider: str
    model: str

    def generate(self, system: str, user: str) -> LLMResponse: ...


MAX_OLLAMA_CTX = 32768  # upper bound to keep memory use reasonable on a laptop

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S | re.I)


# --------------------------------------------------------------------- Ollama
class OllamaClient:
    provider = "ollama"

    def __init__(self, settings: Settings, model: str | None = None):
        self.base_url = settings.ollama_base_url
        self.model = model or settings.ollama_model
        self.num_ctx = settings.ollama_num_ctx
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        self.timeout = settings.llm_timeout_seconds

    def context_size(self, system: str, user: str) -> int:
        """Context window large enough for the prompt plus the answer.

        If prompt + answer exceed num_ctx, Ollama silently drops the *start* of the
        prompt (the instructions and the first sources). Scientific text averages
        roughly 3-4 characters per token; 3 is used as a conservative estimate.
        """
        needed = (len(system) + len(user)) // 3 + self.max_tokens + 256
        size = max(self.num_ctx, -(-needed // 2048) * 2048)  # round up to a multiple of 2048
        return min(size, MAX_OLLAMA_CTX)

    def generate(self, system: str, user: str) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.context_size(system, user),
                "num_predict": self.max_tokens,
            },
        }
        try:
            response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        except requests.exceptions.Timeout as exc:
            raise LLMConnectionError(
                f"Ollama did not respond within {self.timeout}s. The model may still be loading; try again."
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise LLMConnectionError(
                f"Cannot reach Ollama at {self.base_url}. Start the Ollama app (or run `ollama serve`)."
            ) from exc

        if response.status_code == 404:
            raise LLMConfigError(f"Ollama model '{self.model}' is not installed. Run: ollama pull {self.model}")
        if response.status_code >= 400:
            raise LLMResponseError(f"Ollama returned HTTP {response.status_code}: {response.text[:300]}")
        try:
            data = response.json()
            text = data["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMResponseError("Ollama returned an unexpected response format.") from exc
        text = _THINK_BLOCK.sub("", text).strip()  # reasoning models may emit <think> blocks
        if not text:
            raise LLMResponseError("Ollama returned an empty answer.")
        return LLMResponse(text=text, model=self.model, truncated=data.get("done_reason") == "length")


def ollama_status(base_url: str, timeout: float = 1.5) -> tuple[bool, list[str]]:
    """Return (reachable, installed model names) for the sidebar."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=timeout)
        response.raise_for_status()
        return True, sorted(m["name"] for m in response.json().get("models", []))
    except (requests.RequestException, ValueError, KeyError):
        return False, []


# --------------------------------------------------------------------- Gemini
class GeminiClient:
    provider = "gemini"

    def __init__(self, settings: Settings, model: str | None = None):
        if not settings.gemini_api_key:
            raise LLMConfigError("GEMINI_API_KEY is not set. Add it to your .env file.")
        from google import genai

        self._genai = genai
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = model or settings.gemini_model
        self.temperature = settings.llm_temperature
        # Gemini 2.5+ models spend output tokens on internal "thinking", so leave headroom.
        self.max_tokens = max(settings.llm_max_tokens, 8192)

    def generate(self, system: str, user: str) -> LLMResponse:
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=system, temperature=self.temperature, max_output_tokens=self.max_tokens
        )
        for attempt in range(2):  # one retry on rate limiting
            try:
                response = self.client.models.generate_content(model=self.model, contents=user, config=config)
                break
            except errors.ClientError as exc:
                code = getattr(exc, "code", None)
                if code == 429:
                    if attempt == 0:
                        time.sleep(5)
                        continue
                    raise LLMRateLimitError("Gemini rate limit or quota exceeded. Wait a minute and retry.") from exc
                if code in (401, 403) or "API key" in str(exc):
                    raise LLMConfigError("Gemini rejected the API key. Check GEMINI_API_KEY in .env.") from exc
                if code == 404:
                    raise LLMConfigError(f"Gemini model '{self.model}' was not found. Set GEMINI_MODEL in .env.") from exc
                raise LLMResponseError(f"Gemini request failed ({code}).") from exc
            except errors.ServerError as exc:
                raise LLMConnectionError("Gemini is temporarily unavailable. Try again shortly.") from exc
            except (requests.RequestException, OSError) as exc:
                raise LLMConnectionError("Could not connect to the Gemini API.") from exc

        text = (response.text or "").strip()
        finish = ""
        if response.candidates:
            finish = str(getattr(response.candidates[0], "finish_reason", "") or "")
        if not text:
            raise LLMResponseError(f"Gemini returned no text (finish reason: {finish or 'unknown'}).")
        return LLMResponse(text=text, model=self.model, truncated="MAX_TOKENS" in finish)


# --------------------------------------------------------------------- Claude
# Models that accept server-side refusal fallbacks (fallbacks="default").
_FALLBACK_MODELS = ("claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5")


class ClaudeClient:
    provider = "claude"

    def __init__(self, settings: Settings, model: str | None = None):
        if not settings.anthropic_api_key:
            raise LLMConfigError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
        import anthropic

        self._anthropic = anthropic
        # The SDK already retries connection errors, 429 and 5xx with backoff.
        self.client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key, timeout=float(settings.llm_timeout_seconds), max_retries=2
        )
        self.model = model or settings.claude_model
        self.effort = settings.claude_effort
        # Current Claude models think adaptively and thinking counts toward max_tokens.
        self.max_tokens = max(settings.llm_max_tokens, 16000)

    def generate(self, system: str, user: str) -> LLMResponse:
        anthropic = self._anthropic
        # Current Claude models reject temperature/top_p, so none are sent; depth is
        # controlled with the effort setting instead.
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort},
        )
        if self.model in _FALLBACK_MODELS:
            # If a safety classifier declines, let the API retry on a suitable fallback model.
            kwargs.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        try:
            response = self.client.beta.messages.create(**kwargs)
        except anthropic.AuthenticationError as exc:
            raise LLMConfigError("Anthropic rejected the API key. Check ANTHROPIC_API_KEY in .env.") from exc
        except anthropic.PermissionDeniedError as exc:
            raise LLMConfigError("This Anthropic API key cannot use the selected model.") from exc
        except anthropic.NotFoundError as exc:
            raise LLMConfigError(f"Claude model '{self.model}' was not found. Set CLAUDE_MODEL in .env.") from exc
        except anthropic.RateLimitError as exc:
            raise LLMRateLimitError("Anthropic rate limit reached. Wait a moment and retry.") from exc
        except anthropic.APITimeoutError as exc:
            raise LLMConnectionError("The Anthropic API timed out. Try again.") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMConnectionError("Could not connect to the Anthropic API.") from exc
        except anthropic.APIStatusError as exc:
            raise LLMResponseError(f"Anthropic API error (HTTP {exc.status_code}).") from exc

        if response.stop_reason == "refusal":
            raise LLMResponseError("Claude declined to answer this request.")
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        if not text:
            raise LLMResponseError("Claude returned an empty answer.")
        return LLMResponse(text=text, model=response.model, truncated=response.stop_reason == "max_tokens")


# ----------------------------------------------------------------- Extractive
class ExtractiveClient:
    """No-LLM mode. The pipeline formats the top passages itself (see RAGPipeline)."""

    provider = "extractive"
    model = "none"

    def __init__(self, settings: Settings | None = None, model: str | None = None):
        pass

    def generate(self, system: str, user: str) -> LLMResponse:  # pragma: no cover - not used
        raise LLMConfigError("Extractive mode does not generate text.")


# -------------------------------------------------------------------- factory
_CLIENTS = {"ollama": OllamaClient, "gemini": GeminiClient, "claude": ClaudeClient, "extractive": ExtractiveClient}


def get_llm_client(provider: str, settings: Settings, model: str | None = None) -> LLMClient:
    try:
        cls = _CLIENTS[provider]
    except KeyError as exc:
        raise LLMConfigError(f"Unknown LLM provider '{provider}'. Choose one of: {', '.join(_CLIENTS)}.") from exc
    return cls(settings, model=model)
