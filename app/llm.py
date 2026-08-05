"""OpenRouter LLM integration isolated from application logic."""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass

from app.config import DEFAULT_OPENROUTER_MODEL
from app.logger import get_logger

logger = get_logger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class LLMError(RuntimeError):
    """Raised when the LLM provider cannot return an answer."""


@dataclass(frozen=True)
class OpenRouterClient:
    """Minimal OpenRouter chat client with a configurable model."""

    api_key: str
    model: str = DEFAULT_OPENROUTER_MODEL
    timeout: float = 30.0

    def generate(self, prompt: str) -> str:
        """Generate an answer from OpenRouter."""
        if not self.api_key:
            logger.error("openrouter_api_key_missing")
            raise LLMError("OpenRouter API key is missing")

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        request = urllib.request.Request(
            OPENROUTER_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/legal-rag-system",
                "X-Title": "Legal RAG System",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="ignore")[:500]
            logger.error(
                "openrouter_http_error",
                extra={"status_code": exc.code, "model": self.model, "body": error_body},
            )
            raise LLMError("OpenRouter request failed") from exc
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            logger.error(
                "openrouter_unavailable",
                extra={"model": self.model, "error": str(exc)},
            )
            raise LLMError("OpenRouter is unavailable") from exc

        try:
            data = json.loads(body)
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            logger.error("openrouter_invalid_response", extra={"model": self.model})
            raise LLMError("OpenRouter returned an invalid response") from exc
        answer = str(content).strip()
        if not answer:
            logger.error("openrouter_empty_response", extra={"model": self.model})
            raise LLMError("OpenRouter returned an empty response")
        return answer
