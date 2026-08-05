"""OpenRouter LLM integration isolated from application logic."""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass

from app.logger import get_logger

logger = get_logger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
FREE_MODEL = "mistralai/mistral-7b-instruct:free"


class LLMError(RuntimeError):
    """Raised when the LLM provider cannot return an answer."""


@dataclass(frozen=True)
class OpenRouterClient:
    """Minimal OpenRouter chat client using a free instruct model."""

    api_key: str
    model: str = FREE_MODEL
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
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            logger.error("openrouter_http_error", extra={"status_code": exc.code})
            raise LLMError("OpenRouter request failed") from exc
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            logger.error("openrouter_unavailable", extra={"error": str(exc)})
            raise LLMError("OpenRouter is unavailable") from exc

        try:
            data = json.loads(body)
            return str(data["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            logger.error("openrouter_invalid_response")
            raise LLMError("OpenRouter returned an invalid response") from exc
