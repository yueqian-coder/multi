from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from urllib.parse import urlparse


class OpenAICompatibleClientError(RuntimeError):
    pass


@dataclass
class OpenAICompatibleClient:
    """Minimal OpenAI-compatible chat client.

    The API key is read from the environment and is never logged or stored.
    """

    api_key: str = field(repr=False)
    base_url: str
    model: str
    timeout: float = 60

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url.strip())
        local_hosts = {"localhost", "127.0.0.1", "::1"}
        if parsed.scheme == "https" and parsed.netloc:
            return
        if parsed.scheme == "http" and parsed.hostname in local_hosts:
            return
        raise ValueError("Remote OpenAI-compatible providers must use HTTPS.")

    @classmethod
    def from_env(cls) -> "OpenAICompatibleClient | None":
        provider = os.getenv("CLAIMSCOPE_PROVIDER", "").strip().lower()
        if not provider:
            if os.getenv("CLAIMSCOPE_CLAUDE_API_KEY"):
                provider = "claude"
            elif os.getenv("CLAIMSCOPE_GPT_API_KEY"):
                provider = "gpt"
        if provider == "claude":
            api_key = os.getenv("CLAIMSCOPE_CLAUDE_API_KEY") or os.getenv(
                "OPENAI_API_KEY"
            )
            model = (
                os.getenv("CLAIMSCOPE_CLAUDE_MODEL")
                or os.getenv("MODEL_NAME")
                or "claude-sonnet-4-5"
            )
        elif provider == "gpt":
            api_key = os.getenv("CLAIMSCOPE_GPT_API_KEY") or os.getenv(
                "OPENAI_API_KEY"
            )
            model = (
                os.getenv("CLAIMSCOPE_GPT_MODEL")
                or os.getenv("MODEL_NAME")
                or "gpt-5.4-mini"
            )
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            model = os.getenv("MODEL_NAME", "gpt-4o-mini")
        base_url = os.getenv("OPENAI_BASE_URL")
        if not api_key or not base_url:
            return None
        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        timeout: float | None = None,
    ) -> str:
        endpoint = self.base_url.rstrip("/") + "/chat/completions"
        request_payload = {
            "model": self.model,
            "messages": messages,
        }
        # GPT-5 reasoning endpoints commonly reject non-default sampling controls.
        # Reduce request-shape failures while preserving temperature for chat models.
        if not self.model.strip().lower().startswith("gpt-5"):
            request_payload["temperature"] = temperature
        payload = json.dumps(request_payload).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        request_timeout = self.timeout if timeout is None else timeout
        for attempt in range(2):
            try:
                with urllib.request.urlopen(
                    request, timeout=request_timeout
                ) as response:
                    result = json.loads(response.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as exc:
                if attempt == 0 and _is_retryable_http_status(exc.code):
                    continue
                raise OpenAICompatibleClientError(
                    f"LLM chat request failed with HTTP {exc.code}."
                ) from None
            except Exception as exc:
                raise OpenAICompatibleClientError(
                    f"LLM chat request failed with {exc.__class__.__name__}."
                ) from None
        raise OpenAICompatibleClientError("LLM chat request failed.")


def _is_retryable_http_status(status: int) -> bool:
    return status == 429 or 500 <= status <= 599
