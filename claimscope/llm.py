from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass


@dataclass
class OpenAICompatibleClient:
    """Minimal OpenAI-compatible chat client.

    The API key is read from the environment and is never logged or stored.
    """

    api_key: str
    base_url: str
    model: str

    @classmethod
    def from_env(cls) -> "OpenAICompatibleClient | None":
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        if not api_key or not base_url:
            return None
        return cls(
            api_key=api_key,
            base_url=base_url,
            model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        )

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        endpoint = self.base_url.rstrip("/") + "/chat/completions"
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"]
