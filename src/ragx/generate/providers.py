from __future__ import annotations

import random
import time
from typing import Protocol

import httpx

from ragx.cost import CostTracker


class LLMProvider(Protocol):
    provider_id: str

    def generate(self, prompt: str, *, temperature: float = 0.0, seed: int | None = 0) -> str: ...


class MockProvider:
    provider_id = "mock"

    def __init__(self, response: str = "not found in the provided documents") -> None:
        self.response = response

    def generate(self, prompt: str, *, temperature: float = 0.0, seed: int | None = 0) -> str:
        return self.response


class OpenAICompatibleProvider:
    """Adapter for OpenAI-compatible APIs, including local Ollama gateways."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        cost_tracker: CostTracker,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self.provider_id = f"openai-compatible:{model}"
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.cost_tracker = cost_tracker
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def generate(self, prompt: str, *, temperature: float = 0.0, seed: int | None = 0) -> str:
        self.cost_tracker.reserve(0.0)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if seed is not None:
            payload["seed"] = seed
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = httpx.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
                if 400 <= response.status_code < 500:
                    response.raise_for_status()
                response.raise_for_status()
                data = response.json()
                text = data["choices"][0]["message"]["content"]
                if not text:
                    raise RuntimeError("provider returned empty output")
                return str(text)
            except httpx.HTTPStatusError as exc:
                if 400 <= exc.response.status_code < 500:
                    raise
                last_error = exc
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc
            if attempt < self.max_retries:
                time.sleep((2**attempt) * 0.25 + random.uniform(0.0, 0.1))
        raise RuntimeError("provider failed after retries") from last_error
