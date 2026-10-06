from __future__ import annotations

import random
import time

import httpx

from ragx.cost import CostTracker


class AnthropicProvider:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        cost_tracker: CostTracker,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self.provider_id = f"anthropic:{model}"
        self.api_key = api_key
        self.model = model
        self.cost_tracker = cost_tracker
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def generate(self, prompt: str, *, temperature: float = 0.0, seed: int | None = 0) -> str:
        del seed
        self.cost_tracker.reserve(0.0)
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": 2048,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = httpx.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                if 400 <= response.status_code < 500:
                    response.raise_for_status()
                response.raise_for_status()
                blocks = response.json().get("content", [])
                text = "".join(block.get("text", "") for block in blocks if block.get("type") == "text")
                if not text:
                    raise RuntimeError("provider returned empty output")
                return text
            except httpx.HTTPStatusError as exc:
                if 400 <= exc.response.status_code < 500:
                    raise
                last_error = exc
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc
            if attempt < self.max_retries:
                time.sleep((2**attempt) * 0.25 + random.uniform(0.0, 0.1))
        raise RuntimeError("provider failed after retries") from last_error
