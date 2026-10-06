from __future__ import annotations

import random
import time
from typing import Protocol

import httpx

from ragx.cost import CostTracker
from ragx.generate.pricing import TokenUsage, calculate_cost_usd, estimate_input_token_upper_bound


class LLMProvider(Protocol):
    provider_id: str
    last_usage: TokenUsage
    last_cost_usd: float

    def generate(self, prompt: str, *, temperature: float = 0.0, seed: int | None = 0) -> str: ...


class MockProvider:
    provider_id = "mock"

    def __init__(self, response: str = "not found in the provided documents") -> None:
        self.response = response
        self.last_usage = TokenUsage()
        self.last_cost_usd = 0.0

    def generate(self, prompt: str, *, temperature: float = 0.0, seed: int | None = 0) -> str:
        del prompt, temperature, seed
        self.last_usage = TokenUsage()
        self.last_cost_usd = 0.0
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
        input_usd_per_million: float = 0.0,
        output_usd_per_million: float = 0.0,
        max_output_tokens: int = 2048,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        zero_cost_local: bool = False,
    ) -> None:
        self.provider_id = f"openai-compatible:{model}"
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.cost_tracker = cost_tracker
        self.input_usd_per_million = input_usd_per_million
        self.output_usd_per_million = output_usd_per_million
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.zero_cost_local = zero_cost_local
        self.last_usage = TokenUsage()
        self.last_cost_usd = 0.0

    def _maximum_cost(self, prompt: str) -> float:
        usage = TokenUsage(
            input_tokens=estimate_input_token_upper_bound(prompt),
            output_tokens=self.max_output_tokens,
        )
        return calculate_cost_usd(
            usage,
            input_usd_per_million=self.input_usd_per_million,
            output_usd_per_million=self.output_usd_per_million,
        )

    def generate(self, prompt: str, *, temperature: float = 0.0, seed: int | None = 0) -> str:
        maximum_cost = self._maximum_cost(prompt)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": self.max_output_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        last_error: Exception | None = None
        accumulated_cost = 0.0

        for attempt in range(self.max_retries + 1):
            reserved = 0.0
            if not self.zero_cost_local:
                reserved = self.cost_tracker.reserve(maximum_cost)

            try:
                response = httpx.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
                if 400 <= response.status_code < 500:
                    try:
                        response.raise_for_status()
                    finally:
                        if reserved:
                            self.cost_tracker.cancel_reservation(reserved)
                    raise AssertionError("unreachable")
                response.raise_for_status()
                data = response.json()
                text = data["choices"][0]["message"]["content"]
                if not text:
                    raise RuntimeError("provider returned empty output")

                raw_usage = data.get("usage") or {}
                if raw_usage:
                    usage = TokenUsage(
                        input_tokens=int(raw_usage.get("prompt_tokens", 0)),
                        output_tokens=int(raw_usage.get("completion_tokens", 0)),
                    )
                    actual = calculate_cost_usd(
                        usage,
                        input_usd_per_million=self.input_usd_per_million,
                        output_usd_per_million=self.output_usd_per_million,
                    )
                else:
                    usage = TokenUsage(
                        input_tokens=estimate_input_token_upper_bound(prompt),
                        output_tokens=self.max_output_tokens,
                    )
                    actual = maximum_cost

                if self.zero_cost_local:
                    actual = 0.0
                else:
                    self.cost_tracker.commit(actual, reserved_usd=reserved)
                accumulated_cost += actual
                self.last_usage = usage
                self.last_cost_usd = accumulated_cost
                return str(text)

            except httpx.HTTPStatusError as exc:
                if 400 <= exc.response.status_code < 500:
                    raise
                last_error = exc
                if reserved:
                    self.cost_tracker.commit(maximum_cost, reserved_usd=reserved)
                    accumulated_cost += maximum_cost
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc
                if reserved:
                    self.cost_tracker.commit(maximum_cost, reserved_usd=reserved)
                    accumulated_cost += maximum_cost
            except RuntimeError:
                if reserved:
                    self.cost_tracker.commit(maximum_cost, reserved_usd=reserved)
                    accumulated_cost += maximum_cost
                raise

            if attempt < self.max_retries:
                time.sleep((2**attempt) * 0.25 + random.uniform(0.0, 0.1))

        self.last_cost_usd = accumulated_cost
        raise RuntimeError("provider failed after retries") from last_error
