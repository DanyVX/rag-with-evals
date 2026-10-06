from __future__ import annotations

import random
import time

import httpx

from ragx.cost import CostTracker
from ragx.generate.pricing import TokenUsage, calculate_cost_usd, estimate_input_token_upper_bound


class AnthropicProvider:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        cost_tracker: CostTracker,
        input_usd_per_million: float,
        output_usd_per_million: float,
        max_output_tokens: int = 2048,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self.provider_id = f"anthropic:{model}"
        self.api_key = api_key
        self.model = model
        self.cost_tracker = cost_tracker
        self.input_usd_per_million = input_usd_per_million
        self.output_usd_per_million = output_usd_per_million
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.last_usage = TokenUsage()
        self.last_cost_usd = 0.0

    def _maximum_cost(self, prompt: str) -> float:
        return calculate_cost_usd(
            TokenUsage(
                input_tokens=estimate_input_token_upper_bound(prompt),
                output_tokens=self.max_output_tokens,
            ),
            input_usd_per_million=self.input_usd_per_million,
            output_usd_per_million=self.output_usd_per_million,
        )

    def generate(self, prompt: str, *, temperature: float = 0.0, seed: int | None = 0) -> str:
        del seed
        maximum_cost = self._maximum_cost(prompt)
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        last_error: Exception | None = None
        accumulated_cost = 0.0

        for attempt in range(self.max_retries + 1):
            reserved = self.cost_tracker.reserve(maximum_cost)
            try:
                response = httpx.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                if 400 <= response.status_code < 500:
                    try:
                        response.raise_for_status()
                    finally:
                        self.cost_tracker.cancel_reservation(reserved)
                    raise AssertionError("unreachable")
                response.raise_for_status()
                data = response.json()
                blocks = data.get("content", [])
                text = "".join(
                    block.get("text", "") for block in blocks if block.get("type") == "text"
                )
                if not text:
                    raise RuntimeError("provider returned empty output")

                raw_usage = data.get("usage") or {}
                if raw_usage:
                    usage = TokenUsage(
                        input_tokens=int(raw_usage.get("input_tokens", 0)),
                        output_tokens=int(raw_usage.get("output_tokens", 0)),
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

                self.cost_tracker.commit(actual, reserved_usd=reserved)
                accumulated_cost += actual
                self.last_usage = usage
                self.last_cost_usd = accumulated_cost
                return text

            except httpx.HTTPStatusError as exc:
                if 400 <= exc.response.status_code < 500:
                    raise
                last_error = exc
                self.cost_tracker.commit(maximum_cost, reserved_usd=reserved)
                accumulated_cost += maximum_cost
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc
                self.cost_tracker.commit(maximum_cost, reserved_usd=reserved)
                accumulated_cost += maximum_cost
            except RuntimeError:
                self.cost_tracker.commit(maximum_cost, reserved_usd=reserved)
                accumulated_cost += maximum_cost
                raise

            if attempt < self.max_retries:
                time.sleep((2**attempt) * 0.25 + random.uniform(0.0, 0.1))

        self.last_cost_usd = accumulated_cost
        raise RuntimeError("provider failed after retries") from last_error
