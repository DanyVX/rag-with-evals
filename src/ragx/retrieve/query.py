from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Literal, Protocol

QueryMode = Literal["none", "lowercase", "query_expansion"]


class QueryExpander(Protocol):
    def generate(self, prompt: str, *, temperature: float = 0.0, seed: int | None = 0) -> str: ...


@dataclass(slots=True)
class QueryPreprocessResult:
    original: str
    processed: str
    truncated: bool = False


def preprocess_query(
    query: str,
    *,
    mode: QueryMode = "none",
    expander: QueryExpander | None = None,
    max_chars: int = 8_000,
) -> QueryPreprocessResult:
    normalized = unicodedata.normalize("NFKC", query).strip()
    truncated = len(normalized) > max_chars
    normalized = normalized[:max_chars].strip()

    if not normalized:
        return QueryPreprocessResult(original=query, processed="", truncated=truncated)

    if mode == "none":
        processed = normalized
    elif mode == "lowercase":
        processed = normalized.lower()
    elif mode == "query_expansion":
        if expander is None:
            raise ValueError("query_expansion requires an LLM provider")
        prompt = (
            "Rewrite the user query into one concise retrieval query. "
            "Preserve exact identifiers, numbers, acronyms, and negation. "
            "Add useful synonyms only when they do not change meaning. "
            "Return only the retrieval query.\n\n"
            f"USER QUERY:\n{normalized}"
        )
        processed = expander.generate(prompt, temperature=0.0, seed=0).strip()
        if not processed:
            processed = normalized
    else:
        raise ValueError(f"unsupported query preprocessing mode: {mode}")

    return QueryPreprocessResult(
        original=query,
        processed=processed,
        truncated=truncated,
    )
