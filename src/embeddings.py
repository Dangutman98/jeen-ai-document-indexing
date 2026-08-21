"""Embedding generation via the Gemini API.

gemini-embedding-001 returns 3072-dimension vectors by default, but pgvector's
HNSW and IVFFlat indexes only support up to 2000 dimensions -- a 3072-dim
column can be stored but never indexed, so every similarity search would
degrade to a full sequential scan. We request output_dimensionality=1536
(one of Google's own recommended sizes, alongside 3072 and 768) and
L2-renormalize the result, which Google requires for any non-default size
since the API does not renormalize truncated vectors itself.
"""

from __future__ import annotations

import os
import time

from google import genai
from google.genai import types
from google.genai.errors import APIError

from .errors import EmbeddingError

EMBEDDING_MODEL = "gemini-embedding-001"
OUTPUT_DIMENSIONALITY = 1536

# Free-tier embedding quota is small and rate-limited; batch requests and
# back off on transient failures rather than firing one call per chunk.
BATCH_SIZE = 32
MAX_RETRIES = 4
INITIAL_BACKOFF_SECONDS = 2.0


def _client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EmbeddingError(
            "GEMINI_API_KEY is not set. Add it to your .env file (see .env.example)."
        )
    return genai.Client(api_key=api_key)


def _renormalize(vector: list[float]) -> list[float]:
    norm = sum(v * v for v in vector) ** 0.5
    if norm == 0:
        return vector
    return [v / norm for v in vector]


def embed_texts(texts: list[str], *, task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Embed a list of texts, batched, with retry/backoff on transient failures.

    Raises EmbeddingError on non-transient failures or after exhausting retries.
    """
    if not texts:
        return []

    client = _client()
    config = types.EmbedContentConfig(
        task_type=task_type,
        output_dimensionality=OUTPUT_DIMENSIONALITY,
    )

    results: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        results.extend(_embed_batch_with_retry(client, batch, config))
    return results


def _embed_batch_with_retry(
    client: genai.Client, batch: list[str], config: "types.EmbedContentConfig"
) -> list[list[float]]:
    backoff = INITIAL_BACKOFF_SECONDS
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch,
                config=config,
            )
            return [_renormalize(e.values) for e in response.embeddings]
        except APIError as e:
            last_error = e
            # 429 (rate limit) and 5xx are worth retrying; anything else (e.g.
            # 400 bad request, 401 auth) will fail identically on retry.
            status = getattr(e, "code", None)
            if status is not None and status not in (429, 500, 502, 503, 504):
                raise EmbeddingError(f"Embedding request failed ({status}): {e}") from e
            if attempt < MAX_RETRIES:
                time.sleep(backoff)
                backoff *= 2
        except Exception as e:  # network errors etc.
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(backoff)
                backoff *= 2

    raise EmbeddingError(
        f"Embedding request failed after {MAX_RETRIES} attempts: {last_error}"
    ) from last_error


def embed_query(text: str) -> list[float]:
    return embed_texts([text], task_type="RETRIEVAL_QUERY")[0]
