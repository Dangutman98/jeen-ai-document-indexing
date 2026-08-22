"""Unit tests for src/embeddings.py.

Retry/backoff and API-key handling are tested against a mocked client --
these must never make real network calls or actually sleep. The one live
test that touches the real Gemini API is separate and explicitly opt-in
(see tests/test_integration.py).
"""

import math
from unittest.mock import MagicMock, patch

import pytest
from google.genai.errors import APIError

from src.embeddings import _renormalize, embed_texts
from src.errors import EmbeddingError


# ---------------------------------------------------------------- renormalize

def test_renormalize_produces_unit_vector():
    v = [3.0, 4.0]  # 3-4-5 triangle, norm = 5
    out = _renormalize(v)
    norm = math.sqrt(sum(x * x for x in out))
    assert norm == pytest.approx(1.0)
    assert out == pytest.approx([0.6, 0.8])


def test_renormalize_zero_vector_does_not_divide_by_zero():
    v = [0.0, 0.0, 0.0]
    out = _renormalize(v)
    assert out == v  # returned unchanged, no ZeroDivisionError


def test_renormalize_preserves_direction():
    v = [1.0, 2.0, 3.0]
    out = _renormalize(v)
    # renormalized vector should be a positive scalar multiple of the original
    ratios = [o / x for o, x in zip(out, v) if x != 0]
    assert all(r == pytest.approx(ratios[0]) for r in ratios)


# ---------------------------------------------------------------- api key

def test_missing_api_key_raises_embedding_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(EmbeddingError, match="GEMINI_API_KEY"):
        embed_texts(["hello"])


def test_empty_input_returns_empty_list_without_calling_api(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    # Should short-circuit before ever needing a client/key.
    assert embed_texts([]) == []


# ---------------------------------------------------------------- retry/backoff

def _fake_embedding_response(dim=1536, n=1):
    resp = MagicMock()
    resp.embeddings = [MagicMock(values=[0.1] * dim) for _ in range(n)]
    return resp


def test_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
    monkeypatch.setattr("src.embeddings.time.sleep", lambda *_: None)

    mock_client = MagicMock()
    mock_client.models.embed_content.side_effect = [
        APIError(429, {"error": {"message": "rate limited"}}),
        _fake_embedding_response(),
    ]
    with patch("src.embeddings.genai.Client", return_value=mock_client):
        result = embed_texts(["hello"])

    assert len(result) == 1
    assert len(result[0]) == 1536
    assert mock_client.models.embed_content.call_count == 2


def test_does_not_retry_on_400_bad_request(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
    monkeypatch.setattr("src.embeddings.time.sleep", lambda *_: None)

    mock_client = MagicMock()
    mock_client.models.embed_content.side_effect = APIError(
        400, {"error": {"message": "bad request"}}
    )
    with patch("src.embeddings.genai.Client", return_value=mock_client):
        with pytest.raises(EmbeddingError):
            embed_texts(["hello"])

    # 400 is not transient -- must fail on the first attempt, no retries burned.
    assert mock_client.models.embed_content.call_count == 1


def test_exhausts_retries_and_raises(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
    monkeypatch.setattr("src.embeddings.time.sleep", lambda *_: None)

    mock_client = MagicMock()
    mock_client.models.embed_content.side_effect = APIError(
        503, {"error": {"message": "unavailable"}}
    )
    with patch("src.embeddings.genai.Client", return_value=mock_client):
        with pytest.raises(EmbeddingError, match="after 4 attempts"):
            embed_texts(["hello"])

    from src.embeddings import MAX_RETRIES
    assert mock_client.models.embed_content.call_count == MAX_RETRIES


def test_batches_large_input(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
    monkeypatch.setattr("src.embeddings.time.sleep", lambda *_: None)

    from src.embeddings import BATCH_SIZE

    mock_client = MagicMock()
    mock_client.models.embed_content.side_effect = [
        _fake_embedding_response(n=BATCH_SIZE),
        _fake_embedding_response(n=5),
    ]
    texts = ["chunk"] * (BATCH_SIZE + 5)
    with patch("src.embeddings.genai.Client", return_value=mock_client):
        result = embed_texts(texts)

    assert len(result) == BATCH_SIZE + 5
    assert mock_client.models.embed_content.call_count == 2
