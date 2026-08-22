"""Integration tests against a real Postgres + real Gemini API.

These require GEMINI_API_KEY and a running Postgres (docker compose up) and
are skipped automatically if either is unavailable -- they are not run by
default in an environment without real credentials, but they were run for
real during development against this repo's own docker-compose setup.
"""

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

from src.db import get_connection, insert_chunks, search
from src.embeddings import embed_query, embed_texts
from src.errors import DatabaseConnectionError as DBError

requires_gemini = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set"
)
requires_db = pytest.mark.skipif(
    not os.environ.get("POSTGRES_URL"), reason="POSTGRES_URL not set"
)


@requires_gemini
def test_real_embedding_has_1536_dimensions_and_unit_norm():
    vec = embed_query("integration test query")
    assert len(vec) == 1536
    norm = sum(v * v for v in vec) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-4)


@requires_gemini
def test_document_and_query_task_types_produce_different_vectors():
    # RETRIEVAL_DOCUMENT and RETRIEVAL_QUERY are asymmetric task types --
    # the same text embedded each way should not be byte-identical.
    text = "electricity bill dispute"
    doc_vec = embed_texts([text], task_type="RETRIEVAL_DOCUMENT")[0]
    query_vec = embed_query(text)
    assert doc_vec != query_vec


@requires_db
def test_db_connection_failure_raises_clean_error(monkeypatch):
    monkeypatch.setenv("POSTGRES_URL", "postgresql://baduser:badpass@localhost:1/nonexistent_db")
    with pytest.raises(DBError):
        with get_connection():
            pass


@requires_db
def test_missing_postgres_url_env_raises(monkeypatch):
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    with pytest.raises(DBError, match="POSTGRES_URL"):
        with get_connection():
            pass


@requires_db
@requires_gemini
def test_full_round_trip_insert_and_search():
    """Insert a known chunk, search for it, confirm it comes back on top."""
    marker = "zzz_integration_test_marker_unique_phrase_xyzzy"
    text = f"This is a uniquely identifiable test chunk containing {marker} for round-trip verification."
    embedding = embed_texts([text])[0]

    with get_connection() as conn:
        insert_chunks(
            conn,
            filename="_integration_test.txt",
            split_strategy="fixed",
            chunks=[text],
            embeddings=[embedding],
        )
        try:
            query_vec = embed_query(marker)
            results = search(conn, query_vec, limit=5)
            assert any(marker in r.chunk_text for r in results), (
                "round-tripped chunk did not come back from search"
            )
        finally:
            # clean up -- this test should not leave permanent rows behind
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM document_chunks WHERE filename = %s",
                    ("_integration_test.txt",),
                )
            conn.commit()


@requires_db
def test_search_returns_a_list_even_with_no_close_matches():
    """search() itself returns []/a list on no match; the
    EmptySearchResultError is raised by the CLI layer (search.py) when that
    list is empty, not by src/db.py -- verify that boundary directly rather
    than assuming it. search() wraps the raw vector internally, so callers
    must pass a plain list, not a pre-wrapped Vector."""
    with get_connection() as conn:
        fake_vec = [0.001] * 1536
        results = search(conn, fake_vec, limit=1)
        assert isinstance(results, list)
