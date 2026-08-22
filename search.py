#!/usr/bin/env python
"""Semantic search over indexed document chunks.

Usage:
    python search.py --query "login issue"
    python search.py --query "login issue" --limit 3
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from src.db import get_connection, search
from src.embeddings import embed_query
from src.errors import EmptySearchResultError, InvalidArgumentError, PipelineError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic search over indexed document chunks.")
    parser.add_argument("--query", required=True, help="Search query text")
    parser.add_argument("--limit", type=int, default=5, help="Number of results to return (default: 5)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)

    try:
        if args.limit <= 0:
            raise InvalidArgumentError(f"--limit must be positive, got {args.limit}")

        print(f"Embedding query: {args.query!r}")
        query_embedding = embed_query(args.query)

        with get_connection() as conn:
            results = search(conn, query_embedding, limit=args.limit)

        if not results:
            raise EmptySearchResultError(
                "No results found. Has anything been indexed yet? Try index_documents.py first."
            )

        print(f"\nTop {len(results)} result(s) for {args.query!r}:\n")
        for rank, r in enumerate(results, start=1):
            print(f"[{rank}] similarity={r.similarity:.4f}  file={r.filename}  strategy={r.split_strategy}  id={r.id}")
            preview = r.chunk_text[:220].replace("\n", " ")
            print(f"    {preview}{'...' if len(r.chunk_text) > 220 else ''}\n")
        return 0

    except PipelineError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
