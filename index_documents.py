#!/usr/bin/env python
"""Index a PDF or DOCX file into Postgres/pgvector.

Usage:
    python index_documents.py --file ./docs/example.pdf --strategy paragraph
    python index_documents.py --file ./docs/example.docx --strategy fixed --chunk-size 800 --overlap 100
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from src.chunking import STRATEGIES, chunk_text
from src.db import get_connection, insert_chunks
from src.embeddings import embed_texts
from src.errors import PipelineError
from src.extract import extract_text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index a PDF/DOCX file for semantic search.")
    parser.add_argument("--file", required=True, help="Path to a .pdf or .docx file")
    parser.add_argument("--strategy", required=True, choices=STRATEGIES, help="Chunking strategy")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Target chunk size in characters (default: 1000)")
    parser.add_argument("--overlap", type=int, default=150, help="Overlap in characters, fixed strategy only (default: 150)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)

    try:
        print(f"Extracting text from {args.file} ...")
        text = extract_text(args.file)
        print(f"  extracted {len(text):,} characters")

        print(f"Chunking with strategy='{args.strategy}' ...")
        chunks = chunk_text(text, args.strategy, chunk_size=args.chunk_size, overlap=args.overlap)
        print(f"  produced {len(chunks)} chunks")
        if not chunks:
            print("No chunks produced -- nothing to index.")
            return 1

        print(f"Generating embeddings via gemini-embedding-001 ({len(chunks)} chunks) ...")
        chunk_texts = [c.text for c in chunks]
        embeddings = embed_texts(chunk_texts)
        print(f"  received {len(embeddings)} embeddings, {len(embeddings[0])} dims each")

        print("Storing in Postgres ...")
        import os
        filename = os.path.basename(args.file)
        with get_connection() as conn:
            count = insert_chunks(
                conn,
                filename=filename,
                split_strategy=args.strategy,
                chunks=chunk_texts,
                embeddings=embeddings,
            )
        print(f"Done. Inserted {count} rows for '{filename}' (strategy={args.strategy}).")
        return 0

    except PipelineError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
