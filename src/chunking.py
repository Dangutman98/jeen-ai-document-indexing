"""Three chunking strategies: fixed-size with overlap, sentence-based, paragraph-based.

Chunk size/overlap are expressed in characters, not tokens, to keep the module
dependency-free. This is documented in the README as a deliberate simplification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

STRATEGIES = ("fixed", "sentence", "paragraph")

# A conservative sentence boundary: end punctuation followed by whitespace and
# a capital/quote/digit. Avoids splitting on "Dr." or "e.g." in the common case
# without pulling in a full NLP dependency for what is a best-effort splitter.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Zא-ת0-9\"'“])")


@dataclass(frozen=True)
class Chunk:
    text: str
    index: int


def chunk_text(text: str, strategy: str, *, chunk_size: int = 1000, overlap: int = 150) -> list[Chunk]:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Must be one of {STRATEGIES}")
    if strategy == "fixed":
        pieces = _chunk_fixed(text, chunk_size=chunk_size, overlap=overlap)
    elif strategy == "sentence":
        pieces = _chunk_sentence(text, chunk_size=chunk_size)
    else:
        pieces = _chunk_paragraph(text, chunk_size=chunk_size)

    return [Chunk(text=t, index=i) for i, t in enumerate(pieces) if t.strip()]


def _chunk_fixed(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    step = chunk_size - overlap
    return [text[i : i + chunk_size] for i in range(0, len(text), step)]


def _split_sentences(text: str) -> list[str]:
    # Sentence splitting operates within paragraphs so a chunk never silently
    # bridges two unrelated paragraphs.
    sentences: list[str] = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        sentences.extend(s.strip() for s in _SENTENCE_BOUNDARY.split(para) if s.strip())
    return sentences


def _chunk_sentence(text: str, *, chunk_size: int) -> list[str]:
    sentences = _split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sent in sentences:
        # A single sentence longer than chunk_size is kept whole rather than
        # cut mid-word -- sentence boundaries are the unit here, not characters.
        if current and current_len + 1 + len(sent) > chunk_size:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sent)
        current_len += len(sent) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def _chunk_paragraph(text: str, *, chunk_size: int) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Some source formats (notably PDF text layers) carry no blank-line
    # paragraph markers at all -- extraction yields one giant "paragraph".
    # Rather than silently return the whole document as a single chunk,
    # fall back to sentence grouping so the strategy still does its job.
    if len(paragraphs) <= 1 and len(text) > chunk_size:
        return _chunk_sentence(text, chunk_size=chunk_size)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        if current and current_len + 2 + len(para) > chunk_size:
            chunks.append("\n\n".join(current))
            current, current_len = [], 0
        current.append(para)
        current_len += len(para) + 2
    if current:
        chunks.append("\n\n".join(current))
    return chunks
