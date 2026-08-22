"""Three chunking strategies: fixed-size with overlap, sentence-based, paragraph-based.

Chunk size/overlap are expressed in characters, not tokens, to keep the module
dependency-free. This is documented in the README as a deliberate simplification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .errors import InvalidArgumentError

STRATEGIES = ("fixed", "sentence", "paragraph")

# Real-world documents (as opposed to plain prose) are full of short
# standalone paragraphs -- headings, table-cell text, form labels, list
# items -- each separated by a blank line. Treating every paragraph
# boundary as a hard chunk break (needed to avoid bridging unrelated
# paragraphs) then means each of these becomes its own near-empty chunk,
# e.g. a lone "Developer" or "Login" heading. That wastes an embedding API
# call per fragment and pollutes search results with content-free matches.
# Found by testing against real DOCX/PDF files, not the synthetic samples.
MIN_CHUNK_CHARS = 30

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


def _merge_undersized(chunks: list[str], *, min_chars: int = MIN_CHUNK_CHARS) -> list[str]:
    """Fold any chunk shorter than min_chars into a neighbor.

    A heading is folded into whatever follows it (that's usually what it
    introduces); a trailing tiny fragment with nothing after it is folded
    into whatever precedes it instead. Runs of several consecutive tiny
    chunks (e.g. a stack of short labels) keep merging until the combined
    piece clears the threshold or the list is exhausted.
    """
    if len(chunks) <= 1:
        return chunks
    out = list(chunks)
    i = 0
    while i < len(out) and len(out) > 1:
        if len(out[i]) >= min_chars:
            i += 1
            continue
        if i + 1 < len(out):
            out[i + 1] = out[i] + " " + out[i + 1]
            del out[i]
            # re-check the merged result at this same position
        else:
            out[i - 1] = out[i - 1] + " " + out[i]
            del out[i]
            i -= 1
    return out


def _chunk_fixed(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise InvalidArgumentError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise InvalidArgumentError(f"overlap must not be negative, got {overlap}")
    if overlap >= chunk_size:
        raise InvalidArgumentError(
            f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})"
        )
    step = chunk_size - overlap
    return [text[i : i + chunk_size] for i in range(0, len(text), step)]


def _split_sentences_by_paragraph(text: str) -> list[list[str]]:
    # Returns sentences grouped by source paragraph (not flattened) so the
    # packing step below can keep paragraph boundaries as hard chunk breaks --
    # a flat list would lose exactly the information needed for that.
    paragraphs: list[list[str]] = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        sentences = [s.strip() for s in _SENTENCE_BOUNDARY.split(para) if s.strip()]
        if sentences:
            paragraphs.append(sentences)
    return paragraphs


def _chunk_sentence(text: str, *, chunk_size: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for paragraph_sentences in _split_sentences_by_paragraph(text):
        for sent in paragraph_sentences:
            # A single sentence longer than chunk_size is kept whole rather
            # than cut mid-word -- sentence boundaries are the unit here.
            if current and current_len + 1 + len(sent) > chunk_size:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            current.append(sent)
            current_len += len(sent) + 1
        # Paragraph boundary is a hard break: never let the next paragraph's
        # sentences merge into this chunk, even if there's room left.
        if current:
            chunks.append(" ".join(current))
            current, current_len = [], 0

    if current:
        chunks.append(" ".join(current))
    return _merge_undersized(chunks)


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
    return _merge_undersized(chunks)
