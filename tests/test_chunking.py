"""Unit tests for src/chunking.py -- pure logic, no I/O, no network."""

import pytest

from src.chunking import chunk_text, STRATEGIES


# ---------------------------------------------------------------- shared

def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        chunk_text("some text", "nonsense")


def test_empty_text_returns_no_chunks():
    for strategy in STRATEGIES:
        assert chunk_text("", strategy) == []


def test_whitespace_only_text_returns_no_chunks():
    for strategy in STRATEGIES:
        assert chunk_text("   \n\n   \n  ", strategy) == []


def test_chunk_indices_are_sequential_from_zero():
    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    chunks = chunk_text(text, "sentence", chunk_size=20)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_all_strategies_cover_the_source_text():
    # Every strategy should account for the substantive content -- no
    # strategy should silently drop text.
    text = "Alpha beta gamma. Delta epsilon zeta.\n\nEta theta iota. Kappa lambda mu."
    for strategy in STRATEGIES:
        chunks = chunk_text(text, strategy, chunk_size=1000)
        combined = " ".join(c.text for c in chunks)
        for word in ["Alpha", "gamma", "Eta", "lambda"]:
            assert word in combined, f"{strategy} lost word {word!r}"


# ---------------------------------------------------------------- fixed

def test_fixed_produces_expected_chunk_count():
    text = "x" * 1000
    chunks = chunk_text(text, "fixed", chunk_size=100, overlap=0)
    assert len(chunks) == 10


def test_fixed_overlap_repeats_boundary_text():
    text = "0123456789" * 10  # 100 chars
    chunks = chunk_text(text, "fixed", chunk_size=30, overlap=10)
    # chunk 1 ends where chunk 2 begins, minus the 10-char overlap
    assert chunks[0].text[-10:] == chunks[1].text[:10]


def test_fixed_overlap_equal_to_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_text("some text here", "fixed", chunk_size=50, overlap=50)


def test_fixed_overlap_greater_than_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_text("some text here", "fixed", chunk_size=50, overlap=80)


def test_fixed_text_shorter_than_chunk_size_returns_one_chunk():
    chunks = chunk_text("short", "fixed", chunk_size=1000, overlap=100)
    assert len(chunks) == 1
    assert chunks[0].text == "short"


def test_fixed_zero_overlap_produces_no_duplicated_content():
    text = "abcdefghij" * 5  # 50 chars
    chunks = chunk_text(text, "fixed", chunk_size=10, overlap=0)
    assert "".join(c.text for c in chunks) == text


# ---------------------------------------------------------------- sentence

def test_sentence_does_not_split_short_abbreviations_aggressively():
    # Not a hard guarantee (heuristic regex), but the common case should hold.
    text = "Dr. Smith arrived. He was late."
    chunks = chunk_text(text, "sentence", chunk_size=1000)
    assert len(chunks) == 1
    assert "Dr. Smith arrived." in chunks[0].text


def test_sentence_never_bridges_paragraph_boundaries():
    text = "First paragraph sentence one. First paragraph sentence two.\n\nSecond paragraph sentence one."
    chunks = chunk_text(text, "sentence", chunk_size=1000)
    # with a generous chunk_size, still must not merge across \n\n
    combined_texts = [c.text for c in chunks]
    assert not any("First paragraph sentence two. Second paragraph" in t for t in combined_texts)


def test_sentence_longer_than_chunk_size_kept_whole():
    long_sentence = "This is a single very long sentence that exceeds the tiny chunk size limit deliberately."
    chunks = chunk_text(long_sentence, "sentence", chunk_size=20)
    assert len(chunks) == 1
    assert chunks[0].text == long_sentence


def test_sentence_hebrew_text_splits_correctly():
    text = "זו משפט ראשון בעברית. וזה משפט שני בעברית. ומשפט שלישי."
    chunks = chunk_text(text, "sentence", chunk_size=25)
    assert len(chunks) >= 2
    combined = " ".join(c.text for c in chunks)
    assert "משפט ראשון" in combined and "משפט שלישי" in combined


def test_sentence_no_terminal_punctuation_still_returns_chunk():
    text = "just a fragment with no ending punctuation at all"
    chunks = chunk_text(text, "sentence", chunk_size=1000)
    assert len(chunks) == 1
    assert chunks[0].text == text


# ---------------------------------------------------------------- paragraph

def test_paragraph_splits_on_blank_lines():
    text = "Para one line.\n\nPara two line.\n\nPara three line."
    chunks = chunk_text(text, "paragraph", chunk_size=15)
    assert len(chunks) == 3


def test_paragraph_groups_short_paragraphs_together():
    text = "A.\n\nB.\n\nC."
    chunks = chunk_text(text, "paragraph", chunk_size=1000)
    assert len(chunks) == 1
    assert "A." in chunks[0].text and "C." in chunks[0].text


def test_paragraph_fallback_when_no_blank_lines_present():
    # Simulates a PDF text layer with no \n\n markers at all (the real bug
    # found and fixed during development).
    text = "Sentence one is here. Sentence two follows. Sentence three ends it. " * 5
    chunks = chunk_text(text, "paragraph", chunk_size=100)
    assert len(chunks) > 1, "paragraph strategy must not collapse to a single giant chunk"


def test_paragraph_single_short_paragraph_no_fallback_needed():
    text = "Just one short paragraph, no blank lines, under the size limit."
    chunks = chunk_text(text, "paragraph", chunk_size=1000)
    assert len(chunks) == 1
    assert chunks[0].text == text
