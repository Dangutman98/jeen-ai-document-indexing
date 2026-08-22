"""CLI-level argument validation -- these must fail cleanly (no raw
traceback) and, where possible, before spending an API call on bad input."""

from unittest.mock import patch

import search


def test_negative_limit_returns_clean_error_without_calling_embed_query(capsys):
    with patch("search.embed_query") as mock_embed:
        exit_code = search.main(["--query", "anything", "--limit", "-1"])

    assert exit_code == 1
    mock_embed.assert_not_called()
    captured = capsys.readouterr()
    assert "Error:" in captured.err
    assert "Traceback" not in captured.err


def test_zero_limit_returns_clean_error_without_calling_embed_query(capsys):
    with patch("search.embed_query") as mock_embed:
        exit_code = search.main(["--query", "anything", "--limit", "0"])

    assert exit_code == 1
    mock_embed.assert_not_called()
