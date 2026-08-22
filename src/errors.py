"""Exception types for the indexing/search pipeline.

Each maps to one of the mandatory error-handling cases: missing file,
unsupported file type, document with no text, embedding failure, database
connection failure, and empty search results.
"""


class PipelineError(Exception):
    """Base class for all expected, user-facing pipeline failures."""


class FileNotFoundPipelineError(PipelineError):
    pass


class UnsupportedFileTypeError(PipelineError):
    pass


class NoExtractableTextError(PipelineError):
    pass


class EmbeddingError(PipelineError):
    pass


class DatabaseConnectionError(PipelineError):
    pass


class EmptySearchResultError(PipelineError):
    pass


class InvalidArgumentError(PipelineError, ValueError):
    """Bad CLI argument values (chunk-size/overlap/limit combinations, etc).

    Inherits ValueError too so existing call sites that already catch
    ValueError (e.g. library callers of chunk_text() that aren't going
    through the CLI) keep working unchanged, while the CLIs' `except
    PipelineError` also catches it -- one raise satisfies both.
    """
