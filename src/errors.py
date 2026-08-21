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
