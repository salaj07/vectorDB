"""
VectorForge — Custom Exception Classes

Converted into appropriate HTTP responses at the API layer.
"""


class VectorForgeError(Exception):
    """Base exception for all VectorForge errors."""
    pass


class VectorNotFoundError(VectorForgeError):
    """Raised when a vector ID does not exist in the store."""

    def __init__(self, vector_id: str):
        self.vector_id = vector_id
        super().__init__(f"Vector '{vector_id}' not found")


class DuplicateVectorError(VectorForgeError):
    """Raised when inserting a vector with an ID that already exists."""

    def __init__(self, vector_id: str):
        self.vector_id = vector_id
        super().__init__(f"Vector '{vector_id}' already exists")


class DimensionMismatchError(VectorForgeError):
    """Raised when the vector dimension does not match the store dimension."""

    def __init__(self, expected: int, got: int):
        self.expected = expected
        self.got = got
        super().__init__(
            f"Dimension mismatch: expected {expected}, got {got}"
        )


class InvalidSearchParameterError(VectorForgeError):
    """Raised when a search parameter is invalid (k, nprobe, ef_search)."""

    def __init__(self, param: str, value, reason: str = ""):
        self.param = param
        self.value = value
        msg = f"Invalid search parameter '{param}={value}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class IndexNotBuiltError(VectorForgeError):
    """Raised when searching an index that has not been built yet."""

    def __init__(self, index_name: str):
        self.index_name = index_name
        super().__init__(
            f"Index '{index_name}' is not built. Call build() first."
        )
