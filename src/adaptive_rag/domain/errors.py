"""Domain-level errors mapped to HTTP responses at the API boundary."""


class DomainError(Exception):
    """Base class for domain errors."""

    code: str = "domain_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidCollectionIdError(DomainError):
    """Raised when a collection identifier is malformed or unsafe."""

    code = "invalid_collection_id"


class CollectionNotFoundError(DomainError):
    """Raised when a collection does not exist."""

    code = "collection_not_found"


class PathAccessError(DomainError):
    """Raised when a filesystem path is outside an allowed directory."""

    code = "path_access_denied"


class EmbedderCompatibilityError(DomainError):
    """Raised when an index embedder is incompatible with the active embedder."""

    code = "embedder_incompatible"


class ProviderError(DomainError):
    """Raised when an external provider call fails."""

    code = "provider_error"
