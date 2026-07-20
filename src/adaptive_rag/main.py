"""CLI entry point."""

import uvicorn

from adaptive_rag.config.settings import get_settings


def main() -> None:
    """Run the FastAPI application."""
    settings = get_settings()
    uvicorn.run(
        "adaptive_rag.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()
