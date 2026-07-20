"""Structured logging setup."""

from __future__ import annotations

import logging
import sys
from typing import Any

from adaptive_rag.config.settings import LoggingSettings


class StructuredFormatter(logging.Formatter):
    """Simple key=value formatter for production logs."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = " ".join(f"{k}={v}" for k, v in sorted(record.__dict__.items()) if k.startswith("ctx_"))
        return f"{base} {extras}".strip()


def setup_logging(settings: LoggingSettings) -> None:
    """Configure root logger with structured output."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(settings.level)

    handler = logging.StreamHandler(sys.stdout)
    if settings.json_logs:
        handler.setFormatter(logging.Formatter('{"level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}'))
    else:
        handler.setFormatter(
            StructuredFormatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )

    root.addHandler(handler)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger."""
    return logging.getLogger(name)


def log_context(logger: logging.Logger, level: int, message: str, **context: Any) -> None:
    """Emit a log record with ctx_* fields for structured context."""
    extra = {f"ctx_{key}": value for key, value in context.items()}
    logger.log(level, message, extra=extra)
