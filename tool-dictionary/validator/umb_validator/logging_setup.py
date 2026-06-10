"""Structured JSON logging. SPEC v2 §8.

One INFO line per state transition / gate decision / ntfy push; DEBUG per LLM
call; ERROR with stack per error. JSON to stdout -> captured by
`journalctl -u umb-validator`.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    """Configure structlog to emit JSON (or console) to stdout."""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Any:
    """Return a bound structlog logger.

    Typed as `Any`: structlog's bound-logger type is configuration-dependent
    (the filtering wrapper class is built at `configure_logging` time), so a
    precise static type is not available.
    """
    return structlog.get_logger(name)
