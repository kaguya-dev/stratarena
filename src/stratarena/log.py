"""Structured logging setup."""

import logging
import sys
from typing import Any

import structlog


class _StderrLogger:
    """Writes each rendered event to the *current* ``sys.stderr``.

    Looking the stream up on every call (instead of binding it once) keeps logging working
    when stderr is swapped, e.g. by test runners or CLI harnesses.
    """

    def msg(self, message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    debug = info = warning = warn = error = critical = exception = msg


def _stderr_logger_factory(*_: Any) -> _StderrLogger:
    return _StderrLogger()


def configure_logging(level: int = logging.INFO, json: bool = False) -> None:
    """Configure structlog for console (default) or JSON-lines output on stderr."""
    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=_stderr_logger_factory,
        cache_logger_on_first_use=False,
    )


def get_logger(name: str) -> structlog.typing.FilteringBoundLogger:
    logger: structlog.typing.FilteringBoundLogger = structlog.get_logger(name)
    return logger
