"""
Logging Configuration - Structured JSON logging with structlog.

Features:
- JSON output for log aggregation
- ISO timestamp formatting
- Correlation IDs for distributed tracing
- Configurable log levels
"""
from __future__ import annotations

import logging
import sys
from typing import Any, Optional

import structlog


def configure_logging(
    level: str = "INFO",
    json_output: bool = True,
    show_locals: bool = False,
) -> None:
    """
    Configure structured logging for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: If True, output JSON logs. Otherwise, console format.
        show_locals: If True, include local variables in tracebacks.
    """
    # Set up stdlib logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )
    
    # Build processor chain
    processors = [
        # Add log level
        structlog.stdlib.add_log_level,
        # Add logger name
        structlog.stdlib.add_logger_name,
        # Filter by level
        structlog.stdlib.filter_by_level,
        # Add timestamp
        structlog.processors.TimeStamper(fmt="iso"),
        # Add caller info
        structlog.processors.CallsiteParameterAdder(
            parameters=[
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
                structlog.processors.CallsiteParameter.FUNC_NAME,
            ]
        ),
        # Stack info for exceptions
        structlog.processors.StackInfoRenderer(),
        # Format exception info
        structlog.processors.format_exc_info,
        # Decode unicode
        structlog.processors.UnicodeDecoder(),
    ]
    
    # Add final renderer
    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """
    Get a configured logger.
    
    Args:
        name: Optional logger name (defaults to calling module)
        
    Returns:
        Configured structlog logger
    """
    if name:
        return structlog.get_logger(name)
    return structlog.get_logger()


class CorrelationIdProcessor:
    """Add correlation ID to log entries for distributed tracing."""
    
    _correlation_id: Optional[str] = None
    
    @classmethod
    def set_correlation_id(cls, correlation_id: str) -> None:
        """Set the current correlation ID."""
        cls._correlation_id = correlation_id
    
    @classmethod
    def clear_correlation_id(cls) -> None:
        """Clear the current correlation ID."""
        cls._correlation_id = None
    
    def __call__(
        self,
        logger: logging.Logger,
        method_name: str,
        event_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """Add correlation ID to event dict."""
        if self._correlation_id:
            event_dict["correlation_id"] = self._correlation_id
        return event_dict
