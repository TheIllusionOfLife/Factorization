"""Centralized logging configuration for the application."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO", log_file: Optional[str] = None, console_output: bool = True
) -> logging.Logger:
    """
    Configure application-wide logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file. If None, no file logging.
        console_output: Whether to output logs to console (default: True)

    Returns:
        Root logger instance

    Environment Variables:
        LOG_LEVEL: Override default log level (DEBUG, INFO, WARNING, ERROR)
        LOG_FILE: Override default log file path

    Example:
        >>> import os
        >>> os.environ['LOG_LEVEL'] = 'DEBUG'
        >>> logger = setup_logging()
        >>> logger.info("Application started")
    """
    # Parse log level
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear any existing handlers (prevent duplicates)
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (if enabled)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler (if path provided)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.debug("Detailed debug info")
    """
    return logging.getLogger(name)


__all__ = ["setup_logging", "get_logger"]
