"""Tests for logging configuration."""

import logging
import tempfile
from pathlib import Path

from src.logging_config import get_logger, setup_logging


def test_setup_logging_default():
    """Test default logging configuration."""
    logger = setup_logging()
    assert logger.level == logging.INFO
    assert len(logger.handlers) >= 1  # At least console handler


def test_setup_logging_custom_level():
    """Test custom log level."""
    logger = setup_logging(level="DEBUG")
    assert logger.level == logging.DEBUG


def test_setup_logging_with_file():
    """Test file logging."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test.log"
        logger = setup_logging(log_file=str(log_file))

        # Check handlers
        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 1

        # Test actual logging
        test_logger = get_logger("test_module")
        test_logger.info("Test message")

        # Verify file was created and written
        assert log_file.exists()
        content = log_file.read_text()
        assert "Test message" in content


def test_get_logger():
    """Test logger retrieval."""
    logger = get_logger("test_module")
    assert logger.name == "test_module"
    assert isinstance(logger, logging.Logger)


def test_logging_levels_hierarchy():
    """Test that log levels work correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "levels.log"
        setup_logging(level="WARNING", log_file=str(log_file))

        logger = get_logger("test")
        logger.debug("debug message")  # Should NOT appear
        logger.info("info message")  # Should NOT appear
        logger.warning("warning message")  # Should appear
        logger.error("error message")  # Should appear

        content = log_file.read_text()
        assert "debug message" not in content
        assert "info message" not in content
        assert "warning message" in content
        assert "error message" in content


def test_setup_logging_no_console():
    """Test logging without console output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "no_console.log"
        logger = setup_logging(log_file=str(log_file), console_output=False)

        # Should only have file handler, no console handler
        console_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ]
        assert len(console_handlers) == 0

        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 1


def test_setup_logging_creates_log_directory():
    """Test that logging creates parent directories automatically."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "nested" / "dir" / "test.log"
        assert not log_file.parent.exists()

        setup_logging(log_file=str(log_file))

        assert log_file.parent.exists()
        assert log_file.exists()


def test_setup_logging_clears_existing_handlers():
    """Test that setup_logging clears existing handlers to prevent duplicates."""
    # First setup
    logger1 = setup_logging(level="INFO")
    initial_handler_count = len(logger1.handlers)

    # Second setup should clear and recreate
    logger2 = setup_logging(level="DEBUG")
    assert len(logger2.handlers) == initial_handler_count  # Should be same, not doubled
