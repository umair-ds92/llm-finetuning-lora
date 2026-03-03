"""
Logging utilities for the fine-tuning pipeline.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: str = "INFO",
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.
    
    Args:
        name: Name of the logger
        log_file: Path to log file (optional)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
        
    Returns:
        Configured logger instance
    """
    # Remove default logger
    logger.remove()
    
    # Default format
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # Add console handler
    logger.add(
        sys.stderr,
        format=format_string,
        level=level,
        colorize=True,
    )
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            format=format_string,
            level=level,
            rotation="100 MB",  # Rotate when file reaches 100MB
            retention="10 days",  # Keep logs for 10 days
            compression="zip",  # Compress rotated logs
        )
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance by name.
    
    Args:
        name: Name of the logger
        
    Returns:
        Logger instance
    """
    return logger.bind(name=name)