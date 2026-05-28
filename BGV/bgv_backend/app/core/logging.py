"""
app/core/logging.py
───────────────────
Loguru-based logging configuration.
Call `setup_logging()` once at application startup (in main.py).
"""

import sys
from loguru import logger
from app.core.config import settings


def setup_logging() -> None:
    """Configure Loguru sinks: stdout + rotating file."""

    # Remove the default handler
    logger.remove()

    # ── Console sink ──────────────────────────────────────────────────────────
    logger.add(
        sys.stdout,
        level=settings.log_level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
    )

    # ── File sink (rotating, 10 MB per file, 7-day retention) ─────────────────
    logger.add(
        settings.log_file,
        level=settings.log_level,
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        enqueue=True,          # thread-safe async logging
    )

    logger.info(
        f"Logging initialised — level={settings.log_level}, file={settings.log_file}"
    )