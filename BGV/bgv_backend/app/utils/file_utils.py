"""
app/utils/file_utils.py
───────────────────────
File I/O helpers: validation, safe-save, MIME detection, cleanup.
"""

import uuid
import shutil
from pathlib import Path

import magic                         # python-magic — MIME sniffing
from fastapi import UploadFile
from loguru import logger

from app.core.config import settings
from app.core.exceptions import FileTooLargeError, UnsupportedFileTypeError


# ── Allowed MIME types ────────────────────────────────────────────────────────
ALLOWED_MIME_TYPES: set[str] = {
    "application/pdf",
    "image/jpeg",
    "image/png",
    "image/tiff",
    "image/webp",
}


def validate_upload(file: UploadFile, content: bytes) -> None:
    """
    Raise FileValidationError subclasses if:
    - file exceeds MAX_FILE_SIZE_MB
    - MIME type is not in the allow-list
    """
    # Size check
    if len(content) > settings.max_file_size_bytes:
        raise FileTooLargeError(
            f"File '{file.filename}' is {len(content) / 1_048_576:.1f} MB — "
            f"limit is {settings.max_file_size_mb} MB."
        )

    # MIME sniff (ignore client-supplied Content-Type)
    mime = magic.from_buffer(content[:2048], mime=True)
    if mime not in ALLOWED_MIME_TYPES:
        raise UnsupportedFileTypeError(
            f"MIME type '{mime}' is not supported. "
            f"Allowed: {', '.join(sorted(ALLOWED_MIME_TYPES))}"
        )

    logger.debug(f"File validated: name={file.filename}, size={len(content)} B, mime={mime}")


def save_upload(content: bytes, original_filename: str) -> Path:
    """
    Persist uploaded bytes to the uploads directory with a UUID prefix.
    Returns the absolute path of the saved file.
    """
    suffix = Path(original_filename).suffix.lower() or ".bin"
    dest = settings.upload_dir / f"{uuid.uuid4().hex}{suffix}"
    dest.write_bytes(content)
    logger.debug(f"Saved upload → {dest}")
    return dest


def cleanup_file(path: Path) -> None:
    """Delete a temporary file, swallowing errors silently."""
    try:
        if path and path.exists():
            path.unlink()
            logger.debug(f"Cleaned up temp file: {path}")
    except Exception as exc:
        logger.warning(f"Could not delete temp file {path}: {exc}")


def cleanup_dir(path: Path) -> None:
    """Recursively delete a temporary directory."""
    try:
        if path and path.exists():
            shutil.rmtree(path, ignore_errors=True)
            logger.debug(f"Cleaned up temp dir: {path}")
    except Exception as exc:
        logger.warning(f"Could not delete temp dir {path}: {exc}")