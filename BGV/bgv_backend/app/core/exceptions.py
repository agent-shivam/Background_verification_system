"""
app/core/exceptions.py
──────────────────────
Custom exception hierarchy for the BGV system.
All domain exceptions inherit from BGVBaseError so callers
can catch the whole family with a single except clause.
"""

from __future__ import annotations


class BGVBaseError(Exception):
    """Root of the BGV exception tree."""
    http_status: int = 500

    def __init__(self, message: str = "An unexpected error occurred."):
        super().__init__(message)
        self.message = message


# ── File validation ────────────────────────────────────────────────────────────

class FileValidationError(BGVBaseError):
    """Base for all file-related validation failures."""
    http_status = 422


class FileTooLargeError(FileValidationError):
    """Uploaded file exceeds the configured size limit."""


class UnsupportedFileTypeError(FileValidationError):
    """MIME type or extension is not in the allow-list."""


# ── Processing ────────────────────────────────────────────────────────────────

class PreprocessingError(BGVBaseError):
    """Image preprocessing (resize, deskew, binarise) failed."""
    http_status = 500


class PDFConversionError(BGVBaseError):
    """pdf2image / poppler conversion failed."""
    http_status = 500


class OCRExtractionError(BGVBaseError):
    """PaddleOCR could not extract text from the image."""
    http_status = 500


# ── Forgery detection ─────────────────────────────────────────────────────────

class ForgeryDetectionError(BGVBaseError):
    """One or more forgery-detection modules raised an unrecoverable error."""
    http_status = 500


# ── QR / Barcode ──────────────────────────────────────────────────────────────

class QRDecodeError(BGVBaseError):
    """pyzbar could not decode the QR / barcode."""
    http_status = 422


# ── Risk scoring ──────────────────────────────────────────────────────────────

class RiskScoringError(BGVBaseError):
    """Risk scoring engine encountered an unexpected state."""
    http_status = 500