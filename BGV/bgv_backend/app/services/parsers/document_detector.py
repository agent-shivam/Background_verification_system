"""
app/services/parsers/document_detector.py
──────────────────────────────────────────
Keyword / regex heuristics to identify which type of document
was uploaded based on the OCR-extracted text.
"""

from __future__ import annotations

import re

from loguru import logger

from app.schemas.document import DocumentType


# ── Keyword signatures for each document type ─────────────────────────────────

_SIGNATURES: dict[DocumentType, list[str]] = {
    DocumentType.aadhaar: [
        r"aadhaar", r"unique identification", r"uidai", r"\d{4}\s\d{4}\s\d{4}",
        r"government of india", r"enrolment",
    ],
    DocumentType.pan: [
        r"permanent account number", r"income tax", r"\b[A-Z]{5}\d{4}[A-Z]\b",
        r"income tax department", r"pan card",
    ],
    DocumentType.passport: [
        r"passport", r"republic of india", r"nationality", r"place of birth",
        r"date of issue", r"date of expiry", r"[A-Z]{1}\d{7}",  # passport number
        r"machine readable", r"mrz",
    ],
    DocumentType.resume: [
        r"curriculum vitae", r"\bcv\b", r"\bresume\b", r"objective",
        r"work experience", r"employment history", r"references",
        r"skills", r"education", r"projects",
    ],
    DocumentType.graduation: [
        r"bachelor", r"master", r"degree", r"university", r"college",
        r"awarded", r"conferred", r"graduation", r"convocation",
        r"chancellor", r"registrar",
    ],
    DocumentType.marksheet: [
        r"mark sheet", r"marksheet", r"result", r"semester", r"examination",
        r"roll no", r"subject", r"marks obtained", r"total marks",
        r"grade", r"percentage",
    ],
}


def detect_document_type(text: str) -> DocumentType:
    """
    Score each document type against extracted text.
    Returns the type with the highest keyword hit count.
    Falls back to `unknown` if no type reaches the minimum threshold.
    """
    lower = text.lower()
    scores: dict[DocumentType, int] = {}

    for doc_type, patterns in _SIGNATURES.items():
        hits = sum(1 for p in patterns if re.search(p, lower))
        scores[doc_type] = hits
        logger.debug(f"  {doc_type.value}: {hits} keyword hit(s)")

    best_type, best_score = max(scores.items(), key=lambda kv: kv[1])

    if best_score == 0:
        logger.warning("Could not detect document type — returning 'unknown'")
        return DocumentType.unknown

    logger.info(f"Detected document type: {best_type.value} (score={best_score})")
    return best_type