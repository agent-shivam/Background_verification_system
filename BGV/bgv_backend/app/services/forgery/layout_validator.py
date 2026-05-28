"""
app/services/forgery/layout_validator.py
─────────────────────────────────────────
Structural / layout validation for identity documents.

Approach:
- Each document type has expected keyword anchors and rough spatial zones.
- We verify that mandatory keywords appear in the OCR text.
- We check aspect-ratio heuristics (portrait vs landscape, rough proportions).
- We flag if expected section count is far below the minimum.

This is a lightweight heuristic layer — it is NOT template matching via
pixel alignment (which would require reference template images).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from app.schemas.document import DocumentType, LayoutResult


# ── Layout specification per document type ────────────────────────────────────

@dataclass
class LayoutSpec:
    """Minimum requirements for a document layout to be considered valid."""
    required_keywords: list[str] = field(default_factory=list)
    min_keyword_hits: int = 2               # how many must be present
    expected_aspect_ratio: tuple[float, float] = (0.5, 2.5)  # (min, max) w/h
    min_text_lines: int = 4                 # OCR must produce at least N lines


_SPECS: dict[DocumentType, LayoutSpec] = {
    DocumentType.aadhaar: LayoutSpec(
        required_keywords=["aadhaar", "uidai", "unique", "government", "india", "enrolment"],
        min_keyword_hits=2,
        expected_aspect_ratio=(0.55, 1.80),   # typical card ~1.58
        min_text_lines=5,
    ),
    DocumentType.pan: LayoutSpec(
        required_keywords=["permanent", "account", "income", "tax", "department"],
        min_keyword_hits=2,
        expected_aspect_ratio=(0.55, 1.80),
        min_text_lines=5,
    ),
    DocumentType.passport: LayoutSpec(
        required_keywords=["passport", "republic", "india", "nationality", "surname"],
        min_keyword_hits=2,
        expected_aspect_ratio=(0.60, 1.60),
        min_text_lines=8,
    ),
    DocumentType.resume: LayoutSpec(
        required_keywords=["experience", "education", "skills", "email", "name"],
        min_keyword_hits=2,
        expected_aspect_ratio=(0.60, 1.80),
        min_text_lines=10,
    ),
    DocumentType.graduation: LayoutSpec(
        required_keywords=["university", "degree", "awarded", "student", "college"],
        min_keyword_hits=2,
        expected_aspect_ratio=(0.60, 1.80),
        min_text_lines=5,
    ),
    DocumentType.marksheet: LayoutSpec(
        required_keywords=["marks", "examination", "roll", "subject", "result"],
        min_keyword_hits=2,
        expected_aspect_ratio=(0.60, 1.80),
        min_text_lines=6,
    ),
}

_DEFAULT_SPEC = LayoutSpec()


def validate_layout(
    text: str,
    doc_type: DocumentType,
    image: np.ndarray | None = None,
) -> LayoutResult:
    """
    Run layout heuristic checks against the detected document type.

    Parameters
    ----------
    text      : Full OCR text (lowercased inside this function).
    doc_type  : Detected DocumentType enum value.
    image     : Optional original image for aspect-ratio check.

    Returns
    -------
    LayoutResult
    """
    spec = _SPECS.get(doc_type, _DEFAULT_SPEC)
    lower = text.lower()

    violations: list[str] = []

    # ── 1. Keyword presence ───────────────────────────────────────────────────
    hits = sum(1 for kw in spec.required_keywords if re.search(kw, lower))
    if hits < spec.min_keyword_hits:
        violations.append(
            f"keyword_hits={hits} (expected ≥{spec.min_keyword_hits})"
        )

    # ── 2. Minimum text lines ─────────────────────────────────────────────────
    line_count = len([l for l in text.splitlines() if l.strip()])
    if line_count < spec.min_text_lines:
        violations.append(
            f"text_lines={line_count} (expected ≥{spec.min_text_lines})"
        )

    # ── 3. Aspect-ratio check (only if image provided) ────────────────────────
    if image is not None:
        h, w = image.shape[:2]
        ar = w / h if h > 0 else 1.0
        lo, hi = spec.expected_aspect_ratio
        if not (lo <= ar <= hi):
            violations.append(
                f"aspect_ratio={ar:.2f} (expected {lo}–{hi})"
            )

    # ── Verdict ───────────────────────────────────────────────────────────────
    if doc_type == DocumentType.unknown:
        result = LayoutResult.unknown
    elif violations:
        result = LayoutResult.invalid
        logger.info(f"Layout invalid for {doc_type.value}: {'; '.join(violations)}")
    else:
        result = LayoutResult.valid

    logger.debug(f"Layout validation → {result.value} ({doc_type.value})")
    return result