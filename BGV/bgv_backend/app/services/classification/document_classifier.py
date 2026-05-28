"""
app/services/classification/document_classifier.py
────────────────────────────────────────────────────
Advanced Document Classification Layer.

Improves over the basic keyword-score detector by:
  • Weighted keyword scoring (rare/specific keywords score higher)
  • Structural feature signals (number patterns, MRZ, logo text)
  • Confidence percentage per candidate type
  • Returns top-N candidates with scores for audit trail

Used at the very start of the pipeline (before field parsing) so
downstream parsers always receive a high-confidence document type.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from loguru import logger

from app.schemas.document import DocumentType


# ── Weighted signature definitions ────────────────────────────────────────────
# Each entry: (regex_pattern, weight)
# Higher weight = stronger signal for that document type

_WEIGHTED_SIGNATURES: dict[DocumentType, list[tuple[str, float]]] = {
    DocumentType.aadhaar: [
        (r"aadhaar", 3.0),
        (r"uidai", 3.0),
        (r"unique identification authority", 3.0),
        (r"\d{4}\s\d{4}\s\d{4}", 2.5),          # 12-digit grouped number
        (r"enrolment\s*no", 2.0),
        (r"government of india", 1.0),
        (r"dob\s*:", 0.5),
        (r"vid\s*:\s*\d{16}", 2.0),              # Virtual ID
    ],
    DocumentType.pan: [
        (r"permanent account number", 3.0),
        (r"income.tax.department", 3.0),
        (r"\b[A-Z]{5}\d{4}[A-Z]\b", 3.0),        # PAN format
        (r"pan\s*card", 2.0),
        (r"father'?s?\s*name", 1.5),
        (r"income.tax", 1.0),
    ],
    DocumentType.passport: [
        (r"\bpassport\b", 3.0),
        (r"republic of india", 2.0),
        (r"place of birth", 2.0),
        (r"date of expiry", 2.0),
        (r"date of issue", 1.5),
        (r"[A-Z]{1}\d{7}", 2.5),                  # passport number
        (r"[A-Z0-9<]{44}", 2.0),                   # MRZ line
        (r"nationality", 1.0),
        (r"machine readable", 1.5),
    ],
    DocumentType.resume: [
        (r"\bresume\b", 2.5),
        (r"curriculum\s*vitae", 3.0),
        (r"\bcv\b", 1.5),
        (r"work\s*experience", 2.0),
        (r"employment\s*history", 2.0),
        (r"objective\s*:", 1.5),
        (r"skills?\s*:", 1.5),
        (r"education\s*:", 1.5),
        (r"projects?\s*:", 1.0),
        (r"references?\s*(available)?", 1.0),
        (r"[a-z0-9.+-]+@[a-z0-9-]+\.[a-z]{2,}", 1.0),  # email
    ],
    DocumentType.graduation: [
        (r"bachelor\s*of", 2.5),
        (r"master\s*of", 2.5),
        (r"degree\s*of", 2.0),
        (r"awarded\s*(the\s*)?(degree|title)", 3.0),
        (r"convocation", 2.5),
        (r"chancellor", 2.0),
        (r"registrar", 1.5),
        (r"graduation\s*certificate", 3.0),
        (r"university\s*of", 1.0),
    ],
    DocumentType.marksheet: [
        (r"mark\s*sheet", 3.0),
        (r"marks?\s*obtained", 2.5),
        (r"total\s*marks", 2.0),
        (r"roll\s*(no|number)", 2.0),
        (r"semester\s*[ivx\d]+", 2.0),
        (r"result\s*(of|declared)", 1.5),
        (r"subject\s*code", 1.5),
        (r"grade\s*point", 1.5),
        (r"percentage\s*:", 1.0),
        (r"examination\s*(center|centre|board)", 1.5),
    ],
}


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    """Full classification output with confidence breakdown."""
    primary_type: DocumentType
    primary_confidence: float                       # 0.0 – 1.0
    all_scores: dict[DocumentType, float] = field(default_factory=dict)
    matched_signals: dict[DocumentType, list[str]] = field(default_factory=dict)
    is_ambiguous: bool = False


# ── Classifier ────────────────────────────────────────────────────────────────

def classify_document(text: str, min_confidence: float = 0.25) -> ClassificationResult:
    """
    Classify a document from its OCR text using weighted signal matching.

    Parameters
    ----------
    text            : Full OCR-extracted text (any case).
    min_confidence  : Below this confidence threshold the doc is marked ambiguous.

    Returns
    -------
    ClassificationResult with primary type, confidence, and per-type score breakdown.
    """
    lower = text.lower()
    raw_scores: dict[DocumentType, float] = {}
    matched_signals: dict[DocumentType, list[str]] = {}

    for doc_type, patterns in _WEIGHTED_SIGNATURES.items():
        score = 0.0
        hits: list[str] = []
        for pattern, weight in patterns:
            if re.search(pattern, lower):
                score += weight
                hits.append(pattern)
        raw_scores[doc_type] = score
        matched_signals[doc_type] = hits
        logger.debug(f"  {doc_type.value}: raw_score={score:.2f}, hits={len(hits)}")

    total = sum(raw_scores.values())
    if total == 0:
        logger.warning("No classification signals found — returning unknown")
        return ClassificationResult(
            primary_type=DocumentType.unknown,
            primary_confidence=0.0,
            all_scores={dt: 0.0 for dt in DocumentType},
            matched_signals=matched_signals,
            is_ambiguous=True,
        )

    # Normalise to confidence percentages
    normalised = {dt: round(s / total, 4) for dt, s in raw_scores.items()}

    best_type = max(normalised, key=lambda k: normalised[k])
    best_conf = normalised[best_type]

    is_ambiguous = best_conf < min_confidence

    logger.info(
        f"Document classified as {best_type.value} "
        f"(confidence={best_conf:.1%}, ambiguous={is_ambiguous})"
    )

    return ClassificationResult(
        primary_type=best_type,
        primary_confidence=best_conf,
        all_scores=normalised,
        matched_signals=matched_signals,
        is_ambiguous=is_ambiguous,
    )
