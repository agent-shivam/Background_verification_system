"""
app/services/forgery/risk_scorer.py
─────────────────────────────────────
Composite risk-scoring engine.

Each fraud-detection signal contributes a weighted penalty to a 0–100 score.
Score ranges:
    0  – 29  →  Likely Genuine
    30 – 59  →  Suspicious
    60 – 100 →  Likely Fake / Tampered

Weight rationale
────────────────
- ELA tampered    : highest weight — direct pixel manipulation evidence
- QR mismatch     : very high     — data cross-validation failure
- Duplicate region: high          — copy-move forgery
- Layout invalid  : medium        — structural anomaly
- Metadata suspic : medium        — tool / timestamp evidence
- Blur (blurry)   : low           — could be poor scan quality
- AI artifacts    : medium        — probabilistic signal
"""

from __future__ import annotations

from app.core.config import settings
from app.schemas.document import (
    AIArtifactResult,
    BlurResult,
    DocumentType,
    DuplicateRegionResult,
    ELAResult,
    FraudAnalysis,
    LayoutResult,
    MetadataResult,
    QRResult,
    RiskStatus,
)


# ── Score contribution table ──────────────────────────────────────────────────
# Each entry: (enum_value, penalty_points)

_ELA_PENALTIES = {
    ELAResult.clean:      0,
    ELAResult.suspicious: 15,
    ELAResult.tampered:   35,
}

_META_PENALTIES = {
    MetadataResult.normal:     0,
    MetadataResult.suspicious: 12,
    MetadataResult.missing:    8,
}

_BLUR_PENALTIES = {
    BlurResult.sharp:           0,
    BlurResult.slightly_blurry: 5,
    BlurResult.blurry:          10,
}

_DUP_PENALTIES = {
    DuplicateRegionResult.not_detected: 0,
    DuplicateRegionResult.detected:     20,
}

_LAYOUT_PENALTIES = {
    LayoutResult.valid:   0,
    LayoutResult.unknown: 5,
    LayoutResult.invalid: 15,
}

_QR_PENALTIES = {
    QRResult.verified:    -5,    # bonus for confirmed match
    QRResult.not_found:    0,
    QRResult.decode_error: 5,
    QRResult.mismatch:    25,
}

_AI_PENALTIES = {
    AIArtifactResult.not_detected: 0,
    AIArtifactResult.suspected:    12,
}

# QR check only applies to Aadhaar
_QR_APPLICABLE_TYPES = {DocumentType.aadhaar}


def compute_risk_score(
    analysis: FraudAnalysis,
    doc_type: DocumentType,
    ela_score: float = 0.0,
) -> tuple[int, RiskStatus]:
    """
    Compute a composite 0–100 risk score and derive the RiskStatus.

    Parameters
    ----------
    analysis  : Completed FraudAnalysis schema.
    doc_type  : Detected document type.
    ela_score : Raw ELA score (0–100) used as an additive tiebreaker.

    Returns
    -------
    (risk_score: int, status: RiskStatus)
    """
    penalty = 0

    penalty += _ELA_PENALTIES.get(analysis.ela, 0)
    penalty += _META_PENALTIES.get(analysis.metadata, 0)
    penalty += _BLUR_PENALTIES.get(analysis.blur, 0)
    penalty += _DUP_PENALTIES.get(analysis.duplicate_regions, 0)
    penalty += _LAYOUT_PENALTIES.get(analysis.layout_validation, 0)
    penalty += _AI_PENALTIES.get(analysis.ai_artifacts, 0)

    # QR bonus/penalty only for applicable doc types
    if doc_type in _QR_APPLICABLE_TYPES:
        penalty += _QR_PENALTIES.get(analysis.qr_validation, 0)

    # Add a fraction of the raw ELA score to differentiate edge cases
    penalty += int(ela_score * 0.1)

    risk_score = max(0, min(100, penalty))

    if risk_score < settings.risk_low_threshold:
        status = RiskStatus.genuine
    elif risk_score < settings.risk_medium_threshold:
        status = RiskStatus.suspicious
    else:
        status = RiskStatus.fake

    return risk_score, status