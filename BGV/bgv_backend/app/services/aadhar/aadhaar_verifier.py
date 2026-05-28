"""
app/services/aadhaar/aadhaar_verifier.py
──────────────────────────────────────────
Aadhaar Verification Orchestrator

Runs all available Aadhaar-specific checks and produces a unified
AadhaarVerificationResult with per-check details and composite risk score.

Checks performed:
  1. Verhoeff Checksum       — mathematical validity of the 12-digit number
  2. Secure QR Decode        — extract digitally-signed QR payload
  3. QR vs OCR Cross-Match   — golden fraud signal: field consistency
  4. Font Consistency        — detect edited text regions
  5. Layout Authenticity     — UIDAI field positions and structure
  6. ELA (inherited)         — already in main pipeline
  7. Metadata (inherited)    — already in main pipeline
  8. Face Verification note  — placeholder for InsightFace integration
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

from app.services.aadhar.verhoeff import validate_aadhaar_checksum
from app.services.aadhar.secure_qr import (
    decode_secure_qr,
    cross_validate_qr_vs_ocr,
    SecureQRType,
    SecureQRStatus,
)
from app.services.aadhar.font_consistency import analyse_font_consistency
from app.services.aadhar.layout_checker import validate_aadhaar_layout


@dataclass
class AadhaarVerificationResult:
    """Unified result from all Aadhaar-specific checks."""

    # ── Check 1: Verhoeff ──────────────────────────────────────────────────
    verhoeff_valid: bool = False
    verhoeff_reason: str = ""
    verhoeff_penalty: int = 0

    # ── Check 2+3: Secure QR + Cross-Match ────────────────────────────────
    qr_type: str = "not_found"
    qr_fields_found: list[str] = field(default_factory=list)
    qr_cross_validation_status: str = "no_qr"
    qr_field_matches: dict[str, Any] = field(default_factory=dict)
    qr_mismatches: list[str] = field(default_factory=list)
    qr_penalty: int = 0
    qr_summary: str = ""

    # ── Check 4: Font Consistency ──────────────────────────────────────────
    font_status: str = "consistent"
    font_variance_ratio: float = 0.0
    font_suspicious_blocks: int = 0
    font_high_frequency_anomaly: bool = False
    font_compression_inconsistency: bool = False
    font_penalty: int = 0
    font_detail: str = ""

    # ── Check 5: Layout Authenticity ──────────────────────────────────────
    layout_status: str = "unknown"
    layout_issues: list[str] = field(default_factory=list)
    layout_penalty: int = 0
    layout_detail: str = ""

    # ── Composite ──────────────────────────────────────────────────────────
    total_aadhaar_penalty: int = 0
    aadhaar_risk_level: str = "low"      # "low" | "medium" | "high" | "critical"
    checks_passed: int = 0
    checks_failed: int = 0
    summary: str = ""
    corrected_fields: dict[str, Any] = field(default_factory=dict)


def run_aadhaar_verification(
    image: np.ndarray,
    ocr_fields: dict[str, Any],
) -> AadhaarVerificationResult:
    """
    Run the complete Aadhaar-specific verification suite.

    Parameters
    ----------
    image       : BGR OpenCV image of the Aadhaar card
    ocr_fields  : Dict of OCR-extracted fields (may be mutated to correct errors)

    Returns
    -------
    AadhaarVerificationResult with all check results and composite score
    """
    result = AadhaarVerificationResult()
    # Working copy of fields — we may correct from QR
    fields = dict(ocr_fields)

    # ── 1. Verhoeff Checksum ───────────────────────────────────────────────
    logger.info("Aadhaar Check 1: Verhoeff checksum")
    aadhaar_num = fields.get("aadhaar_number", "")
    if aadhaar_num:
        verhoeff = validate_aadhaar_checksum(str(aadhaar_num))
        result.verhoeff_valid = verhoeff["valid"]
        result.verhoeff_reason = verhoeff["reason"]
        result.verhoeff_penalty = verhoeff["risk_penalty"]
        if verhoeff["valid"]:
            result.checks_passed += 1
            logger.info(f"✅ Verhoeff: PASS")
        else:
            result.checks_failed += 1
            logger.warning(f"❌ Verhoeff: FAIL — {verhoeff['reason']}")
    else:
        result.verhoeff_valid = False
        result.verhoeff_reason = "Aadhaar number not found in OCR"
        result.verhoeff_penalty = 15
        result.checks_failed += 1

    # ── 2+3. Secure QR Decode + Cross-Validation ───────────────────────────
    
    logger.info("Aadhaar Check 2+3: Secure QR + OCR cross-match")
    try:
        qr_payload = decode_secure_qr(image)
        result.qr_type = qr_payload.qr_type.value
        result.qr_fields_found = qr_payload.fields_found

        # Cross-validate QR against OCR fields
        xval = cross_validate_qr_vs_ocr(qr_payload, fields)
        result.qr_cross_validation_status = xval["status"]
        result.qr_field_matches = xval.get("field_matches", {})
        result.qr_mismatches = xval.get("mismatches", [])
        result.qr_penalty = xval["risk_penalty"]
        result.qr_summary = xval["summary"]

        # Accept QR-corrected fields (QR is authoritative)
        if xval["status"] in (SecureQRStatus.verified.value, SecureQRStatus.partial_match.value):
            if qr_payload.name and not fields.get("name"):
                fields["name"] = qr_payload.name
                result.corrected_fields["name"] = qr_payload.name
            if qr_payload.dob and not fields.get("dob"):
                fields["dob"] = qr_payload.dob
                result.corrected_fields["dob"] = qr_payload.dob

        qr_ok = xval["status"] in (
            SecureQRStatus.verified.value,
            SecureQRStatus.partial_match.value,
            SecureQRStatus.no_qr.value,  # absence isn't failure
        )
        if xval["status"] == SecureQRStatus.verified.value:
            result.checks_passed += 1
            logger.info(f"✅ QR cross-validation: VERIFIED")
        elif xval["status"] == SecureQRStatus.mismatch.value:
            result.checks_failed += 1
            logger.warning(f"❌ QR cross-validation: MISMATCH — {result.qr_mismatches}")
        else:
            # no_qr, partial, suspicious — neutral
            logger.info(f"⚠️ QR cross-validation: {xval['status']}")

    except Exception as exc:
        logger.warning(f"QR verification error: {exc}")
        result.qr_type = "error"
        result.qr_cross_validation_status = "error"
        result.qr_penalty = 5
        result.qr_summary = f"QR check failed: {exc}"

    # ── 4. Font Consistency ────────────────────────────────────────────────
    logger.info("Aadhaar Check 4: Font consistency")
    try:
        font_result = analyse_font_consistency(image)
        result.font_status = font_result.status
        result.font_variance_ratio = font_result.variance_ratio
        result.font_suspicious_blocks = font_result.suspicious_blocks
        result.font_high_frequency_anomaly = font_result.high_frequency_anomaly
        result.font_compression_inconsistency = font_result.compression_inconsistency
        result.font_penalty = font_result.risk_penalty
        result.font_detail = font_result.detail

        if font_result.status == "consistent":
            result.checks_passed += 1
            logger.info(f"✅ Font consistency: CONSISTENT")
        elif font_result.status == "suspicious":
            logger.warning(f"⚠️ Font consistency: SUSPICIOUS")
        else:
            result.checks_failed += 1
            logger.warning(f"❌ Font consistency: INCONSISTENT")

    except Exception as exc:
        logger.warning(f"Font consistency error: {exc}")
        result.font_status = "error"
        result.font_detail = str(exc)

    # ── 5. Layout Authenticity ─────────────────────────────────────────────
    logger.info("Aadhaar Check 5: Layout authenticity")
    try:
        layout_result = validate_aadhaar_layout(image, fields)
        result.layout_status = layout_result["status"]
        result.layout_issues = layout_result.get("issues", [])
        result.layout_penalty = layout_result["risk_penalty"]
        result.layout_detail = layout_result["detail"]

        if layout_result["status"] == "valid":
            result.checks_passed += 1
            logger.info(f"✅ Layout: VALID")
        elif layout_result["status"] == "suspicious":
            logger.warning(f"⚠️ Layout: SUSPICIOUS")
        else:
            result.checks_failed += 1
            logger.warning(f"❌ Layout: INVALID")

    except Exception as exc:
        logger.warning(f"Layout check error: {exc}")
        result.layout_status = "error"
        result.layout_detail = str(exc)

    # ── Composite Risk ─────────────────────────────────────────────────────
    total = (
        result.verhoeff_penalty
        + result.qr_penalty
        + result.font_penalty
        + result.layout_penalty
    )
    result.total_aadhaar_penalty = max(-5, min(100, total))

    if result.total_aadhaar_penalty <= 5:
        result.aadhaar_risk_level = "low"
    elif result.total_aadhaar_penalty <= 20:
        result.aadhaar_risk_level = "medium"
    elif result.total_aadhaar_penalty <= 40:
        result.aadhaar_risk_level = "high"
    else:
        result.aadhaar_risk_level = "critical"

    # Propagate corrected fields back
    ocr_fields.update(result.corrected_fields)

    total_checks = result.checks_passed + result.checks_failed
    result.summary = (
        f"Aadhaar verification: {result.checks_passed}/{total_checks} checks passed | "
        f"penalty={result.total_aadhaar_penalty} | risk={result.aadhaar_risk_level} | "
        f"verhoeff={result.verhoeff_valid} | qr={result.qr_cross_validation_status} | "
        f"font={result.font_status} | layout={result.layout_status}"
    )
    logger.info(result.summary)
    return result
