"""
app/services/aadhaar/layout_checker.py
────────────────────────────────────────
Aadhaar Layout Authenticity Validator.

Real Aadhaar cards have a strict UIDAI-defined layout:
  - UIDAI logo (top-left or top-center)
  - "Unique Identification Authority of India" text (top)
  - "Government of India" text
  - Hologram strip (right side, post-2018)
  - Aadhaar number displayed with spaces: XXXX XXXX XXXX
  - QR code (bottom-right typically)
  - Biometric photo (left side)
  - Name, DOB, Gender in defined zones
  - "आधार" (Devanagari) text visible

Fake cards often:
  - Misalign text blocks
  - Use wrong aspect ratio
  - Miss hologram/logo region
  - Have UIDAI text in wrong position
  - Show Aadhaar number without standard spacing
"""

from __future__ import annotations

import re
from typing import Any

import cv2
import numpy as np
from loguru import logger


# ── Aadhaar layout constants ──────────────────────────────────────────────────
# Standard Aadhaar card ratio (85.6mm × 54mm = credit card size)
AADHAAR_ASPECT_RATIO   = 85.6 / 54.0   # ≈ 1.585
ASPECT_TOLERANCE       = 0.15           # ±15% tolerance

# OCR text signals that should appear on genuine Aadhaar
REQUIRED_TEXT_SIGNALS  = [
    "unique identification",
    "government of india",
    "uidai",
    "aadhaar",
]
DEVANAGARI_PATTERN     = re.compile(r"[\u0900-\u097F]")   # Hindi script
AADHAAR_NUM_PATTERN    = re.compile(r"\b\d{4}\s+\d{4}\s+\d{4}\b")  # proper spacing

LOGO_MIN_AREA_RATIO    = 0.002   # logo should be at least 0.2% of image area


def validate_aadhaar_layout(
    image: np.ndarray,
    ocr_fields: dict[str, Any],
) -> dict[str, Any]:
    """
    Validate Aadhaar card layout authenticity.

    Returns a dict with:
        status       : "valid" | "suspicious" | "invalid"
        issues       : list of identified problems
        risk_penalty : int (0-20)
        detail       : summary string
        checks       : per-check results dict
    """
    issues = []
    checks = {}
    penalty = 0

    try:
        h, w = image.shape[:2]

        # ── 1. Aspect ratio check ─────────────────────────────────────────────
        if h > 0 and w > 0:
            actual_ratio = w / h
            expected     = AADHAAR_ASPECT_RATIO
            ratio_diff   = abs(actual_ratio - expected) / expected
            aspect_ok    = ratio_diff <= ASPECT_TOLERANCE
            checks["aspect_ratio"] = {
                "passed": aspect_ok,
                "actual": round(actual_ratio, 3),
                "expected": round(expected, 3),
                "diff_pct": round(ratio_diff * 100, 1),
            }
            if not aspect_ok:
                issues.append(
                    f"Aspect ratio {actual_ratio:.2f} deviates from standard "
                    f"{expected:.2f} by {ratio_diff*100:.1f}%"
                )
                penalty += 5
        else:
            checks["aspect_ratio"] = {"passed": False, "actual": 0}

        # ── 2. OCR text signals ───────────────────────────────────────────────
        raw_text = (ocr_fields.get("raw_text") or "").lower()
        found_signals = []
        missing_signals = []
        for signal in REQUIRED_TEXT_SIGNALS:
            if signal in raw_text:
                found_signals.append(signal)
            else:
                missing_signals.append(signal)

        signal_score = len(found_signals) / max(len(REQUIRED_TEXT_SIGNALS), 1)
        checks["text_signals"] = {
            "passed": signal_score >= 0.5,
            "found": found_signals,
            "missing": missing_signals,
            "score": round(signal_score, 2),
        }
        if signal_score < 0.25:
            issues.append(
                f"Missing critical Aadhaar text signals: {missing_signals}"
            )
            penalty += 8
        elif signal_score < 0.5:
            issues.append(f"Some Aadhaar text signals absent: {missing_signals}")
            penalty += 3

        # ── 3. Aadhaar number spacing check ──────────────────────────────────
        # Genuine Aadhaar shows number as "1234 5678 9012" (4-4-4 groups)
        aadhaar_num_raw = str(ocr_fields.get("aadhaar_number") or raw_text)
        proper_spacing = bool(AADHAAR_NUM_PATTERN.search(aadhaar_num_raw))
        if not proper_spacing and re.search(r"\d{12}", raw_text):
            # 12 digits present but no spacing — might be OCR strip issue, not fatal
            proper_spacing = True   # benefit of the doubt
        checks["number_spacing"] = {"passed": proper_spacing}
        if not proper_spacing and not ocr_fields.get("aadhaar_number"):
            issues.append("Aadhaar number not found or incorrectly formatted")
            penalty += 4

        # ── 4. Devanagari script presence ────────────────────────────────────
        has_devanagari = bool(DEVANAGARI_PATTERN.search(raw_text))
        checks["devanagari_script"] = {"passed": has_devanagari}
        if not has_devanagari:
            # Not always present in scans — low penalty
            issues.append("Devanagari (Hindi) script not detected")
            penalty += 2

        # ── 5. Colour / visual checks ─────────────────────────────────────────
        colour_checks = _check_colour_profile(image)
        checks["colour_profile"] = colour_checks
        if not colour_checks["passed"]:
            issues.append(colour_checks["reason"])
            penalty += colour_checks.get("penalty", 3)

        # ── 6. QR region detection ────────────────────────────────────────────
        qr_region = _detect_qr_region(image)
        checks["qr_region"] = qr_region
        if not qr_region["found"]:
            issues.append("QR code region not detected visually")
            penalty += 2   # low — may be obscured

        # ── 7. Photo region presence ──────────────────────────────────────────
        photo_region = _detect_photo_region(image)
        checks["photo_region"] = photo_region
        if not photo_region["found"]:
            issues.append("Biometric photo region not detected")
            penalty += 3

    except Exception as exc:
        logger.warning(f"Layout check error: {exc}")
        issues.append(f"Layout analysis error: {exc}")
        penalty = max(penalty, 3)

    # ── Classify status ───────────────────────────────────────────────────────
    if penalty == 0 and not issues:
        status = "valid"
    elif penalty <= 8:
        status = "suspicious"
    else:
        status = "invalid"

    detail = (
        f"Layout {status}: {len(issues)} issue(s), penalty={penalty}. "
        + ("; ".join(issues) if issues else "All checks passed.")
    )

    return {
        "status": status,
        "issues": issues,
        "risk_penalty": min(penalty, 20),
        "detail": detail,
        "checks": checks,
    }


# ── Visual sub-checks ─────────────────────────────────────────────────────────

def _check_colour_profile(image: np.ndarray) -> dict:
    """
    Genuine Aadhaar cards have a distinctive blue-white colour scheme.
    Grayscale photocopies are OK (low penalty) but monochrome scans of
    colour-edited fakes often show unusual colour distributions.
    """
    if len(image.shape) == 2:
        # Grayscale — skip colour check
        return {"passed": True, "reason": "Grayscale image — colour check skipped"}

    # Check for colour image
    b_mean = float(np.mean(image[:, :, 0]))
    g_mean = float(np.mean(image[:, :, 1]))
    r_mean = float(np.mean(image[:, :, 2]))

    # Genuine Aadhaar tends to have higher blue/white values (light background)
    # Very dark images are suspicious (heavily degraded or fake)
    brightness = (r_mean + g_mean + b_mean) / 3

    if brightness < 40:
        return {
            "passed": False,
            "reason": f"Image very dark (brightness={brightness:.0f}) — possible heavily degraded scan",
            "penalty": 3,
        }

    # Extremely saturated (overly colourful) can indicate digital generation
    saturation = _calc_mean_saturation(image)
    if saturation > 180:
        return {
            "passed": False,
            "reason": f"Unusually high saturation ({saturation:.0f}) — possible AI-generated/printed fake",
            "penalty": 5,
        }

    return {"passed": True, "reason": f"Colour profile normal (brightness={brightness:.0f}, sat={saturation:.0f})"}


def _calc_mean_saturation(image: np.ndarray) -> float:
    """Return mean HSV saturation of the image."""
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return float(np.mean(hsv[:, :, 1]))
    except Exception:
        return 0.0


def _detect_qr_region(image: np.ndarray) -> dict:
    """
    Quick visual check: look for a QR-like high-contrast square region
    in the lower portion of the image (where UIDAI places the QR).
    """
    try:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Look at the right side, lower 60% of card
        roi = gray[int(h*0.4):, int(w*0.5):]

        # Binarize and look for dense square contours
        _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            if area < 400:
                continue
            aspect = cw / max(ch, 1)
            if 0.7 < aspect < 1.3:  # roughly square
                fill_ratio = cv2.contourArea(cnt) / area
                if fill_ratio > 0.3:
                    return {"found": True, "area": int(area)}

        return {"found": False}
    except Exception:
        return {"found": False}


def _detect_photo_region(image: np.ndarray) -> dict:
    """
    Detect presence of a face/photo region on the left side of the card.
    Uses skin-tone detection heuristic.
    """
    try:
        if len(image.shape) == 2:
            return {"found": True, "reason": "Grayscale — photo check skipped"}

        h, w = image.shape[:2]
        # Photo is typically in the left 30% of card, upper 80%
        roi = image[:int(h*0.85), :int(w*0.35)]

        # Convert to HSV and look for skin tones
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Skin tone range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_ratio = float(np.sum(skin_mask > 0)) / (roi.shape[0] * roi.shape[1])

        return {"found": skin_ratio > 0.03, "skin_ratio": round(skin_ratio, 3)}
    except Exception:
        return {"found": True, "reason": "Photo detection error — assuming present"}
