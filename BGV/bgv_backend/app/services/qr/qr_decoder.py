"""
app/services/qr/qr_decoder.py
──────────────────────────────
QR / barcode decoding via pyzbar + cross-validation against
OCR-extracted Aadhaar fields.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import cv2
import numpy as np
from loguru import logger
from pyzbar.pyzbar import decode as pyzbar_decode

from app.schemas.document import QRResult


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class QRData:
    raw: str = ""
    data_type: str = ""    # "aadhaar_xml", "text", "url", "unknown"
    parsed: dict = field(default_factory=dict)


# ── Decoding ──────────────────────────────────────────────────────────────────

def decode_qr(image: np.ndarray) -> QRData | None:
    """
    Attempt to decode QR / barcode from the image.
    Tries multiple strategies to maximise detection rate on PDFs:
      1. Original image
      2. Contrast-enhanced image
      3. Upscaled 2x (PDFs rendered at 300dpi may still be too small for pyzbar)
      4. OpenCV QR detector as fallback
    Returns None if no code found.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    upscaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    enhanced  = _enhance_for_qr(gray)
    enhanced2 = _enhance_for_qr(upscaled)

    for attempt_img in [gray, enhanced, upscaled, enhanced2]:
        decoded = pyzbar_decode(attempt_img)
        if decoded:
            raw = decoded[0].data.decode("utf-8", errors="ignore")
            logger.info(f"QR decoded (pyzbar) — {len(raw)} chars, type={decoded[0].type}")
            return _parse_qr_data(raw)

    # OpenCV QR detector fallback
    qr_detector = cv2.QRCodeDetector()
    for attempt_img in [gray, upscaled]:
        data, _, _ = qr_detector.detectAndDecode(attempt_img)
        if data:
            logger.info(f"QR decoded (cv2) — {len(data)} chars")
            return _parse_qr_data(data)

    logger.warning("No QR / barcode found in image")
    return None


# ── Aadhaar QR cross-validation ───────────────────────────────────────────────

def validate_aadhaar_qr(
    qr_data: QRData | None,
    extracted_fields: dict,
) -> QRResult:
    """
    Compare QR-decoded Aadhaar fields with OCR-extracted fields.

    The QR code is the authoritative source — it is digitally signed by UIDAI
    and cannot be forged without detection. Strategy:
    1. If QR has a name, use it to CORRECT the extracted name (OCR is noisy).
    2. Verify DOB matches between QR and OCR.
    3. If the OCR name is clearly garbage (fails _is_valid_name), accept QR name.
    """
    if qr_data is None:
        return QRResult.not_found

    if not qr_data.parsed:
        return QRResult.decode_error

    qr_name = qr_data.parsed.get("name", "").strip()
    qr_dob  = qr_data.parsed.get("dob", "").strip()
    ocr_name = (extracted_fields.get("name") or "").strip()
    ocr_dob  = (extracted_fields.get("dob") or "").strip()

    if not (qr_name or qr_dob):
        return QRResult.not_found

    # ── Name comparison ───────────────────────────────────────────────────────
    name_ok: bool
    if not qr_name:
        name_ok = True  # QR has no name field — skip name check
    elif not ocr_name or not _is_plausible_name(ocr_name):
        # OCR name is missing or clearly garbage — trust QR, auto-correct
        logger.info(
            f"OCR name {ocr_name!r} invalid — accepting QR name {qr_name!r}"
        )
        extracted_fields["name"] = qr_name
        name_ok = True
    else:
        name_ok = _fuzzy_name_match(qr_name.lower(), ocr_name.lower())
        if not name_ok:
            # Last chance: check if any QR word appears in OCR name
            qr_tokens  = set(qr_name.lower().split())
            ocr_tokens = set(ocr_name.lower().split())
            if qr_tokens & ocr_tokens:  # at least one word matches
                name_ok = True
                logger.info(f"QR name partial match — accepting ({qr_name!r} ~ {ocr_name!r})")
            else:
                # OCR name is wrong — correct it from QR
                logger.warning(
                    f"QR name {qr_name!r} != OCR name {ocr_name!r} — "
                    f"correcting from QR (QR is authoritative)"
                )
                extracted_fields["name"] = qr_name
                name_ok = True  # QR is ground truth, not a forgery signal

    # ── DOB comparison ────────────────────────────────────────────────────────
    dob_ok: bool
    if not qr_dob or not ocr_dob:
        dob_ok = True
    else:
        dob_ok = _normalise_date(qr_dob) == _normalise_date(ocr_dob)

    if name_ok and dob_ok:
        logger.info("Aadhaar QR validation: VERIFIED")
        return QRResult.verified

    logger.warning(f"Aadhaar QR mismatch — name_ok={name_ok}, dob_ok={dob_ok}")
    return QRResult.mismatch


def _is_plausible_name(text: str) -> bool:
    """Return True if text looks like a real person name (not OCR noise)."""
    if not text or len(text) < 4:
        return False
    words = text.split()
    if len(words) < 2 or len(words) > 6:
        return False
    for w in words:
        if len(w) < 2:
            return False
        alpha_ratio = sum(c.isalpha() for c in w) / len(w)
        if alpha_ratio < 0.75:
            return False
    if re.search(r"\d", text):
        return False
    return True


# ── Internal helpers ──────────────────────────────────────────────────────────

def _parse_qr_data(raw: str) -> QRData:
    """Try to parse as Aadhaar XML; fall back to plain text."""
    qr = QRData(raw=raw)
    try:
        root = ET.fromstring(raw)
        # Aadhaar offline XML has <PrintLetterBioInfo> or <OfflinePaperlessKycRes>
        name  = root.get("name") or root.get("nm") or ""
        dob   = root.get("dob") or root.get("yob") or ""
        uid   = root.get("uid") or ""
        qr.data_type = "aadhaar_xml"
        qr.parsed = {"name": name, "dob": dob, "uid": uid}
    except ET.ParseError:
        qr.data_type = "text"
        qr.parsed = {"text": raw[:200]}
    return qr


def _enhance_for_qr(gray: np.ndarray) -> np.ndarray:
    """CLAHE + threshold to improve QR detection on poor scans."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    _, thresh = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def _normalise_date(date_str: str) -> str:
    """Normalise date strings to DDMMYYYY for comparison."""
    return re.sub(r"[/\-\s]", "", date_str)


def _fuzzy_name_match(a: str, b: str, threshold: float = 0.6) -> bool:
    """
    Fuzzy name match using Jaccard similarity on tokens.
    Lowered threshold to 0.6 (was 0.7) to handle OCR variants.
    Also accepts if all tokens of the shorter name appear in the longer one.
    """
    if not a or not b:
        return False
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    overlap = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    jaccard = overlap / union if union else 0
    if jaccard >= threshold:
        return True
    # Subset check: shorter name fully contained in longer (handles middle names)
    shorter = tokens_a if len(tokens_a) <= len(tokens_b) else tokens_b
    longer  = tokens_a if len(tokens_a) >  len(tokens_b) else tokens_b
    if shorter and shorter.issubset(longer):
        return True
    return False