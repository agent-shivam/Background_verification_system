"""
app/services/aadhaar/secure_qr.py
───────────────────────────────────
Aadhaar Secure QR Validation — most powerful authenticity check.

Modern Aadhaar cards (post-2018) contain a digitally signed QR code
created by UIDAI using RSA-SHA256. The QR payload is a compressed
binary blob that cannot be forged without the UIDAI private key.

What we validate:
1. QR exists and is decodable
2. QR type: text/XML (old) vs compressed binary (secure/new)
3. If structured XML: parse name, DOB, gender, address, UID
4. Cross-match QR fields vs OCR-extracted fields (name, DOB, UID)
5. Completeness: all expected fields present
6. UID from QR vs OCR matches exactly

Note on digital signature:
  Full cryptographic verification requires the UIDAI public certificate
  (available from https://uidai.gov.in/en/916-developer-section/api-and-sdk/3478-qr-code.html).
  We implement structural + field-level validation which catches:
    - Fake QR codes (missing structure)
    - Tampered field values (mismatch with OCR)
    - Non-Aadhaar QR (wrong format)
  True signature verification is available via the `cryptography` library
  but requires the UIDAI cert bundle — flagged for enterprise integration.
"""
from __future__ import annotations
try:
    from pyaadhaar.decode import AadhaarSecureQr
    PYAADHAAR_AVAILABLE = True
except Exception:
    PYAADHAAR_AVAILABLE = False


import re
import zlib
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import cv2
import numpy as np
from loguru import logger

try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False
    logger.warning("pyzbar not available — QR detection limited to OpenCV")


# ── Enums & Data classes ──────────────────────────────────────────────────────

class SecureQRType(str, Enum):
    secure_v2       = "secure_v2"       # Post-2018 compressed binary
    xml_signed      = "xml_signed"      # XML with digital signature
    xml_plain       = "xml_plain"       # Older plain XML
    text_plain      = "text_plain"      # Very old text format
    unknown         = "unknown"
    not_found       = "not_found"


class SecureQRStatus(str, Enum):
    verified        = "verified"        # QR decoded + all fields match OCR
    partial_match   = "partial_match"   # QR decoded, some fields match
    mismatch        = "mismatch"        # QR decoded, fields don't match OCR
    no_qr           = "no_qr"          # No QR code found in image
    decode_error    = "decode_error"    # QR found but unreadable
    suspicious      = "suspicious"      # QR found but invalid structure


@dataclass
class SecureQRPayload:
    """Parsed data from Aadhaar QR code."""
    qr_type: SecureQRType = SecureQRType.not_found
    raw_bytes: bytes = b""
    name: str = ""
    dob: str = ""
    gender: str = ""
    address: str = ""
    uid_last4: str = ""       # Only last 4 digits of UID (privacy)
    uid_full: str = ""        # Full UID if available in QR
    pincode: str = ""
    mobile_hash: str = ""     # Hashed mobile (not reversible)
    email_hash: str = ""      # Hashed email (not reversible)
    has_photo: bool = False
    xml_raw: str = ""
    parse_error: str = ""
    fields_found: list[str] = field(default_factory=list)


# ── QR Extraction ─────────────────────────────────────────────────────────────

def _extract_qr_bytes(image: np.ndarray) -> bytes | None:
    """
    Multi-strategy QR extraction from image.
    Returns raw bytes of the first QR found, or None.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    candidates = [gray]

    # Strategy 2: CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    candidates.append(clahe.apply(gray))

    # Strategy 3: 2× upscale (small QR in PDFs)
    up2 = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    candidates.append(up2)

    # Strategy 4: Otsu threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates.append(thresh)

    # pyzbar (most reliable for Aadhaar compressed QR)
    if PYZBAR_AVAILABLE:
        for img in candidates:
            try:
                results = pyzbar_decode(img)
                for r in results:
                    if r.type in ("QRCODE", "QR CODE"):
                        logger.info(f"QR found via pyzbar — {len(r.data)} bytes")
                        return r.data
                # Also accept non-QR barcodes as fallback
                if results:
                    logger.info(f"Barcode found via pyzbar — type={results[0].type}")
                    return results[0].data
            except Exception:
                continue

    # OpenCV QR fallback
    qr_det = cv2.QRCodeDetector()
    for img in candidates[:2]:
        try:
            data, _, _ = qr_det.detectAndDecode(img)
            if data:
                logger.info(f"QR found via OpenCV — {len(data)} chars")
                return data.encode("utf-8", errors="ignore")
        except Exception:
            continue

    logger.warning("No QR code found in image")
    return None


# ── Payload Parsing ───────────────────────────────────────────────────────────

def _try_decompress(raw: bytes) -> bytes | None:
    """Attempt zlib decompression (Secure QR V2 format)."""
        
    try:
        return zlib.decompress(raw)
    except Exception:
        pass

    try:
        return zlib.decompress(raw, -15)
    except Exception:
        pass

    # Aadhaar Secure QR often starts after metadata bytes
    for offset in range(1, 25):
        try:
            return zlib.decompress(raw[offset:])
        except Exception:
            continue

    for offset in range(1, 25):
        try:
            return zlib.decompress(raw[offset:], -15)
        except Exception:
            continue

    return None





def _parse_secure_v2_payload(decompressed: bytes) -> SecureQRPayload:
    """
    Parse Aadhaar Secure QR V2 payload.

    Priority:
    1. pyaadhaar decoder
    2. XML fallback
    3. heuristic extraction
    """

    payload = SecureQRPayload(qr_type=SecureQRType.secure_v2)
    payload.raw_bytes = decompressed

    # =========================================================
    # STRATEGY 1
    # pyaadhaar decoder
    # =========================================================

    if PYAADHAAR_AVAILABLE:
        try:
            obj = AadhaarSecureQr(decompressed)

            data = {}

            # Different pyaadhaar versions expose data differently
            if hasattr(obj, "data"):
                data = obj.data

            elif hasattr(obj, "__dict__"):
                data = obj.__dict__

            # -------------------------
            # Extract fields safely
            # -------------------------

            payload.name = str(
                data.get("name", "")
            ).strip()

            payload.dob = str(
                data.get("dob", "")
            ).strip()

            payload.gender = str(
                data.get("gender", "")
            ).strip()

            payload.address = str(
                data.get("address", "")
            ).strip()

            payload.pincode = str(
                data.get("pincode", "")
            ).strip()

            uid = re.sub(
                r"\D",
                "",
                str(data.get("uid", ""))
            )

            if len(uid) == 12:
                payload.uid_full = uid
                payload.uid_last4 = uid[-4:]

            # -------------------------
            # Build fields_found
            # -------------------------

            if payload.name:
                payload.fields_found.append("name")

            if payload.dob:
                payload.fields_found.append("dob")

            if payload.gender:
                payload.fields_found.append("gender")

            if payload.address:
                payload.fields_found.append("address")

            if payload.pincode:
                payload.fields_found.append("pincode")

            if payload.uid_full:
                payload.fields_found.append("uid")

            logger.info(
                f"Secure QR decoded via pyaadhaar — "
                f"fields={payload.fields_found}"
            )

            if payload.fields_found:
                return payload

        except Exception as exc:
            logger.warning(
                f"pyaadhaar decode failed: {exc}"
            )

    # =========================================================
    # STRATEGY 2
    # XML fallback
    # =========================================================

    try:
        text = decompressed.decode(
            "utf-8",
            errors="ignore"
        )

        if "<" in text and ">" in text:
            logger.info(
                "Secure QR appears XML-based"
            )

            return _parse_xml_payload(
                text.encode(),
                payload
            )

    except Exception:
        pass

    # =========================================================
    # STRATEGY 3
    # Heuristic extraction fallback
    # =========================================================

    try:
        text = decompressed.decode(
            "utf-8",
            errors="replace"
        )

        parts = re.split(
            r"[\x00-\x1f]+",
            text
        )

        readable = [
            p.strip()
            for p in parts
            if len(p.strip()) >= 3
        ]

        for part in readable:

            clean = str(part).strip()

            # DOB
            if re.match(
                r"\d{2}[/-]\d{2}[/-]\d{4}",
                clean
            ):
                payload.dob = clean
                payload.fields_found.append("dob")

            # UID
            uid = re.sub(r"\D", "", clean)

            if len(uid) == 12:
                payload.uid_full = uid
                payload.uid_last4 = uid[-4:]
                payload.fields_found.append("uid")

            # Name
            if re.match(
                r"^[A-Z][a-z]+(?: [A-Z][a-z]+){0,4}$",
                clean
            ):
                payload.name = clean
                payload.fields_found.append("name")

            # Pincode
            if re.match(r"^\d{6}$", clean):
                payload.pincode = clean
                payload.fields_found.append("pincode")

        payload.fields_found = list(
            set(payload.fields_found)
        )

        logger.info(
            f"Secure QR heuristic parse — "
            f"fields={payload.fields_found}"
        )

    except Exception as exc:
        payload.parse_error = str(exc)

        logger.warning(
            f"Secure QR heuristic parse error: {exc}"
        )

    return payload






def _parse_xml_payload(raw: bytes, base: SecureQRPayload | None = None) -> SecureQRPayload:
    """
    Parse Aadhaar XML QR payload (older format / offline XML).

    XML root: <PrintLetterBioInfo> or <OfflinePaperlessKycRes>
    Attributes: name, dob, yob, gender, co, dist, state, pc, uid
    """
    payload = base or SecureQRPayload(qr_type=SecureQRType.xml_plain)

    try:
        text = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else raw
        payload.xml_raw = text[:2000]  # store for debugging

        # Strip BOM and leading whitespace
        text = text.strip().lstrip("\ufeff")

        root = ET.fromstring(text)
        payload.qr_type = SecureQRType.xml_signed if root.get("signature") else SecureQRType.xml_plain

        # Name
        name = root.get("name") or root.get("nm") or ""
        if name:
            payload.name = name
            payload.fields_found.append("name")

        # DOB / Year of birth
        dob = root.get("dob") or ""
        yob = root.get("yob") or ""
        if dob:
            payload.dob = dob
            payload.fields_found.append("dob")
        elif yob:
            payload.dob = yob
            payload.fields_found.append("dob_year")

        # Gender
        gender = root.get("gender") or root.get("gentype") or ""
        if gender:
            payload.gender = gender
            payload.fields_found.append("gender")

        # UID (last 4 digits in newer XML, full in older)
        uid = root.get("uid") or ""
        if uid:
            clean_uid = re.sub(r"\D", "", uid)
            if len(clean_uid) == 12:
                payload.uid_full = clean_uid
                payload.uid_last4 = clean_uid[-4:]
            elif len(clean_uid) == 4:
                payload.uid_last4 = clean_uid
            if clean_uid:
                payload.fields_found.append("uid")

        # Address components
        addr_parts = []
        for attr in ["co", "house", "street", "lm", "loc", "vtc", "subdist", "dist", "state", "country"]:
            val = root.get(attr, "")
            if val:
                addr_parts.append(val)
        if addr_parts:
            payload.address = ", ".join(addr_parts)
            payload.fields_found.append("address")

        # Pincode
        pc = root.get("pc") or ""
        if re.match(r"^\d{6}$", pc):
            payload.pincode = pc
            payload.fields_found.append("pincode")

        # Photo present (base64 photo in XML)
        photo_node = root.find(".//Pht") or root.find(".//pht")
        if photo_node is not None and photo_node.text:
            payload.has_photo = True
            payload.fields_found.append("photo")

        # Mobile / email hash
        mobile_hash = root.get("m") or root.get("mobile_hash") or ""
        if mobile_hash:
            payload.mobile_hash = mobile_hash
            payload.fields_found.append("mobile_hash")

        logger.info(f"Aadhaar XML QR parsed — type={payload.qr_type.value}, fields={payload.fields_found}")

    except ET.ParseError as exc:
        payload.parse_error = f"XML parse error: {exc}"
        logger.warning(f"Aadhaar QR XML parse failed: {exc}")
    except Exception as exc:
        payload.parse_error = str(exc)
        logger.warning(f"Aadhaar QR parse unexpected error: {exc}")

    return payload


def _parse_text_payload(raw: bytes) -> SecureQRPayload:
    """Parse older plain-text Aadhaar QR."""
    payload = SecureQRPayload(qr_type=SecureQRType.text_plain)
    text = raw.decode("utf-8", errors="ignore")

    # Look for 12-digit UID
    uid_match = re.search(r"\b([2-9]\d{11})\b", text)
    if uid_match:
        payload.uid_full = uid_match.group(1)
        payload.uid_last4 = payload.uid_full[-4:]
        payload.fields_found.append("uid")

    # DOB
    dob_match = re.search(r"\b(\d{2}[/-]\d{2}[/-]\d{4}|\d{4}[/-]\d{2}[/-]\d{2})\b", text)
    if dob_match:
        payload.dob = dob_match.group(1)
        payload.fields_found.append("dob")

    logger.info(f"Text QR parsed — fields: {payload.fields_found}")
    return payload


# ── Main Decode Function ──────────────────────────────────────────────────────

def decode_secure_qr(image: np.ndarray) -> SecureQRPayload:
    """
    Decode and parse Aadhaar QR code from image.

    Tries all known Aadhaar QR formats:
    1. Secure V2 (post-2018 compressed binary)
    2. XML with signature (2017-2018)
    3. Plain XML (pre-2017)
    4. Plain text (very old)
    """
    raw = _extract_qr_bytes(image)

    if raw is None:
        return SecureQRPayload(qr_type=SecureQRType.not_found)

    if len(raw) == 0:
        return SecureQRPayload(qr_type=SecureQRType.not_found, parse_error="Empty QR")

    # Strategy 1: Try zlib decompress (Secure V2 format — starts with specific magic)
    decompressed = _try_decompress(raw)
    if decompressed and len(decompressed) > 20:
        logger.info(f"Secure QR V2 decompressed: {len(raw)}→{len(decompressed)} bytes")
        return _parse_secure_v2_payload(decompressed)

    # Strategy 2: Try as XML
    try:
        text = raw.decode("utf-8", errors="ignore").strip()
        if text.startswith("<") and ">" in text:
            return _parse_xml_payload(raw)
    except Exception:
        pass

    # Strategy 3: Try plain text with UID

    text_preview = raw[:50].decode("utf-8", errors="ignore")

    if any(str(c).isdigit() for c in text_preview):
    
        return _parse_text_payload(raw)

    # Unknown format
    payload = SecureQRPayload(qr_type=SecureQRType.unknown)
    payload.raw_bytes = raw[:200]
    payload.parse_error = f"Unknown QR format — {len(raw)} bytes"
    logger.warning(f"Unknown Aadhaar QR format: {len(raw)} bytes")
    return payload


# ── OCR Cross-Validation ──────────────────────────────────────────────────────

def cross_validate_qr_vs_ocr(
    qr: SecureQRPayload,
    ocr_fields: dict[str, Any],
) -> dict[str, Any]:
    """
    Cross-validate QR-extracted fields against OCR-extracted fields.

    This is the 'golden weapon' — any mismatch is a strong fraud signal.
    The QR is cryptographically signed; OCR is just pixel reading.

    Returns a detailed cross-validation report.
    """
    results: dict[str, Any] = {
        "status": SecureQRStatus.no_qr.value,
        "qr_type": qr.qr_type.value,
        "fields_in_qr": qr.fields_found,
        "field_matches": {},
        "mismatches": [],
        "matches": [],
        "risk_penalty": 0,
        "summary": "",
    }

    if qr.qr_type == SecureQRType.not_found:
        results["summary"] = "No QR code found — cannot perform QR cross-validation"
        results["risk_penalty"] = 5   # slight suspicion, not fatal
        return results

    if qr.parse_error and not qr.fields_found:
        results["status"] = SecureQRStatus.decode_error.value
        results["summary"] = f"QR decode error: {qr.parse_error}"
        results["risk_penalty"] = 8
        return results

    if not qr.fields_found:
        results["status"] = SecureQRStatus.suspicious.value
        results["summary"] = "QR found but no Aadhaar fields extracted — suspicious structure"
        results["risk_penalty"] = 5
        return results

    match_count = 0
    mismatch_count = 0

    # ── Name comparison ───────────────────────────────────────────────────────
    if qr.name and ocr_fields.get("name"):
        ocr_name = str(ocr_fields["name"]).strip()
        qr_name = qr.name.strip()
        name_match = _fuzzy_match(qr_name.lower(), ocr_name.lower(), threshold=0.55)
        results["field_matches"]["name"] = {
            "qr": qr_name,
            "ocr": ocr_name,
            "match": name_match,
        }
        if name_match:
            results["matches"].append("name")
            match_count += 1
            # Correct OCR with authoritative QR value
            ocr_fields["name"] = qr_name
        else:
            results["mismatches"].append(f"name: QR='{qr_name}' vs OCR='{ocr_name}'")
            mismatch_count += 1

    # ── DOB comparison ────────────────────────────────────────────────────────
    if qr.dob and ocr_fields.get("dob"):
        qr_dob_norm = _norm_date(qr.dob)
        ocr_dob_norm = _norm_date(str(ocr_fields["dob"]))
        # Also compare just the year (YOB-only QR)
        qr_year = qr.dob[:4] if len(qr.dob) >= 4 and qr.dob[:4].isdigit() else ""
        ocr_year = str(ocr_fields["dob"])[:4] if ocr_fields.get("dob") else ""
        dob_match = (qr_dob_norm == ocr_dob_norm) or (qr_year and qr_year == ocr_year)

        results["field_matches"]["dob"] = {
            "qr": qr.dob,
            "ocr": ocr_fields.get("dob"),
            "match": dob_match,
        }
        if dob_match:
            results["matches"].append("dob")
            match_count += 1
        else:
            results["mismatches"].append(f"dob: QR='{qr.dob}' vs OCR='{ocr_fields.get('dob')}'")
            mismatch_count += 1

    # ── UID comparison ────────────────────────────────────────────────────────
    ocr_uid = re.sub(r"\D", "", str(ocr_fields.get("aadhaar_number", "")))
    if qr.uid_full and ocr_uid:
        uid_match = qr.uid_full == ocr_uid
        results["field_matches"]["aadhaar_number"] = {
            "qr": f"****{qr.uid_last4}",  # Mask for privacy
            "ocr": f"****{ocr_uid[-4:]}",
            "match": uid_match,
        }
        if uid_match:
            results["matches"].append("aadhaar_number")
            match_count += 1
        else:
            results["mismatches"].append(f"aadhaar_number: last4 QR='{qr.uid_last4}' vs OCR='{ocr_uid[-4:]}'")
            mismatch_count += 1
    elif qr.uid_last4 and ocr_uid and len(ocr_uid) >= 4:
        uid_last4_match = qr.uid_last4 == ocr_uid[-4:]
        results["field_matches"]["aadhaar_last4"] = {
            "qr": f"****{qr.uid_last4}",
            "ocr": f"****{ocr_uid[-4:]}",
            "match": uid_last4_match,
        }
        if uid_last4_match:
            results["matches"].append("aadhaar_last4")
            match_count += 1
        else:
            results["mismatches"].append(f"aadhaar last4: QR='{qr.uid_last4}' vs OCR='{ocr_uid[-4:]}'")
            mismatch_count += 1

    # ── Gender comparison ─────────────────────────────────────────────────────
    if qr.gender and ocr_fields.get("gender"):
        qr_g = qr.gender[0].upper()  # M/F/T
        ocr_g = str(ocr_fields["gender"])[0].upper()
        gender_match = qr_g == ocr_g
        results["field_matches"]["gender"] = {
            "qr": qr.gender,
            "ocr": ocr_fields["gender"],
            "match": gender_match,
        }
        if gender_match:
            results["matches"].append("gender")
            match_count += 1

    # ── Determine overall status ──────────────────────────────────────────────
    checked = match_count + mismatch_count
    if checked == 0:
        results["status"] = SecureQRStatus.suspicious.value
        results["risk_penalty"] = 5
        results["summary"] = "QR decoded but no comparable fields found in OCR"
    elif mismatch_count == 0:
        results["status"] = SecureQRStatus.verified.value
        results["risk_penalty"] = -5  # bonus for confirmed match
        results["summary"] = (
            f"QR VERIFIED — {match_count}/{checked} fields match "
            f"(type: {qr.qr_type.value})"
        )
    elif mismatch_count >= 2:
        results["status"] = SecureQRStatus.mismatch.value
        results["risk_penalty"] = 30
        results["summary"] = (
            f"QR MISMATCH — {mismatch_count} field(s) differ: "
            + "; ".join(results["mismatches"])
        )
    else:
        results["status"] = SecureQRStatus.partial_match.value
        results["risk_penalty"] = 10
        results["summary"] = (
            f"QR PARTIAL — {match_count} match, {mismatch_count} mismatch: "
            + "; ".join(results["mismatches"])
        )

    logger.info(f"QR cross-validation: {results['status']} — {results['summary']}")
    return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm_date(s: str) -> str:
    """Normalise date to DDMMYYYY digits only."""
    return re.sub(r"[^0-9]", "", s)


def _fuzzy_match(a: str, b: str, threshold: float = 0.6) -> bool:
    """Token-based fuzzy name match."""
    a = str(a).strip()
    b = str(b).strip() 
    if not a or not b:
        return False
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    overlap = len(ta & tb)
    union = len(ta | tb)
    if union == 0:
        return False
    jaccard = overlap / union
    if jaccard >= threshold:
        return True
    # Subset match (handles missing middle name)
    shorter = ta if len(ta) <= len(tb) else tb
    longer = ta if len(ta) > len(tb) else tb
    if shorter and shorter.issubset(longer):
        return True
    return False
