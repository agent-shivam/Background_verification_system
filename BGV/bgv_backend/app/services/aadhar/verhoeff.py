"""
app/services/aadhaar/verhoeff.py
─────────────────────────────────
Verhoeff Algorithm — Aadhaar checksum validation.

UIDAI uses the Verhoeff algorithm (not Luhn) to validate all 12-digit
Aadhaar numbers. The last digit is a check digit computed using three
tables: multiplication, permutation, and inverse.

A number that fails this check is DEFINITIVELY invalid — it cannot be
a real Aadhaar number issued by UIDAI.

Reference: https://en.wikipedia.org/wiki/Verhoeff_algorithm
"""

from __future__ import annotations
from loguru import logger


# ── Verhoeff tables ────────────────────────────────────────────────────────────

# Multiplication table (d5)
_D5 = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]

# Permutation table (p8)
_P8 = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]

# Inverse table (inv)
_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def _verhoeff_validate(number: str) -> bool:
    """
    Return True if the digit string passes the Verhoeff check.
    The check digit is the last digit; iterate right-to-left.
    """
    c = 0
    for i, ch in enumerate(reversed(number)):
        c = _D5[c][_P8[i % 8][int(ch)]]
    return c == 0


def _verhoeff_generate(number: str) -> str:
    """
    Generate and append a Verhoeff check digit to `number`.
    Returns the full number with check digit appended.
    """
    c = 0
    padded = number + "0"
    for i, ch in enumerate(reversed(padded)):
        c = _D5[c][_P8[i % 8][int(ch)]]
    check = _INV[c]
    return number + str(check)


# ── Public API ────────────────────────────────────────────────────────────────

def validate_aadhaar_checksum(aadhaar_number: str) -> dict:
    """
    Validate an Aadhaar number using the Verhoeff algorithm.

    Parameters
    ----------
    aadhaar_number : str
        Raw Aadhaar string (may contain spaces/hyphens; will be cleaned).

    Returns
    -------
    dict with keys:
        valid         : bool   — True if checksum passes
        clean_number  : str    — digits-only form
        reason        : str    — human-readable explanation
        risk_penalty  : int    — suggested risk score addition (0 = ok, 25 = definite fail)
    """
    # Strip non-digits
    clean = "".join(c for c in str(aadhaar_number) if c.isdigit())

    # Basic length check
    if len(clean) != 12:
        return {
            "valid": False,
            "clean_number": clean,
            "reason": f"Aadhaar must be 12 digits, got {len(clean)}",
            "risk_penalty": 25,
        }

    # First digit cannot be 0 or 1
    if clean[0] in ("0", "1"):
        return {
            "valid": False,
            "clean_number": clean,
            "reason": f"Aadhaar cannot start with '{clean[0]}' (UIDAI rule)",
            "risk_penalty": 25,
        }

    # Verhoeff checksum
    try:
        passes = _verhoeff_validate(clean)
    except Exception as exc:
        logger.warning(f"Verhoeff validation error: {exc}")
        passes = False

    if passes:
        logger.info(f"Aadhaar {clean[:4]}**** Verhoeff: VALID")
        return {
            "valid": True,
            "clean_number": clean,
            "reason": "Verhoeff checksum passed — mathematically valid Aadhaar number",
            "risk_penalty": 0,
        }
    else:
        logger.warning(f"Aadhaar {clean[:4]}**** Verhoeff: FAILED — checksum mismatch")
        return {
            "valid": False,
            "clean_number": clean,
            "reason": "Verhoeff checksum FAILED — this is NOT a valid Aadhaar number",
            "risk_penalty": 30,
        }
