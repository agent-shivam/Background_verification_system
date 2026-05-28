"""
app/services/parsers/field_parser.py
─────────────────────────────────────
Regex + spaCy NER-based field extraction for each document type.
Returns typed schema objects defined in app/schemas/document.py.
"""

from __future__ import annotations

import re
from typing import Any

import spacy
from loguru import logger

from app.schemas.document import (
    DocumentType,
    AadhaarFields,
    PANFields,
    PassportFields,
    ResumeFields,
    CertificateFields,
    ExtractedFields,
)

# ── spaCy model ───────────────────────────────────────────────────────────────
# We use the small English model; load lazily on first call.
_nlp: spacy.Language | None = None


def _get_nlp() -> spacy.Language:
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded: en_core_web_sm")
        except OSError:
            logger.warning(
                "spaCy model 'en_core_web_sm' not found — "
                "run: python -m spacy download en_core_web_sm"
            )
            _nlp = spacy.blank("en")
    return _nlp


# ══════════════════════════════════════════════════════════════════════════════
# Top-level dispatcher
# ══════════════════════════════════════════════════════════════════════════════

def parse_fields(text: str, doc_type: DocumentType) -> dict[str, Any]:
    """Return a dict of extracted fields for the given document type."""
    parsers = {
        DocumentType.aadhaar:    _parse_aadhaar,
        DocumentType.pan:        _parse_pan,
        DocumentType.passport:   _parse_passport,
        DocumentType.resume:     _parse_resume,
        DocumentType.graduation: _parse_certificate,
        DocumentType.marksheet:  _parse_certificate,
    }
    parser = parsers.get(doc_type, _parse_generic)
    fields = parser(text)
    fields["raw_text"] = text
    return fields


# ══════════════════════════════════════════════════════════════════════════════
# Per-document parsers
# ══════════════════════════════════════════════════════════════════════════════

# ── Aadhaar ───────────────────────────────────────────────────────────────────

def _parse_aadhaar(text: str) -> dict[str, Any]:
    fields = AadhaarFields()

    # Aadhaar number: 12 digits optionally grouped as XXXX XXXX XXXX
    m = re.search(r"\b(\d{4}[\s-]?\d{4}[\s-]?\d{4})\b", text)
    if m:
        fields.aadhaar_number = re.sub(r"[\s-]", "", m.group(1))

    # DOB: DD/MM/YYYY or DD-MM-YYYY
    m = re.search(r"\b(\d{2}[/\-]\d{2}[/\-]\d{4})\b", text)
    if m:
        fields.dob = m.group(1)

    # Gender
    m = re.search(r"\b(male|female|transgender)\b", text, re.IGNORECASE)
    if m:
        fields.gender = m.group(1).capitalize()

    # Pincode (6-digit)
    m = re.search(r"\b(\d{6})\b", text)
    if m:
        fields.pincode = m.group(1)

    # Full address
    fields.address = _extract_full_address(text)

    # Name — multi-strategy:
    # 1. Look for name adjacent to Aadhaar number (printed on front of card)
    # 2. Look for name near DOB line
    # 3. Fall back to spaCy NER
    import re as _re
    name_candidate = None

    # Pattern: Proper name (Title Case) on its own line near the Aadhaar number
    for pat in [
        r"(?:^|\n)([A-Z][a-z]+(?: [A-Z][a-z]+){1,3})\s*\n.*?(?:DOB|D\.O\.B|\d{2}/\d{2}/\d{4})",
        r"([A-Z][a-z]+(?: [A-Z][a-z]+){1,3})\s*\n.*?(?:\d{4}\s?\d{4}\s?\d{4})",
    ]:
        m = _re.search(pat, text, _re.DOTALL | _re.MULTILINE)
        if m:
            cand = m.group(1).strip()
            if _is_valid_name(cand):
                name_candidate = cand
                break

    fields.name = name_candidate or _extract_person_name(text)

    logger.debug(f"Aadhaar fields: {fields.model_dump(exclude={'raw_text'})}")
    return fields.model_dump()


# ── PAN ───────────────────────────────────────────────────────────────────────

def _parse_pan(text: str) -> dict[str, Any]:
    fields = PANFields()

    # PAN number: AAAAA1234A
    m = re.search(r"\b([A-Z]{5}\d{4}[A-Z])\b", text)
    if m:
        fields.pan_number = m.group(1)

    # DOB
    m = re.search(r"\b(\d{2}[/\-]\d{2}[/\-]\d{4})\b", text)
    if m:
        fields.dob = m.group(1)

    fields.name = _extract_person_name(text)

    # Father name: line after "Father's Name" label
    m = re.search(r"father['\u2019s]*\s*name[:\s]+([A-Z\s]+)", text, re.IGNORECASE)
    if m:
        fields.father_name = m.group(1).strip().title()

    logger.debug(f"PAN fields: {fields.model_dump(exclude={'raw_text'})}")
    return fields.model_dump()


# ── Passport ──────────────────────────────────────────────────────────────────

def _parse_passport(text: str) -> dict[str, Any]:
    fields = PassportFields()

    # Passport number: letter + 7 digits
    m = re.search(r"\b([A-Z]\d{7})\b", text)
    if m:
        fields.passport_number = m.group(1)

    # All dates (pick first = DOB, second = expiry if present)
    dates = re.findall(r"\b(\d{2}[/\-]\d{2}[/\-]\d{4})\b", text)
    if dates:
        fields.dob = dates[0]
    if len(dates) >= 2:
        fields.expiry_date = dates[-1]

    # Nationality
    m = re.search(r"nationality[:\s]+([A-Z]+)", text, re.IGNORECASE)
    if m:
        fields.nationality = m.group(1).strip().title()

    # MRZ lines (two consecutive lines of A-Z0-9< ≥ 40 chars)
    mrz_lines = re.findall(r"[A-Z0-9<]{40,}", text)
    if mrz_lines:
        fields.mrz_line1 = mrz_lines[0]
    if len(mrz_lines) >= 2:
        fields.mrz_line2 = mrz_lines[1]

    fields.surname = _extract_person_name(text)

    logger.debug(f"Passport fields: {fields.model_dump(exclude={'raw_text'})}")
    return fields.model_dump()


# ── Resume ────────────────────────────────────────────────────────────────────

def _parse_resume(text: str) -> dict[str, Any]:
    fields = ResumeFields()
    nlp = _get_nlp()
    doc = nlp(text[:5000])  # limit to first 5000 chars for speed

    # Email
    m = re.search(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", text, re.IGNORECASE)
    if m:
        fields.email = m.group(0)

    # Phone
    m = re.search(r"(\+?[\d\s\-()]{7,15})", text)
    if m:
        fields.phone = m.group(1).strip()

    # Name: first PERSON entity
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            fields.name = ent.text
            break

    # Skills: look for a "Skills" section and grab the next lines
    skill_match = re.search(
        r"skills?[:\-\s]+([\s\S]{0,400}?)(?:\n{2,}|education|experience|$)",
        text, re.IGNORECASE,
    )
    if skill_match:
        raw_skills = skill_match.group(1)
        fields.skills = [s.strip() for s in re.split(r"[,\n•\|·]", raw_skills) if s.strip()]

    logger.debug(f"Resume fields: name={fields.name}, email={fields.email}")
    return fields.model_dump()


# ── Graduation / Marksheet ────────────────────────────────────────────────────

def _parse_certificate(text: str) -> dict[str, Any]:
    fields = CertificateFields()

    # Year: 4-digit year
    m = re.search(r"\b(20\d{2}|19\d{2})\b", text)
    if m:
        fields.year = m.group(1)

    # Percentage / CGPA
    m = re.search(r"(\d{1,3}(?:\.\d{1,2})?)\s*%", text)
    if m:
        fields.percentage = m.group(0)

    # Roll number
    m = re.search(r"roll\s*(?:no|number)[:\s]+([A-Z0-9\-/]+)", text, re.IGNORECASE)
    if m:
        fields.roll_number = m.group(1).strip()

    fields.student_name = _extract_person_name(text)

    # Institution: line containing "University" or "College"
    m = re.search(r"([A-Z][^\n]*(?:university|college|institute)[^\n]*)", text, re.IGNORECASE)
    if m:
        fields.institution = m.group(1).strip()

    logger.debug(f"Certificate fields: {fields.model_dump(exclude={'raw_text'})}")
    return fields.model_dump()


# ── Generic fallback ──────────────────────────────────────────────────────────

def _parse_generic(text: str) -> dict[str, Any]:
    return ExtractedFields(raw_text=text).model_dump()


# ── Helper ────────────────────────────────────────────────────────────────────

def _extract_full_address(text: str) -> str | None:
    """
    Extract a full postal address from OCR text.

    Strategy (highest → lowest confidence):

    1. **Label-anchored block** — locate an "Address:" / "S/O" / "C/O" label and
       capture the multi-line block that follows until a sentinel (Aadhaar number,
       DOB line, gender keyword, or two blank lines).

    2. **Pincode-anchored block** — walk backwards from the 6-digit pincode,
       collecting contiguous non-empty lines that look like address components
       (they contain digits, commas, directional keywords, or known locality
       suffixes like Nagar / Road / Street / Colony / District / State).

    3. **Heuristic line scan** — collect every line that looks address-like
       (contains a number + word, or contains locality/city keywords) and
       return them joined.

    Returns a single cleaned string or None if nothing plausible found.
    """
    import re as _re

    # ── helpers ───────────────────────────────────────────────────────────────
    _LOCALITY_WORDS = _re.compile(
        r"\b(road|rd|street|st|nagar|colony|sector|phase|block|district|"
        r"tehsil|taluka|village|town|city|state|near|opp|opposite|plot|"
        r"flat|house|h\.no|floor|lane|marg|vihar|enclave|extension|park|"
        r"ward|area|locality|mohalla|chowk|bazaar|bazar|market|circle|"
        r"cross|layout|scheme|society|apartment|apt|residency|residences|"
        r"complex|tower|building|bhawan|gali|bypass|highway|nh|sh)\b",
        _re.IGNORECASE,
    )

    _SENTINELS = _re.compile(
        r"(\b\d{4}\s?\d{4}\s?\d{4}\b"            # Aadhaar number
        r"|\b(DOB|D\.O\.B|Date of Birth)\b"       # DOB label
        r"|\b(Male|Female|Transgender)\b"         # Gender
        r"|\b(PAN|Permanent Account)\b"           # PAN label
        r"|^\s*$)",                               # blank line (used per-line)
        _re.IGNORECASE | _re.MULTILINE,
    )

    def _clean_line(ln: str) -> str:
        return _re.sub(r"\s+", " ", ln).strip()

    def _is_address_line(ln: str) -> bool:
        ln = ln.strip()
        if not ln or len(ln) < 4:
            return False
        if _re.search(r"\d", ln) and _re.search(r"[A-Za-z]", ln):
            return True
        if _LOCALITY_WORDS.search(ln):
            return True
        return False

    lines = text.splitlines()

    # ── Strategy 1: label-anchored ────────────────────────────────────────────
    label_pat = _re.compile(
        r"(?:^|\n)\s*(?:address|addr|s/o|c/o|w/o|d/o|h\.?\s*no\.?)[:\s]+",
        _re.IGNORECASE,
    )
    m = label_pat.search(text)
    if m:
        start_idx = text.index("\n", m.start()) + 1 if "\n" in text[m.start():] else m.end()
        # collect text from the label match end
        after = text[m.end():]
        addr_lines: list[str] = []
        consecutive_blanks = 0
        for ln in after.splitlines():
            cln = _clean_line(ln)
            # stop at sentinels
            if _re.search(
                r"(\b\d{4}\s?\d{4}\s?\d{4}\b|\b(DOB|D\.O\.B|Date of Birth)\b|\b(Male|Female|Transgender)\b)",
                cln, _re.IGNORECASE
            ):
                break
            if not cln:
                consecutive_blanks += 1
                if consecutive_blanks >= 2:
                    break
                continue
            consecutive_blanks = 0
            addr_lines.append(cln)
            if len(addr_lines) >= 6:   # cap at 6 lines
                break
        if addr_lines:
            return ", ".join(addr_lines)

    # ── Strategy 2: pincode-anchored ─────────────────────────────────────────
    pin_match = _re.search(r"\b(\d{6})\b", text)
    if pin_match:
        pin_line_idx: int | None = None
        for i, ln in enumerate(lines):
            if pin_match.group(1) in ln:
                pin_line_idx = i
                break

        if pin_line_idx is not None:
            # collect lines around the pincode (2 before, 3 after)
            start = max(0, pin_line_idx - 4)
            end   = min(len(lines), pin_line_idx + 3)
            block = [_clean_line(lines[j]) for j in range(start, end)]
            addr_block = [ln for ln in block if _is_address_line(ln) or pin_match.group(1) in ln]
            if len(addr_block) >= 2:
                return ", ".join(addr_block)

    # ── Strategy 3: heuristic scan ───────────────────────────────────────────
    addr_lines = [_clean_line(ln) for ln in lines if _is_address_line(ln)]
    if len(addr_lines) >= 2:
        return ", ".join(addr_lines[:6])

    return None


def _extract_person_name(text: str, preferred_names: list[str] | None = None) -> str | None:
    """
    Extract person name with multi-strategy approach:
    1. Look for name after known label patterns (most reliable for Indian IDs)
    2. Use spaCy NER on a clean excerpt of the text
    3. Fall back to spaCy on full text

    OCR noise produces many short garbage tokens that spaCy misidentifies as PERSON.
    We filter out tokens that look like OCR artifacts (< 3 chars, non-alpha majority).
    """
    import re as _re

    # Strategy 1: label-based extraction (Aadhaar / PAN style)
    label_patterns = [
        r"Name[:\s]+([A-Z][a-z]+(\s+[A-Z][a-z]+){1,3})",       # "Name: Ajay Sharma"
        r"([A-Z][a-z]+(\s+[A-Z][a-z]+){1,3})\s*\n\s*\d{2}/",  # Name line before DOB
    ]
    for pat in label_patterns:
        m = _re.search(pat, text)
        if m:
            candidate = m.group(1).strip()
            if _is_valid_name(candidate):
                return candidate

    # Strategy 2: spaCy on a shorter, cleaner window
    # Focus on the last 1500 chars (Aadhaar prints name near the bottom)
    clean_window = text[-1500:]
    nlp = _get_nlp()
    for window in [clean_window, text[:2000]]:
        doc = nlp(window)
        candidates = [
            ent.text.strip()
            for ent in doc.ents
            if ent.label_ == "PERSON" and _is_valid_name(ent.text.strip())
        ]
        if candidates:
            # Prefer longer names (more words = more likely to be a real full name)
            candidates.sort(key=lambda n: (-len(n.split()), -len(n)))
            return candidates[0]

    return None


def _is_valid_name(text: str) -> bool:
    """Return True if text looks like a plausible person name."""
    import re as _re
    if not text or len(text) < 4:
        return False
    words = text.split()
    if len(words) < 2 or len(words) > 5:
        return False
    # Each word should be mostly alphabetic and at least 2 chars
    for w in words:
        if len(w) < 2:
            return False
        alpha_ratio = sum(c.isalpha() for c in w) / len(w)
        if alpha_ratio < 0.8:
            return False
    # No digits allowed in a name
    if _re.search(r"\d", text):
        return False
    return True