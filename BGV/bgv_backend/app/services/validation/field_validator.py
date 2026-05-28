"""
app/services/validation/field_validator.py
───────────────────────────────────────────
Validation Engine — business logic checks on extracted document fields.

Runs AFTER OCR parsing and VLM analysis to apply deterministic rules:
  • Format validation  (Luhn-like checks, regex patterns)
  • Date logic         (DOB vs expiry consistency, future dates)
  • Cross-field checks (name on PAN vs Aadhaar, DOB consistency)
  • Completeness check (required fields present)

Returns a structured ValidationReport that feeds into the risk scorer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from loguru import logger

from app.schemas.document import DocumentType
from app.services.aadhar.verhoeff import validate_aadhaar_checksum

# final\bgv_system_enhanced\app\services\validation\field_validator.py
# final\bgv_system_enhanced\app\services\aadhar\verhoeff.py
# ── Result structures ─────────────────────────────────────────────────────────

@dataclass
class FieldCheck:
    field_name: str
    passed: bool
    message: str
    severity: str = "medium"   # "low" | "medium" | "high"


@dataclass
class ValidationReport:
    checks: list[FieldCheck] = field(default_factory=list)
    passed: int = 0
    failed: int = 0
    critical_failures: int = 0
    validation_score: float = 1.0   # 1.0 = fully valid; 0.0 = all checks failed
    summary: str = ""

    def add(self, check: FieldCheck) -> None:
        self.checks.append(check)
        if check.passed:
            self.passed += 1
        else:
            self.failed += 1
            if check.severity == "high":
                self.critical_failures += 1

    def finalise(self) -> None:
        total = self.passed + self.failed
        self.validation_score = round(self.passed / total, 4) if total else 1.0
        self.summary = (
            f"{self.passed}/{total} checks passed "
            f"({self.critical_failures} critical failure(s))"
        )
        logger.info(f"Validation: {self.summary}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _luhn_check(number: str) -> bool:
    """Luhn algorithm for numeric ID validation."""
    digits = [int(d) for d in number if d.isdigit()]
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]
    total = sum(odd_digits) + sum(sum(divmod(d * 2, 10)) for d in even_digits)
    return total % 10 == 0


# ── Per-document validators ───────────────────────────────────────────────────

def _validate_aadhaar(fields: dict[str, Any], report: ValidationReport) -> None:
    num = fields.get("aadhaar_number", "")

    # Format: 12 digits
    if num:
        clean = re.sub(r"\D", "", str(num))
        report.add(FieldCheck(
            "aadhaar_number_format",
            len(clean) == 12,
            f"Aadhaar must be 12 digits (got {len(clean)})",
            severity="high",
        ))
        # Aadhaar cannot start with 0 or 1
        report.add(FieldCheck(
            "aadhaar_first_digit",
            clean[0] not in ("0", "1") if len(clean) >= 1 else False,
            "First digit of Aadhaar cannot be 0 or 1",
            severity="high",
        ))
        # Verhoeff checksum — mathematical proof of validity
        if len(clean) == 12:
            v = validate_aadhaar_checksum(clean)
            report.add(FieldCheck(
                "aadhaar_verhoeff_checksum",
                v["valid"],
                v["reason"],
                severity="high",
            ))
    else:
        report.add(FieldCheck("aadhaar_number_present", False,
                              "Aadhaar number not found", severity="high"))

    # DOB sanity
    dob = _parse_date(fields.get("dob"))
    if dob:
        today = date.today()
        report.add(FieldCheck(
            "dob_not_future",
            dob < today,
            f"DOB {dob} is in the future" if dob >= today else "DOB OK",
            severity="high",
        ))
        age = (today - dob).days // 365
        report.add(FieldCheck(
            "dob_realistic_age",
            0 < age < 120,
            f"Age computed as {age} years — {'suspicious' if age >= 120 else 'OK'}",
            severity="medium",
        ))
    else:
        report.add(FieldCheck("dob_present", False, "DOB not found", severity="medium"))

    # Name
    report.add(FieldCheck(
        "name_present",
        bool(fields.get("name")),
        "Name field extracted" if fields.get("name") else "Name not detected",
        severity="medium",
    ))

    # Gender
    gender = (fields.get("gender") or "").lower()
    report.add(FieldCheck(
        "gender_valid",
        gender in ("male", "female", "transgender", ""),
        f"Gender value '{gender}' is valid" if gender else "Gender not found",
        severity="low",
    ))

    # Pincode: 6 digits, must not be 000000
    pin = fields.get("pincode", "")
    if pin:
        report.add(FieldCheck(
            "pincode_valid",
            bool(re.match(r"^[1-9]\d{5}$", str(pin))),
            f"Pincode {pin} valid" if re.match(r"^[1-9]\d{5}$", str(pin)) else f"Invalid pincode: {pin}",
            severity="low",
        ))


def _validate_pan(fields: dict[str, Any], report: ValidationReport) -> None:
    pan = fields.get("pan_number", "")
    if pan:
        valid_format = bool(re.match(r"^[A-Z]{5}\d{4}[A-Z]$", str(pan)))
        report.add(FieldCheck(
            "pan_format",
            valid_format,
            f"PAN format {'valid' if valid_format else 'INVALID'}: {pan}",
            severity="high",
        ))
        # 4th character encodes entity type
        valid_4th = pan[3] in "ABCFGHLJPTK" if len(pan) >= 4 else False
        report.add(FieldCheck(
            "pan_entity_code",
            valid_4th,
            f"PAN 4th char '{pan[3] if len(pan) >= 4 else '?'}' entity code valid" if valid_4th else "Invalid PAN entity code",
            severity="medium",
        ))
    else:
        report.add(FieldCheck("pan_number_present", False, "PAN number not found", severity="high"))

    dob = _parse_date(fields.get("dob"))
    if dob:
        report.add(FieldCheck(
            "pan_dob_valid",
            dob < date.today(),
            "DOB is valid" if dob < date.today() else "DOB is in the future",
            severity="high",
        ))
    report.add(FieldCheck(
        "pan_name_present",
        bool(fields.get("name")),
        "Name present" if fields.get("name") else "Name missing",
        severity="medium",
    ))


def _validate_passport(fields: dict[str, Any], report: ValidationReport) -> None:
    num = fields.get("passport_number", "")
    if num:
        valid = bool(re.match(r"^[A-Z]\d{7}$", str(num)))
        report.add(FieldCheck(
            "passport_number_format",
            valid,
            f"Passport number {'valid' if valid else 'INVALID'}: {num}",
            severity="high",
        ))
    else:
        report.add(FieldCheck("passport_number_present", False, "Passport number missing", severity="high"))

    dob = _parse_date(fields.get("dob"))
    expiry = _parse_date(fields.get("expiry_date"))
    today = date.today()

    if dob:
        report.add(FieldCheck("passport_dob_past", dob < today,
                              "DOB is valid" if dob < today else "DOB in future", severity="high"))
    if expiry:
        is_expired = expiry < today
        report.add(FieldCheck(
            "passport_not_expired",
            not is_expired,
            "Passport is expired" if is_expired else f"Passport valid until {expiry}",
            severity="high" if is_expired else "low",
        ))
    if dob and expiry:
        report.add(FieldCheck(
            "expiry_after_dob",
            expiry > dob,
            "Expiry is after DOB" if expiry > dob else "Expiry date before DOB — impossible",
            severity="high",
        ))

    # MRZ check
    mrz1 = fields.get("mrz_line1", "")
    report.add(FieldCheck(
        "mrz_present",
        bool(mrz1 and len(mrz1) >= 40),
        "MRZ line detected" if mrz1 else "MRZ line not found",
        severity="medium",
    ))


def _validate_certificate(fields: dict[str, Any], report: ValidationReport) -> None:
    year = fields.get("year")
    if year:
        try:
            yr = int(year)
            valid = 1950 <= yr <= date.today().year
            report.add(FieldCheck(
                "certificate_year_valid",
                valid,
                f"Certificate year {yr} is {'valid' if valid else 'suspicious'}",
                severity="medium",
            ))
        except ValueError:
            report.add(FieldCheck("certificate_year_format", False,
                                  f"Cannot parse year: {year}", severity="low"))
    else:
        report.add(FieldCheck("certificate_year_present", False,
                              "Year not found in certificate", severity="low"))

    pct = fields.get("percentage", "")
    if pct:
        nums = re.findall(r"\d+(?:\.\d+)?", str(pct))
        if nums:
            val = float(nums[0])
            report.add(FieldCheck(
                "percentage_range",
                0 <= val <= 100,
                f"Percentage {val}% in valid range" if 0 <= val <= 100 else f"Impossible percentage: {val}%",
                severity="high" if val > 100 else "low",
            ))

    report.add(FieldCheck(
        "institution_present",
        bool(fields.get("institution")),
        "Institution name found" if fields.get("institution") else "Institution name missing",
        severity="medium",
    ))


def _validate_resume(fields: dict[str, Any], report: ValidationReport) -> None:
    email = fields.get("email", "")
    if email:
        valid = bool(re.match(r"^[\w.+-]+@[\w-]+\.[a-z]{2,}$", str(email), re.IGNORECASE))
        report.add(FieldCheck(
            "email_format",
            valid,
            f"Email {'valid' if valid else 'invalid'}: {email}",
            severity="low",
        ))
    else:
        report.add(FieldCheck("email_present", False, "No email found in resume", severity="low"))

    phone = fields.get("phone", "")
    if phone:
        digits = re.sub(r"\D", "", str(phone))
        valid = 7 <= len(digits) <= 15
        report.add(FieldCheck(
            "phone_length",
            valid,
            f"Phone digit count {len(digits)}: {'OK' if valid else 'suspicious'}",
            severity="low",
        ))

    report.add(FieldCheck(
        "resume_has_skills",
        bool(fields.get("skills")),
        "Skills section found" if fields.get("skills") else "No skills listed",
        severity="low",
    ))


# ── Dispatcher ────────────────────────────────────────────────────────────────

_VALIDATORS = {
    DocumentType.aadhaar:    _validate_aadhaar,
    DocumentType.pan:        _validate_pan,
    DocumentType.passport:   _validate_passport,
    DocumentType.graduation: _validate_certificate,
    DocumentType.marksheet:  _validate_certificate,
    DocumentType.resume:     _validate_resume,
}


def validate_fields(
    fields: dict[str, Any],
    doc_type: DocumentType,
) -> ValidationReport:
    """
    Run all business-logic checks for the given document type.

    Returns a ValidationReport with per-field check results and an
    overall validation_score (1.0 = all passed, 0.0 = all failed).
    """
    report = ValidationReport()
    validator = _VALIDATORS.get(doc_type)

    if validator is None:
        logger.debug(f"No validator registered for {doc_type.value} — skipping")
        report.add(FieldCheck("doc_type_supported", False,
                              f"No validation rules for {doc_type.value}", severity="low"))
    else:
        validator(fields, report)

    report.finalise()
    return report
