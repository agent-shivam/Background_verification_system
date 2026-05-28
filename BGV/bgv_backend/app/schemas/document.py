"""
app/schemas/document.py
───────────────────────
Typed Pydantic v2 models for all API request / response payloads.
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════════════════
# Enumerations
# ══════════════════════════════════════════════════════════════════════════════

class DocumentType(str, Enum):
    aadhaar    = "aadhaar"
    pan        = "pan"
    passport   = "passport"
    resume     = "resume"
    graduation = "graduation_certificate"
    marksheet  = "marksheet"
    unknown    = "unknown"


class RiskStatus(str, Enum):
    genuine     = "Likely Genuine"
    suspicious  = "Suspicious"
    fake        = "Likely Fake / Tampered"


class ELAResult(str, Enum):
    clean       = "clean"
    suspicious  = "suspicious"
    tampered    = "tampered"


class MetadataResult(str, Enum):
    normal      = "normal"
    suspicious  = "suspicious"
    missing     = "missing"


class BlurResult(str, Enum):
    sharp       = "sharp"
    slightly_blurry = "slightly_blurry"
    blurry      = "blurry"


class DuplicateRegionResult(str, Enum):
    not_detected = "not_detected"
    detected     = "detected"


class LayoutResult(str, Enum):
    valid       = "valid"
    invalid     = "invalid"
    unknown     = "unknown"


class QRResult(str, Enum):
    verified    = "verified"
    mismatch    = "mismatch"
    not_found   = "not_found"
    decode_error = "decode_error"


class AIArtifactResult(str, Enum):
    not_detected = "not_detected"
    suspected    = "suspected"


class OCRBox(BaseModel):
    text: str
    confidence: float
    bbox: list[list[int]]


# ══════════════════════════════════════════════════════════════════════════════
# Sub-schemas
# ══════════════════════════════════════════════════════════════════════════════

class FraudAnalysis(BaseModel):
    ela:               ELAResult              = Field(..., description="Error Level Analysis result")
    metadata:          MetadataResult         = Field(..., description="EXIF / file metadata analysis")
    blur:              BlurResult             = Field(..., description="Image sharpness / blur check")
    duplicate_regions: DuplicateRegionResult  = Field(..., description="ORB-based copy-move detection")
    layout_validation: LayoutResult           = Field(..., description="Template / layout consistency")
    qr_validation:     QRResult               = Field(..., description="QR cross-validation (Aadhaar)")
    ai_artifacts:      AIArtifactResult       = Field(..., description="AI-generated content detection")
    ela_score:         float                  = Field(..., ge=0, le=100, description="Raw ELA anomaly score (0–100)")
    blur_score:        float                  = Field(..., ge=0, description="Laplacian variance (higher = sharper)")


class ExtractedFields(BaseModel):
    """
    Generic container — actual keys depend on document type.
    Use `model_extra` (pydantic v2 extra='allow') to carry dynamic fields.
    """
    model_config = {"extra": "allow"}

    raw_text: str = Field(default="", description="Full concatenated OCR text")


# ── Aadhaar specific ──────────────────────────────────────────────────────────
class AadhaarFields(ExtractedFields):
    name:         str | None = None
    dob:          str | None = None
    gender:       str | None = None
    aadhaar_number: str | None = None
    address:      str | None = None
    pincode:      str | None = None


# ── PAN specific ──────────────────────────────────────────────────────────────
class PANFields(ExtractedFields):
    name:         str | None = None
    father_name:  str | None = None
    dob:          str | None = None
    pan_number:   str | None = None


# ── Passport specific ─────────────────────────────────────────────────────────
class PassportFields(ExtractedFields):
    surname:      str | None = None
    given_names:  str | None = None
    nationality:  str | None = None
    dob:          str | None = None
    passport_number: str | None = None
    expiry_date:  str | None = None
    mrz_line1:    str | None = None
    mrz_line2:    str | None = None


# ── Resume specific ───────────────────────────────────────────────────────────
class ResumeFields(ExtractedFields):
    name:         str | None = None
    email:        str | None = None
    phone:        str | None = None
    skills:       list[str]  = Field(default_factory=list)
    education:    list[str]  = Field(default_factory=list)
    experience:   list[str]  = Field(default_factory=list)


# ── Certificate / Marksheet ───────────────────────────────────────────────────
class CertificateFields(ExtractedFields):
    student_name:   str | None = None
    institution:    str | None = None
    degree:         str | None = None
    year:           str | None = None
    roll_number:    str | None = None
    percentage:     str | None = None


# ══════════════════════════════════════════════════════════════════════════════
# Top-level response schemas
# ══════════════════════════════════════════════════════════════════════════════

class ExtractResponse(BaseModel):
    """Response for POST /extract"""
    document_type:   DocumentType
    extracted_fields: dict[str, Any]   = Field(..., description="Parsed document fields")
    llm_credentials: dict[str, Any]    = Field(default_factory=dict, description="LLM-extracted credential fields (name, DOB, document numbers, address, etc.)")
    confidence:      float             = Field(..., ge=0, le=1, description="OCR confidence (0–1)")
    page_count:      int               = Field(default=1)
    processing_time_ms: float          = Field(..., description="End-to-end processing time")


class VerifyResponse(BaseModel):
    """Response for POST /verify"""
    document_type:   DocumentType
    extracted_fields: dict[str, Any]
    fraud_analysis:  FraudAnalysis
    risk_score:      int               = Field(..., ge=0, le=100, description="Composite risk score (0=safe, 100=fake)")
    status:          RiskStatus
    confidence:      float             = Field(..., ge=0, le=1)
    page_count:      int               = Field(default=1)
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Response for GET /health"""
    status:  str = "ok"
    version: str
    env:     str

# ══════════════════════════════════════════════════════════════════════════════
# Enhanced schemas — VLM + Validation additions
# ══════════════════════════════════════════════════════════════════════════════

class VLMFraudSignal(BaseModel):
    signal: str
    severity: str  # "low" | "medium" | "high"


class VLMAnalysis(BaseModel):
    """Structured output from Claude VLM layer."""
    document_type_confirmed: str = "unknown"
    vlm_confidence: float = Field(default=0.0, ge=0, le=1)
    field_validation: dict[str, Any] = Field(default_factory=dict)
    logical_inconsistencies: list[str] = Field(default_factory=list)
    fraud_signals: list[VLMFraudSignal] = Field(default_factory=list)
    font_consistency: str = "unknown"
    layout_authenticity: str = "unknown"
    seal_signature_present: bool = False
    visible_tampering: bool = False
    overall_assessment: str = "unknown"
    reasoning: str = ""
    suggested_risk_adjustment: int = Field(default=0, ge=-20, le=40)
    vlm_available: bool = True
    vlm_provider: str = "none"
    vlm_model: str = "none"


class FieldCheckResult(BaseModel):
    field_name: str
    passed: bool
    message: str
    severity: str = "medium"


class ValidationReport(BaseModel):
    checks: list[FieldCheckResult] = Field(default_factory=list)
    passed: int = 0
    failed: int = 0
    critical_failures: int = 0
    validation_score: float = Field(default=1.0, ge=0, le=1)
    summary: str = ""


class ClassificationScore(BaseModel):
    document_type: str
    confidence: float


class ClassificationResult(BaseModel):
    primary_type: str
    primary_confidence: float = Field(ge=0, le=1)
    all_scores: dict[str, float] = Field(default_factory=dict)
    is_ambiguous: bool = False


class AadhaarVerification(BaseModel):
    """Aadhaar-specific verification results (only populated for Aadhaar documents)."""
    # Verhoeff checksum
    verhoeff_valid: bool = False
    verhoeff_reason: str = ""
    verhoeff_penalty: int = 0

    # Secure QR
    qr_type: str = "not_found"
    qr_fields_found: list[str] = Field(default_factory=list)
    qr_cross_validation_status: str = "no_qr"
    qr_field_matches: dict[str, Any] = Field(default_factory=dict)
    qr_mismatches: list[str] = Field(default_factory=list)
    qr_penalty: int = 0
    qr_summary: str = ""

    # Font consistency
    font_status: str = "consistent"
    font_variance_ratio: float = 0.0
    font_suspicious_blocks: int = 0
    font_high_frequency_anomaly: bool = False
    font_compression_inconsistency: bool = False
    font_penalty: int = 0
    font_detail: str = ""

    # Layout authenticity
    layout_status: str = "unknown"
    layout_issues: list[str] = Field(default_factory=list)
    layout_penalty: int = 0
    layout_detail: str = ""

    # Composite
    total_aadhaar_penalty: int = 0
    aadhaar_risk_level: str = "low"
    checks_passed: int = 0
    checks_failed: int = 0
    summary: str = ""
    corrected_fields: dict[str, Any] = Field(default_factory=dict)


class EnhancedVerifyResponse(BaseModel):
    """Extended response for POST /verify with full pipeline output."""
    # Core (same as before)
    document_type: DocumentType
    extracted_fields: dict[str, Any]
    llm_credentials: dict[str, Any] = Field(default_factory=dict, description="LLM-extracted credential fields (name, DOB, document numbers, address, etc.)")
    fraud_analysis: FraudAnalysis
    risk_score: int = Field(..., ge=0, le=100)
    status: RiskStatus
    confidence: float = Field(..., ge=0, le=1)
    page_count: int = Field(default=1)
    processing_time_ms: float

    # Enhanced layers
    classification: ClassificationResult
    vlm_analysis: VLMAnalysis
    validation: ValidationReport

    # Aadhaar-specific deep verification (None for non-Aadhaar docs)
    aadhaar_verification: AadhaarVerification | None = Field(
        default=None,
        description="Deep Aadhaar-specific checks: Verhoeff, Secure QR, font consistency, layout"
    )

    # Audit trail
    pipeline_stages: list[str] = Field(
        default_factory=list,
        description="Ordered list of pipeline stages executed"
    )
