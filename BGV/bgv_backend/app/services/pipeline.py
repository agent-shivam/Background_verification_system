"""
app/services/pipeline.py
────────────────────────
Central orchestration service — full enterprise-grade pipeline for document extraction and verification.
Both /extract and /verify call into this module:
  • /extract  → lightweight path (OCR + classification + parsing)
  • /verify   → full pipeline including VLM + fraud + validation
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from app.core.config import settings
from app.schemas.document import (
    DocumentType,
    FraudAnalysis,
    QRResult,
    LayoutResult,
    ELAResult,
    MetadataResult,
    BlurResult,
    DuplicateRegionResult,
    AIArtifactResult,
    RiskStatus,
    ExtractResponse,
    VerifyResponse,
    EnhancedVerifyResponse,
    VLMAnalysis,
    VLMFraudSignal,
    AadhaarVerification,
    ValidationReport as ValidationReportSchema,
    FieldCheckResult,
    ClassificationResult as ClassificationResultSchema,
)
from app.services.ocr.engine import run_ocr
from app.services.preprocessing.image_processor import (
    preprocess_image,
    pdf_to_images,
    load_image,
)
from app.services.layout.structure_engine import analyse_layout
from app.services.classification.document_classifier import classify_document
from app.services.parsers.field_parser import parse_fields
from app.services.forgery.ela import run_ela
from app.services.forgery.blur import detect_blur
from app.services.forgery.metadata import analyse_metadata
from app.services.forgery.duplicate_regions import detect_duplicate_regions
from app.services.forgery.layout_validator import validate_layout
from app.services.forgery.ai_artifact_detector import detect_ai_artifacts
from app.services.forgery.risk_scorer import compute_risk_score
from app.services.ocr.cleaner import clean_ocr_text
from app.services.qr.qr_decoder import decode_qr, validate_aadhaar_qr
from app.services.vlm.claude_vlm import run_vlm_analysis
from app.services.validation.field_validator import validate_fields
from app.services.llm.credential_extractor import extract_credentials_with_llm
from app.services.aadhar.aadhaar_verifier import run_aadhaar_verification


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_images(file_path: Path) -> list[np.ndarray]:
    """Return a list of BGR images from a PDF or an image file."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return pdf_to_images(file_path)
    return [load_image(file_path)]


def _run_ocr_on_pages(images: list[np.ndarray]) -> tuple[str, float]:
    """
    Preprocess each page and run OCR.
    Returns (full_text, mean_confidence).
    """
    all_lines: list[str] = []
    all_confidences: list[float] = []

    for idx, img in enumerate(images):
        preprocessed = preprocess_image(img)
        result = run_ocr(preprocessed, lang="en")
        all_lines.extend(result.lines)
        all_confidences.append(result.confidence)
        logger.debug(f"Page {idx + 1}: {len(result.lines)} lines, conf={result.confidence:.2%}")

    full_text = "\n".join(all_lines)
    mean_conf = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
    return full_text, mean_conf


def _vlm_dict_to_schema(raw: dict) -> VLMAnalysis:
    """Convert raw VLM dict to Pydantic schema."""
    fraud_signals = [
        VLMFraudSignal(signal=s.get("signal", ""), severity=s.get("severity", "medium"))
        for s in raw.get("fraud_signals", [])
        if isinstance(s, dict)
    ]
    return VLMAnalysis(
        document_type_confirmed=raw.get("document_type_confirmed", "unknown"),
        vlm_confidence=float(raw.get("vlm_confidence", 0.0)),
        field_validation=raw.get("field_validation", {}),
        logical_inconsistencies=raw.get("logical_inconsistencies", []),
        fraud_signals=fraud_signals,
        font_consistency=raw.get("font_consistency", "unknown"),
        layout_authenticity=raw.get("layout_authenticity", "unknown"),
        seal_signature_present=bool(raw.get("seal_signature_present", False)),
        visible_tampering=bool(raw.get("visible_tampering", False)),
        overall_assessment=raw.get("overall_assessment", "unknown"),
        reasoning=raw.get("reasoning", ""),
        suggested_risk_adjustment=int(raw.get("suggested_risk_adjustment", 0)),
        vlm_available=raw.get("vlm_available", True),
    )


def _validation_to_schema(report) -> ValidationReportSchema:
    """Convert internal ValidationReport dataclass to Pydantic schema."""
    checks = [
        FieldCheckResult(
            field_name=c.field_name,
            passed=c.passed,
            message=c.message,
            severity=c.severity,
        )
        for c in report.checks
    ]
    return ValidationReportSchema(
        checks=checks,
        passed=report.passed,
        failed=report.failed,
        critical_failures=report.critical_failures,
        validation_score=report.validation_score,
        summary=report.summary,
    )


def _classification_to_schema(cls_result) -> ClassificationResultSchema:
    return ClassificationResultSchema(
        primary_type=cls_result.primary_type.value,
        primary_confidence=cls_result.primary_confidence,
        all_scores={k.value: v for k, v in cls_result.all_scores.items()},
        is_ambiguous=cls_result.is_ambiguous,
    )


def _apply_vlm_risk_adjustment(base_score: int, vlm: VLMAnalysis) -> int:
    """Apply VLM's suggested risk delta, clamped to [0, 100]."""
    if not vlm.vlm_available:
        return base_score
    # Extra penalty if VLM sees visible tampering
    extra = 15 if vlm.visible_tampering else 0
    adjusted = base_score + vlm.suggested_risk_adjustment + extra
    return max(0, min(100, adjusted))


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def run_extract_pipeline(file_path: Path) -> ExtractResponse:
    """
    Lightweight pipeline: Preprocess → OCR → Classify → Parse.
    Used by POST /extract.
    """
    t0 = time.perf_counter()
    stages = []

    # Stage 1 — Load
    images = _load_images(file_path)
    stages.append("load_images")

    # Stage 2 — OCR (with preprocessing inside)
    full_text, confidence = _run_ocr_on_pages(images)
    full_text = clean_ocr_text(full_text)
    stages.append("ocr_extraction")

    # Stage 3 — Classification (enhanced weighted classifier)
    cls_result = classify_document(full_text)
    doc_type = cls_result.primary_type
    stages.append("document_classification")

    # Stage 4 — LLM credential extraction (text-only LLM on clean OCR text)
    logger.info("Running LLM credential extraction...")
    llm_credentials = extract_credentials_with_llm(full_text)
    stages.append("llm_credential_extraction")

    # Stage 5 — Field parsing
    fields = parse_fields(full_text, doc_type)
    stages.append("field_parsing")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        f"Extract pipeline done — type={doc_type.value}, "
        f"confidence={confidence:.2%}, time={elapsed_ms:.0f}ms"
    )

    return ExtractResponse(
        document_type=doc_type,
        extracted_fields=fields,
        llm_credentials=llm_credentials,
        confidence=round(confidence, 4),
        page_count=len(images),
        processing_time_ms=round(elapsed_ms, 1),
    )


def run_verify_pipeline(file_path: Path) -> EnhancedVerifyResponse:
    """
    Full enterprise verification pipeline (9 stages):
      1. Image loading
      2. Preprocessing (denoise/resize/deskew/sharpen)
      3. Layout detection (PPStructure regions)
      4. OCR extraction (PaddleOCR bilingual)
      5. Document classification (weighted signal scoring)
      6. Field parsing (regex + spaCy NER)
      7. VLM understanding (provider-agnostic — semantic validation)
      8. Fraud/tamper detection (ELA, blur, ORB, metadata, AI artifacts)
      9. Validation engine (business rule checks)
     10. Risk scoring (composite 0–100 with VLM adjustment)

    Used by POST /verify.
    """
    t0 = time.perf_counter()
    stages: list[str] = []

    # ── Stage 1: Load images ────────────────────────────────────────────────
    images = _load_images(file_path)
    primary_image = images[0]
    stages.append("load_images")
    logger.info(f"Loaded {len(images)} page(s)")

    # ── Stage 2: Layout detection ───────────────────────────────────────────
    layout_regions = analyse_layout(primary_image)
    stages.append("layout_detection")

    # ── Stage 3: OCR (preprocessing happens inside) ─────────────────────────
    full_text, confidence = _run_ocr_on_pages(images)
    full_text = clean_ocr_text(full_text)
    stages.append("ocr_extraction")

    # ── Stage 4: Document Classification ────────────────────────────────────
    cls_result = classify_document(full_text)
    doc_type = cls_result.primary_type
    stages.append("document_classification")
    logger.info(
        f"Classified as: {doc_type.value} "
        f"(confidence={cls_result.primary_confidence:.1%}, "
        f"ambiguous={cls_result.is_ambiguous})"
    )

    # ── Stage 5: Structured field parsing ───────────────────────────────────
    fields = parse_fields(full_text, doc_type)
    stages.append("field_parsing")

    # ── Stage 5.5: LLM credential extraction (text-only, runs after OCR) ────
    logger.info("Running LLM credential extraction...")
    llm_credentials = extract_credentials_with_llm(full_text)
    stages.append("llm_credential_extraction")

    # ── Stage 6: VLM Understanding (provider-agnostic) ───────────────────────
    import os as _os
    _vlm_provider = _os.environ.get("VLM_PROVIDER", "openrouter")
    _vlm_model    = _os.environ.get("VLM_MODEL", "")
    logger.info(f"Running VLM analysis via {_vlm_provider} / {_vlm_model}...")
    vlm_raw = run_vlm_analysis(
        image=primary_image,
        ocr_text=full_text,
        parsed_fields=fields,
        doc_type=doc_type.value,
    )
    vlm_analysis = _vlm_dict_to_schema(vlm_raw)
    stages.append("vlm_understanding")

    # ── Stage 7: Fraud/Tamper detection suite ───────────────────────────────
    ela_result, ela_score = run_ela(primary_image)
    blur_result, blur_score = detect_blur(primary_image)
    meta_result, _meta_dict = analyse_metadata(file_path)
    dup_result = detect_duplicate_regions(primary_image)
    layout_result = validate_layout(full_text, doc_type, primary_image)
    ai_result = detect_ai_artifacts(primary_image)
    stages.append("fraud_detection")

    # ── Stage 8: QR validation (Aadhaar — legacy decode for FraudAnalysis) ──
    qr_result: QRResult
    if doc_type == DocumentType.aadhaar:
        qr_data = decode_qr(primary_image)
        qr_result = validate_aadhaar_qr(qr_data, fields)
    else:
        qr_result = QRResult.not_found
    stages.append("qr_validation")

    # ── Stage 8.5: Deep Aadhaar Verification Suite ───────────────────────────
    # Runs ONLY for Aadhaar cards. Includes:
    #   • Verhoeff checksum (mathematical validity)
    #   • Secure QR decode + OCR cross-match (STRONGEST signal)
    #   • Font consistency analysis (detect edited text)
    #   • Layout authenticity (UIDAI structure validation)
    aadhaar_verification_schema: AadhaarVerification | None = None
    if doc_type == DocumentType.aadhaar:
        logger.info("Running deep Aadhaar verification suite (Verhoeff + SecureQR + Font + Layout)...")
        aadhaar_result = run_aadhaar_verification(primary_image, fields)
        aadhaar_verification_schema = AadhaarVerification(
            verhoeff_valid=aadhaar_result.verhoeff_valid,
            verhoeff_reason=aadhaar_result.verhoeff_reason,
            verhoeff_penalty=aadhaar_result.verhoeff_penalty,
            qr_type=aadhaar_result.qr_type,
            qr_fields_found=aadhaar_result.qr_fields_found,
            qr_cross_validation_status=aadhaar_result.qr_cross_validation_status,
            qr_field_matches=aadhaar_result.qr_field_matches,
            qr_mismatches=aadhaar_result.qr_mismatches,
            qr_penalty=aadhaar_result.qr_penalty,
            qr_summary=aadhaar_result.qr_summary,
            font_status=aadhaar_result.font_status,
            font_variance_ratio=aadhaar_result.font_variance_ratio,
            font_suspicious_blocks=aadhaar_result.font_suspicious_blocks,
            font_high_frequency_anomaly=aadhaar_result.font_high_frequency_anomaly,
            font_compression_inconsistency=aadhaar_result.font_compression_inconsistency,
            font_penalty=aadhaar_result.font_penalty,
            font_detail=aadhaar_result.font_detail,
            layout_status=aadhaar_result.layout_status,
            layout_issues=aadhaar_result.layout_issues,
            layout_penalty=aadhaar_result.layout_penalty,
            layout_detail=aadhaar_result.layout_detail,
            total_aadhaar_penalty=aadhaar_result.total_aadhaar_penalty,
            aadhaar_risk_level=aadhaar_result.aadhaar_risk_level,
            checks_passed=aadhaar_result.checks_passed,
            checks_failed=aadhaar_result.checks_failed,
            summary=aadhaar_result.summary,
            corrected_fields=aadhaar_result.corrected_fields,
        )
        # Upgrade legacy QR result from deep verification if better info available
        if aadhaar_result.qr_cross_validation_status == "verified":
            qr_result = QRResult.verified
        elif aadhaar_result.qr_cross_validation_status == "mismatch":
            qr_result = QRResult.mismatch
        stages.append("aadhaar_deep_verification")
        logger.info(f"Aadhaar deep verification: {aadhaar_result.summary}")

    # ── Stage 9: Business rule validation ────────────────────────────────────
    validation_report = validate_fields(fields, doc_type)
    stages.append("validation_engine")

    # ── Assemble FraudAnalysis ────────────────────────────────────────────────
    fraud = FraudAnalysis(
        ela=ela_result,
        metadata=meta_result,
        blur=blur_result,
        duplicate_regions=dup_result,
        layout_validation=layout_result,
        qr_validation=qr_result,
        ai_artifacts=ai_result,
        ela_score=ela_score,
        blur_score=blur_score,
    )

    # ── Risk Scoring (with VLM adjustment) ───────────────────────────────────
    base_score, status = compute_risk_score(fraud, doc_type, ela_score)

    # Apply VLM-suggested adjustment
    risk_score = _apply_vlm_risk_adjustment(base_score, vlm_analysis)

    # Apply Aadhaar deep verification penalty (Verhoeff + QR + Font + Layout)
    if aadhaar_verification_schema is not None:
        risk_score = min(100, risk_score + aadhaar_verification_schema.total_aadhaar_penalty)
        logger.info(
            f"After Aadhaar deep verification: risk_score={risk_score} "
            f"(aadhaar_penalty={aadhaar_verification_schema.total_aadhaar_penalty})"
        )

    # If validation has critical failures, push score up
    if validation_report.critical_failures > 0:
        risk_score = min(100, risk_score + validation_report.critical_failures * 8)

    # Re-derive status from final score
    if risk_score < settings.risk_low_threshold:
        status = RiskStatus.genuine
    elif risk_score < settings.risk_medium_threshold:
        status = RiskStatus.suspicious
    else:
        status = RiskStatus.fake

    stages.append("risk_scoring")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        f"Verify pipeline done — type={doc_type.value}, "
        f"risk={risk_score}, status={status.value}, "
        f"vlm={vlm_analysis.overall_assessment}, "
        f"validation={validation_report.summary}, "
        f"time={elapsed_ms:.0f}ms"
    )

    return EnhancedVerifyResponse(
        # Core fields
        document_type=doc_type,
        extracted_fields=fields,
        llm_credentials=llm_credentials,
        fraud_analysis=fraud,
        risk_score=risk_score,
        status=status,
        confidence=round(confidence, 4),
        page_count=len(images),
        processing_time_ms=round(elapsed_ms, 1),
        # Enhanced layers
        classification=_classification_to_schema(cls_result),
        vlm_analysis=vlm_analysis,
        validation=_validation_to_schema(validation_report),
        aadhaar_verification=aadhaar_verification_schema,
        pipeline_stages=stages,
    )