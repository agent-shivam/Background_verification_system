"""
app/api/routes/documents.py
────────────────────────────
POST /extract  — OCR + classification + field parsing (lightweight).
POST /verify   — Full enterprise pipeline: OCR → VLM → Fraud → Validation → Risk Score.
POST /analyze  — Alias for /verify that returns a human-readable summary alongside JSON.

All endpoints accept multipart/form-data with a single `file` field.
Supported formats: PDF, JPEG, PNG, TIFF.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from loguru import logger

from app.core.exceptions import BGVBaseError, FileValidationError
from app.schemas.document import ExtractResponse, EnhancedVerifyResponse
from app.services.pipeline import run_extract_pipeline, run_verify_pipeline
from app.utils.file_utils import cleanup_file, save_upload, validate_upload

router = APIRouter(tags=["Documents"])


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _read_and_save(file: UploadFile) -> Path:
    content = await file.read()
    try:
        validate_upload(file, content)
    except FileValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": type(exc).__name__, "message": exc.message},
        ) from exc
    return save_upload(content, file.filename or "upload.bin")


def _handle_bgv_error(exc: BGVBaseError) -> HTTPException:
    return HTTPException(
        status_code=exc.http_status,
        detail={"error": type(exc).__name__, "message": exc.message},
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /extract
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/extract",
    response_model=ExtractResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract document fields (lightweight)",
    description=(
        "Lightweight pipeline: preprocess → OCR → classify → parse fields.\n\n"
        "No VLM, no fraud detection. Use `/verify` for full analysis."
    ),
)
async def extract(
    file: UploadFile = File(
        ...,
        description="Document image or PDF (max 20 MB). Supported: PDF, JPEG, PNG, TIFF.",
    ),
) -> ExtractResponse:
    saved_path: Path | None = None
    try:
        saved_path = await _read_and_save(file)
        logger.info(f"[/extract] Processing: {saved_path.name}")
        result = await asyncio.get_event_loop().run_in_executor(
            None, run_extract_pipeline, saved_path
        )
        return result
    except HTTPException:
        raise
    except BGVBaseError as exc:
        raise _handle_bgv_error(exc) from exc
    except Exception as exc:
        logger.exception(f"[/extract] Unexpected error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "InternalError", "message": "An unexpected error occurred."},
        ) from exc
    finally:
        if saved_path:
            cleanup_file(saved_path)


# ─────────────────────────────────────────────────────────────────────────────
# POST /verify   (full enterprise pipeline)
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/verify",
    response_model=EnhancedVerifyResponse,
    status_code=status.HTTP_200_OK,
    summary="Full enterprise document verification",
    description=(
        "Complete 9-stage background verification pipeline:\n\n"
        "1. **Image Loading** — PDF/image ingestion\n"
        "2. **Preprocessing** — denoise, resize, deskew, sharpen\n"
        "3. **Layout Detection** — PPStructure region analysis\n"
        "4. **OCR Extraction** — PaddleOCR bilingual text extraction\n"
        "5. **Document Classification** — weighted signal scoring\n"
        "6. **Field Parsing** — regex + spaCy NER structured extraction\n"
        "7. **VLM Understanding** — Vision model semantic analysis (provider-agnostic)\n"
        "8. **Fraud/Tamper Detection** — ELA, blur, ORB, metadata, AI artifact checks\n"
        "9. **Validation Engine** — business rule + format cross-checks\n"
        "10. **Risk Scoring** — composite 0–100 score with VLM adjustment\n\n"
        "Returns risk score (0 = safe, 100 = fake), status label, VLM reasoning, "
        "and structured validation report."
    ),
)
async def verify(
    file: UploadFile = File(
        ...,
        description="Document image or PDF (max 20 MB). Supported: PDF, JPEG, PNG, TIFF.",
    ),
) -> EnhancedVerifyResponse:
    saved_path: Path | None = None
    try:
        saved_path = await _read_and_save(file)
        logger.info(f"[/verify] Processing: {saved_path.name}")
        result = await asyncio.get_event_loop().run_in_executor(
            None, run_verify_pipeline, saved_path
        )
        return result
    except HTTPException:
        raise
    except BGVBaseError as exc:
        raise _handle_bgv_error(exc) from exc
    except Exception as exc:
        logger.exception(f"[/verify] Unexpected error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "InternalError", "message": "An unexpected error occurred."},
        ) from exc
    finally:
        if saved_path:
            cleanup_file(saved_path)


# ─────────────────────────────────────────────────────────────────────────────
# POST /analyze  (alias of /verify with human-readable summary wrapper)
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/analyze",
    status_code=status.HTTP_200_OK,
    summary="Analyze document — returns verification result with human summary",
    description=(
        "Runs the same full pipeline as `/verify` but wraps the result with a "
        "plain-English summary field for easier consumption by front-end apps."
    ),
)
async def analyze(
    file: UploadFile = File(...),
) -> dict:
    saved_path: Path | None = None
    try:
        saved_path = await _read_and_save(file)
        logger.info(f"[/analyze] Processing: {saved_path.name}")
        result: EnhancedVerifyResponse = await asyncio.get_event_loop().run_in_executor(
            None, run_verify_pipeline, saved_path
        )

        # Build human-readable summary
        vlm = result.vlm_analysis
        val = result.validation
        fraud = result.fraud_analysis

        high_fraud = [
            s.signal for s in vlm.fraud_signals if s.severity == "high"
        ]
        failed_checks = [c.message for c in val.checks if not c.passed and c.severity == "high"]

        summary_lines = [
            f"Document Type: {result.document_type.value.replace('_', ' ').title()}",
            f"Verification Status: {result.status.value}",
            f"Risk Score: {result.risk_score}/100",
            f"OCR Confidence: {result.confidence:.0%}",
            f"VLM Assessment: {vlm.overall_assessment.title()} — {vlm.reasoning}",
            f"Layout Authenticity: {vlm.layout_authenticity.replace('_', ' ').title()}",
            f"Font Consistency: {vlm.font_consistency.replace('_', ' ').title()}",
            f"Visible Tampering Detected: {'Yes ⚠️' if vlm.visible_tampering else 'No ✓'}",
            f"Seal/Signature Present: {'Yes ✓' if vlm.seal_signature_present else 'Not detected'}",
            f"Validation: {val.summary}",
        ]
        if high_fraud:
            summary_lines.append(f"High-Severity Fraud Signals: {'; '.join(high_fraud)}")
        if failed_checks:
            summary_lines.append(f"Critical Field Issues: {'; '.join(failed_checks)}")
        if vlm.logical_inconsistencies:
            summary_lines.append(f"Logical Inconsistencies: {'; '.join(vlm.logical_inconsistencies)}")

        return {
            **result.model_dump(),
            "human_summary": "\n".join(summary_lines),
        }

    except HTTPException:
        raise
    except BGVBaseError as exc:
        raise _handle_bgv_error(exc) from exc
    except Exception as exc:
        logger.exception(f"[/analyze] Unexpected error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "InternalError", "message": "An unexpected error occurred."},
        ) from exc
    finally:
        if saved_path:
            cleanup_file(saved_path)