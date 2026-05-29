Comprehensive README for BGV System Enhanced
markdown

# BGV System Enhanced — Complete Project Encyclopedia

> **AI/ML-Powered Background Verification System**  
> Version 2.0.0 | Python 3.11 | FastAPI + Gradio | Multi-layer Document Forgery Detection

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [Architecture Overview](#2-architecture-overview)
3. [Project Structure](#3-project-structure)
4. [The Pipeline — Every Stage Explained](#4-the-pipeline--every-stage-explained)
5. [Every Module — What, Why, How](#5-every-module--what-why-how)
6. [Library Encyclopedia](#6-library-encyclopedia)
7. [API Reference](#7-api-reference)
8. [Configuration & Environment Variables](#8-configuration--environment-variables)
9. [Schemas & Data Models](#9-schemas--data-models)
10. [Fraud Detection Deep Dive](#10-fraud-detection-deep-dive)
11. [Aadhaar-Specific Verification Suite](#11-aadhaar-specific-verification-suite)
12. [Risk Scoring Engine](#12-risk-scoring-engine)
13. [Frontend (Gradio UI)](#13-frontend-gradio-ui)
14. [How to Run](#14-how-to-run)
15. [Error Handling Hierarchy](#15-error-handling-hierarchy)

---

## 1. What Is This Project?

BGV System Enhanced is an enterprise-grade **Background Verification** system that:

- **Accepts** identity documents (Aadhaar, PAN, Passport) and educational documents (Graduation Certificate, Marksheet, Resume) as PDF or image uploads.
- **Extracts** all structured fields (name, DOB, document number, address, institution, marks, etc.) using multi-engine OCR + regex + NLP.
- **Verifies** authenticity through a 10-stage AI/ML pipeline combining computer-vision forgery detection, vision-language model (VLM) reasoning, and deterministic rule-based validation.
- **Scores** each document 0–100 risk score and classifies it as **Likely Genuine**, **Suspicious**, or **Likely Fake / Tampered**.
- **Exposes** results through a REST API (FastAPI) and a browser-based UI (Gradio).

**Primary use case:** HR background verification — automate the check of employee-submitted documents for authenticity before onboarding.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Client Layer                               │
│   Gradio UI (frontend.py)   ←→   REST API Consumers                │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │ HTTP multipart/form-data
┌──────────────────────────────────▼──────────────────────────────────┐
│                        FastAPI Backend                              │
│   main.py → create_app() → POST /extract | /verify | /analyze      │
│   CORS middleware · BGVBaseError handler · Uvicorn ASGI server      │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────┐
│                     Pipeline Orchestrator                           │
│                   app/services/pipeline.py                          │
│   run_extract_pipeline()  ←→  run_verify_pipeline()                 │
└────┬────────┬────────┬────────┬────────┬────────┬────────┬──────────┘
     │        │        │        │        │        │        │
  Preproc   OCR    Classify  Parse    VLM    Forgery  Validate
  (cv2)  (Paddle  (regex  (regex+  (OpenRouter (ELA/Blur/ (Pydantic
         +Tesser  weights) spaCy)  /NVIDIA    ORB/Meta/  rules +
         act)                     NIM)        Layout/AI) Verhoeff)
                                              │
                                         Risk Scorer
                                        (0-100 score)
```

Two pipeline modes:
- **`/extract`** — lightweight: Preprocess → OCR → Classify → LLM credentials → Parse fields. Fast, no fraud detection.
- **`/verify`** (and alias `/analyze`) — full: all 10 stages including VLM, fraud suite, Aadhaar deep verification, validation, and risk scoring.

---

## 3. Project Structure

```
final/
├── frontend.py                          # Gradio UI — runs independently of backend
└── bgv_system_enhanced/
    ├── main.py                          # Entry point — loads .env, creates ASGI app
    ├── requirements.txt                 # All Python dependencies
    ├── .env                             # API keys and configuration (not committed)
    ├── readme.md                        # Original brief readme
    ├── test_vlm.py                      # Manual VLM integration test script
    ├── sample_docs/                     # Empty folder for test documents
    ├── logs/                            # Runtime log files (loguru output)
    └── app/
        ├── __init__.py
        ├── uploads/                     # Temp storage for in-flight uploads
        ├── api/
        │   ├── __init__.py              # FastAPI app factory (create_app)
        │   └── routes/
        │       ├── documents.py         # POST /extract, /verify, /analyze
        │       └── health.py            # GET /health
        ├── core/
        │   ├── config.py                # Pydantic Settings singleton
        │   ├── exceptions.py            # Custom exception hierarchy
        │   └── logging.py               # Loguru setup
        ├── schemas/
        │   └── document.py              # All Pydantic v2 request/response models
        ├── services/
        │   ├── pipeline.py              # ← CENTRAL ORCHESTRATOR
        │   ├── preprocessing/
        │   │   └── image_processor.py   # PDF→image, denoise, deskew, sharpen
        │   ├── ocr/
        │   │   ├── engine.py            # PaddleOCR + Tesseract fusion
        │   │   └── cleaner.py           # OCR text post-processing
        │   ├── classification/
        │   │   └── document_classifier.py  # Weighted keyword classifier
        │   ├── parsers/
        │   │   ├── field_parser.py      # Regex + spaCy NER per doc type
        │   │   ├── document_detector.py # Legacy type detector
        │   │   └── region_detector.py   # Region extraction helper
        │   ├── llm/
        │   │   └── credential_extractor.py  # NVIDIA NIM LLM credential extraction
        │   ├── vlm/
        │   │   └── claude_vlm.py        # VLM analysis (OpenRouter model chain)
        │   ├── forgery/
        │   │   ├── ela.py               # Error Level Analysis
        │   │   ├── blur.py              # Laplacian blur / sharpness
        │   │   ├── metadata.py          # EXIF metadata inspection
        │   │   ├── duplicate_regions.py # ORB copy-move detection
        │   │   ├── layout_validator.py  # Document structure validation
        │   │   ├── ai_artifact_detector.py  # AI-generated image detection
        │   │   └── risk_scorer.py       # Composite 0–100 risk scorer
        │   ├── aadhar/                  # Aadhaar-specific deep verification
        │   │   ├── aadhaar_verifier.py  # Orchestrates all Aadhaar checks
        │   │   ├── verhoeff.py          # Verhoeff checksum algorithm
        │   │   ├── secure_qr.py         # UIDAI Secure QR decode + cross-match
        │   │   ├── font_consistency.py  # Frequency-domain font analysis
        │   │   └── layout_checker.py    # UIDAI layout structure validation
        │   ├── qr/
        │   │   └── qr_decoder.py        # Generic QR/barcode decode
        │   ├── validation/
        │   │   └── field_validator.py   # Business-rule field checks
        │   └── layout/
        │       └── structure_engine.py  # PPStructure layout detection wrapper
        └── utils/
            └── file_utils.py            # Upload validation, MIME sniff, save/cleanup
```

---

## 4. The Pipeline — Every Stage Explained

### Full Verify Pipeline (`run_verify_pipeline`)

```
Stage 1 ── Image Loading
    PDF → pdf2image (poppler) → list of BGR numpy arrays
    Image → cv2.imread → single BGR numpy array

Stage 2 ── Layout Detection
    PaddleOCR PPStructure on primary image
    Identifies regions: header, table, figure, text blocks
    Used as structural metadata (not directly in field extraction)

Stage 3 ── OCR Extraction
    For each page:
      1. preprocess_image(img):
           resize to 1600px width
           grayscale → denoise (fastNlMeansDenoising)
           adaptive Gaussian threshold (binarise)
           deskew (Hough line transform)
           unsharp mask (sharpen)
      2. PaddleOCR (English) → text lines + confidence
      3. PaddleOCR (Hindi) → merged (for Aadhaar Devanagari text)
      4. Tesseract (if available) → merged by confidence
    Output: full_text (joined lines), mean_confidence

Stage 4 ── Document Classification
    classify_document(full_text):
      Weighted regex pattern matching against 6 document types
      Normalises raw scores to confidence percentages
      Flags ambiguous if top confidence < 25%
    Output: ClassificationResult (primary_type, confidence, all_scores)

Stage 5 ── Field Parsing
    parse_fields(full_text, doc_type):
      Dispatches to type-specific parser:
        Aadhaar  → regex for 12-digit UID, DOB, gender, address, pincode
        PAN      → regex for PAN format [A-Z]{5}\d{4}[A-Z], father name
        Passport → MRZ line parsing, passport number, expiry
        Resume   → spaCy NER for PERSON, ORG, GPE entities + email/phone regex
        Cert/Mark → institution, degree, roll number, marks, year
    Output: dict of typed fields

Stage 5.5 ── LLM Credential Extraction
    extract_credentials_with_llm(full_text):
      Sends OCR text to NVIDIA NIM (nvidia/llama-3.3-nemotron-super-49b-v1)
      System prompt demands strict JSON with 19 credential fields
      Returns structured dict with name, DOB, document number, address, etc.
      Fallback returns null for all fields on API failure

Stage 6 ── VLM Understanding
    run_vlm_analysis(image, ocr_text, parsed_fields, doc_type):
      Encodes image as base64 JPEG
      Tries 3 OpenRouter models in chain:
        1. nvidia/nemotron-3-nano-omni-30b-a3b-reasoning
        2. google/gemma-4-26b-a4b-it:free
        3. moonshotai/kimi-vl-a3b-thinking:free
      System prompt asks for JSON with:
        - document_type_confirmed
        - vlm_confidence (0–1)
        - field_validation (per-field OCR correctness)
        - logical_inconsistencies (list)
        - fraud_signals (list with severity)
        - font_consistency, layout_authenticity
        - seal_signature_present, visible_tampering
        - overall_assessment (genuine/suspicious/fake)
        - reasoning (2–3 sentences)
        - suggested_risk_adjustment (-20 to +40)
    Output: VLMAnalysis schema

Stage 7 ── Fraud/Tamper Detection Suite (7 independent checks)
    run_ela(image)              → ELA: pixel-level editing detection
    detect_blur(image)          → Laplacian variance sharpness
    analyse_metadata(file_path) → EXIF software tag / timestamp mismatch
    detect_duplicate_regions(image) → ORB copy-move keypoint matching
    validate_layout(text, doc_type, image) → document structure check
    detect_ai_artifacts(image)  → DCT + noise + edge AI detection
    decode_qr + validate_aadhaar_qr → QR field cross-validation

Stage 8 ── Aadhaar Deep Verification (Aadhaar cards only)
    run_aadhaar_verification(image, fields):
      Check 1: Verhoeff checksum on 12-digit UID
      Check 2: Secure QR decode (pyzbar + OpenCV)
      Check 3: QR vs OCR cross-match (name, DOB, UID)
      Check 4: Font consistency (frequency domain variance analysis)
      Check 5: Layout authenticity (aspect ratio, UIDAI text positions)
    Produces AadhaarVerification with per-check penalties

Stage 9 ── Business Rule Validation
    validate_fields(fields, doc_type):
      Required field presence checks (severity: high)
      Date format validation (DD/MM/YYYY patterns)
      DOB consistency (not in future, not >120 years ago)
      PAN format regex ([A-Z]{5}\d{4}[A-Z])
      Aadhaar Verhoeff checksum (re-run for validation report)
      Expiry date checks (passport not expired)
    Output: ValidationReport (checks, passed, failed, critical_failures)

Stage 10 ── Risk Scoring
    compute_risk_score(fraud, doc_type, ela_score):
      Sums weighted penalties from each fraud check
    + vlm_analysis.suggested_risk_adjustment
    + 15 if vlm.visible_tampering
    + aadhaar_verification.total_aadhaar_penalty (Aadhaar only)
    + 8 per critical validation failure
    Clamp to [0, 100]
    Derive status:
      0–29  → Likely Genuine
      30–59 → Suspicious
      60–100 → Likely Fake / Tampered
```

---

## 5. Every Module — What, Why, How

### `main.py`
**What:** Entry point. Loads `.env` via `python-dotenv`, calls `create_app()`, exposes the ASGI `app` object for uvicorn/gunicorn.  
**Why:** Separates app creation from startup so the factory can be imported in tests without side effects.  
**How:** `uvicorn.run("main:app", host=..., port=..., reload=settings.debug)`

---

### `app/api/__init__.py` — App Factory
**What:** `create_app()` builds and configures the FastAPI instance.  
**Why:** Factory pattern makes the app testable and keeps startup logic away from the entry point.  
**How:**
- Sets CORS to wide-open `*` (tighten in production)
- Registers global `BGVBaseError` exception handler → returns structured JSON error
- Includes two routers: `health_router` and `document_router`

---

### `app/api/routes/documents.py` — API Endpoints
**What:** Three endpoints: `/extract` (lightweight), `/verify` (full), `/analyze` (full + human summary).  
**Why:** Separates fast extraction (no AI cost) from full verification (VLM API calls).  
**How:**
- `_read_and_save()` reads file bytes, calls `validate_upload()`, saves to `app/uploads/`
- Runs pipeline in thread executor (`run_in_executor`) since pipeline is synchronous CPU/IO work
- `finally:` block always calls `cleanup_file()` to delete temp files
- `/analyze` enriches the `/verify` response with a pre-built plain-English summary string

---

### `app/core/config.py` — Settings
**What:** Pydantic `BaseSettings` singleton loaded from `.env`.  
**Why:** Single source of truth for all config — no scattered `os.getenv()` calls.  
**How:** `pydantic-settings` auto-reads from `.env` file. Properties `allowed_ext_list` and `max_file_size_bytes` are computed. Injects `VLM_API_KEY` and `VLM_TIMEOUT` into `os.environ` (needed because `claude_vlm.py` reads env directly).

---

### `app/core/exceptions.py` — Exception Hierarchy
**What:** Custom exception tree rooted at `BGVBaseError`.  
**Why:** Allows callers to catch the whole family with one `except BGVBaseError`, and the global handler maps each to an appropriate HTTP status code.  
**How:** Each subclass sets `http_status` at the class level. Subclasses: `FileValidationError` (422), `FileTooLargeError`, `UnsupportedFileTypeError`, `PreprocessingError` (500), `PDFConversionError`, `OCRExtractionError`, `ForgeryDetectionError`, `QRDecodeError` (422), `RiskScoringError`.

---

### `app/core/logging.py` — Logging Setup
**What:** Configures Loguru to write to both console and `logs/bgv.log`.  
**Why:** Loguru is simpler and more powerful than Python's stdlib `logging`. Rotating file logs are essential for debugging production issues.

---

### `app/services/preprocessing/image_processor.py` — Image Preprocessing
**What:** Converts PDFs to images and applies 5-step image enhancement.  
**Why:** PaddleOCR accuracy is extremely sensitive to image quality. Raw scans often have noise, skew, and poor contrast. Each preprocessing step measurably improves OCR confidence.  
**How (step by step):**
1. `_resize()` → scale to 1600px width with `cv2.INTER_CUBIC` (bicubic interpolation)
2. `_to_grayscale()` → `cv2.COLOR_BGR2GRAY` (OCR doesn't need color)
3. `_denoise()` → `cv2.fastNlMeansDenoising()` with h=10 (removes scanner noise while preserving text edges)
4. `_binarise()` → `cv2.adaptiveThreshold` ADAPTIVE_THRESH_GAUSSIAN_C (handles uneven illumination unlike global threshold)
5. Deskew → `cv2.HoughLinesP` to detect dominant line angle → `cv2.warpAffine` to rotate

For PDFs: `pdf2image.convert_from_path()` at 300 DPI using Poppler, then `cv2.cvtColor(RGB→BGR)`.

---

### `app/services/ocr/engine.py` — OCR Engine
**What:** Multi-engine OCR fusing PaddleOCR and Tesseract.  
**Why:** No single OCR engine is best for all cases. PaddleOCR excels at rotated/low-contrast/Hindi text; Tesseract excels at clean printed English text. Fusion gives higher accuracy and confidence than either alone.  
**How:**
- PaddleOCR loaded lazily with `@lru_cache(maxsize=1)` — avoids re-initializing 200MB models on every request
- Two PaddleOCR instances: `lang="en"` and `lang="hi"` (for Aadhaar Devanagari)
- OCR results are deduplicated: near-identical strings within the same bounding box are collapsed
- `OCRResult` container exposes `.lines` (text list), `.confidence` (mean), `.boxes` (raw boxes with bbox)

---

### `app/services/ocr/cleaner.py` — OCR Text Cleaner
**What:** Post-processes raw OCR text to remove artifacts.  
**Why:** OCR engines produce noise characters, duplicate spaces, garbled symbols. Field parsers and classifiers work much better on clean text.  
**How:** Strips non-printable characters, normalises whitespace, removes common OCR noise patterns (`|||`, `---`, etc.).

---

### `app/services/classification/document_classifier.py` — Document Classifier
**What:** Classifies a document into one of 6 types using weighted regex pattern scoring.  
**Why:** Downstream parsers are type-specific — the classifier determines which parser runs. Weighted scoring (versus simple keyword count) prevents misclassification when two types share keywords.  
**How:**
- `_WEIGHTED_SIGNATURES` dict maps each `DocumentType` to a list of `(regex_pattern, weight)` tuples
- High-weight patterns (3.0): unique strings like `"aadhaar"`, `"permanent account number"`, PAN regex `[A-Z]{5}\d{4}[A-Z]`
- Low-weight patterns (0.5–1.0): common strings like `"government of india"`, `"dob :"` 
- Raw scores normalised to confidence percentages across all 6 types
- Result flagged as `is_ambiguous` if top confidence < 25%

---

### `app/services/parsers/field_parser.py` — Field Parser
**What:** Type-specific regex + spaCy NER extraction of structured fields from OCR text.  
**Why:** Each document type has unique field formats and regex patterns. A single generic parser would have poor precision.  
**How:** Dispatches to 5 specialized parsers:
- **Aadhaar:** regex for `\d{4}\s\d{4}\s\d{4}` (UID), DOB `\d{2}/\d{2}/\d{4}`, gender keywords, address block after `Address:`, pincode `\d{6}`
- **PAN:** regex for PAN `[A-Z]{5}\d{4}[A-Z]`, father name after `Father's Name:`, DOB, name
- **Passport:** MRZ line 1 and 2 (44-char uppercase), passport number `[A-Z]\d{7}`, expiry, nationality
- **Resume:** spaCy `en_core_web_sm` NER for `PERSON` (name), `ORG` (companies), `GPE` (locations) + email regex + phone regex
- **Cert/Marksheet:** institution name, degree title, roll number, marks/percentage, passing year

---

### `app/services/llm/credential_extractor.py` — LLM Credential Extractor
**What:** Sends OCR text to NVIDIA NIM LLM (text-only, no image) for structured credential extraction.  
**Why:** Rule-based regex parsers miss edge cases (unusual formatting, rotated text, line breaks mid-field). An LLM understands context and can extract fields even from messy OCR output.  
**How:**
- Default model: `nvidia/llama-3.3-nemotron-super-49b-v1` via `https://integrate.api.nvidia.com/v1`
- System prompt demands a strict 19-field JSON schema with `null` for missing fields
- Uses `httpx` (async-capable HTTP client) for the API call
- Returns dict with fields: `full_name`, `father_spouse_name`, `date_of_birth`, `gender`, `document_number`, `secondary_number`, `issue_date`, `expiry_date`, `address_line`, `city`, `state`, `pincode`, `nationality`, `issuing_authority`, `degree_title`, `institution_name`, `marks_or_grade`, `passing_year`, `phone_number`, `email_address`, `extraction_notes`

---

### `app/services/vlm/claude_vlm.py` — VLM Analysis
**What:** Sends document image + OCR text to a Vision-Language Model for semantic document understanding and fraud assessment.  
**Why:** Computer-vision heuristics (ELA, blur, etc.) detect pixel-level tampering but miss semantic fraud: a correct-looking Aadhaar with someone else's name, wrong date logic, or a fake institution stamp. VLMs can "read" the document holistically.  
**How:**
- Image encoded as base64 JPEG at quality=90
- OpenRouter API with model fallback chain (3 models tried in order)
- System prompt instructs the model to act as a document verification expert and return a structured JSON assessment
- Handles OpenRouter's `reasoning_content` field (some models return JSON in a different key)
- `_clean_json()` strips markdown fences from model responses before parsing
- `suggested_risk_adjustment` field in the JSON lets the VLM influence the final risk score by ±40 points

---

## 6. Library Encyclopedia

| Library | Version | Category | What | Why | How Used |
|---|---|---|---|---|---|
| **fastapi** | 0.110.2 | Web Framework | High-performance async Python web framework | Auto-generates OpenAPI docs, native Pydantic integration, async support | Hosts REST API, handles routing, request parsing, response serialization |
| **uvicorn[standard]** | 0.29.0 | ASGI Server | Lightning-fast ASGI server | Production-grade server for FastAPI; supports HTTP/1.1 and WebSockets | Entry point for the backend process |
| **python-multipart** | 0.0.9 | Web | Multipart form data parser | FastAPI needs this to handle `UploadFile` (file uploads) | File upload parsing in `/extract`, `/verify`, `/analyze` |
| **paddlepaddle** | 2.6.2 | ML Framework | PaddlePaddle deep learning framework | PaddleOCR requires this as its backend | Underpins all PaddleOCR operations |
| **paddleocr** | 2.7.3 | OCR | State-of-art OCR engine from Baidu | Best-in-class accuracy on Hindi, rotated text, and degraded document scans | Text extraction in `ocr/engine.py` (English + Hindi dual engines) |
| **opencv-python-headless** | 4.9.0.80 | Computer Vision | OpenCV image processing (no GUI) | Industry standard for image manipulation; headless = no X11 dependency | Preprocessing, ELA, blur detection, ORB matching, QR decode, deskew, all forgery checks |
| **Pillow** | 10.3.0 | Image I/O | Python Imaging Library | ELA needs JPEG re-save (PIL handles quality parameter); EXIF reading | ELA in-memory JPEG re-compress, metadata EXIF extraction |
| **pdf2image** | 1.17.0 | PDF | Converts PDF pages to PIL images | PDFs must be rasterized before OCR | `pdf_to_images()` → 300 DPI PNG per page |
| **pyzbar** | 0.1.9 | QR/Barcode | Zbar-based QR and barcode decoder | Reliable QR decode including DataMatrix; better than OpenCV for dense QR | Aadhaar QR decode in `secure_qr.py` and `qr_decoder.py` |
| **spacy** | 3.7.4 | NLP | Industrial NLP library | Named Entity Recognition for resume parsing (names, organizations, locations) | `field_parser.py` loads `en_core_web_sm` for PERSON/ORG/GPE entities |
| **pydantic** | 2.7.1 | Validation | Data validation library | Type-safe request/response schemas; auto-validates all API I/O | All schemas in `app/schemas/document.py`; `BaseSettings` config |
| **pydantic-settings** | 2.2.1 | Config | Pydantic extension for settings | Reads `.env` file automatically into typed Settings class | `app/core/config.py` Settings singleton |
| **loguru** | 0.7.2 | Logging | Structured, beautiful logging | Zero-config logging with file rotation, colors, structured context | Every service module uses `from loguru import logger` |
| **python-dotenv** | 1.0.1 | Config | `.env` file loader | Loads secrets from `.env` into `os.environ` before app starts | `load_dotenv()` at top of `main.py` and `credential_extractor.py` |
| **numpy** | 1.26.4 | Numerics | N-dimensional array library | All image data is numpy arrays; mathematical operations on pixel data | Used everywhere images are manipulated |
| **scipy** | 1.13.0 | Scientific | Scientific computing | Statistical operations for frequency analysis in AI artifact detection | DCT analysis in `ai_artifact_detector.py` |
| **piexif** | 1.1.3 | EXIF | EXIF metadata reader/writer | Pure-Python EXIF parsing; no system dependency | Reads Software tag, DateTime, DateTimeOriginal in `metadata.py` |
| **python-magic-bin** | 0.4.14 | MIME | File type detection via magic bytes | Prevents MIME spoofing — validates actual file content not just extension | `validate_upload()` in `file_utils.py` |
| **aiofiles** | 23.2.1 | Async I/O | Async file operations | Non-blocking file reads for async FastAPI endpoints | File streaming in async route handlers |
| **httpx** | 0.27.0 | HTTP Client | Modern async HTTP client | Used for LLM API calls; supports both sync and async | `credential_extractor.py` calls NVIDIA NIM |
| **ujson** | 5.9.0 | JSON | Ultra-fast JSON serializer | 2–3× faster than stdlib `json`; important for large API responses | JSON parsing of VLM/LLM responses |
| **anthropic** | ≥0.25.0 | AI SDK | Anthropic Claude API client | Official SDK for Claude API integration | Installed as dependency; `claude_vlm.py` currently uses direct HTTP via OpenRouter |
| **gradio** | (frontend) | UI | ML demo UI framework | Build browser UIs with pure Python; no frontend code needed | `frontend.py` — entire UI is ~400 lines of Python |
| **requests** | (frontend) | HTTP | Synchronous HTTP client | Simple file upload to backend from Gradio | `frontend.py` uses it to call the FastAPI backend |

---

## 7. API Reference

### `GET /health`
Returns server status, version, and environment.
```json
{"status": "ok", "version": "2.0.0", "env": "development"}
```

### `POST /extract`
**Purpose:** Lightweight OCR + classification + field extraction. No fraud detection. Fast.  
**Input:** `multipart/form-data` with `file` field (PDF, JPEG, PNG, TIFF, max 20MB)  
**Returns:** `ExtractResponse`
```json
{
  "document_type": "aadhaar",
  "extracted_fields": {"name": "Rahul Sharma", "aadhaar_number": "1234 5678 9012", ...},
  "llm_credentials": {"full_name": "Rahul Sharma", "date_of_birth": "15/08/1990", ...},
  "confidence": 0.9341,
  "page_count": 1,
  "processing_time_ms": 1240.5
}
```

### `POST /verify`
**Purpose:** Full 10-stage enterprise verification pipeline.  
**Input:** Same as `/extract`  
**Returns:** `EnhancedVerifyResponse`
```json
{
  "document_type": "aadhaar",
  "extracted_fields": {...},
  "llm_credentials": {...},
  "fraud_analysis": {
    "ela": "clean",
    "blur": "sharp",
    "metadata": "normal",
    "duplicate_regions": "not_detected",
    "layout_validation": "valid",
    "qr_validation": "verified",
    "ai_artifacts": "not_detected",
    "ela_score": 4.2,
    "blur_score": 512.3
  },
  "risk_score": 12,
  "status": "Likely Genuine",
  "confidence": 0.9341,
  "page_count": 1,
  "processing_time_ms": 8340.2,
  "classification": {"primary_type": "aadhaar", "primary_confidence": 0.87, ...},
  "vlm_analysis": {
    "document_type_confirmed": "aadhaar",
    "vlm_confidence": 0.92,
    "font_consistency": "consistent",
    "layout_authenticity": "authentic",
    "seal_signature_present": false,
    "visible_tampering": false,
    "overall_assessment": "genuine",
    "reasoning": "The document shows consistent UIDAI formatting...",
    "suggested_risk_adjustment": -5,
    "fraud_signals": []
  },
  "validation": {
    "checks": [...],
    "passed": 8, "failed": 0, "critical_failures": 0,
    "validation_score": 1.0,
    "summary": "8/8 checks passed"
  },
  "aadhaar_verification": {
    "verhoeff_valid": true,
    "qr_cross_validation_status": "verified",
    "font_status": "consistent",
    "layout_status": "authentic",
    "total_aadhaar_penalty": 0,
    "aadhaar_risk_level": "low",
    "summary": "4/4 Aadhaar checks passed"
  },
  "pipeline_stages": ["load_images", "layout_detection", "ocr_extraction", ...]
}
```

### `POST /analyze`
Same as `/verify` but adds a `human_summary` string field — a pre-formatted plain-English report for frontend display.

---

## 8. Configuration & Environment Variables

All settings live in `.env` in the `bgv_system_enhanced/` directory:

```env
# ── Application ──────────────────────────────────────────────────────────────
APP_ENV=development
DEBUG=true
LOG_LEVEL=DEBUG

# ── Server ───────────────────────────────────────────────────────────────────
HOST=0.0.0.0
PORT=8000

# ── File Upload ──────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB=20
ALLOWED_EXTENSIONS=pdf,jpg,jpeg,png,tiff

# ── PDF Rendering (Poppler) ──────────────────────────────────────────────────
POPPLER_PATH=C:/poppler/Library/bin       # Windows; blank on Linux if in PATH

# ── OCR ──────────────────────────────────────────────────────────────────────
OCR_LANG=en
OCR_USE_GPU=false

# ── Risk Score Thresholds ────────────────────────────────────────────────────
RISK_LOW_THRESHOLD=30                     # below this = Genuine
RISK_MEDIUM_THRESHOLD=60                  # below this = Suspicious; above = Fake

# ── VLM (Vision Language Model via OpenRouter) ───────────────────────────────
VLM_API_KEY=sk-or-v1-...                  # OpenRouter API key
VLM_TIMEOUT=60

# ── LLM (Text only, Credential Extraction via NVIDIA NIM) ────────────────────
LLM_API_KEY=                              # NVIDIA NIM key (falls back to VLM_API_KEY)
LLM_BASE_URL=https://integrate.api.nvidia.com/v1
LLM_MODEL=nvidia/llama-3.3-nemotron-super-49b-v1
LLM_TIMEOUT=60

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_FILE=logs/bgv.log
```

**Important note:** `config.py` injects `VLM_API_KEY` and `VLM_TIMEOUT` into `os.environ` after loading, because `claude_vlm.py` reads those directly from the environment.

---

## 9. Schemas & Data Models

All schemas are Pydantic v2 models in `app/schemas/document.py`.

### Enumerations
| Enum | Values |
|---|---|
| `DocumentType` | `aadhaar`, `pan`, `passport`, `resume`, `graduation_certificate`, `marksheet`, `unknown` |
| `RiskStatus` | `Likely Genuine`, `Suspicious`, `Likely Fake / Tampered` |
| `ELAResult` | `clean`, `suspicious`, `tampered` |
| `MetadataResult` | `normal`, `suspicious`, `missing` |
| `BlurResult` | `sharp`, `slightly_blurry`, `blurry` |
| `DuplicateRegionResult` | `not_detected`, `detected` |
| `LayoutResult` | `valid`, `invalid`, `unknown` |
| `QRResult` | `verified`, `mismatch`, `not_found`, `decode_error` |
| `AIArtifactResult` | `not_detected`, `suspected` |

### Key Models
- **`FraudAnalysis`** — aggregates all 7 forgery checks + raw numeric scores
- **`VLMAnalysis`** — full VLM assessment including fraud signals and reasoning
- **`VLMFraudSignal`** — individual signal with severity (`low`/`medium`/`high`)
- **`ValidationReport`** — list of `FieldCheckResult` items with pass/fail/severity
- **`AadhaarVerification`** — per-check results for Verhoeff, QR, font, layout + composite penalty
- **`ClassificationResult`** — confidence breakdown across all 6 document types
- **`ExtractResponse`** — response for `/extract`
- **`EnhancedVerifyResponse`** — full response for `/verify` and `/analyze`

---

## 10. Fraud Detection Deep Dive

### ELA — Error Level Analysis (`forgery/ela.py`)

**What it detects:** Regions that were saved at a different JPEG quality than the rest of the image — a hallmark of copy-paste editing.

**Algorithm:**
1. Re-compress the image at quality=90 into a memory buffer
2. Compute absolute pixel difference: `diff = |original - recompressed| × 10`
3. Mean of diff (0–255 scale) → rescaled to 0–100 score
4. Score ≥ 30 → suspicious; ≥ 60 → tampered

**Penalty in risk score:** 0 (clean), 15 (suspicious), 35 (tampered)

**Limitations:** Only works on JPEG images. PDFs rendered at 300 DPI may have artificially high ELA scores. Less effective on screenshots.

---

### Blur Detection (`forgery/blur.py`)

**What it detects:** Abnormally blurry images that may be hiding editing artifacts, or very sharp images that suggest AI upscaling.

**Algorithm:** Laplacian variance — applies `cv2.Laplacian` (second-order derivative filter) and computes `np.var`. Higher variance = more edges = sharper image.

**Thresholds:**
- Variance < 80 → blurry
- Variance 80–200 → slightly blurry  
- Variance > 200 → sharp

**Penalty:** 0 (sharp), 5 (slightly blurry), 10 (blurry)

---

### EXIF Metadata Analysis (`forgery/metadata.py`)

**What it detects:** Signs that an image was processed in editing software.

**Checks:**
1. **Software tag** — if EXIF Software field contains: Photoshop, GIMP, Lightroom, Affinity, Paint.net, Canva, Snapseed, Pixlr, Corel, Adobe → `suspicious`
2. **DateTime mismatch** — if `DateTime` ≠ `DateTimeOriginal` → `suspicious` (image was modified after capture)
3. **No EXIF** — `missing` (common for forged scans that strip metadata)
4. **Corrupt EXIF** — `suspicious`
5. **PDFs** — marked as `not_applicable` (no EXIF in PDFs)

**Penalty:** 0 (normal), 8 (missing), 12 (suspicious)

---

### Copy-Move Detection (`forgery/duplicate_regions.py`)

**What it detects:** Regions that were copied and pasted within the same image (e.g., duplicating a stamp or signature).

**Algorithm:** ORB (Oriented FAST and Rotated BRIEF) self-matching:
1. Detect up to 1000 ORB keypoints + binary descriptors
2. BruteForce Hamming distance match against all other descriptors
3. For every "good" match (Hamming distance < 40), check if the two keypoints are spatially distant (> 50px)
4. Spatially distant pairs with similar descriptors = copy-move evidence
5. ≥ 10 such pairs → `detected`

**Penalty:** 0 (not_detected), 20 (detected)

---

### Layout Validation (`forgery/layout_validator.py`)

**What it detects:** Documents whose text content doesn't match the expected structural signals for the identified document type.

**How:** Re-runs the weighted classifier signal-matching but looks for specific required patterns. For example, a document classified as Aadhaar should have the UIDAI text, the 12-digit number pattern, and DOB field.

**Penalty:** 0 (valid), 5 (unknown), 15 (invalid)

---

### AI Artifact Detection (`forgery/ai_artifact_detector.py`)

**What it detects:** Images generated or heavily modified by AI tools (Stable Diffusion, DALL-E, Midjourney).

**Three independent checks (2 of 3 must fire to flag `suspected`):**

1. **DCT Frequency Analysis** — divides image into 8×8 blocks, applies 2D DCT, measures ratio of mid-frequency energy to total. AI images show unnaturally high mid-frequency energy (ringing/over-sharpening). Threshold: > 35% mid-frequency ratio.

2. **Noise Uniformity** — computes local noise in 16×16 patches. Real scans have spatially correlated noise; AI images have unnaturally flat or periodic noise. Threshold: std-of-local-noise / global-noise > 0.92.

3. **Edge Density Abnormality** — Canny edge detection, measure fraction of edge pixels. Below 2% = suspiciously smooth (AI softness); above 40% = suspiciously over-sharpened.

**Penalty:** 0 (not_detected), 12 (suspected)

---

## 11. Aadhaar-Specific Verification Suite

Aadhaar cards get an additional 4-check deep verification pass in `app/services/aadhar/`.

### Check 1: Verhoeff Checksum (`verhoeff.py`)

**What:** UIDAI uses the Verhoeff algorithm (not Luhn) for all 12-digit Aadhaar numbers. The last digit is a check digit computed using three lookup tables.

**Why it's powerful:** A number that fails this check is **definitively invalid** — it cannot be a real UIDAI-issued Aadhaar number, no matter how authentic the document looks.

**Tables used:**
- `_D5`: Multiplication table (dihedral group D5)
- `_P8`: Permutation table (8-cycle permutation)
- `_INV`: Inverse table

**Algorithm:** Iterate digits right-to-left. `c = D5[c][P8[i % 8][digit]]`. Result must be 0 for valid number.

**Additional check:** First digit cannot be 0 or 1 (UIDAI rule).

**Penalty:** 0 (valid), 25–30 (failed)

---

### Check 2 & 3: Secure QR Decode + Cross-Match (`secure_qr.py`)

**What:** Modern Aadhaar cards (post-2018) contain a digitally-signed QR code with compressed binary payload created by UIDAI using RSA-SHA256.

**QR Types:**
- `secure_v2` — post-2018, compressed binary, strongest signal
- `xml_signed` — XML with digital signature
- `xml_plain` — older plain XML
- `text_plain` — very old text format

**Decode strategy (multi-strategy):**
1. Try pyzbar (ZBar library) — best for dense QR
2. Try OpenCV `QRCodeDetector`
3. Try enhanced contrast version if both fail

**Cross-validation:** Compares QR payload fields vs OCR-extracted fields:
- Name match (fuzzy, handles OCR errors)
- DOB match (after normalizing date formats)
- UID last 4 digits match
- Address match (partial)

**Outcome:** `verified` (all match), `partial_match`, `mismatch` (most suspicious), `no_qr`, `decode_error`

**Penalty:** −5 (verified bonus), 0 (not_found), 5 (decode_error), 25 (mismatch — very high penalty for data mismatch)

---

### Check 4: Font Consistency (`font_consistency.py`)

**What:** Detects text regions that were digitally edited (different font rendering, different JPEG compression artifacts around the text).

**Why:** Fake Aadhaar cards are often made by editing name/DOB/photo on a real card scan. The editing changes the local texture and compression signature.

**Algorithm:**
1. Divide image into 32×32 pixel blocks
2. For each block, compute Laplacian variance (texture energy)
3. Global mean and std of all block variances
4. Blocks with variance > mean + 2.5×std are "suspicious" (too sharp = pasted content)
5. DCT high-frequency analysis per block to detect compression inconsistency
6. Variance ratio = suspicious blocks / total text blocks

**Thresholds:** > 25% → suspicious; > 45% → tampered

**Penalty:** scales with variance ratio and number of suspicious blocks

---

### Check 5: Layout Authenticity (`layout_checker.py`)

**What:** Validates that the Aadhaar card follows UIDAI's standardized layout.

**Checks:**
- Aspect ratio: 85.6mm × 54mm = 1.585 (standard credit card size), ±15% tolerance
- Required text signals: "UIDAI", "Unique Identification Authority", "Government of India", "आधार" (Devanagari)
- Aadhaar number format: `XXXX XXXX XXXX` with spaces
- VID field present (post-2018 cards)
- Expected field labels present: DOB, Gender/Linga, Address/Pata

**Penalty:** escalates with number of layout issues

---

## 12. Risk Scoring Engine

The final risk score (0–100) is assembled in `pipeline.py` from multiple sources:

```
Base score (from compute_risk_score):
  ELA penalty:          0 / 15 / 35
  Metadata penalty:     0 / 8 / 12
  Blur penalty:         0 / 5 / 10
  Duplicate penalty:    0 / 20
  Layout penalty:       0 / 5 / 15
  AI artifact penalty:  0 / 12
  QR penalty (Aadhaar): -5 / 0 / 5 / 25
  ELA raw score bonus:  ela_score × 0.1

+ VLM adjustment:
  vlm_analysis.suggested_risk_adjustment   (-20 to +40)
  +15 if vlm_analysis.visible_tampering

+ Aadhaar deep verification penalty (Aadhaar only):
  aadhaar_verification.total_aadhaar_penalty

+ Validation critical failures:
  +8 per critical_failure

= Final score (clamped 0–100)

Status derivation:
  0–29  → Likely Genuine
  30–59 → Suspicious  
  60–100 → Likely Fake / Tampered
```

**Design principle:** The scoring is additive — each check adds evidence of tampering. A genuinely clean document should score near 0 (no red flags). Only severe multiple signals push a document to 60+.

---

## 13. Frontend (Gradio UI)

`frontend.py` is a standalone Gradio application that provides a browser interface for the BGV system.

**How to run:** `python frontend.py` (requires the backend to be running at `http://localhost:8000`)

**Two tabs:**

**Tab 1 — Quick Extract:**
- Upload a document
- Shows: document type, OCR confidence, page count, processing time
- Displays: extracted fields (formatted markdown), raw JSON

**Tab 2 — Full Verification:**
- Upload a document  
- Shows: risk score with color-coded badge (green/yellow/red), verification status, processing time
- Panels: Extracted Fields, Fraud Detection (7 checks with traffic light icons), VLM Analysis, Field Validation, Raw JSON

**Color coding for risk:**
- Green (#16a34a): score < 30 — Likely Genuine
- Amber (#d97706): score 30–59 — Suspicious
- Red (#dc2626): score ≥ 60 — Likely Fake / Tampered

**Status icons used in fraud panel:**
- 🟢 clean / verified / valid / sharp / normal
- 🟡 suspicious / slightly_blurry / missing / unknown
- 🔴 tampered / detected / blurry / mismatch / invalid / suspected

---

## 14. How to Run

### Prerequisites
- Python 3.11
- Poppler installed (for PDF support)
  - Windows: download from https://github.com/oschwartz10612/poppler-windows/releases
  - Linux: `sudo apt-get install poppler-utils`
  - macOS: `brew install poppler`
- spaCy model: `python -m spacy download en_core_web_sm`

### Installation
```bash
cd bgv_system_enhanced
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Configuration
Copy `.env` and fill in your API keys:
```env
VLM_API_KEY=sk-or-v1-...      # OpenRouter key for VLM
LLM_API_KEY=nvapi-...         # NVIDIA NIM key for LLM (optional)
POPPLER_PATH=C:/poppler/Library/bin  # Windows only
```

### Start Backend
```bash
cd bgv_system_enhanced
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# Or simply:
python main.py
```

### Start Frontend
```bash
# From the final/ directory
python frontend.py
```

### API Documentation
Interactive Swagger UI: `http://localhost:8000/docs`  
ReDoc: `http://localhost:8000/redoc`

### Quick Test
```bash
curl -X POST http://localhost:8000/extract \
  -F "file=@your_aadhaar.jpg"
```

---

## 15. Error Handling Hierarchy

```
BGVBaseError (HTTP 500)
├── FileValidationError (HTTP 422)
│   ├── FileTooLargeError       — file > MAX_FILE_SIZE_MB
│   └── UnsupportedFileTypeError — MIME type not in allow-list
├── PreprocessingError (HTTP 500) — cv2 resize/deskew/binarize failure
├── PDFConversionError (HTTP 500) — pdf2image/poppler failure
├── OCRExtractionError (HTTP 500) — PaddleOCR complete failure
├── ForgeryDetectionError (HTTP 500) — forgery module unrecoverable error
├── QRDecodeError (HTTP 422)    — pyzbar cannot decode QR
└── RiskScoringError (HTTP 500) — unexpected scoring state
```

**Design:** All BGVBaseErrors are caught by the FastAPI global exception handler and returned as structured JSON:
```json
{"error": "FileTooLargeError", "message": "File 'doc.pdf' is 25.3 MB — limit is 20 MB."}
```

Individual service modules (ELA, blur, metadata, etc.) have internal `try/except` that return safe fallback values rather than propagating exceptions — this ensures partial failures don't kill the entire pipeline response.

---