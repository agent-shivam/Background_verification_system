# BGV System Enhanced

AI/ML-powered Background Verification System for automated document authenticity checks.

---

## What It Does

Accepts identity and educational documents (Aadhaar, PAN, Passport, Graduation Certificate, Marksheet, Resume) as PDF or image uploads, then runs them through a 10-stage pipeline that:

- Extracts structured fields (name, DOB, document number, address, institution, marks, etc.)
- Detects forgery using computer vision, VLM reasoning, and rule-based validation
- Scores each document 0–100 and classifies it as **Likely Genuine**, **Suspicious**, or **Likely Fake / Tampered**

---

## Project Setup

### Prerequisites

- Python 3.11
- Poppler (for PDF support)
  - **Windows:** Download from https://github.com/oschwartz10612/poppler-windows/releases
  - **Linux:** `sudo apt-get install poppler-utils`
  - **macOS:** `brew install poppler`

### Installation

```bash
# 1. Navigate to project directory
cd bgv_system_enhanced

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Download spaCy language model
python -m spacy download en_core_web_sm
```

### Configuration

Create a `.env` file inside `bgv_system_enhanced/` with the following:

```env
# Application
APP_ENV=development
DEBUG=true
LOG_LEVEL=DEBUG

# Server
HOST=0.0.0.0
PORT=8000

# File Upload
MAX_FILE_SIZE_MB=20
ALLOWED_EXTENSIONS=pdf,jpg,jpeg,png,tiff

# PDF Rendering (Windows only — leave blank on Linux/macOS if Poppler is in PATH)
POPPLER_PATH=C:/poppler/Library/bin

# VLM — Vision Language Model via OpenRouter
VLM_API_KEY=sk-or-v1-...
VLM_TIMEOUT=60

# LLM — Text-only credential extraction via NVIDIA NIM
LLM_API_KEY=nvapi-...
LLM_BASE_URL=https://integrate.api.nvidia.com/v1
LLM_MODEL=nvidia/llama-3.3-nemotron-super-49b-v1
LLM_TIMEOUT=60

# Risk Score Thresholds
RISK_LOW_THRESHOLD=30
RISK_MEDIUM_THRESHOLD=60
```

### Running the Backend

```bash
cd bgv_system_enhanced
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Or simply:
python main.py
```

### Running the Frontend (Gradio UI)

```bash
# From the final/ directory (backend must be running first)
python frontend.py
```

---

## Required Libraries

| Library | Version | Purpose |
|---|---|---|
| fastapi | 0.110.2 | REST API framework |
| uvicorn[standard] | 0.29.0 | ASGI server |
| paddlepaddle | 2.6.2 | PaddleOCR backend |
| paddleocr | 2.7.3 | OCR engine (English + Hindi) |
| opencv-python-headless | 4.9.0.80 | Image processing and forgery detection |
| Pillow | 10.3.0 | Image I/O and ELA |
| pdf2image | 1.17.0 | PDF to image conversion |
| pyzbar | 0.1.9 | QR/barcode decoding |
| spacy | 3.7.4 | NLP / Named Entity Recognition |
| pydantic | 2.7.1 | Schema validation |
| pydantic-settings | 2.2.1 | `.env` config loading |
| loguru | 0.7.2 | Logging |
| numpy | 1.26.4 | Numerical operations |
| scipy | 1.13.0 | DCT frequency analysis |
| httpx | 0.27.0 | HTTP client for LLM/VLM API calls |
| piexif | 1.1.3 | EXIF metadata reading |
| python-magic-bin | 0.4.14 | MIME type detection |
| aiofiles | 23.2.1 | Async file I/O |
| ujson | 5.9.0 | Fast JSON parsing |
| python-multipart | 0.0.9 | File upload parsing |
| python-dotenv | 1.0.1 | `.env` file loader |
| gradio | latest | Browser UI (frontend only) |
| requests | latest | HTTP calls from Gradio to backend |

Install all at once:

```bash
pip install -r requirements.txt
```

---

## API Documentation

Interactive docs available at runtime:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### Endpoints

#### `GET /health`
Returns server status.

#### `POST /extract`
Lightweight extraction — OCR + classification + field parsing. No fraud detection. Fast.

**Input:** `multipart/form-data` with a `file` field (PDF, JPEG, PNG, TIFF — max 20 MB)

#### `POST /verify`
Full 10-stage pipeline including fraud detection, VLM analysis, Aadhaar deep verification, and risk scoring.

**Input:** Same as `/extract`

#### `POST /analyze`
Same as `/verify` but also returns a plain-English `human_summary` field for display.

---

## Sample Inputs & Outputs

### Sample Input

Upload an Aadhaar card image via curl:

```bash
curl -X POST http://localhost:8000/extract \
  -F "file=@aadhaar_sample.jpg"
```

Or via Swagger UI at http://localhost:8000/docs — click **POST /verify → Try it out**, attach a file, and hit Execute.

---

### Sample Output — `/extract`

```json
{
  "document_type": "aadhaar",
  "extracted_fields": {
    "name": "Rahul Sharma",
    "aadhaar_number": "1234 5678 9012",
    "dob": "15/08/1990",
    "gender": "Male",
    "address": "123 Main Street, New Delhi",
    "pincode": "110001"
  },
  "llm_credentials": {
    "full_name": "Rahul Sharma",
    "date_of_birth": "15/08/1990",
    "document_number": "1234 5678 9012"
  },
  "confidence": 0.93,
  "page_count": 1,
  "processing_time_ms": 1240.5
}
```

---

### Sample Output — `/verify`

```json
{
  "document_type": "aadhaar",
  "risk_score": 12,
  "status": "Likely Genuine",
  "confidence": 0.93,
  "extracted_fields": {
    "name": "Rahul Sharma",
    "aadhaar_number": "1234 5678 9012",
    "dob": "15/08/1990"
  },
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
  "vlm_analysis": {
    "overall_assessment": "genuine",
    "vlm_confidence": 0.92,
    "visible_tampering": false,
    "font_consistency": "consistent",
    "layout_authenticity": "authentic",
    "reasoning": "The document shows consistent UIDAI formatting with no signs of editing.",
    "suggested_risk_adjustment": -5,
    "fraud_signals": []
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
  "validation": {
    "passed": 8,
    "failed": 0,
    "critical_failures": 0,
    "validation_score": 1.0,
    "summary": "8/8 checks passed"
  },
  "page_count": 1,
  "processing_time_ms": 8340.2
}
```

---

### Risk Score Reference

| Score Range | Status |
|---|---|
| 0 – 29 | ✅ Likely Genuine |
| 30 – 59 | ⚠️ Suspicious |
| 60 – 100 | ❌ Likely Fake / Tampered |

---

## Supported Document Types

- Aadhaar Card
- PAN Card
- Passport
- Resume
- Graduation Certificate
- Marksheet
