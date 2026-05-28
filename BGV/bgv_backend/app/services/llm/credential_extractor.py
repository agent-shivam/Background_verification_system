"""
app/services/llm/credential_extractor.py
─────────────────────────────────────────
LLM-based credential extraction layer.

After OCR produces raw text, this module sends that text to an LLM
(text-only, no image) and asks it to extract ONLY credential fields:
  • Personal identifiers  (name, DOB, gender, father/spouse name)
  • Document numbers      (Aadhaar UID, PAN, passport number, roll no.)
  • Addresses             (full address, city, state, PIN)
  • Dates                 (issue date, expiry date, DOB)
  • Issuing authority     (bank name, institution, board, ministry)
  • Educational details   (degree, marks, percentage, grade)
  • Contact info          (phone, email)

Configure via .env (falls back to VLM_API_KEY if LLM_API_KEY is absent):
    LLM_API_KEY    = <your key>     (optional — uses VLM_API_KEY if blank)
    LLM_BASE_URL   = <base URL>     (default: NVIDIA NIM)
    LLM_MODEL      = <model string> (default: nvidia/llama-3.3-nemotron-super-49b-v1)
    LLM_TIMEOUT    = 60
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
_DEFAULT_MODEL    = "nvidia/llama-3.3-nemotron-super-49b-v1"

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a precise credential-extraction engine for Indian identity and educational documents.

Given raw OCR text from a scanned document, extract ONLY the credential fields below.
Return a single valid JSON object — no prose, no markdown fences, no extra keys.

JSON schema (include every key; use null if a field is not found):
{
  "full_name":          "<string | null>",
  "father_spouse_name": "<string | null>",
  "date_of_birth":      "<DD/MM/YYYY | null>",
  "gender":             "<Male|Female|Other | null>",
  "document_number":    "<primary ID number, e.g. Aadhaar UID / PAN / Passport No / Roll No | null>",
  "secondary_number":   "<secondary ID if present, e.g. VID / enrollment no | null>",
  "issue_date":         "<DD/MM/YYYY | null>",
  "expiry_date":        "<DD/MM/YYYY | null>",
  "address_line":       "<street / locality | null>",
  "city":               "<string | null>",
  "state":              "<string | null>",
  "pincode":            "<6-digit string | null>",
  "nationality":        "<string | null>",
  "issuing_authority":  "<ministry / bank / university / board | null>",
  "degree_title":       "<e.g. B.Tech Computer Science | null>",
  "institution_name":   "<school / college / university name | null>",
  "marks_or_grade":     "<e.g. 85.4% / 8.7 CGPA / Grade A | null>",
  "passing_year":       "<YYYY | null>",
  "phone_number":       "<10-digit string | null>",
  "email_address":      "<string | null>",
  "extraction_notes":   "<brief note on any ambiguity or low-confidence fields | null>"
}
"""

# ── HTTP call ─────────────────────────────────────────────────────────────────

def _call_llm(
    api_key: str,
    base_url: str,
    model: str,
    ocr_text: str,
    timeout: float,
) -> str:
    """POST to an OpenAI-compatible /chat/completions endpoint (text only)."""

    user_content = (
        f"Extract credentials from the following OCR text. "
        f"Return only JSON per the schema.\n\n"
        f"--- OCR TEXT START ---\n{ocr_text[:4000]}\n--- OCR TEXT END ---"
    )

    payload = {
        "model": model,
        "max_tokens": 1024,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }

    url  = f"{base_url.rstrip('/')}/chat/completions"
    resp = httpx.post(url, headers=headers, json=payload, timeout=timeout)

    if resp.status_code != 200:
        logger.warning(
            f"LLM credential extractor HTTP {resp.status_code}: {resp.text[:300]}"
        )
        resp.raise_for_status()

    data    = resp.json()
    content = data["choices"][0]["message"]["content"]
    return content.strip() if isinstance(content, str) else ""


# ── JSON cleanup ──────────────────────────────────────────────────────────────

def _clean_json(raw: str) -> dict[str, Any]:
    """Strip markdown fences and parse JSON safely."""
    text = raw.strip()
    # Remove ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        parts = text.split("```")
        text  = parts[1] if len(parts) > 1 else text
        if text.lower().startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


# ── Config loader ─────────────────────────────────────────────────────────────

def _load_config() -> tuple[str, str, str, float]:
    """
    Returns (api_key, base_url, model, timeout).
    LLM_API_KEY  → falls back to VLM_API_KEY if not set.
    """
    api_key  = (
        os.environ.get("LLM_API_KEY",  "").strip()
        or os.environ.get("VLM_API_KEY", "").strip()
    )
    base_url = os.environ.get("LLM_BASE_URL", _DEFAULT_BASE_URL).strip()
    model    = os.environ.get("LLM_MODEL",    _DEFAULT_MODEL).strip()
    timeout  = float(os.environ.get("LLM_TIMEOUT", "60"))
    return api_key, base_url, model, timeout


# ── Fallback ──────────────────────────────────────────────────────────────────

def _fallback(reason: str) -> dict[str, Any]:
    return {
        "full_name":          None,
        "father_spouse_name": None,
        "date_of_birth":      None,
        "gender":             None,
        "document_number":    None,
        "secondary_number":   None,
        "issue_date":         None,
        "expiry_date":        None,
        "address_line":       None,
        "city":               None,
        "state":              None,
        "pincode":            None,
        "nationality":        None,
        "issuing_authority":  None,
        "degree_title":       None,
        "institution_name":   None,
        "marks_or_grade":     None,
        "passing_year":       None,
        "phone_number":       None,
        "email_address":      None,
        "extraction_notes":   f"LLM extraction skipped: {reason}",
        "llm_available":      False,
        "llm_model":          "none",
    }


# ── Public API ────────────────────────────────────────────────────────────────

def extract_credentials_with_llm(ocr_text: str) -> dict[str, Any]:
    """
    Send clean OCR text to the configured LLM and return a structured
    dict of credential fields only.

    Falls back gracefully on any error — the pipeline never crashes.

    Requires in .env:
        LLM_API_KEY  = <your key>   (or VLM_API_KEY as fallback)
    """
    if not ocr_text or not ocr_text.strip():
        return _fallback("Empty OCR text — nothing to extract")

    api_key, base_url, model, timeout = _load_config()

    if not api_key:
        logger.warning("LLM_API_KEY / VLM_API_KEY not set — skipping LLM credential extraction")
        return _fallback("API key not configured")

    logger.info(f"LLM credential extraction → model={model}, url={base_url}")

    try:
        raw_text = _call_llm(api_key, base_url, model, ocr_text, timeout)
        result   = _clean_json(raw_text)

        logger.info(
            f"LLM credential extraction done — "
            f"name={result.get('full_name')}, "
            f"doc_no={result.get('document_number')}"
        )

        result["llm_available"] = True
        result["llm_model"]     = model
        return result

    except json.JSONDecodeError as exc:
        logger.warning(f"LLM credential extractor returned invalid JSON: {exc}")
        return _fallback(f"JSON parse error: {exc}")

    except Exception as exc:
        logger.warning(f"LLM credential extraction failed: {exc}")
        return _fallback(str(exc))
