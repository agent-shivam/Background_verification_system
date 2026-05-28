"""
app/services/vlm/claude_vlm.py
──────────────────────────────
OpenRouter VLM layer — free vision models with auto-fallback.

Model chain (tried in order, first live endpoint wins):
  1. deepseek/deepseek-vl2:free          — free, vision, 256K ctx
  2. google/gemma-4-26b-a4b-it:free      — free, vision, MoE, 262K ctx
  3. moonshotai/kimi-vl-a3b-thinking:free — free, vision fallback

Configure via .env:
    VLM_API_KEY  = sk-or-v1-...
    VLM_TIMEOUT  = 60   (optional)
"""

from __future__ import annotations
import re
import base64
import json
import os
from typing import Any

import cv2
import httpx
import numpy as np
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# ── Model chain ───────────────────────────────────────────────────────────────

VLM_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Tried in order — first model with live endpoints wins.
# All confirmed free + vision-capable on OpenRouter as of May 2026.
VLM_MODELS = [
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    # "google/gemma-4-31b-it:free",           # Google Gemma 4 31B — free, multimodal      
    # "moonshotai/kimi-vl-a3b-thinking:free", # Kimi VL — free vision fallback
]

_active_model: str = VLM_MODELS[0]


# ── Image encoding ────────────────────────────────────────────────────────────

def _numpy_to_base64(image: np.ndarray, quality: int = 90) -> str:
    """Encode OpenCV BGR/Gray image → base64 JPEG string."""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are an expert document verification AI specialising in Indian identity "
    "and educational documents (Aadhaar, PAN, Passport, Marksheets, Graduation "
    "Certificates, Resumes).\n\n"
    "Analyse the provided document image together with the OCR-extracted text below. "
    "Return ONLY a valid JSON object — no prose, no markdown code fences.\n\n"
    "Required JSON structure (every key mandatory):\n"
    "{\n"
    '  "document_type_confirmed": "<aadhaar|pan|passport|resume|graduation_certificate|marksheet|unknown>",\n'
    '  "vlm_confidence": <0.0-1.0>,\n'
    '  "field_validation": {\n'
    '    "<field_name>": {"ocr_value": "<val>", "looks_correct": true/false, "note": "<optional>"}\n'
    "  },\n"
    '  "logical_inconsistencies": ["<issue>", ...],\n'
    '  "fraud_signals": [\n'
    '    {"signal": "<description>", "severity": "low|medium|high"}\n'
    "  ],\n"
    '  "font_consistency": "consistent|inconsistent|unknown",\n'
    '  "layout_authenticity": "authentic|suspicious|fake|unknown",\n'
    '  "seal_signature_present": true/false,\n'
    '  "visible_tampering": true/false,\n'
    '  "overall_assessment": "genuine|suspicious|fake",\n'
    '  "reasoning": "<2-3 sentence explanation>",\n'
    '  "suggested_risk_adjustment": <integer -20 to 40>\n'
    "}"
)


def _user_message_text(ocr_text: str, parsed_fields: dict, doc_type: str) -> str:
    return (
        f"Detected document type (OCR heuristics): {doc_type}\n\n"
        f"OCR text (first 3000 chars):\n{ocr_text[:3000]}\n\n"
        f"Parsed fields:\n{json.dumps(parsed_fields, indent=2, default=str)[:1500]}\n\n"
        "Analyse the image and text above. Validate all fields, check for font "
        "inconsistencies, layout authenticity, logical errors, and tampering. "
        "Respond with the JSON only."
    )


# ── HTTP call ─────────────────────────────────────────────────────────────────

def _call_vlm(
    api_key: str,
    model: str,
    image_b64: str,
    user_text: str,
    timeout: float,
) -> dict[str, Any]:
    """POST to OpenRouter /chat/completions for a single model."""

    # Fold system prompt into user turn for broadest model compatibility
    combined_text = f"{_SYSTEM_PROMPT}\n\n---\n\n{user_text}"

    payload = {
        "model": model,
        "max_tokens": 1024,
        "temperature": 0.1,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                        },
                    },
                    {"type": "text", "text": combined_text},
                ],
            }
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.environ.get("APP_URL", "http://localhost:8000"),
    }

    url = f"{VLM_BASE_URL}/chat/completions"
    resp = httpx.post(url, headers=headers, json=payload, timeout=timeout)

    if resp.status_code != 200:
        # Log full body — shows "No endpoints found" vs auth errors
        err_body = resp.text[:400]
        logger.warning(f"OpenRouter HTTP {resp.status_code} [{model}]: {err_body}")
        resp.raise_for_status()

    return resp.json()


def _try_models(
    api_key: str,
    image_b64: str,
    user_text: str,
    timeout: float,
) -> tuple[dict[str, Any], str]:
    """
    Try VLM_MODELS in order. Returns (response_json, model_used).
    Raises last exception if all fail.
    """
    last_exc: Exception = RuntimeError("No models tried")
    for model in VLM_MODELS:
        try:
            logger.info(f"VLM attempting → {model}")
            resp = _call_vlm(api_key, model, image_b64, user_text, timeout)
            choices = resp.get("choices")
            if not choices:
                raise ValueError(f"Empty choices: {json.dumps(resp)[:200]}")
            logger.info(f"VLM succeeded → {model}")
            return resp, model
        except Exception as exc:
            logger.warning(f"VLM [{model}] failed: {exc}")
            last_exc = exc
    raise last_exc


# ── JSON cleanup ──────────────────────────────────────────────────────────────
def _clean_json(raw_text: str) -> dict:
    """
    Robust JSON extraction from VLM responses.
    """

    if not raw_text:
        return {}

    raw_text = str(raw_text).strip()

    # Remove markdown fences
    raw_text = re.sub(
        r"```json",
        "",
        raw_text,
        flags=re.IGNORECASE,
    )

    raw_text = re.sub(
        r"```",
        "",
        raw_text,
    )

    raw_text = raw_text.strip()

    # -------------------------------------------------
    # Extract JSON object
    # -------------------------------------------------

    match = re.search(
        r"\{.*\}",
        raw_text,
        re.DOTALL,
    )

    if not match:
        raise ValueError(
            f"No JSON found in VLM output:\n{raw_text[:500]}"
        )

    json_text = match.group(0)

    # Remove trailing commas
    json_text = re.sub(
        r",(\s*[}\]])",
        r"\1",
        json_text,
    )

    return json.loads(json_text)




# ── Config ────────────────────────────────────────────────────────────────────

def _load_config() -> tuple[str, float]:
    api_key = os.environ.get("VLM_API_KEY", "").strip()
    timeout = float(os.environ.get("VLM_TIMEOUT", "60"))
    return api_key, timeout


# ── Public API ────────────────────────────────────────────────────────────────

def run_vlm_analysis(
    image: np.ndarray,
    ocr_text: str,
    parsed_fields: dict[str, Any],
    doc_type: str,
) -> dict[str, Any]:
    """
    Run VLM analysis on a document image via OpenRouter.

    Tries models in order:
      1. nvidia/nemotron-3-nano-omni-30b-a3b-reasoning
      2. google/gemma-4-26b-a4b-it:free
      3. moonshotai/kimi-vl-a3b-thinking:free

    Requires in .env:  VLM_API_KEY = sk-or-v1-...
    Returns structured dict; safe fallback on any failure.
    """
    global _active_model
    api_key, timeout = _load_config()

    if not api_key:
        logger.warning("VLM_API_KEY not set — skipping VLM analysis")
        return _fallback_result("VLM_API_KEY not configured")

    logger.info(f"VLM starting — {len(VLM_MODELS)} model(s) in chain")

    try:
        image_b64 = _numpy_to_base64(image)
        user_text = _user_message_text(ocr_text, parsed_fields, doc_type)

        raw_resp, used_model = _try_models(api_key, image_b64, user_text, timeout)
        _active_model = used_model

        message = raw_resp["choices"][0]["message"]

        # ---------------------------------------------------------
        # OpenRouter / NVIDIA reasoning models may return:
        #   content = None
        #   reasoning = "actual output"
        #   reasoning_content = "actual output"
        # ---------------------------------------------------------

        content = (
            message.get("content")
            or message.get("reasoning")
            or message.get("reasoning_content")
        )

        if content is None:
            raise ValueError(f"VLM returned empty content: {raw_resp}")

        # ---------------------------------------------------------
        # Handle OpenAI-style structured content
        # ---------------------------------------------------------

        if isinstance(content, list):
            raw_text = " ".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict)
            ).strip()

        elif isinstance(content, str):
            raw_text = content.strip()

        else:
            raw_text = str(content).strip()

        # ---------------------------------------------------------
        # Extract JSON safely
        # ---------------------------------------------------------

        result = _clean_json(raw_text)
                

        logger.info(
            f"VLM done — model={used_model}, "
            f"assessment={result.get('overall_assessment')}, "
            f"confidence={result.get('vlm_confidence')}"
        )
        result["vlm_provider"]  = "openrouter"
        result["vlm_model"]     = used_model
        result["vlm_available"] = True
        return result

    except Exception as exc:
        logger.warning(f"All VLM models failed: {exc}")
        return _fallback_result(str(exc))


def _fallback_result(reason: str) -> dict[str, Any]:
    return {
        "document_type_confirmed": "unknown",
        "vlm_confidence": 0.0,
        "field_validation": {},
        "logical_inconsistencies": [],
        "fraud_signals": [],
        "font_consistency": "unknown",
        "layout_authenticity": "unknown",
        "seal_signature_present": False,
        "visible_tampering": False,
        "overall_assessment": "unknown",
        "reasoning": f"VLM analysis skipped: {reason}",
        "suggested_risk_adjustment": 0,
        "vlm_available": False,
        "vlm_provider": "openrouter",
        "vlm_model": _active_model,
    }
