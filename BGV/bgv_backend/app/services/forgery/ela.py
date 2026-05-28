"""
app/services/forgery/ela.py
────────────────────────────
Error Level Analysis (ELA) for detecting JPEG editing artefacts.

Algorithm:
1. Re-compress the image at a fixed low quality (ELA quality).
2. Compute the absolute pixel difference between original and re-compressed.
3. Amplify differences for visibility & scoring.
4. High mean difference → regions were saved at different quality levels
   → indicator of localised editing / compositing.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
from loguru import logger
from PIL import Image

from app.schemas.document import ELAResult

# ── Tuning constants ──────────────────────────────────────────────────────────
ELA_QUALITY: int = 90          # JPEG re-save quality
SCALE_FACTOR: int = 10         # amplify diff for scoring
SUSPICIOUS_THRESHOLD: float = 30.0    # mean diff  → suspicious
TAMPERED_THRESHOLD: float = 60.0     # mean diff  → tampered


def run_ela(image: np.ndarray) -> tuple[ELAResult, float]:
    """
    Run Error Level Analysis on a BGR (or grayscale) OpenCV image.

    Returns
    -------
    (ELAResult, ela_score)
        ela_score is the mean amplified-difference value (0–100 scale).
    """
    try:
        # Convert to RGB PIL for JPEG re-save
        if len(image.shape) == 2:
            pil_img = Image.fromarray(image).convert("RGB")
        else:
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Re-save at reduced quality into an in-memory buffer
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=ELA_QUALITY)
        buf.seek(0)
        recompressed = Image.open(buf).convert("RGB")

        # Absolute difference → amplify
        orig_arr = np.array(pil_img, dtype=np.float32)
        reco_arr = np.array(recompressed, dtype=np.float32)
        diff = np.abs(orig_arr - reco_arr) * SCALE_FACTOR
        diff = np.clip(diff, 0, 255)

        # Score: mean of the amplified diff image (0–255 range → rescale to 0–100)
        raw_mean = float(np.mean(diff))
        ela_score = min(
            100.0,
            raw_mean * 100.0 / 255.0
        )

        # Classify
        if ela_score >= TAMPERED_THRESHOLD:
            result = ELAResult.tampered
        elif ela_score >= SUSPICIOUS_THRESHOLD:
            result = ELAResult.suspicious
        else:
            result = ELAResult.clean

        logger.debug(f"ELA → score={ela_score:.2f}, result={result.value}")
        return result, round(ela_score, 2)

    except Exception as exc:
        logger.warning(f"ELA failed ({exc}) — defaulting to suspicious")
        return ELAResult.suspicious, 50.0