"""
app/services/forgery/blur.py
──────────────────────────────
Blur / sharpness detection using Laplacian variance.

Why blur matters for forgery detection:
- Authentic scanned documents have consistent, moderate sharpness.
- Heavily blurred images may be hiding editing artefacts.
- Very low blur variance on a specific region can indicate a pasted patch.

We compute the global Laplacian variance and classify into three bands.
"""

from __future__ import annotations

import cv2
import numpy as np
from loguru import logger

from app.schemas.document import BlurResult

# ── Thresholds (tuned for 1200-px-wide scans) ────────────────────────────────
BLUR_THRESHOLD: float    = 80.0   # below → blurry
SLIGHTLY_BLURRY: float   = 200.0  # below → slightly blurry; above → sharp


def detect_blur(image: np.ndarray) -> tuple[BlurResult, float]:
    """
    Compute Laplacian variance to estimate image sharpness.

    Parameters
    ----------
    image : np.ndarray
        Grayscale or BGR image (uint8).

    Returns
    -------
    (BlurResult, blur_score)
        blur_score is the Laplacian variance (higher = sharper).
    """
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Laplacian variance
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        variance = float(np.var(lap))

        if variance < BLUR_THRESHOLD:
            result = BlurResult.blurry
        elif variance < SLIGHTLY_BLURRY:
            result = BlurResult.slightly_blurry
        else:
            result = BlurResult.sharp

        logger.debug(f"Blur → variance={variance:.2f}, result={result.value}")
        return result, round(variance, 2)

    except Exception as exc:
        logger.warning(f"Blur detection failed: {exc}")
        return BlurResult.slightly_blurry, 0.0