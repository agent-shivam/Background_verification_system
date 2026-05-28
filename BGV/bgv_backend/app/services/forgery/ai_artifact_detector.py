"""
app/services/forgery/ai_artifact_detector.py
──────────────────────────────────────────────
Heuristic detection of AI-generated / AI-edited document images.

Modern AI image generators (Stable Diffusion, DALL-E, Midjourney) leave
distinctive statistical signatures in the frequency domain.  Without a
dedicated deep-learning classifier we use a set of signal-processing
heuristics that catch many common artefacts:

1. **DCT frequency analysis** — natural document scans have a smooth
   frequency fall-off.  AI images often show unnatural energy at mid
   frequencies (ringing / over-sharpening).

2. **Noise texture consistency** — real scans have spatially correlated
   noise. AI images often have extremely flat noise floors or periodic
   noise patterns from the diffusion process.

3. **Edge density abnormality** — AI upscaled/generated text tends to
   produce too-crisp or too-smooth edges compared with authentic scans.

These are lightweight CPU checks (no GPU required).  They are intentionally
conservative — a single failed check only adds to the score; all three must
fire to return `suspected`.
"""

from __future__ import annotations

import cv2
import numpy as np
from loguru import logger

from app.schemas.document import AIArtifactResult

# ── Thresholds ────────────────────────────────────────────────────────────────
DCT_MID_RATIO_THRESHOLD: float = 0.35   # mid-freq energy / total energy
NOISE_UNIFORMITY_THRESHOLD: float = 0.92  # std-of-local-noise / global-noise
EDGE_DENSITY_LOW: float = 0.02          # below → suspiciously smooth
EDGE_DENSITY_HIGH: float = 0.40         # above → suspiciously over-sharpened


def detect_ai_artifacts(image: np.ndarray) -> AIArtifactResult:
    """
    Run heuristic AI-artifact checks on the image.

    Returns
    -------
    AIArtifactResult.suspected  — if ≥ 2 checks fire
    AIArtifactResult.not_detected — otherwise
    """
    try:
        gray = _to_gray(image)
        flags: list[bool] = [
            _check_dct_anomaly(gray),
            _check_noise_uniformity(gray),
            _check_edge_density(gray),
        ]
        positives = sum(flags)
        result = (
            AIArtifactResult.suspected
            if positives >= 2
            else AIArtifactResult.not_detected
        )
        logger.debug(
            f"AI artifact detection → flags={flags}, "
            f"positives={positives}, result={result.value}"
        )
        return result
    except Exception as exc:
        logger.warning(f"AI artifact detection failed: {exc}")
        return AIArtifactResult.not_detected


# ── Internal checks ───────────────────────────────────────────────────────────

def _to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def _check_dct_anomaly(gray: np.ndarray) -> bool:
    """
    Check for unnatural mid-frequency energy in the DCT spectrum.
    Authentic scans concentrate energy at low frequencies.
    """
    # Resize to 256×256 for consistent analysis
    resized = cv2.resize(gray, (256, 256)).astype(np.float32)
    dct = cv2.dct(resized)

    total_energy = float(np.sum(dct ** 2)) + 1e-9
    # Mid-frequency band: rows/cols 32–96
    mid = dct[32:96, 32:96]
    mid_energy = float(np.sum(mid ** 2))

    ratio = mid_energy / total_energy
    logger.debug(f"DCT mid-freq ratio={ratio:.4f}")
    return ratio > DCT_MID_RATIO_THRESHOLD


def _check_noise_uniformity(gray: np.ndarray) -> bool:
    """
    Measure spatial uniformity of local noise.
    AI images often have suspiciously flat or periodic noise patterns.
    """
    # Estimate noise via Laplacian residual
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    # Divide into 8×8 grid, compute std per cell
    h, w = lap.shape
    cell_h, cell_w = h // 8, w // 8
    if cell_h == 0 or cell_w == 0:
        return False

    cell_stds: list[float] = []
    for r in range(8):
        for c in range(8):
            cell = lap[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w]
            cell_stds.append(float(np.std(cell)))

    if not cell_stds:
        return False

    mean_std = np.mean(cell_stds)
    global_std = float(np.std(gray))

    # High ratio means very uniform noise → suspicious
    if global_std < 1e-3:
        return False
    uniformity = mean_std / (global_std + 1e-9)
    logger.debug(f"Noise uniformity={uniformity:.4f}")
    return uniformity > NOISE_UNIFORMITY_THRESHOLD


def _check_edge_density(gray: np.ndarray) -> bool:
    """
    Check if edge density is outside the expected range for document scans.
    """
    edges = cv2.Canny(gray, 50, 150)
    density = float(np.count_nonzero(edges)) / (gray.shape[0] * gray.shape[1])
    logger.debug(f"Edge density={density:.4f}")
    return density < EDGE_DENSITY_LOW or density > EDGE_DENSITY_HIGH