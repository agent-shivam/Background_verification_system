"""
app/services/aadhaar/font_consistency.py
─────────────────────────────────────────
Font Consistency Detection for Aadhaar cards.

Fake Aadhaar cards are often created by:
1. Photographing/scanning a real card
2. Digitally editing specific fields (name, DOB, photo)
3. Re-printing or using as-is

The editing leaves detectable traces:
- Different font rendering engine (Windows GDI vs UIDAI's renderer)
- Different anti-aliasing profile
- Different JPEG compression artefacts around edited text
- Inconsistent stroke width
- Different letter spacing (kerning)

This module uses OpenCV frequency analysis + local variance to flag
regions with inconsistent text rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from loguru import logger


# ── Tuning constants ───────────────────────────────────────────────────────────
FONT_SUSPICIOUS_THRESHOLD = 0.25   # variance ratio above this = suspicious
FONT_TAMPERED_THRESHOLD   = 0.45   # variance ratio above this = likely tampered
BLOCK_SIZE                = 32     # analyse in NxN pixel blocks
MIN_TEXT_DENSITY          = 0.05   # skip near-empty blocks


@dataclass
class FontConsistencyResult:
    """Results of font consistency analysis."""
    status: str = "consistent"          # "consistent" | "suspicious" | "inconsistent"
    variance_ratio: float = 0.0         # 0=uniform, 1=highly inconsistent
    suspicious_blocks: int = 0
    total_text_blocks: int = 0
    high_frequency_anomaly: bool = False
    compression_inconsistency: bool = False
    risk_penalty: int = 0
    detail: str = ""
    block_map: list[dict] = field(default_factory=list)  # per-block analysis


def analyse_font_consistency(image: np.ndarray) -> FontConsistencyResult:
    """
    Analyse font consistency across an Aadhaar card image.

    Uses two complementary methods:
    1. Block-level frequency analysis (DCT-based) — detects different
       JPEG quality zones that indicate edited text regions.
    2. Local stroke variance — detects text with different stroke widths
       or anti-aliasing profiles (different font renderer).

    Returns FontConsistencyResult with risk assessment.
    """
    result = FontConsistencyResult()

    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape

        # Need meaningful image size
        if h < 100 or w < 100:
            result.status = "consistent"
            result.detail = "Image too small for font analysis"
            return result

        # ── Method 1: Block DCT frequency analysis ────────────────────────────
        # Edited regions re-compressed at different JPEG quality than original.
        # High-frequency content (text edges) should be uniformly compressed.
        dct_variances = []
        block_results = []

        step = BLOCK_SIZE
        for y in range(0, h - step, step):
            for x in range(0, w - step, step):
                block = gray[y:y+step, x:x+step].astype(np.float32)

                # Skip near-empty blocks (no text)
                text_density = np.mean(block < 200) / (block.size)
                if text_density < MIN_TEXT_DENSITY:
                    continue

                # DCT of the block
                dct_block = cv2.dct(block)

                # High-frequency energy (upper-right of DCT matrix)
                hf_energy = float(np.mean(np.abs(dct_block[step//2:, step//2:])))
                lf_energy = float(np.mean(np.abs(dct_block[:step//2, :step//2])))
                ratio = hf_energy / (lf_energy + 1e-6)

                dct_variances.append(ratio)
                block_results.append({
                    "x": x, "y": y,
                    "hf_ratio": round(ratio, 4),
                    "text_density": round(text_density, 3),
                })

        if len(dct_variances) < 5:
            result.status = "consistent"
            result.detail = "Insufficient text blocks for analysis"
            return result

        # Measure spread: uniform quality → low variance of ratios
        variance_array = np.array(dct_variances)
        mean_ratio = float(np.mean(variance_array))
        std_ratio = float(np.std(variance_array))
        cv = std_ratio / (mean_ratio + 1e-6)  # coefficient of variation

        result.total_text_blocks = len(dct_variances)

        # Flag suspicious blocks (outliers beyond 2σ)
        upper_bound = mean_ratio + 2 * std_ratio
        suspicious = sum(1 for r in dct_variances if r > upper_bound)
        result.suspicious_blocks = suspicious

        # ── Method 2: Local stroke width variance ─────────────────────────────
        # Detect non-uniform anti-aliasing by measuring edge sharpness distribution
        edge_sharpness_scores = _analyse_edge_sharpness(gray)
        sharpness_cv = float(np.std(edge_sharpness_scores) / (np.mean(edge_sharpness_scores) + 1e-6))

        # High-frequency anomaly: unusually sharp edges in specific regions
        # (typical of pasted text from different source)
        result.high_frequency_anomaly = sharpness_cv > 0.8

        # ── Compression inconsistency via block artefacts ─────────────────────
        result.compression_inconsistency = _detect_compression_inconsistency(gray)

        # ── Composite variance ratio ──────────────────────────────────────────
        # Blend DCT CV with sharpness CV
        result.variance_ratio = round(min(1.0, cv * 0.6 + sharpness_cv * 0.4), 4)

        # ── Classification ────────────────────────────────────────────────────
        anomaly_bonus = 0.15 if result.high_frequency_anomaly else 0
        anomaly_bonus += 0.10 if result.compression_inconsistency else 0
        final_ratio = result.variance_ratio + anomaly_bonus

        if final_ratio >= FONT_TAMPERED_THRESHOLD:
            result.status = "inconsistent"
            result.risk_penalty = 20
            result.detail = (
                f"Font inconsistency DETECTED — variance_ratio={result.variance_ratio:.3f}, "
                f"suspicious_blocks={suspicious}/{result.total_text_blocks}, "
                f"hf_anomaly={result.high_frequency_anomaly}, "
                f"compression_inconsistency={result.compression_inconsistency}"
            )
        elif final_ratio >= FONT_SUSPICIOUS_THRESHOLD:
            result.status = "suspicious"
            result.risk_penalty = 10
            result.detail = (
                f"Font rendering suspicious — variance_ratio={result.variance_ratio:.3f}, "
                f"suspicious_blocks={suspicious}/{result.total_text_blocks}"
            )
        else:
            result.status = "consistent"
            result.risk_penalty = 0
            result.detail = (
                f"Font rendering consistent — variance_ratio={result.variance_ratio:.3f}"
            )

        # Store block map (top-5 most suspicious for reporting)
        sorted_blocks = sorted(block_results, key=lambda b: b["hf_ratio"], reverse=True)
        result.block_map = sorted_blocks[:5]

        logger.info(f"Font consistency: {result.status} (ratio={result.variance_ratio:.3f})")

    except Exception as exc:
        logger.warning(f"Font consistency analysis failed: {exc}")
        result.status = "consistent"
        result.detail = f"Analysis error: {exc}"

    return result


def _analyse_edge_sharpness(gray: np.ndarray) -> np.ndarray:
    """
    Compute per-block edge sharpness using Laplacian.
    Returns array of sharpness scores per block.
    """
    h, w = gray.shape
    scores = []
    step = BLOCK_SIZE * 2

    for y in range(0, h - step, step):
        for x in range(0, w - step, step):
            block = gray[y:y+step, x:x+step]
            lap = cv2.Laplacian(block, cv2.CV_64F)
            scores.append(float(np.var(lap)))

    if not scores:
        return np.array([0.0])
    return np.array(scores)


def _detect_compression_inconsistency(gray: np.ndarray) -> bool:
    """
    Detect JPEG blocking artefacts that are inconsistent across the image.

    Real Aadhaar cards have uniform JPEG compression quality.
    Edited cards may have regions with different blocking artefact levels.
    """
    h, w = gray.shape
    blocking_scores = []
    step = 48

    for y in range(0, h - step, step):
        for x in range(0, w - step, step):
            block = gray[y:y+step, x:x+step].astype(np.float32)

            # Measure blockiness: difference at 8-pixel boundaries (JPEG block size)
            if block.shape[0] >= 16 and block.shape[1] >= 16:
                # Horizontal boundary diff at y=8, 16, 24...
                h_diffs = []
                for by in range(8, step - 8, 8):
                    row_above = block[by-1, :]
                    row_below = block[by, :]
                    h_diffs.append(float(np.mean(np.abs(row_above - row_below))))

                if h_diffs:
                    blocking_scores.append(np.mean(h_diffs))

    if len(blocking_scores) < 4:
        return False

    arr = np.array(blocking_scores)
    cv = float(np.std(arr) / (np.mean(arr) + 1e-6))
    return cv > 1.2  # highly inconsistent blocking = different compression zones
