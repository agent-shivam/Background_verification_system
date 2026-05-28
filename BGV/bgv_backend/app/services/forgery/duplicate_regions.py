"""
app/services/forgery/duplicate_regions.py
──────────────────────────────────────────
ORB (Oriented FAST and Rotated BRIEF) keypoint-based copy-move detection.

Strategy:
1. Detect ORB keypoints + descriptors across the full image.
2. BruteForce match descriptors against themselves (excluding self-matches).
3. For every "good" match, measure spatial distance between the two keypoints.
4. If two spatially-distant regions share many similar descriptors it suggests
   a copy-paste operation (copy-move forgery).

Tuning parameters are conservative to minimise false positives on real docs
with repetitive elements (logos, borders, watermarks).
"""

from __future__ import annotations

import cv2
import numpy as np
from loguru import logger

from app.schemas.document import DuplicateRegionResult

# ── Tuning constants ──────────────────────────────────────────────────────────
N_FEATURES: int = 1000          # max ORB keypoints to detect
MATCH_DISTANCE: int = 40        # Hamming distance threshold for "good" match
MIN_SPATIAL_DIST: float = 50.0  # px; close keypoints are NOT copy-move pairs
DETECTION_THRESHOLD: int = 10   # min suspicious pairs to flag detection


def detect_duplicate_regions(image: np.ndarray) -> DuplicateRegionResult:
    """
    Detect copy-move forgery using ORB self-matching.

    Parameters
    ----------
    image : np.ndarray
        Grayscale or BGR image (uint8).

    Returns
    -------
    DuplicateRegionResult
    """
    try:
        # Grayscale required for ORB
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # ── Detect keypoints & descriptors ────────────────────────────────────
        orb = cv2.ORB_create(nfeatures=N_FEATURES)
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        if descriptors is None or len(descriptors) < 2:
            logger.debug("ORB: insufficient keypoints — skipping copy-move check")
            return DuplicateRegionResult.not_detected

        # ── Self-match ────────────────────────────────────────────────────────
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        # k=2 so we can apply ratio test
        matches = bf.knnMatch(descriptors, descriptors, k=2)

        suspicious_pairs: int = 0

        for m, n in matches:
            # Skip self-match (same descriptor index)
            if m.queryIdx == m.trainIdx:
                continue
            # Hamming distance filter
            if m.distance > MATCH_DISTANCE:
                continue
            # Reject ratio-test failures (Lowe's ratio test)
            if n.distance == 0 or m.distance / n.distance > 0.75:
                continue

            # Spatial distance between the two matched keypoints
            pt1 = np.array(keypoints[m.queryIdx].pt)
            pt2 = np.array(keypoints[m.trainIdx].pt)
            dist = float(np.linalg.norm(pt1 - pt2))

            if dist > MIN_SPATIAL_DIST:
                suspicious_pairs += 1

        result = (
            DuplicateRegionResult.detected
            if suspicious_pairs >= DETECTION_THRESHOLD
            else DuplicateRegionResult.not_detected
        )

        logger.debug(
            f"ORB → keypoints={len(keypoints)}, "
            f"suspicious_pairs={suspicious_pairs}, result={result.value}"
        )
        return result

    except Exception as exc:
        logger.warning(f"Duplicate region detection failed: {exc}")
        return DuplicateRegionResult.not_detected