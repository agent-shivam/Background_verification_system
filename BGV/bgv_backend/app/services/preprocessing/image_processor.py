"""
app/services/preprocessing/image_processor.py
──────────────────────────────────────────────
Converts PDF → images and applies image-enhancement pipeline
(grayscale, denoising, deskew, contrast normalisation) so PaddleOCR
gets the cleanest possible input.
"""

from __future__ import annotations

from pathlib import Path

from app.core.config import settings


import cv2
import numpy as np
from loguru import logger

from app.core.exceptions import PDFConversionError, PreprocessingError


# ── PDF → images ──────────────────────────────────────────────────────────────
def pdf_to_images(pdf_path: Path, dpi: int = 300) -> list[np.ndarray]:
    """
    Render every page of a PDF to an OpenCV BGR image array.
    Uses pdf2image (poppler wrapper).
    """
    try:
        from pdf2image import convert_from_path

        pil_pages = convert_from_path(
            str(pdf_path),
            dpi=dpi,
            poppler_path=settings.poppler_path
        )

        images: list[np.ndarray] = []

        for page in pil_pages:
            bgr = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            images.append(bgr)

        logger.info(f"PDF → {len(images)} page(s) at {dpi} dpi: {pdf_path.name}")

        return images

    except Exception as exc:
        logger.exception(exc)

        raise PDFConversionError(
            f"PDF conversion failed for '{pdf_path.name}'"
        ) from exc

# ── Helpers ───────────────────────────────────────────────────────────────────

def _resize(img: np.ndarray, target_width: int = 1600) -> np.ndarray:
    h, w = img.shape[:2]
    if w == target_width:
        return img
    scale = target_width / w
    new_h = int(h * scale)
    return cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_CUBIC)


def _to_grayscale(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)


def _binarise(gray: np.ndarray) -> np.ndarray:
    """Adaptive Gaussian threshold — handles uneven illumination well."""
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )


def _deskew(img: np.ndarray) -> np.ndarray:
    """
    Estimate skew angle via Hough lines and rotate to correct it.
    Skips correction if estimated angle is < 0.5 ° (noise).
    """
    coords = np.column_stack(np.where(img > 0))
    if len(coords) < 5:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    # minAreaRect returns angles in [-90, 0); map to [-45, 45]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    logger.debug(f"Deskewed by {angle:.2f}°")
    return rotated


def load_image(path: Path) -> np.ndarray:
    """Load any supported image file as a BGR NumPy array."""
    img = cv2.imread(str(path))
    if img is None:
        raise PreprocessingError(f"cv2.imread returned None for '{path}'")
    return img


def image_to_pil(img: np.ndarray):
    """Convert OpenCV BGR array → PIL Image (RGB)."""
    from PIL import Image
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))



# ── Core preprocessing pipeline ───────────────────────────────────────────────

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline:
    1. Resize
    2. Grayscale
    3. Denoise
    4. Adaptive threshold
    5. Deskew
    """

    try:
        img = _resize(img, target_width=1200)

        gray = _to_grayscale(img)

        denoised = _denoise(gray)

        binary = _binarise(denoised)

        deskewed = _deskew(binary)

        logger.debug(f"Preprocessed image shape: {deskewed.shape}")

        return deskewed

    except Exception as exc:
        logger.exception(exc)

        raise PreprocessingError(
            "Image preprocessing failed"
        ) from exc