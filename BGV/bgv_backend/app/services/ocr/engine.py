"""
app/services/ocr/engine.py
──────────────────────────
Multi-engine OCR: Tesseract + PaddleOCR fusion.

• Both engines run on every image; results are merged by confidence.
• Tesseract handles clean printed text well; PaddleOCR excels at
  rotated / low-contrast / Hindi text.
• The fused OCRResult returns the best-confidence line for every
  detected text region, deduplicating near-identical strings.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import numpy as np

import pytesseract

# -------------------------------------------------------------------
# Explicit Windows Tesseract path
# -------------------------------------------------------------------

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


from loguru import logger

from app.core.config import settings
from app.core.exceptions import OCRExtractionError


# ── Engine initialisation ─────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_paddle_ocr():
    """Build and cache PaddleOCR engines (English + Hindi)."""
    from paddleocr import PaddleOCR

    logger.info("Initialising PaddleOCR engines…")

    ocr_en = PaddleOCR(
        use_angle_cls=True,
        lang="en",
        use_gpu=False,
        det_db_box_thresh=0.5,
        det_db_unclip_ratio=1.8,
        rec_batch_num=6,
        cls_batch_num=6,
        show_log=False,
    )

    ocr_hi = PaddleOCR(
        use_angle_cls=True,
        lang="hi",
        use_gpu=False,
        show_log=False,
    )

    logger.info("PaddleOCR engines ready.")
    return {"en": ocr_en, "hi": ocr_hi}




def _tesseract_available() -> bool:
    """
    Return True if pytesseract + Tesseract binary work.
    """

    try:
        import os
        import pytesseract

        tess_path = pytesseract.pytesseract.tesseract_cmd

        if not os.path.exists(tess_path):
            logger.warning(
                f"Tesseract executable not found: {tess_path}"
            )
            return False

        version = pytesseract.get_tesseract_version()

        logger.info(
            f"Tesseract available — version={version}"
        )

        return True

    except Exception as exc:
        logger.warning(
            f"Tesseract unavailable: {exc}"
        )
        return False



# ── OCRResult ─────────────────────────────────────────────────────────────────

class OCRResult:
    """Container for fused OCR output from a single image."""

    def __init__(self, boxes: list[dict]):
        """
        Parameters
        ----------
        boxes : list of dicts with keys: text, confidence, bbox, engine
        """
        self._boxes = boxes or []

    # ── core properties ───────────────────────────────────────────────────────

    @property
    def lines(self) -> list[str]:
        """Ordered list of recognised text lines (highest-confidence first within ties)."""
        return [b["text"] for b in self._boxes]

    @property
    def full_text(self) -> str:
        return "\n".join(self.lines)

    @property
    def confidence(self) -> float:
        """Mean confidence across all detected tokens."""
        if not self._boxes:
            return 0.0
        return sum(b["confidence"] for b in self._boxes) / len(self._boxes)

    @property
    def bboxes(self) -> list[dict]:
        """Bounding boxes with text, confidence, engine tag."""
        return list(self._boxes)

    # legacy alias used in pipeline
    @property
    def boxes(self) -> list[dict]:
        return self.bboxes


# ── Tesseract helper ──────────────────────────────────────────────────────────

def _run_tesseract(image: np.ndarray, lang: str = "eng") -> list[dict]:
    """
    Run Tesseract on *image* and return a list of word-level boxes.
    Returns [] gracefully when Tesseract is unavailable.
    """
    try:
        import pytesseract
        import cv2

        # Tesseract works best on 8-bit grayscale / BGR
        if len(image.shape) == 2:
            tess_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            tess_img = image

        # --psm 3  = fully automatic page segmentation
        # --oem 3  = LSTM + legacy
        tess_lang = "hin+eng" if lang == "hi" else "eng"
        data = pytesseract.image_to_data(
            tess_img,
            lang=tess_lang,
            config="--psm 3 --oem 3",
            output_type=pytesseract.Output.DICT,
        )

        boxes: list[dict] = []
        n = len(data["text"])
        for i in range(n):
            text = data["text"][i].strip()
            if not text:
                continue
            try:
                conf = float(data["conf"][i])
            except (ValueError, TypeError):
                conf = 0.0
            if conf < 0:          # Tesseract returns -1 for non-text blocks
                continue
            conf_norm = conf / 100.0  # normalise to [0, 1]

            x, y, w, h = (
                data["left"][i],
                data["top"][i],
                data["width"][i],
                data["height"][i],
            )
            bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
            boxes.append(
                {"text": text, "confidence": conf_norm, "bbox": bbox, "engine": "tesseract"}
            )

        logger.debug(f"Tesseract: {len(boxes)} tokens extracted")
        return boxes

    except Exception as exc:
        logger.warning(f"Tesseract skipped: {exc}")
        return []


# ── PaddleOCR helper ──────────────────────────────────────────────────────────

def _run_paddle(image: np.ndarray, lang: str = "en") -> list[dict]:
    """
    Run PaddleOCR on *image* and return line-level boxes.
    """
    import cv2

    ocr_engines = _get_paddle_ocr()
    ocr = ocr_engines.get(lang) or ocr_engines["en"]

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    raw = ocr.ocr(image, cls=True)

    boxes: list[dict] = []
    for page in (raw or []):
        if not page:
            continue
        for line in page:
            bbox, (text, conf) = line
            boxes.append(
                {
                    "text": text.strip(),
                    "confidence": float(conf),
                    "bbox": bbox,
                    "engine": "paddle",
                }
            )

    logger.debug(f"PaddleOCR: {len(boxes)} lines extracted")
    return boxes


# ── Fusion ────────────────────────────────────────────────────────────────────

def _normalise(s: str) -> str:
    """Collapse whitespace and lower-case for dedup comparison."""
    return re.sub(r"\s+", " ", s).strip().lower()


def _fuse_results(
    paddle_boxes: list[dict],
    tess_boxes: list[dict],
    conf_threshold: float = 0.30,
) -> list[dict]:
    """
    Merge Tesseract word tokens into PaddleOCR line results.

    Strategy:
    1. Start with PaddleOCR lines as the primary set (they carry layout order).
    2. Aggregate Tesseract tokens into pseudo-lines by grouping words on the
       same Y-band; compute average confidence per pseudo-line.
    3. For each Paddle line, if a Tesseract pseudo-line covers the same text
       region AND has *higher* confidence, substitute the Paddle text with the
       Tesseract text.
    4. Append any Tesseract pseudo-lines whose text is NOT already present
       in the Paddle output (i.e. Paddle missed them entirely).
    5. Filter out boxes below *conf_threshold*.

    This gives us the best of both worlds:
    - PaddleOCR's superior layout detection and Hindi support
    - Tesseract's strong English accuracy on clean document images
    """

    # --- Step 1: aggregate Tesseract tokens into pseudo-lines ----------------
    if tess_boxes:
        tess_boxes_sorted = sorted(tess_boxes, key=lambda b: (b["bbox"][0][1], b["bbox"][0][0]))
        pseudo_lines: list[dict] = []
        current_tokens: list[dict] = []
        current_y: float | None = None
        Y_BAND = 12  # px — tokens within this band share the same line

        for tok in tess_boxes_sorted:
            tok_y = tok["bbox"][0][1]
            if current_y is None or abs(tok_y - current_y) <= Y_BAND:
                current_tokens.append(tok)
                current_y = tok_y if current_y is None else (current_y + tok_y) / 2
            else:
                if current_tokens:
                    pseudo_lines.append(_merge_tokens(current_tokens))
                current_tokens = [tok]
                current_y = tok_y

        if current_tokens:
            pseudo_lines.append(_merge_tokens(current_tokens))
    else:
        pseudo_lines = []

    # --- Step 2: build lookup of already-present Paddle text ----------------
    paddle_norm = {_normalise(b["text"]) for b in paddle_boxes if b["text"]}

    # --- Step 3: substitute where Tesseract wins in confidence --------------
    fused: list[dict] = []
    for pb in paddle_boxes:
        best = pb
        pb_norm = _normalise(pb["text"])
        for tl in pseudo_lines:
            tl_norm = _normalise(tl["text"])
            # same content (or very close) — keep higher confidence version
            if _texts_similar(pb_norm, tl_norm) and tl["confidence"] > pb["confidence"]:
                best = {**pb, "text": tl["text"], "confidence": tl["confidence"], "engine": "fused"}
                break
        fused.append(best)

    # --- Step 4: append Tesseract-only lines Paddle missed ------------------
    for tl in pseudo_lines:
        tl_norm = _normalise(tl["text"])
        if not tl_norm:
            continue
        already_present = any(_texts_similar(tl_norm, _normalise(b["text"])) for b in fused)
        if not already_present:
            fused.append({**tl, "engine": "tesseract_only"})

    # --- Step 5: filter low-confidence noise --------------------------------
    fused = [b for b in fused if b["confidence"] >= conf_threshold and b["text"]]

    return fused


def _merge_tokens(tokens: list[dict]) -> dict:
    """Combine a list of word tokens into a single pseudo-line dict."""
    text = " ".join(t["text"] for t in tokens)
    avg_conf = sum(t["confidence"] for t in tokens) / len(tokens)
    # bounding box = union of all token boxes
    all_pts = [pt for t in tokens for pt in t["bbox"]]
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    bbox = [
        [min(xs), min(ys)],
        [max(xs), min(ys)],
        [max(xs), max(ys)],
        [min(xs), max(ys)],
    ]
    return {"text": text, "confidence": avg_conf, "bbox": bbox, "engine": "tesseract"}


def _texts_similar(a: str, b: str, threshold: float = 0.75) -> bool:
    """Simple overlap ratio to detect near-duplicate text strings."""
    if not a or not b:
        return False
    shorter, longer = sorted([a, b], key=len)
    if not longer:
        return False
    # character-level overlap (quick Jaccard on trigrams)
    def trigrams(s: str) -> set:
        return {s[i : i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else {s}
    tA, tB = trigrams(shorter), trigrams(longer)
    inter = len(tA & tB)
    union = len(tA | tB)
    return (inter / union) >= threshold if union else False


# ── Public API ────────────────────────────────────────────────────────────────

def run_ocr(
    image: np.ndarray,
    lang: str = "en",
) -> OCRResult:
    """
    Run multi-engine OCR (Tesseract + PaddleOCR) on a preprocessed NumPy image
    and return a fused OCRResult.

    Parameters
    ----------
    image : np.ndarray
        Preprocessed image (BGR or grayscale).
    lang : str
        'en' for English-primary docs, 'hi' for Hindi / bilingual docs.
    """
    try:
        # ── PaddleOCR (primary) ───────────────────────────────────────────────
        paddle_boxes = _run_paddle(image, lang=lang)

        # ── Tesseract (secondary, if available) ───────────────────────────────
        tess_lang = "hi" if lang == "hi" else "eng"
        if _tesseract_available():
            tess_boxes = _run_tesseract(image, lang=tess_lang)
            logger.info("Multi-engine OCR: fusing Tesseract + PaddleOCR results")
            fused_boxes = _fuse_results(paddle_boxes, tess_boxes)
        else:
            logger.info("Tesseract not available — using PaddleOCR only")
            fused_boxes = paddle_boxes

        result = OCRResult(fused_boxes)
        logger.debug(
            f"OCR done — {len(result.lines)} lines, "
            f"avg confidence={result.confidence:.2%}"
        )
        return result

    except Exception as exc:
        logger.exception(exc)
        raise OCRExtractionError("Multi-engine OCR extraction failed") from exc