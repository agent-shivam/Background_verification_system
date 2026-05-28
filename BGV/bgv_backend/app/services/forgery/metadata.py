"""
app/services/forgery/metadata.py
──────────────────────────────────
EXIF / file metadata analysis.

Red-flag heuristics:
- Software tag reveals image-editing tools (Photoshop, GIMP, etc.)
- Creation timestamp ≠ modification timestamp (significant gap)
- EXIF GPS / serial-number fields stripped (common after editing)
- File has no EXIF at all (some forged scans strip all metadata)
- piexif can't parse the EXIF block (structure corruption)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from app.schemas.document import MetadataResult

# Tools that strongly suggest post-processing
_EDITING_TOOLS: tuple[str, ...] = (
    "photoshop", "gimp", "lightroom", "affinity", "paint.net",
    "canva", "snapseed", "pixlr", "corel", "adobe",
)


def analyse_metadata(file_path: Path) -> tuple[MetadataResult, dict[str, Any]]:
    """
    Inspect EXIF metadata from an image file.

    Returns
    -------
    (MetadataResult, metadata_dict)
        metadata_dict contains raw findings for audit / logging.
    """
    findings: dict[str, Any] = {}

    # PDFs have no EXIF data — skip image-based analysis entirely
    if Path(file_path).suffix.lower() == ".pdf":
        logger.debug("Metadata: PDF file — EXIF analysis not applicable, skipping")
        findings["exif"] = "not_applicable_pdf"
        return MetadataResult.missing, findings

    try:
        import piexif
        from PIL import Image

        img = Image.open(file_path)
        info = img.info or {}
        exif_bytes = info.get("exif", b"")

        if not exif_bytes:
            findings["exif"] = "absent"
            logger.debug("Metadata: no EXIF data found → suspicious")
            return MetadataResult.missing, findings

        try:
            exif_data = piexif.load(exif_bytes)
        except Exception as parse_exc:
            findings["exif"] = f"corrupt ({parse_exc})"
            logger.warning(f"Metadata: EXIF parse error → {parse_exc}")
            return MetadataResult.suspicious, findings

        # ── Software tag check ────────────────────────────────────────────────
        software_tag = exif_data.get("0th", {}).get(piexif.ImageIFD.Software, b"")
        if isinstance(software_tag, bytes):
            software_str = software_tag.decode("utf-8", errors="ignore").lower()
        else:
            software_str = str(software_tag).lower()
        findings["software"] = software_str or "not_set"

        if any(tool in software_str for tool in _EDITING_TOOLS):
            logger.warning(f"Metadata: editing tool detected → '{software_str}'")
            return MetadataResult.suspicious, findings

        # ── DateTime vs DateTimeOriginal mismatch ─────────────────────────────
        dt_tag = exif_data.get("0th", {}).get(piexif.ImageIFD.DateTime, b"")
        dto_tag = exif_data.get("Exif", {}).get(piexif.ExifIFD.DateTimeOriginal, b"")

        dt  = dt_tag.decode("utf-8", errors="ignore")  if isinstance(dt_tag,  bytes) else str(dt_tag)
        dto = dto_tag.decode("utf-8", errors="ignore") if isinstance(dto_tag, bytes) else str(dto_tag)

        findings["datetime"] = dt or "not_set"
        findings["datetime_original"] = dto or "not_set"

        if dt and dto and dt != dto:
            logger.info(f"Metadata: DateTime mismatch — modified={dt}, original={dto}")
            return MetadataResult.suspicious, findings

        # ── All checks passed ─────────────────────────────────────────────────
        logger.debug("Metadata analysis: normal")
        return MetadataResult.normal, findings

    except Exception as exc:
        logger.warning(f"Metadata analysis failed: {exc}")
        findings["error"] = str(exc)
        return MetadataResult.suspicious, findings