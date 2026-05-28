from typing import Dict, List
from app.schemas.ocr import OCRBox


def detect_regions(boxes: List[OCRBox]) -> Dict[str, OCRBox]:
    regions = {}

    for box in boxes:
        text = box.text.lower()

        if "roll" in text:
            regions["roll_number"] = box

        elif "name" in text:
            regions["student_name"] = box

        elif "year" in text:
            regions["year"] = box

    return regions