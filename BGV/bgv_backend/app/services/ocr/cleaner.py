import re


def clean_ocr_text(text: str) -> str:
    text = text.replace("\n\n", "\n")

    replacements = {
        "0l": "01",
        "O": "0",
        "|": "1",
    }

    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)

    text = re.sub(r"[ ]{2,}", " ", text)

    return text.strip()