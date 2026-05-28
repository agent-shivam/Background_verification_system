from paddleocr import PPStructure
from loguru import logger


table_engine = PPStructure(
    show_log=False,
    layout=True,
    table=True,
    ocr=True
)


def analyse_layout(image):
    """
    Analyse document structure:
    - tables
    - headers
    - text regions
    - layout blocks
    """

    result = table_engine(image)

    logger.info(
        f"Detected {len(result)} layout regions"
    )

    return result