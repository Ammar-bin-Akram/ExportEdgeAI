"""
Shared mango HSV mask utility.

Extracts a binary mask of the mango surface from a BGR image using the
same yellow + green + black HSV colour ranges defined in Settings.
This mask is used by:
    - MotionDetector  (motion detection)
    - FrameExtractor  (peak frame selection)
    - MangoDefectDetector  (defect analysis — passed as external mask)
"""

import cv2
import numpy as np
from config.settings import Settings


def create_mango_hsv_mask(image: np.ndarray, settings=None) -> np.ndarray:
    """
    Create a binary mask isolating mango pixels via HSV colour filtering.

    Combines three colour ranges (yellow, green, black) with a
    morphological open+close cleanup.  Works best on **raw** (unprocessed)
    ROI images where belt vs mango colours are clearly separable.

    Args:
        image:    BGR image (any resolution).
        settings: Optional Settings instance.  Uses default if None.

    Returns:
        Binary mask  (255 = mango, 0 = background), same H×W as input.
    """
    if settings is None:
        settings = Settings()

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask_yellow = cv2.inRange(
        hsv,
        np.array(settings.HSV_YELLOW_LOWER),
        np.array(settings.HSV_YELLOW_UPPER),
    )
    mask_green = cv2.inRange(
        hsv,
        np.array(settings.HSV_GREEN_LOWER),
        np.array(settings.HSV_GREEN_UPPER),
    )
    mask_black = cv2.inRange(
        hsv,
        np.array(settings.HSV_BLACK_LOWER),
        np.array(settings.HSV_BLACK_UPPER),
    )

    combined = cv2.bitwise_or(mask_yellow, mask_green)
    combined = cv2.bitwise_or(combined, mask_black)

    # Morphological cleanup (same 7×7 ellipse used everywhere)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    return combined


__all__ = ['create_mango_hsv_mask']
