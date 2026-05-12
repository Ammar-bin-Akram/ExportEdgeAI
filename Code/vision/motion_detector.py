"""Motion detection utilities"""
import cv2
import numpy as np
from config.settings import Settings
from vision.mask_utils import create_mango_hsv_mask


class MotionDetector:
    """Handles motion detection using background subtraction and color filtering"""
    
    def __init__(self, settings=None):
        self.settings = settings or Settings()
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=self.settings.BG_HISTORY,
            varThreshold=self.settings.BG_VAR_THRESHOLD,
            detectShadows=False
        )
    
    def detect_motion(self, roi):
        """
        Detect motion in ROI using combined motion and color masks
        
        Args:
            roi: Region of Interest frame
        
        Returns:
            Tuple of (combined_mask, motion_area)
        """
        motion_mask = self._create_motion_mask(roi)
        hsv_mask = self._create_hsv_mask(roi)
        
        combined = cv2.bitwise_and(motion_mask, hsv_mask)
        combined = self._apply_morphological_cleanup(combined)
        
        motion_area = self._calculate_motion_area(combined)
        
        return combined, motion_area
    
    def _create_motion_mask(self, roi):
        """Create motion detection mask using background subtraction"""
        fgmask = self.fgbg.apply(roi)
        fgmask = cv2.GaussianBlur(fgmask, (5, 5), 0)
        _, motion_bin = cv2.threshold(fgmask, 50, 255, cv2.THRESH_BINARY)
        return motion_bin
    
    def _create_hsv_mask(self, roi):
        """Create HSV color mask for mango detection (delegates to shared utility)"""
        return create_mango_hsv_mask(roi, self.settings)
    
    def _apply_morphological_cleanup(self, mask):
        """Apply morphological operations to clean up mask"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        return cleaned
    
    def _calculate_motion_area(self, mask):
        """Calculate total motion area from contours"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return sum(cv2.contourArea(c) for c in contours)
    
    def is_motion_detected(self, motion_area):
        """Check if motion area exceeds threshold"""
        return motion_area > self.settings.MOTION_AREA_THRESHOLD


# Backward compatibility function
def create_motion_detector():
    """Create motion detector - procedural wrapper"""
    return MotionDetector()
