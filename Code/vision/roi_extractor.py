"""ROI (Region of Interest) extraction utilities"""
import cv2
import numpy as np
from config.settings import Settings


class ROIExtractor:
    """Handles extraction of Region of Interest from frames"""
    
    def __init__(self, settings=None):
        self.settings = settings or Settings()
    
    def extract_roi(self, frame):
        """
        Extract ROI from frame based on configured coordinates
        
        Args:
            frame: Input image frame
        
        Returns:
            Cropped ROI region
        """
        x1, y1, x2, y2 = self.settings.ROI_COORDS
        return frame[y1:y2, x1:x2]
    
    def apply_roi_mask(self, frame):
        """
        Apply ROI as a mask to the frame (black out non-ROI areas)
        
        Args:
            frame: Input image frame
        
        Returns:
            Masked frame with only ROI visible
        """
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        x1, y1, x2, y2 = self.settings.ROI_COORDS
        mask[y1:y2, x1:x2] = 255
        
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        return masked_frame
    
    def visualize_roi(self, frame, color=(0, 255, 0), thickness=2):
        """
        Draw ROI rectangle on frame for visualization
        
        Args:
            frame: Input image frame
            color: Color of rectangle (B, G, R)
            thickness: Line thickness
        
        Returns:
            Frame with ROI rectangle drawn
        """
        frame_copy = frame.copy()
        x1, y1, x2, y2 = self.settings.ROI_COORDS
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness)
        return frame_copy


# Backward compatibility function
def extract_roi(frame):
    """Extract ROI - procedural wrapper"""
    extractor = ROIExtractor()
    return extractor.extract_roi(frame)
