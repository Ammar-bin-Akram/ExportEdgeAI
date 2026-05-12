"""Frame extraction and motion detection"""
import cv2
import numpy as np
from config.settings import Settings
from vision.mask_utils import create_mango_hsv_mask


class FrameExtractor:
    """Handles video frame extraction with motion detection"""
    
    def __init__(self, settings=None):
        self.settings = settings or Settings()
        self.motion_detector = None
    
    def extract_peak_frames(self, video_path, display_debug=True):
        """
        Extract peak frames from video based on motion detection
        
        Args:
            video_path: Path to video file
            display_debug: Whether to display debug windows
        
        Returns:
            List of tuples (frame, metadata_dict)
        """
        cap = cv2.VideoCapture(video_path)
        fgbg = cv2.createBackgroundSubtractorMOG2(
            history=self.settings.BG_HISTORY,
            varThreshold=self.settings.BG_VAR_THRESHOLD,
            detectShadows=False
        )
        
        frame_idx = 0
        in_motion = False
        motion_buffer = []
        low_motion_counter = 0
        peak_frames = []
        
        print(f"Processing video: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            roi = self._extract_roi(frame)
            
            if roi.size == 0:
                break
            
            combined_mask, motion_area = self._process_frame_for_motion(frame, roi, fgbg)
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            
            if frame_idx % 30 == 0:
                print(f"Frame {frame_idx}: motion_area={motion_area:.0f}, in_motion={in_motion}")
            
            if not in_motion and motion_area > self.settings.MOTION_AREA_THRESHOLD:
                in_motion = True
                motion_buffer = []
                low_motion_counter = 0
                print(f"Motion started at frame {frame_idx}")
            
            if in_motion:
                motion_buffer.append((frame.copy(), motion_area, frame_idx, timestamp_ms))
                
                if motion_area < self.settings.MOTION_AREA_THRESHOLD:
                    low_motion_counter += 1
                else:
                    low_motion_counter = 0
                
                if low_motion_counter >= self.settings.MOTION_END_FRAMES:
                    in_motion = False
                    if motion_buffer:
                        best_frame, best_area, best_idx, best_time = max(
                            motion_buffer, key=lambda x: x[1]
                        )
                        
                        metadata = {
                            'frame_idx': best_idx,
                            'timestamp_ms': best_time,
                            'motion_area': best_area
                        }
                        
                        peak_frames.append((best_frame, metadata))
                        print(f"Found peak frame at index {best_idx}, area={best_area:.0f}")
                    
                    motion_buffer = []
                    low_motion_counter = 0
            
            if display_debug:
                self._display_debug_view(frame, combined_mask)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        if in_motion and motion_buffer:
            best_frame, best_area, best_idx, best_time = max(motion_buffer, key=lambda x: x[1])
            metadata = {
                'frame_idx': best_idx,
                'timestamp_ms': best_time,
                'motion_area': best_area
            }
            peak_frames.append((best_frame, metadata))
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"Total peak frames extracted: {len(peak_frames)}")
        return peak_frames
    
    def _extract_roi(self, frame):
        """Extract Region of Interest from frame"""
        x1, y1, x2, y2 = self.settings.ROI_COORDS
        return frame[y1:y2, x1:x2]
    
    def _process_frame_for_motion(self, frame, roi, fgbg):
        """Process frame for motion detection"""
        motion_mask = self._create_motion_mask(roi, fgbg)
        hsv_mask = self._create_hsv_mask(roi)
        
        combined = cv2.bitwise_and(motion_mask, hsv_mask)
        combined = self._apply_morphological_cleanup(combined)
        
        motion_area = self._calculate_motion_area(combined)
        
        return combined, motion_area
    
    def _create_motion_mask(self, roi, fgbg):
        """Create motion detection mask"""
        fgmask = fgbg.apply(roi)
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
    
    def _display_debug_view(self, frame, combined_mask):
        """Display debug visualization windows"""
        debug = frame.copy()
        x1, y1, x2, y2 = self.settings.ROI_COORDS
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Frame (with ROI)", debug)
        cv2.imshow("ROI Combined", combined_mask)


# Backward compatibility functions
def create_background_subtractor():
    """Create and return background subtractor"""
    settings = Settings()
    return cv2.createBackgroundSubtractorMOG2(
        history=settings.BG_HISTORY,
        varThreshold=settings.BG_VAR_THRESHOLD,
        detectShadows=False
    )

def extract_peak_frames(video_path, display_debug=True):
    """Extract peak frames - procedural wrapper"""
    extractor = FrameExtractor()
    return extractor.extract_peak_frames(video_path, display_debug)

def process_frame_for_motion(frame, roi, fgbg):
    """Process frame for motion - procedural wrapper"""
    extractor = FrameExtractor()
    return extractor._process_frame_for_motion(frame, roi, fgbg)
