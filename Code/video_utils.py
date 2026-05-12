"""Video processing and motion detection functions"""
import cv2
import numpy as np
import config

def create_background_subtractor():
    """Create and return background subtractor"""
    return cv2.createBackgroundSubtractorMOG2(
        history=config.BG_HISTORY,
        varThreshold=config.BG_VAR_THRESHOLD,
        detectShadows=False
    )

def extract_roi(frame, roi_coords=None):
    """Extract Region of Interest from frame"""
    if roi_coords is None:
        roi_coords = config.ROI_COORDS
    x1, y1, x2, y2 = roi_coords
    return frame[y1:y2, x1:x2]

def create_motion_mask(roi, fgbg):
    """Create motion detection mask"""
    fgmask = fgbg.apply(roi)
    fgmask = cv2.GaussianBlur(fgmask, (5, 5), 0)
    _, motion_bin = cv2.threshold(fgmask, 50, 255, cv2.THRESH_BINARY)
    return motion_bin

def create_hsv_mask(roi):
    """Create HSV color mask for mango detection (delegates to shared utility)"""
    from vision.mask_utils import create_mango_hsv_mask
    return create_mango_hsv_mask(roi)

def apply_morphological_cleanup(mask):
    """Apply morphological operations to clean up mask"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned

def calculate_motion_area(mask):
    """Calculate total motion area from contours"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(cv2.contourArea(c) for c in contours)

def process_frame_for_motion(frame, roi, fgbg):
    """
    Process single frame and return combined mask and motion area
    
    Returns:
        tuple: (combined_mask, motion_area)
    """
    motion_mask = create_motion_mask(roi, fgbg)
    hsv_mask = create_hsv_mask(roi)
    
    combined = cv2.bitwise_and(motion_mask, hsv_mask)
    combined = apply_morphological_cleanup(combined)
    
    motion_area = calculate_motion_area(combined)
    
    return combined, motion_area

def display_debug_view(frame, roi_coords, combined_mask):
    """Display debug visualization windows"""
    debug = frame.copy()
    x1, y1, x2, y2 = roi_coords
    cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("Frame (with ROI)", debug)
    cv2.imshow("ROI Combined", combined_mask)

def extract_peak_frames(video_path, display_debug=True):
    """
    Extract peak frames from video based on motion detection
    
    Returns:
        list: List of tuples (frame, metadata_dict)
    """
    cap = cv2.VideoCapture(video_path)
    fgbg = create_background_subtractor()
    
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
        roi = extract_roi(frame)
        
        if roi.size == 0:
            print("[!] ROI is empty — check coordinates")
            break
        
        combined_mask, motion_area = process_frame_for_motion(frame, roi, fgbg)
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        
        # Debug output
        if frame_idx % 30 == 0:  # Print every 30 frames to reduce clutter
            print(f"Frame {frame_idx}: motion_area={motion_area:.0f}, in_motion={in_motion}")
        
        # Detect start of motion
        if not in_motion and motion_area > config.MOTION_AREA_THRESHOLD:
            in_motion = True
            motion_buffer = []
            low_motion_counter = 0
            print(f"Motion started at frame {frame_idx}")
        
        # If in motion, keep buffering frames
        if in_motion:
            motion_buffer.append((frame.copy(), motion_area, frame_idx, timestamp_ms))
            
            # Check if motion ended
            if motion_area < config.MOTION_AREA_THRESHOLD:
                low_motion_counter += 1
            else:
                low_motion_counter = 0
            
            if low_motion_counter >= config.MOTION_END_FRAMES:
                # Motion event ended → pick peak frame
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
        
        # Debug visualization
        if display_debug:
            display_debug_view(frame, config.ROI_COORDS, combined_mask)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Handle edge case: motion still ongoing at end of video
    if in_motion and motion_buffer:
        best_frame, best_area, best_idx, best_time = max(motion_buffer, key=lambda x: x[1])
        metadata = {
            'frame_idx': best_idx,
            'timestamp_ms': best_time,
            'motion_area': best_area
        }
        peak_frames.append((best_frame, metadata))
        print(f"Found peak frame at index {best_idx} (end of video), area={best_area:.0f}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Total peak frames extracted: {len(peak_frames)}")
    return peak_frames
