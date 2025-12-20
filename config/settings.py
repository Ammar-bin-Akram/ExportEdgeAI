"""Configuration settings for mango disease detection pipeline"""
import os
from pathlib import Path

# Get the project root directory (parent of Code folder)
PROJECT_ROOT = Path(__file__).parent.parent.parent
CODE_DIR = Path(__file__).parent.parent


class Settings:
    """Centralized configuration settings"""
    
    # Video settings
    VIDEO_SOURCE = str(PROJECT_ROOT / "live_recording3.mp4")
    # VIDEO_SOURCE = "rtsp://admin:admin123@192.168.1.108:554/cam/realmonitor?channel=1&subtype=1"
    
    # ROI coordinates (x1, y1, x2, y2)
    ROI_COORDS = (162, 31, 498, 465)
    
    # Motion detection parameters
    MOTION_AREA_THRESHOLD = 5000
    MOTION_END_FRAMES = 10
    
    # HSV color ranges for mango detection
    HSV_YELLOW_LOWER = [12, 90, 90]
    HSV_YELLOW_UPPER = [40, 255, 255]
    HSV_GREEN_LOWER = [30, 40, 40]
    HSV_GREEN_UPPER = [90, 255, 255]
    HSV_BLACK_LOWER = [0, 0, 0]
    HSV_BLACK_UPPER = [180, 255, 50]
    
    # Background subtraction parameters
    BG_HISTORY = 200
    BG_VAR_THRESHOLD = 40
    
    # Model settings
    MODEL_PATH = str(PROJECT_ROOT / "models" / "mobilenet_mango_model.tflite")
    SEGMENTATION_MODEL_PATH = str(PROJECT_ROOT / "models" / "deeplabv3_mobilenet_mango (1).pth")
    INPUT_SHAPE = (150, 150, 3)
    NUM_CLASSES = 5
    CLASS_NAMES = ['Alternaria', 'Anthracnose', 'Black Mould Rot', 'Healthy', 'Stem end Rot']
    
    # Segmentation settings
    ENABLE_SEGMENTATION = True
    SEGMENTATION_THRESHOLD = 0.5
    
    # Output settings
    OUTPUT_DIR = str(PROJECT_ROOT / "relevant_frames3")
    SAVE_INTERMEDIATE_STEPS = True
    DISPLAY_DEBUG = True


# Create a global instance for backward compatibility
settings = Settings()

# Export individual settings for backward compatibility
VIDEO_SOURCE = settings.VIDEO_SOURCE
ROI_COORDS = settings.ROI_COORDS
MOTION_AREA_THRESHOLD = settings.MOTION_AREA_THRESHOLD
MOTION_END_FRAMES = settings.MOTION_END_FRAMES
HSV_YELLOW_LOWER = settings.HSV_YELLOW_LOWER
HSV_YELLOW_UPPER = settings.HSV_YELLOW_UPPER
HSV_GREEN_LOWER = settings.HSV_GREEN_LOWER
HSV_GREEN_UPPER = settings.HSV_GREEN_UPPER
HSV_BLACK_LOWER = settings.HSV_BLACK_LOWER
HSV_BLACK_UPPER = settings.HSV_BLACK_UPPER
BG_HISTORY = settings.BG_HISTORY
BG_VAR_THRESHOLD = settings.BG_VAR_THRESHOLD
MODEL_PATH = settings.MODEL_PATH
SEGMENTATION_MODEL_PATH = settings.SEGMENTATION_MODEL_PATH
INPUT_SHAPE = settings.INPUT_SHAPE
NUM_CLASSES = settings.NUM_CLASSES
CLASS_NAMES = settings.CLASS_NAMES
ENABLE_SEGMENTATION = settings.ENABLE_SEGMENTATION
SEGMENTATION_THRESHOLD = settings.SEGMENTATION_THRESHOLD
OUTPUT_DIR = settings.OUTPUT_DIR
SAVE_INTERMEDIATE_STEPS = settings.SAVE_INTERMEDIATE_STEPS
DISPLAY_DEBUG = settings.DISPLAY_DEBUG
