"""Configuration constants for mango disease detection pipeline"""
import os
from pathlib import Path

# Get the project root directory (parent of Code folder)
PROJECT_ROOT = Path(__file__).parent.parent
CODE_DIR = Path(__file__).parent

# Video settings
# Default file used when user selects "Video File"
VIDEO_SOURCE = str(PROJECT_ROOT / "live_recording1.mp4")

# Default live stream used when user selects "Camera Feed"
# CAMERA_SOURCE = ""
CAMERA_SOURCE = "rtsp://admin:admin123@192.168.1.108:554/cam/realmonitor?channel=1&subtype=1"

# ROI coordinates (x1, y1, x2, y2)
ROI_COORDS = (162, 31, 498, 465)

# Motion detection parameters
MOTION_AREA_THRESHOLD = 5000
MOTION_END_FRAMES = 5

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
SEGMENTATION_MODEL_PATH = str(PROJECT_ROOT / "models" / "deeplabv3_mobilenet_mango.pth")
INPUT_SHAPE = (150, 150, 3)
NUM_CLASSES = 5
CLASS_NAMES = ['Alternaria', 'Anthracnose', 'Black Mould Rot', 'Healthy', 'Stem end Rot']

# Segmentation settings
ENABLE_SEGMENTATION = True  # Set to False to disable segmentation
SEGMENTATION_THRESHOLD = 0.3  # Only segment if disease confidence > 50%

# Output settings
OUTPUT_DIR = str(PROJECT_ROOT / "relevant_frames1")
SAVE_INTERMEDIATE_STEPS = True
DISPLAY_DEBUG = True
