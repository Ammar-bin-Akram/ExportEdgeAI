import cv2
import torch
import sys
sys.path.insert(0, 'D:/FYP/Code')
from models.segmentation import SegmentationModel
from config.settings import Settings

settings = Settings()
model = SegmentationModel(settings=settings)
model.load()

# Create a test image
test_img = cv2.imread('D:/FYP/relevant_frames1/peak_000156_7750_original.jpg')
roi = test_img[31:465, 162:498]  # Extract ROI manually
print(f'ROI shape: {roi.shape}')

# Try segmentation
mask, overlay, pct = model.segment(roi)
print(f'Mask shape: {mask.shape}')
print(f'Disease percentage: {pct}')
