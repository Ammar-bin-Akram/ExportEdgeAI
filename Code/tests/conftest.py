"""
Shared fixtures for the mango inspection test suite.
Run all tests from the Code/ directory:
    pytest tests/ -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# ── Ensure the Code/ directory is on the import path ───────────────────
CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


# ── Reusable image fixtures ───────────────────────────────────────────

@pytest.fixture
def dummy_bgr_image():
    """A 224×224 BGR image filled with a mango-like orange colour."""
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img[:, :] = (30, 130, 210)  # BGR: orange-ish
    return img


@pytest.fixture
def dummy_bgr_image_small():
    """A 150×150 BGR image for model-input-size tests."""
    img = np.zeros((150, 150, 3), dtype=np.uint8)
    img[:, :] = (20, 120, 200)
    return img


@pytest.fixture
def dummy_binary_mask():
    """A 224×224 binary mask with a known white rectangle (50×50)."""
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[80:130, 80:130] = 255  # 50×50 white region
    return mask


@pytest.fixture
def dummy_grayscale():
    """A 224×224 single-channel grayscale image."""
    return np.full((224, 224), 128, dtype=np.uint8)


@pytest.fixture
def sample_defect_analysis_dict():
    """Realistic defect analysis dict as produced by the pipeline."""
    return {
        "surface_quality_score": 87.0,
        "color_uniformity_score": 91.0,
        "total_defect_percentage": 1.23,
        "defect_count": 2,
        "dark_spot_count": 1,
        "brown_spot_count": 1,
        "export_grade_impact": "minimal",
    }


@pytest.fixture
def sample_pipeline_result(dummy_bgr_image, sample_defect_analysis_dict):
    """A single mango result dict matching the streamlit pipeline format."""
    return {
        "frame_idx": 0,
        "timestamp_ms": 1500.0,
        "original": dummy_bgr_image,
        "preprocessed": dummy_bgr_image,
        "prediction": {
            "class_name": "Healthy",
            "confidence": 0.9512,
            "class_idx": 3,
        },
        "defect_analysis": sample_defect_analysis_dict,
        "segmentation": {
            "disease_percentage": 0.45,
            "mask": np.zeros((224, 224), dtype=np.uint8),
            "overlay": dummy_bgr_image.copy(),
            "segmentation_time": 0.123,
        },
    }
