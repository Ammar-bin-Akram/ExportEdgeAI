"""
Unit tests for preprocessing.py — image transforms used before classification.
"""
import numpy as np
import pytest

from preprocessing import (
    sharpen,
    enhance_contrast,
    remove_shadow,
    denoise_light,
    color_correct,
    unsharp_mask,
    resize_high_quality,
    preprocess_image,
    preprocess_image_minimal,
)


class TestIndividualTransforms:
    """Each transform must preserve dtype / channels and not crash."""

    @pytest.mark.parametrize("fn", [
        sharpen, enhance_contrast, remove_shadow, denoise_light,
        color_correct, preprocess_image, preprocess_image_minimal,
    ])
    def test_output_shape_matches_input(self, dummy_bgr_image, fn):
        result = fn(dummy_bgr_image)
        assert result.shape == dummy_bgr_image.shape

    @pytest.mark.parametrize("fn", [
        sharpen, enhance_contrast, remove_shadow, denoise_light,
        color_correct, preprocess_image, preprocess_image_minimal,
    ])
    def test_output_dtype_uint8(self, dummy_bgr_image, fn):
        result = fn(dummy_bgr_image)
        assert result.dtype == np.uint8

    def test_unsharp_mask_shape(self, dummy_bgr_image):
        result = unsharp_mask(dummy_bgr_image, sigma=1.0, strength=0.3)
        assert result.shape == dummy_bgr_image.shape

    def test_unsharp_mask_dtype(self, dummy_bgr_image):
        result = unsharp_mask(dummy_bgr_image, sigma=1.0, strength=0.3)
        assert result.dtype == np.uint8


class TestResizeHighQuality:

    def test_default_target_size(self, dummy_bgr_image):
        result = resize_high_quality(dummy_bgr_image)
        assert result.shape[:2] == (150, 150)

    def test_custom_target_size(self, dummy_bgr_image):
        result = resize_high_quality(dummy_bgr_image, target_size=(100, 100))
        assert result.shape[:2] == (100, 100)

    def test_preserves_channels(self, dummy_bgr_image):
        result = resize_high_quality(dummy_bgr_image)
        assert result.shape[2] == 3


class TestPipelineOrdering:
    """Full pipelines should produce valid images."""

    def test_preprocess_image_is_valid(self, dummy_bgr_image):
        result = preprocess_image(dummy_bgr_image)
        assert result.min() >= 0 and result.max() <= 255

    def test_preprocess_image_minimal_is_valid(self, dummy_bgr_image):
        result = preprocess_image_minimal(dummy_bgr_image)
        assert result.min() >= 0 and result.max() <= 255
