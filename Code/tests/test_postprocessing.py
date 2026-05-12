"""
Unit tests for postprocessing.py — frame-level image enhancement transforms.
"""
import numpy as np
import pytest

from postprocessing import blur, contrast, deblur, remove_shadow, smooth_image


class TestPostprocessingTransforms:
    """Every postprocessing function must keep shape, dtype, and value range."""

    @pytest.mark.parametrize("fn", [blur, contrast, deblur, remove_shadow, smooth_image])
    def test_preserves_shape(self, dummy_bgr_image, fn):
        result = fn(dummy_bgr_image)
        assert result.shape == dummy_bgr_image.shape

    @pytest.mark.parametrize("fn", [blur, contrast, deblur, remove_shadow, smooth_image])
    def test_preserves_dtype(self, dummy_bgr_image, fn):
        result = fn(dummy_bgr_image)
        assert result.dtype == np.uint8

    @pytest.mark.parametrize("fn", [blur, contrast, deblur, remove_shadow, smooth_image])
    def test_pixel_values_in_range(self, dummy_bgr_image, fn):
        result = fn(dummy_bgr_image)
        assert result.min() >= 0 and result.max() <= 255

    @pytest.mark.parametrize("fn", [blur, contrast, deblur, remove_shadow, smooth_image])
    def test_handles_small_image(self, fn):
        small = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        result = fn(small)
        assert result.shape == small.shape
