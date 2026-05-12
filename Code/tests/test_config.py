"""
Unit tests for config.settings — verifies default values, types, and
that the global `settings` instance is consistent.
"""
import pytest
from config.settings import Settings, settings, CLASS_NAMES, PROJECT_ROOT, CODE_DIR


class TestSettingsDefaults:
    """Verify that Settings class attributes have the expected defaults."""

    def test_roi_coords_is_tuple_of_four(self):
        assert isinstance(Settings.ROI_COORDS, tuple)
        assert len(Settings.ROI_COORDS) == 4

    def test_motion_area_threshold_positive(self):
        assert Settings.MOTION_AREA_THRESHOLD > 0

    def test_class_names_has_five_entries(self):
        assert len(Settings.CLASS_NAMES) == 5

    def test_class_names_contains_healthy(self):
        assert "Healthy" in Settings.CLASS_NAMES

    def test_num_classes_matches_class_names(self):
        assert Settings.NUM_CLASSES == len(Settings.CLASS_NAMES)

    def test_input_shape_is_3d(self):
        assert len(Settings.INPUT_SHAPE) == 3
        assert Settings.INPUT_SHAPE[2] == 3  # RGB channels

    def test_segmentation_threshold_in_range(self):
        assert 0.0 <= Settings.SEGMENTATION_THRESHOLD <= 1.0

    def test_hsv_ranges_are_lists_of_three(self):
        for attr in ("HSV_YELLOW_LOWER", "HSV_YELLOW_UPPER",
                      "HSV_GREEN_LOWER", "HSV_GREEN_UPPER",
                      "HSV_BLACK_LOWER", "HSV_BLACK_UPPER"):
            val = getattr(Settings, attr)
            assert isinstance(val, list) and len(val) == 3, f"{attr} must be a list of 3 ints"

    def test_bg_history_positive(self):
        assert Settings.BG_HISTORY > 0


class TestGlobalInstance:
    """Ensure module-level exports mirror the Settings class."""

    def test_global_instance_type(self):
        assert isinstance(settings, Settings)

    def test_class_names_export(self):
        assert CLASS_NAMES == Settings.CLASS_NAMES

    def test_project_root_exists(self):
        assert PROJECT_ROOT.exists()

    def test_code_dir_exists(self):
        assert CODE_DIR.exists()
