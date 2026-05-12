"""
Unit tests for core/pipeline_factory.py — Factory pattern & abstract pipeline.
"""
import pytest
import numpy as np

from core.base_pipeline import BaseFruitPipeline
from core.pipeline_factory import PipelineFactory


# ── A concrete dummy pipeline for testing ─────────────────────────────

class _DummyPipeline(BaseFruitPipeline):
    """Minimal concrete implementation used only for testing."""

    def load_models(self):
        pass

    def preprocess_image(self, image):
        return image

    def classify(self, preprocessed_image):
        return ("Healthy", 0.95)

    def get_fruit_type(self):
        return "test_fruit"


class TestPipelineFactory:

    def setup_method(self):
        """Clear the registry before each test."""
        PipelineFactory._pipelines.clear()

    def test_register_and_create(self):
        PipelineFactory.register_pipeline("test_fruit", _DummyPipeline)
        pipeline = PipelineFactory.create_pipeline("test_fruit")
        assert isinstance(pipeline, _DummyPipeline)

    def test_case_insensitive_registration(self):
        PipelineFactory.register_pipeline("Mango", _DummyPipeline)
        pipeline = PipelineFactory.create_pipeline("mango")
        assert isinstance(pipeline, _DummyPipeline)

    def test_unknown_fruit_raises(self):
        with pytest.raises(ValueError, match="Unknown fruit type"):
            PipelineFactory.create_pipeline("banana")

    def test_get_available_fruits(self):
        PipelineFactory.register_pipeline("apple", _DummyPipeline)
        PipelineFactory.register_pipeline("mango", _DummyPipeline)
        fruits = PipelineFactory.get_available_fruits()
        assert set(fruits) == {"apple", "mango"}

    def test_empty_registry(self):
        assert PipelineFactory.get_available_fruits() == []


class TestBasePipelineInterface:
    """Test the concrete methods inherited from BaseFruitPipeline."""

    @pytest.fixture
    def pipeline(self):
        from types import SimpleNamespace
        p = _DummyPipeline()
        p.settings = SimpleNamespace(ENABLE_SEGMENTATION=False)
        return p

    def test_process_single_frame(self, pipeline, dummy_bgr_image):
        result = pipeline.process_single_frame(dummy_bgr_image)
        assert result["class_name"] == "Healthy"
        assert result["confidence"] == 0.95
        assert result["original"] is dummy_bgr_image

    def test_segment_if_needed_disabled(self, pipeline, dummy_bgr_image):
        """With no settings or segmentation model, returns (None, None, 0)."""
        # settings is None → the call would raise, so let's give it one
        from types import SimpleNamespace
        pipeline.settings = SimpleNamespace(ENABLE_SEGMENTATION=False)
        mask, overlay, pct = pipeline.segment_if_needed(dummy_bgr_image, "Healthy", 0.9)
        assert mask is None and overlay is None and pct == 0

    def test_process_video_raises_without_extractor(self, pipeline):
        with pytest.raises(RuntimeError, match="Frame extractor not initialized"):
            pipeline.process_video("nonexistent.mp4")
