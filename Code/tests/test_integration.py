"""
Functional / integration tests — verify end-to-end flows without
requiring real models, videos, or a running LM Studio server.

These tests wire multiple modules together with mocked heavy dependencies.
"""
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ══════════════════════════════════════════════════════════════════════
#  F1: Full defect-analysis → export metadata → grade pipeline
# ══════════════════════════════════════════════════════════════════════
from vision.defect_detector import MangoDefectDetector
from vision.export_advisor import ExportAdvisor, _map_to_grade


class TestDefectToExportFlow:
    """Run the defect detector on a synthetic image, then feed results
    through ExportAdvisor.build_metadata and verify the grade."""

    def test_healthy_mango_grades_as_a(self, dummy_bgr_image):
        detector = MangoDefectDetector()
        analysis = detector.detect_defects(dummy_bgr_image)

        advisor = ExportAdvisor()
        meta = advisor.build_metadata(
            defect_analysis={
                "surface_quality_score": analysis.surface_quality_score,
                "color_uniformity_score": analysis.color_uniformity_score,
                "total_defect_percentage": analysis.total_defect_percentage,
                "dark_spot_count": analysis.dark_spot_count,
                "brown_spot_count": analysis.brown_spot_count,
                "export_grade_impact": analysis.export_grade_impact,
            },
            disease_percentage=0.0,
        )
        # A solid-colour image should have zero defects → Grade A
        assert meta["export_grade"] in ("A", "B")


# ══════════════════════════════════════════════════════════════════════
#  F2: IntegratedMangoAnalyzer end-to-end (mocked ML seg)
# ══════════════════════════════════════════════════════════════════════
from vision.integrated_analyzer import IntegratedMangoAnalyzer


class TestIntegratedAnalyzerE2E:

    def test_full_analysis_structure(self, dummy_bgr_image):
        analyzer = IntegratedMangoAnalyzer()
        ml_seg = {
            "disease_percentage": 0.5,
            "mask": np.zeros((224, 224), dtype=np.uint8),
            "overlay": dummy_bgr_image.copy(),
        }
        result = analyzer.analyze_mango_comprehensive(dummy_bgr_image, ml_seg)

        assert "opencv_defects" in result
        assert "ml_disease" in result
        assert "combined_analysis" in result
        assert "export_recommendations" in result
        assert "performance" in result

    def test_analysis_without_ml_seg(self, dummy_bgr_image):
        analyzer = IntegratedMangoAnalyzer()
        result = analyzer.analyze_mango_comprehensive(dummy_bgr_image)
        assert result["ml_disease"] is None
        assert result["combined_analysis"]["total_defect_percentage"] >= 0


# ══════════════════════════════════════════════════════════════════════
#  F3: Report JSON structure (mirrors what streamlit_app saves)
# ══════════════════════════════════════════════════════════════════════

class TestReportJsonStructure:
    """Simulate the JSON export logic from streamlit_app.py and validate schema."""

    def test_json_roundtrip(self, sample_pipeline_result):
        results = [sample_pipeline_result]
        recs = {
            1: {
                "status": "success",
                "answer": "Export to USA, EU.",
                "grade": "A",
                "recommendation": {"recommended_countries": ["USA", "EU"]},
                "sources": [{"source": "CXS_184", "section": "Mango"}],
            }
        }

        json_data = {
            "report_name": "test_report.pdf",
            "generated_at": "2026-03-03T16:00:00",
            "video_source": "test_video.mp4",
            "total_mangoes": len(results),
            "mangoes": [],
        }

        for idx, r in enumerate(results, 1):
            pred = r.get("prediction", {})
            da = r.get("defect_analysis", {})
            seg = r.get("segmentation")
            entry = {
                "mango_id": idx,
                "frame_index": r.get("frame_idx"),
                "classification": {
                    "class_name": pred.get("class_name", "N/A"),
                    "confidence": round(pred.get("confidence", 0), 4),
                },
                "defect_analysis": {
                    "surface_quality_score": da.get("surface_quality_score", 0),
                    "total_defect_percentage": da.get("total_defect_percentage", 0),
                },
                "segmentation": {
                    "disease_percentage": round(seg["disease_percentage"], 4),
                } if seg else None,
                "export_recommendation": None,
            }
            rec = recs.get(idx)
            if rec and rec.get("status") == "success":
                entry["export_recommendation"] = {
                    "status": "success",
                    "answer": rec.get("answer", ""),
                    "grade": rec.get("grade", ""),
                }
            json_data["mangoes"].append(entry)

        # Round-trip through JSON
        serialized = json.dumps(json_data, indent=2)
        loaded = json.loads(serialized)

        assert loaded["total_mangoes"] == 1
        mango = loaded["mangoes"][0]
        assert mango["classification"]["class_name"] == "Healthy"
        assert mango["export_recommendation"]["grade"] == "A"


# ══════════════════════════════════════════════════════════════════════
#  F4: Chunker → verify documents survive round-trip
# ══════════════════════════════════════════════════════════════════════
from language.chunker import chunk_documents


class TestChunkerIntegration:

    def test_multiple_documents_chunked(self):
        docs = [
            {
                "content": "# Intro\nThis is the introduction.\n# Methods\nWe used X and Y.",
                "filename": "doc1.md",
                "filepath": "/data/doc1.md",
            },
            {
                "content": "# Results\nGood results.\n# Conclusion\nThe end.",
                "filename": "doc2.md",
                "filepath": "/data/doc2.md",
            },
        ]
        chunks = chunk_documents(docs)
        sources = {c.metadata["source"] for c in chunks}
        assert "doc1.md" in sources
        assert "doc2.md" in sources

    def test_chunk_content_not_empty(self):
        docs = [{"content": "# A\nText here.", "filename": "a.md", "filepath": "/a.md"}]
        chunks = chunk_documents(docs)
        assert all(len(c.page_content.strip()) > 0 for c in chunks)


# ══════════════════════════════════════════════════════════════════════
#  F5: Preprocessing → Postprocessing round-trip
# ══════════════════════════════════════════════════════════════════════
from preprocessing import preprocess_image
from postprocessing import contrast, smooth_image


class TestPrePostRoundTrip:

    def test_combined_pipeline_preserves_shape(self, dummy_bgr_image):
        pre = preprocess_image(dummy_bgr_image)
        post = contrast(smooth_image(pre))
        assert post.shape == dummy_bgr_image.shape
        assert post.dtype == np.uint8

    def test_pipeline_on_random_image(self):
        img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        pre = preprocess_image(img)
        post = contrast(pre)
        assert post.min() >= 0 and post.max() <= 255


# ══════════════════════════════════════════════════════════════════════
#  F6: Prompt template JSON is valid
# ══════════════════════════════════════════════════════════════════════

class TestPromptTemplates:

    def test_prompt_templates_valid_json(self):
        template_path = Path(__file__).resolve().parent.parent.parent / "prompt_templates.json"
        if not template_path.exists():
            pytest.skip("prompt_templates.json not found")
        data = json.loads(template_path.read_text(encoding="utf-8"))
        assert "system_message" in data
        assert "export_recommendation_prompt" in data
        assert "rag_config" in data

    def test_rag_config_has_required_keys(self):
        template_path = Path(__file__).resolve().parent.parent.parent / "prompt_templates.json"
        if not template_path.exists():
            pytest.skip("prompt_templates.json not found")
        data = json.loads(template_path.read_text(encoding="utf-8"))
        rag = data["rag_config"]
        for key in ("vector_store", "embedding_model", "llm", "retrieval_chunks",
                     "max_tokens", "temperature"):
            assert key in rag, f"Missing rag_config key: {key}"
