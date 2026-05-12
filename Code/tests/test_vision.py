"""
Unit tests for vision modules:
  - defect_detector (DefectRegion, DefectAnalysis, MangoDefectDetector)
  - export_advisor  (_map_to_grade, parse_structured_recommendation, build_metadata)
  - integrated_analyzer (_generate_quality_assessment, _predict_export_suitability)
  - report_generator helpers (_img_to_b64, _ns)
  - segmentation_utils helpers (create_segmentation_overlay, get_disease_statistics)
"""
import numpy as np
import pytest
from dataclasses import asdict

# ══════════════════════════════════════════════════════════════════════
#  DefectRegion & DefectAnalysis dataclasses
# ══════════════════════════════════════════════════════════════════════
from vision.defect_detector import DefectRegion, DefectAnalysis


class TestDefectRegion:

    def test_fields_present(self):
        region = DefectRegion(
            type="dark_spot",
            contour=np.array([[0, 0]]),
            area=100.0,
            area_pct=0.5,
            center=(50, 50),
            severity="minor",
            bounding_box=(40, 40, 20, 20),
            confidence=0.88,
            mean_intensity=45.0,
        )
        assert region.type == "dark_spot"
        assert region.severity == "minor"
        assert region.confidence == 0.88

    def test_default_mean_intensity(self):
        region = DefectRegion(
            type="brown_spot", contour=np.array([[0, 0]]),
            area=50.0, area_pct=0.25, center=(10, 10),
            severity="moderate", bounding_box=(5, 5, 10, 10),
            confidence=0.7,
        )
        assert region.mean_intensity == 0.0


class TestDefectAnalysis:

    def test_construction(self):
        analysis = DefectAnalysis(
            total_defect_area=200.0,
            total_defect_percentage=1.5,
            mango_area=13000.0,
            defect_count=2,
            dark_spot_count=1,
            brown_spot_count=1,
            defect_regions=[],
            color_uniformity_score=90.0,
            surface_quality_score=87.0,
            export_grade_impact="minimal",
            processing_time=0.05,
        )
        assert analysis.defect_count == 2
        assert analysis.export_grade_impact == "minimal"


# ══════════════════════════════════════════════════════════════════════
#  MangoDefectDetector — lightweight tests (no real images needed)
# ══════════════════════════════════════════════════════════════════════
from vision.defect_detector import MangoDefectDetector


class TestMangoDefectDetector:

    @pytest.fixture
    def detector(self):
        return MangoDefectDetector()

    def test_default_config_keys(self, detector):
        assert "dark_threshold" in detector.config
        assert "brown_hue_low" in detector.config

    def test_detect_defects_returns_analysis(self, detector, dummy_bgr_image):
        result = detector.detect_defects(dummy_bgr_image)
        assert isinstance(result, DefectAnalysis)
        assert result.processing_time >= 0

    def test_detect_defects_scores_in_range(self, detector, dummy_bgr_image):
        result = detector.detect_defects(dummy_bgr_image)
        assert 0 <= result.color_uniformity_score <= 100
        assert 0 <= result.surface_quality_score <= 100

    def test_detect_defects_grade_impact_valid(self, detector, dummy_bgr_image):
        result = detector.detect_defects(dummy_bgr_image)
        assert result.export_grade_impact in ("minimal", "moderate", "significant")

    def test_calculate_surface_quality_no_defects(self, detector):
        score = detector.calculate_surface_quality_score([])
        assert score == 100.0

    def test_analyze_color_uniformity_returns_number(self, detector, dummy_bgr_image):
        # Create a simple mask
        mask = np.ones((224, 224), dtype=np.uint8) * 255
        score = detector.analyze_color_uniformity(dummy_bgr_image, mask)
        assert isinstance(score, (int, float))
        assert 0 <= score <= 100


# ══════════════════════════════════════════════════════════════════════
#  ExportAdvisor — _map_to_grade, parse_structured_recommendation, build_metadata
# ══════════════════════════════════════════════════════════════════════
from vision.export_advisor import _map_to_grade, parse_structured_recommendation, ExportAdvisor


class TestMapToGrade:

    def test_grade_a(self):
        assert _map_to_grade(92.0, "minimal", 1.0) == "A"

    def test_grade_b_moderate_impact(self):
        assert _map_to_grade(80.0, "moderate", 3.5) == "B"

    def test_grade_b_minimal_impact_higher_defect(self):
        assert _map_to_grade(85.0, "minimal", 4.0) == "B"

    def test_grade_c_significant(self):
        assert _map_to_grade(60.0, "significant", 8.0) == "C"

    def test_grade_c_high_defect(self):
        assert _map_to_grade(80.0, "minimal", 6.0) == "C"


class TestParseStructuredRecommendation:

    SAMPLE_LLM_OUTPUT = """\
**RECOMMENDED COUNTRIES:**
- USA (Grade A tolerance up to 2% defects)
- EU (Codex standard CXS 184)
- Japan (JAS standard, grade A)

**NOT RECOMMENDED:**
- None

**ACTIONABLE STEPS:**
1. Ensure hot water treatment at 48°C for 60 minutes
2. Apply appropriate labelling per EU Reg 543/2011
3. Obtain phytosanitary certificate

**CONDITIONS:**
- Must meet ISPM 15 packaging requirements
- Irradiation certificate required for USA
"""

    def test_returns_dict_with_four_keys(self):
        result = parse_structured_recommendation(self.SAMPLE_LLM_OUTPUT)
        assert set(result.keys()) == {
            "recommended_countries", "not_recommended",
            "actionable_steps", "conditions",
        }

    def test_recommended_countries_parsed(self):
        result = parse_structured_recommendation(self.SAMPLE_LLM_OUTPUT)
        assert len(result["recommended_countries"]) == 3
        assert any("USA" in c for c in result["recommended_countries"])

    def test_not_recommended_empty_when_none(self):
        result = parse_structured_recommendation(self.SAMPLE_LLM_OUTPUT)
        assert result["not_recommended"] == []

    def test_actionable_steps_parsed(self):
        result = parse_structured_recommendation(self.SAMPLE_LLM_OUTPUT)
        assert len(result["actionable_steps"]) >= 2

    def test_conditions_parsed(self):
        result = parse_structured_recommendation(self.SAMPLE_LLM_OUTPUT)
        assert len(result["conditions"]) >= 1

    def test_empty_input(self):
        result = parse_structured_recommendation("")
        assert all(v == [] for v in result.values())


class TestExportAdvisorBuildMetadata:

    def test_returns_all_expected_keys(self, sample_defect_analysis_dict):
        advisor = ExportAdvisor()
        meta = advisor.build_metadata(sample_defect_analysis_dict, disease_percentage=0.45)
        expected_keys = {
            "surface_quality_score", "color_uniformity_score",
            "total_defect_percentage", "dark_spot_count", "brown_spot_count",
            "export_grade_impact", "export_grade", "disease_percentage",
        }
        assert set(meta.keys()) == expected_keys

    def test_grade_derived_correctly(self, sample_defect_analysis_dict):
        advisor = ExportAdvisor()
        meta = advisor.build_metadata(sample_defect_analysis_dict)
        assert meta["export_grade"] == "A"  # minimal impact, 1.23% defect → Grade A

    def test_empty_dict_defaults(self):
        advisor = ExportAdvisor()
        meta = advisor.build_metadata({})
        assert meta["surface_quality_score"] == 0.0
        assert meta["export_grade"] == "C"  # unknown impact → C


# ══════════════════════════════════════════════════════════════════════
#  IntegratedMangoAnalyzer — quality assessment & export suitability
# ══════════════════════════════════════════════════════════════════════
from vision.integrated_analyzer import IntegratedMangoAnalyzer


class TestIntegratedAnalyzerGrading:

    @pytest.fixture
    def analyzer(self):
        return IntegratedMangoAnalyzer()

    def _make_opencv_result(self, defect_pct, uniformity, surface_quality, regions=None):
        """Build a minimal DefectAnalysis for testing."""
        return DefectAnalysis(
            total_defect_area=0,
            total_defect_percentage=defect_pct,
            mango_area=50000,
            defect_count=0,
            dark_spot_count=0,
            brown_spot_count=0,
            defect_regions=regions or [],
            color_uniformity_score=uniformity,
            surface_quality_score=surface_quality,
            export_grade_impact="minimal",
            processing_time=0.01,
        )

    def test_grade_a_premium(self, analyzer):
        cv_result = self._make_opencv_result(1.0, 90, 95)
        qa = analyzer._generate_quality_assessment(cv_result, None, 1.0)
        assert "Grade A" in qa["grade_category"]

    def test_grade_b_standard(self, analyzer):
        cv_result = self._make_opencv_result(3.0, 75, 80)
        qa = analyzer._generate_quality_assessment(cv_result, None, 3.0)
        assert "Grade B" in qa["grade_category"]

    def test_grade_c_local(self, analyzer):
        cv_result = self._make_opencv_result(7.0, 60, 50)
        qa = analyzer._generate_quality_assessment(cv_result, None, 7.0)
        assert "Grade C" in qa["grade_category"]

    def test_processing_grade(self, analyzer):
        cv_result = self._make_opencv_result(15.0, 40, 30)
        qa = analyzer._generate_quality_assessment(cv_result, None, 15.0)
        assert "Processing" in qa["grade_category"]

    def test_overall_score_in_range(self, analyzer):
        cv_result = self._make_opencv_result(2.0, 85, 88)
        qa = analyzer._generate_quality_assessment(cv_result, None, 2.0)
        assert 0 <= qa["overall_score"] <= 100

    def test_issues_list_type(self, analyzer):
        cv_result = self._make_opencv_result(0.5, 95, 98)
        qa = analyzer._generate_quality_assessment(cv_result, None, 0.5)
        assert isinstance(qa["issues"], list)


class TestExportSuitability:

    @pytest.fixture
    def analyzer(self):
        return IntegratedMangoAnalyzer()

    def test_grade_a_markets(self, analyzer):
        qa = {"grade_category": "Grade A (Premium Export)", "overall_score": 95, "issues": []}
        recs = analyzer._predict_export_suitability(qa)
        assert "USA" in recs["suitable_markets"]
        assert recs["restrictions"] == []

    def test_grade_b_markets(self, analyzer):
        qa = {"grade_category": "Grade B (Standard Export)", "overall_score": 75, "issues": []}
        recs = analyzer._predict_export_suitability(qa)
        assert "Middle East" in recs["suitable_markets"]

    def test_rag_feature_description_generated(self, analyzer):
        qa = {"grade_category": "Grade A (Premium Export)", "overall_score": 92, "issues": []}
        recs = analyzer._predict_export_suitability(qa)
        assert isinstance(recs["feature_description"], str)
        assert len(recs["feature_description"]) > 0


# ══════════════════════════════════════════════════════════════════════
#  ReportGenerator helpers — _img_to_b64, _ns
# ══════════════════════════════════════════════════════════════════════
from vision.report_generator import _img_to_b64, _ns


class TestImgToB64:

    def test_returns_non_empty_string(self, dummy_bgr_image):
        result = _img_to_b64(dummy_bgr_image)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_valid_base64(self, dummy_bgr_image):
        import base64
        b64 = _img_to_b64(dummy_bgr_image)
        decoded = base64.b64decode(b64)
        assert decoded[:4] == b"\x89PNG"  # PNG magic bytes


class TestNsHelper:

    def test_simple_dict(self):
        ns = _ns({"a": 1, "b": "hello"})
        assert ns.a == 1
        assert ns.b == "hello"

    def test_nested_dict(self):
        ns = _ns({"outer": {"inner": 42}})
        assert ns.outer.inner == 42

    def test_keep_dict_keys(self):
        ns = _ns({"probs": {"A": 0.9, "B": 0.1}}, keep_dict_keys={"probs"})
        assert isinstance(ns.probs, dict)
        assert ns.probs["A"] == 0.9


# ══════════════════════════════════════════════════════════════════════
#  segmentation_utils — pure helper functions
# ══════════════════════════════════════════════════════════════════════
from segmentation_utils import (
    create_segmentation_overlay,
    extract_diseased_region,
    get_disease_statistics,
)


class TestCreateSegmentationOverlay:

    def test_output_shape(self, dummy_bgr_image, dummy_binary_mask):
        overlay = create_segmentation_overlay(dummy_bgr_image, dummy_binary_mask)
        assert overlay.shape == dummy_bgr_image.shape

    def test_output_dtype(self, dummy_bgr_image, dummy_binary_mask):
        overlay = create_segmentation_overlay(dummy_bgr_image, dummy_binary_mask)
        assert overlay.dtype == np.uint8


class TestExtractDiseasedRegion:

    def test_masked_region_zeros_outside(self, dummy_bgr_image, dummy_binary_mask):
        result = extract_diseased_region(dummy_bgr_image, dummy_binary_mask)
        # Pixels where mask=0 should be black
        assert result[0, 0].sum() == 0  # top-left corner is outside the mask rect

    def test_preserves_shape(self, dummy_bgr_image, dummy_binary_mask):
        result = extract_diseased_region(dummy_bgr_image, dummy_binary_mask)
        assert result.shape == dummy_bgr_image.shape


class TestGetDiseaseStatistics:

    def test_empty_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        stats = get_disease_statistics(mask)
        assert stats["num_regions"] == 0
        assert stats["total_area"] == 0

    def test_single_region(self, dummy_binary_mask):
        stats = get_disease_statistics(dummy_binary_mask)
        assert stats["num_regions"] >= 1
        assert stats["total_area"] > 0

    def test_bounding_boxes_present(self, dummy_binary_mask):
        stats = get_disease_statistics(dummy_binary_mask)
        assert isinstance(stats["bounding_boxes"], list)
