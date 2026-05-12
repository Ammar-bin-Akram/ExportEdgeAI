# ExportEdge AI — Test Suite Documentation

> **Project:** ExportEdge AI — Mango Quality Inspection & Export Advisory System  
> **Framework:** pytest 9.0.2  
> **Total tests:** 337  
> **Run command:** `python tests/generate_report.py <category>` from `Code/`

---

## Table of Contents

1. [Test Suite Overview](#1-test-suite-overview)
2. [Fixtures & Shared Test Data](#2-fixtures--shared-test-data)
3. [Unit Tests — Configuration (`test_config.py`)](#3-unit-tests--configuration)
4. [Unit Tests — Preprocessing (`test_preprocessing.py`)](#4-unit-tests--preprocessing)
5. [Unit Tests — Postprocessing (`test_postprocessing.py`)](#5-unit-tests--postprocessing)
6. [Unit Tests — Pipeline Factory (`test_pipeline_factory.py`)](#6-unit-tests--pipeline-factory)
7. [Unit Tests — Vision Modules (`test_vision.py`)](#7-unit-tests--vision-modules)
8. [Unit Tests — Regex & Chunker (`test_regex_and_chunker.py`)](#8-unit-tests--regex--chunker)
9. [Data Validation Tests (`test_data_validation.py`)](#9-data-validation-tests)
10. [Performance Tests (`test_performance.py`)](#10-performance-tests)
11. [Integration Tests — Functional Flows (`test_integration.py`)](#11-integration-tests--functional-flows)
12. [Integration Tests — Full Pipeline (`test_integration_pipeline.py`)](#12-integration-tests--full-pipeline)
13. [Reliability Tests (`test_reliability.py`)](#13-reliability-tests)
14. [Streamlit UI Tests (`test_streamlit_app.py`)](#14-streamlit-ui-tests)

---

## 1. Test Suite Overview

| File | Category | Tests | Module(s) Under Test |
|---|---|---|---|
| `test_config.py` | Unit | 11 | `config.settings` |
| `test_preprocessing.py` | Unit | 19 | `preprocessing` |
| `test_postprocessing.py` | Unit | 25 | `postprocessing` |
| `test_pipeline_factory.py` | Unit | 8 | `core.pipeline_factory`, `core.base_pipeline` |
| `test_vision.py` | Unit | 40 | `vision.defect_detector`, `vision.export_advisor`, `vision.integrated_analyzer`, `vision.report_generator`, `segmentation_utils` |
| `test_regex_and_chunker.py` | Unit | 19 | `regex_checking`, `language.chunker` |
| `test_data_validation.py` | Data Validation | 81 | `language.chunker` + real SOP files |
| `test_performance.py` | Performance | 13 | `vision.defect_detector`, `vision.export_advisor`, `language.chunker`, FAISS |
| `test_integration.py` | Integration | 10 | Multi-module end-to-end flows |
| `test_integration_pipeline.py` | Integration | 49 | Full pipeline: vision → RAG → LLM |
| `test_reliability.py` | Reliability | 25 | All deterministic components |
| `test_streamlit_app.py` | UI / Streamlit | 49 | `streamlit_app.py` via `AppTest` |
| **Total** | | **349** | |

---

## 2. Fixtures & Shared Test Data

Defined in `conftest.py`. All image fixtures are synthetic NumPy arrays — no real camera or model files required.

| Fixture | Type | Description |
|---|---|---|
| `dummy_bgr_image` | `ndarray (224, 224, 3) uint8` | Solid mid-grey BGR image; used as a stand-in for a mango frame |
| `dummy_bgr_image_small` | `ndarray (150, 150, 3) uint8` | Smaller grey image for resize tests |
| `dummy_binary_mask` | `ndarray (224, 224) uint8` | Binary mask with a white rectangle in the centre (simulates a segmented region) |
| `dummy_grayscale` | `ndarray (224, 224) uint8` | Greyscale version of `dummy_bgr_image` |
| `sample_defect_analysis_dict` | `dict` | Realistic defect analysis dict: `surface_quality_score=87`, `color_uniformity_score=91`, `total_defect_percentage=1.23`, `export_grade_impact="minimal"` |
| `sample_pipeline_result` | `dict` | Full mango inspection result: classification "Healthy", confidence 0.95, segmentation mask, defect analysis |

---

## 3. Unit Tests — Configuration

**File:** `test_config.py`  
**Module under test:** `config.settings` (`Settings` class, `settings` singleton, `CLASS_NAMES`, `PROJECT_ROOT`, `CODE_DIR`)

### Class `TestSettingsDefaults`

Verifies that every `Settings` class attribute has the expected type and value range.

| Test | Input | Expected Output |
|---|---|---|
| `test_roi_coords_is_tuple_of_four` | `Settings.ROI_COORDS` | `isinstance(…, tuple)` and `len == 4` |
| `test_motion_area_threshold_positive` | `Settings.MOTION_AREA_THRESHOLD` | `> 0` |
| `test_class_names_has_five_entries` | `Settings.CLASS_NAMES` | `len == 5` |
| `test_class_names_contains_healthy` | `Settings.CLASS_NAMES` | `"Healthy" in CLASS_NAMES` |
| `test_num_classes_matches_class_names` | `Settings.NUM_CLASSES`, `Settings.CLASS_NAMES` | `NUM_CLASSES == len(CLASS_NAMES)` |
| `test_input_shape_is_3d` | `Settings.INPUT_SHAPE` | `len == 3` and `INPUT_SHAPE[2] == 3` (RGB channels) |
| `test_segmentation_threshold_in_range` | `Settings.SEGMENTATION_THRESHOLD` | `0.0 ≤ value ≤ 1.0` |
| `test_hsv_ranges_are_lists_of_three` | Six HSV boundary attributes (yellow/green/black lower & upper) | Each is a `list` with exactly 3 elements |
| `test_bg_history_positive` | `Settings.BG_HISTORY` | `> 0` |

### Class `TestGlobalInstance`

Verifies the module-level exports mirror the class.

| Test | Input | Expected Output |
|---|---|---|
| `test_global_instance_type` | `settings` singleton | `isinstance(settings, Settings)` |
| `test_class_names_export` | `CLASS_NAMES` export | `CLASS_NAMES == Settings.CLASS_NAMES` |
| `test_project_root_exists` | `PROJECT_ROOT` path | `Path.exists() == True` |
| `test_code_dir_exists` | `CODE_DIR` path | `Path.exists() == True` |

---

## 4. Unit Tests — Preprocessing

**File:** `test_preprocessing.py`  
**Module under test:** `preprocessing` — image enhancement functions run before classification.

### Class `TestIndividualTransforms`

Parametrised over 7 functions: `sharpen`, `enhance_contrast`, `remove_shadow`, `denoise_light`, `color_correct`, `preprocess_image`, `preprocess_image_minimal`.

| Test | Input | Expected Output |
|---|---|---|
| `test_output_shape_matches_input` | `dummy_bgr_image (224×224×3)` passed through each fn | Output `shape == (224, 224, 3)` |
| `test_output_dtype_uint8` | Same | Output `dtype == uint8` |
| `test_unsharp_mask_shape` | `dummy_bgr_image`, `sigma=1.0`, `strength=0.3` | Output `shape == (224, 224, 3)` |
| `test_unsharp_mask_dtype` | Same | Output `dtype == uint8` |

### Class `TestResizeHighQuality`

| Test | Input | Expected Output |
|---|---|---|
| `test_default_target_size` | `dummy_bgr_image` with no target size arg | Output `shape[:2] == (150, 150)` |
| `test_custom_target_size` | `dummy_bgr_image`, `target_size=(100, 100)` | Output `shape[:2] == (100, 100)` |
| `test_preserves_channels` | `dummy_bgr_image` | Output `shape[2] == 3` (channels preserved) |

### Class `TestPipelineOrdering`

| Test | Input | Expected Output |
|---|---|---|
| `test_preprocess_image_is_valid` | `dummy_bgr_image` through `preprocess_image` | All pixels in `[0, 255]` |
| `test_preprocess_image_minimal_is_valid` | `dummy_bgr_image` through `preprocess_image_minimal` | All pixels in `[0, 255]` |

---

## 5. Unit Tests — Postprocessing

**File:** `test_postprocessing.py`  
**Module under test:** `postprocessing` — frame-level enhancement after capture.  
**Functions tested:** `blur`, `contrast`, `deblur`, `remove_shadow`, `smooth_image`

### Class `TestPostprocessingTransforms`

All four sub-tests are parametrised across all five functions.

| Test | Input | Expected Output |
|---|---|---|
| `test_preserves_shape` | `dummy_bgr_image (224×224×3)` | Output `shape == input.shape` |
| `test_preserves_dtype` | `dummy_bgr_image` | Output `dtype == uint8` |
| `test_pixel_values_in_range` | `dummy_bgr_image` | All pixels `∈ [0, 255]` |
| `test_handles_small_image` | Random `(32×32×3) uint8` array | Output `shape == (32, 32, 3)` — confirms no hard-coded size dependency |

---

## 6. Unit Tests — Pipeline Factory

**File:** `test_pipeline_factory.py`  
**Modules under test:** `core.pipeline_factory.PipelineFactory`, `core.base_pipeline.BaseFruitPipeline`

A local `_DummyPipeline` concrete subclass is defined for testing. It always classifies as `"Healthy"` with confidence `0.95`.

### Class `TestPipelineFactory`

| Test | Input | Expected Output |
|---|---|---|
| `test_register_and_create` | Register `"test_fruit"` → `_DummyPipeline`; call `create_pipeline("test_fruit")` | Returns a `_DummyPipeline` instance |
| `test_case_insensitive_registration` | Register `"Mango"`, then `create_pipeline("mango")` | Returns `_DummyPipeline` (key normalised to lowercase) |
| `test_unknown_fruit_raises` | `create_pipeline("banana")` on empty registry | Raises `ValueError` matching `"Unknown fruit type"` |
| `test_get_available_fruits` | Register `"apple"` and `"mango"` | `get_available_fruits()` returns `{"apple", "mango"}` |
| `test_empty_registry` | Fresh registry (cleared in `setup_method`) | `get_available_fruits() == []` |

### Class `TestBasePipelineInterface`

Uses `_DummyPipeline` with `settings.ENABLE_SEGMENTATION = False`.

| Test | Input | Expected Output |
|---|---|---|
| `test_process_single_frame` | `dummy_bgr_image` | Result dict has `class_name="Healthy"`, `confidence=0.95`, `original is dummy_bgr_image` |
| `test_segment_if_needed_disabled` | `dummy_bgr_image`, class `"Healthy"`, conf `0.9`, segmentation disabled | Returns `(None, None, 0)` |
| `test_process_video_raises_without_extractor` | `"nonexistent.mp4"` path, no frame extractor set | Raises `RuntimeError` matching `"Frame extractor not initialized"` |

---

## 7. Unit Tests — Vision Modules

**File:** `test_vision.py`  
**Modules under test:** `vision.defect_detector`, `vision.export_advisor`, `vision.integrated_analyzer`, `vision.report_generator`, `segmentation_utils`

---

### `DefectRegion` and `DefectAnalysis` dataclasses

#### Class `TestDefectRegion`

| Test | Input | Expected Output |
|---|---|---|
| `test_fields_present` | `DefectRegion(type="dark_spot", area=100, severity="minor", confidence=0.88, …)` | `region.type == "dark_spot"`, `region.severity == "minor"`, `region.confidence == 0.88` |
| `test_default_mean_intensity` | `DefectRegion` constructed without `mean_intensity` | `region.mean_intensity == 0.0` |

#### Class `TestDefectAnalysis`

| Test | Input | Expected Output |
|---|---|---|
| `test_construction` | `DefectAnalysis(defect_count=2, export_grade_impact="minimal", …)` | `analysis.defect_count == 2`, `analysis.export_grade_impact == "minimal"` |

---

### `MangoDefectDetector`

#### Class `TestMangoDefectDetector`

| Test | Input | Expected Output |
|---|---|---|
| `test_default_config_keys` | Freshly instantiated `MangoDefectDetector()` | `config` dict contains `"dark_threshold"` and `"brown_hue_low"` |
| `test_detect_defects_returns_analysis` | `dummy_bgr_image` | Returns a `DefectAnalysis` instance with `processing_time ≥ 0` |
| `test_detect_defects_scores_in_range` | `dummy_bgr_image` | `0 ≤ color_uniformity_score ≤ 100` and `0 ≤ surface_quality_score ≤ 100` |
| `test_detect_defects_grade_impact_valid` | `dummy_bgr_image` | `export_grade_impact ∈ {"minimal", "moderate", "significant"}` |
| `test_calculate_surface_quality_no_defects` | Empty list `[]` (no defect regions) | Returns `100.0` |
| `test_analyze_color_uniformity_returns_number` | `dummy_bgr_image` + full-white 224×224 mask | Returns `float ∈ [0, 100]` |

---

### `ExportAdvisor` — Grade Mapping & Recommendation Parsing

#### Class `TestMapToGrade`

`_map_to_grade(quality_score, impact, defect_pct) → "A" | "B" | "C"`

| Test | Input | Expected Output |
|---|---|---|
| `test_grade_a` | `quality=92, impact="minimal", defect=1.0` | `"A"` |
| `test_grade_b_moderate_impact` | `quality=80, impact="moderate", defect=3.5` | `"B"` |
| `test_grade_b_minimal_impact_higher_defect` | `quality=85, impact="minimal", defect=4.0` | `"B"` |
| `test_grade_c_significant` | `quality=60, impact="significant", defect=8.0` | `"C"` |
| `test_grade_c_high_defect` | `quality=80, impact="minimal", defect=6.0` | `"C"` |

#### Class `TestParseStructuredRecommendation`

Input: multi-section LLM text with `**RECOMMENDED COUNTRIES:**`, `**NOT RECOMMENDED:**`, `**ACTIONABLE STEPS:**`, `**CONDITIONS:**` headers.

| Test | Input | Expected Output |
|---|---|---|
| `test_returns_dict_with_four_keys` | Sample LLM output string | Dict keys exactly `{recommended_countries, not_recommended, actionable_steps, conditions}` |
| `test_recommended_countries_parsed` | Same | `len(recommended_countries) == 3`, `"USA"` present |
| `test_not_recommended_empty_when_none` | `"Not Recommended: None"` section | `not_recommended == []` |
| `test_actionable_steps_parsed` | Same | `len(actionable_steps) ≥ 2` |
| `test_conditions_parsed` | Same | `len(conditions) ≥ 1` |
| `test_empty_input` | `""` | All four lists are empty |

#### Class `TestExportAdvisorBuildMetadata`

| Test | Input | Expected Output |
|---|---|---|
| `test_returns_all_expected_keys` | `sample_defect_analysis_dict`, `disease_percentage=0.45` | Dict contains exactly the 8 expected keys |
| `test_grade_derived_correctly` | `sample_defect_analysis_dict` (minimal impact, 1.23% defect) | `export_grade == "A"` |
| `test_empty_dict_defaults` | `{}` | `surface_quality_score == 0.0`, `export_grade == "C"` (unknown impact defaults to lowest) |

---

### `IntegratedMangoAnalyzer`

#### Class `TestIntegratedAnalyzerGrading`

`_generate_quality_assessment(cv_result, ml_result, disease_pct) → dict`

| Test | Input | Expected Output |
|---|---|---|
| `test_grade_a_premium` | `defect_pct=1.0, uniformity=90, surface=95` | `grade_category` contains `"Grade A"` |
| `test_grade_b_standard` | `defect_pct=3.0, uniformity=75, surface=80` | `grade_category` contains `"Grade B"` |
| `test_grade_c_local` | `defect_pct=7.0, uniformity=60, surface=50` | `grade_category` contains `"Grade C"` |
| `test_processing_grade` | `defect_pct=15.0, uniformity=40, surface=30` | `grade_category` contains `"Processing"` |
| `test_overall_score_in_range` | `defect_pct=2.0, uniformity=85, surface=88` | `0 ≤ overall_score ≤ 100` |
| `test_issues_list_type` | `defect_pct=0.5, uniformity=95, surface=98` | `issues` field is a `list` |

#### Class `TestExportSuitability`

`_predict_export_suitability(quality_assessment) → dict`

| Test | Input | Expected Output |
|---|---|---|
| `test_grade_a_markets` | `grade_category="Grade A (Premium Export)", overall_score=95` | `"USA" ∈ suitable_markets`, `restrictions == []` |
| `test_grade_b_markets` | `grade_category="Grade B (Standard Export)", overall_score=75` | `"Middle East" ∈ suitable_markets` |
| `test_rag_feature_description_generated` | `grade_category="Grade A (Premium Export)", overall_score=92` | `feature_description` is non-empty string |

---

### `ReportGenerator` helpers

#### Class `TestImgToB64`

| Test | Input | Expected Output |
|---|---|---|
| `test_returns_non_empty_string` | `dummy_bgr_image` | Returns non-empty `str` |
| `test_valid_base64` | `dummy_bgr_image` | Base64-decoded bytes start with `\x89PNG` (valid PNG magic bytes) |

#### Class `TestNsHelper`

`_ns(dict, keep_dict_keys=None)` converts dicts to dot-accessible namespaces.

| Test | Input | Expected Output |
|---|---|---|
| `test_simple_dict` | `{"a": 1, "b": "hello"}` | `ns.a == 1`, `ns.b == "hello"` |
| `test_nested_dict` | `{"outer": {"inner": 42}}` | `ns.outer.inner == 42` |
| `test_keep_dict_keys` | `{"probs": {"A": 0.9, "B": 0.1}}`, `keep_dict_keys={"probs"}` | `ns.probs` stays a plain `dict`, `ns.probs["A"] == 0.9` |

---

### `segmentation_utils` helpers

#### Class `TestCreateSegmentationOverlay`

| Test | Input | Expected Output |
|---|---|---|
| `test_output_shape` | `dummy_bgr_image (224×224×3)` + `dummy_binary_mask (224×224)` | `overlay.shape == (224, 224, 3)` |
| `test_output_dtype` | Same | `overlay.dtype == uint8` |

#### Class `TestExtractDiseasedRegion`

| Test | Input | Expected Output |
|---|---|---|
| `test_masked_region_zeros_outside` | `dummy_bgr_image` + `dummy_binary_mask` | Pixel `[0, 0]` (outside the mask rectangle) is black (all channels = 0) |
| `test_preserves_shape` | Same | `result.shape == dummy_bgr_image.shape` |

#### Class `TestGetDiseaseStatistics`

| Test | Input | Expected Output |
|---|---|---|
| `test_empty_mask` | All-zero `(100×100) uint8` mask | `num_regions == 0`, `total_area == 0` |
| `test_single_region` | `dummy_binary_mask` (contains a white rectangle) | `num_regions ≥ 1`, `total_area > 0` |
| `test_bounding_boxes_present` | `dummy_binary_mask` | `bounding_boxes` field is a `list` |

---

## 8. Unit Tests — Regex & Chunker

**File:** `test_regex_and_chunker.py`  
**Modules under test:** `regex_checking` (TOC extraction), `language.chunker` (markdown section splitting)

---

### TOC Extraction

Sample markdown used across tests:
```
# Table of Contents
- [Intro](#intro)
- [Details](#details)

# Intro
Welcome to the doc.

# Details
Some details here.
```

#### Class `TestExtractToc`

| Test | Input | Expected Output |
|---|---|---|
| `test_finds_toc_block` | `SAMPLE_MD` above | Returns a non-`None` match containing `"Table of Contents"` |
| `test_returns_none_when_no_toc` | `"# Intro\nHello world"` | Returns `None` |
| `test_case_insensitive` | `"# table of contents\n- item\n# Next\nBody"` | Returns a non-`None` match |

#### Class `TestRemoveToc`

| Test | Input | Expected Output |
|---|---|---|
| `test_removes_toc_section` | `SAMPLE_MD` | `"Table of Contents"` no longer in result |
| `test_preserves_other_sections` | `SAMPLE_MD` | `"# Intro"` and `"# Details"` remain in result |
| `test_no_toc_returns_unchanged` | `"# Intro\nHello"` | Returned string equals input exactly |

---

### Markdown Section Extraction

#### Class `TestExtractMarkdownSections`

`extract_markdown_sections(text) → list[dict]` — splits markdown into headings + body pairs.

| Test | Input | Expected Output |
|---|---|---|
| `test_returns_list` | `"# Heading\nContent"` | Returns a `list` |
| `test_single_section` | `"# Title\nParagraph text."` | List of 1 item; `heading == "Title"`, `level == 1` |
| `test_multiple_sections` | `"# A\nText A\n## B\nText B\n# C\nText C"` | 3 sections; headings `["A", "B", "C"]` |
| `test_heading_levels` | `"# H1\n## H2\n### H3\n"` | `levels == [1, 2, 3]` |
| `test_empty_string` | `""` | Returns `[]` |

---

### Document Chunker

#### Class `TestChunkDocuments`

`chunk_documents(docs, chunk_size=1000) → list[Document]`

| Test | Input | Expected Output |
|---|---|---|
| `test_returns_list_of_documents` | Single doc with 2 sections | Returns non-empty `list` |
| `test_chunk_has_metadata` | Same | Each chunk's `metadata` contains `"source"`, `"chunk_id"`, `"section_heading"` |
| `test_chunk_ids_are_sequential` | Same | `chunk_id` values are `[0, 1, 2, …]` |
| `test_large_section_gets_split` | One section with ~5 600 characters, `chunk_size=500` | More than 1 chunk produced; all have `chunk_type == "subsection"` |
| `test_small_section_single_chunk` | `"# Small\nTiny."`, `chunk_size=1000` | Exactly 1 chunk with `chunk_type == "section"` |

---

## 9. Data Validation Tests

**File:** `test_data_validation.py`  
**Purpose:** Validate the structural quality of chunks produced from the 7 real SOP Markdown files in `data_for_llm/`.  
**Inputs:** 7 Markdown documents loaded from disk — no API calls or model files needed.

SOP files validated:
- `45_Mangoes.md`
- `APHIS-2006-0121-0010_content.md`
- `CXS_184e(Middle East& South Asia).md`
- `Final-Mango-Brochure.md`
- `Mango_Inspection_Instructions[1]USA.md`
- `Manual For Export of Mangoes.md`
- `mango_content.md`

---

### Class `TestFileAvailability` — 4 tests + 14 parametrised

| Test | Input | Expected Output |
|---|---|---|
| `test_data_directory_exists` | `data_for_llm/` path | Directory exists on disk |
| `test_sop_file_exists` *(×7)* | Each of the 7 filenames | File is present at expected path |
| `test_sop_file_is_non_empty` *(×7)* | Each file's `stat().st_size` | `> 0` bytes |
| `test_all_seven_files_present` | All 7 filenames checked at once | No missing files (empty list) |

### Class `TestChunkCountAndContent` — 5 tests

| Test | Input | Expected Output |
|---|---|---|
| `test_chunking_produces_chunks` | All 7 SOP docs passed to `chunk_documents()` | `len(chunks) > 0` |
| `test_minimum_chunk_count` | Same | `len(chunks) ≥ 20` (7 files × at least 1 section) |
| `test_no_chunk_has_empty_page_content` | All chunks | No chunk has empty or whitespace-only `page_content` |
| `test_no_chunk_page_content_is_whitespace_only` | All chunks | Same check (belt-and-braces) |
| `test_chunk_count_matches_raw_doc_count` | Chunk sources vs. input filenames | Set of `source` values in chunks == set of input filenames |

### Class `TestMetadataCompleteness` — 8 tests

Required keys: `source`, `filepath`, `chunk_id`, `section_heading`, `heading_level`, `chunk_type`, `chunk_size`.

| Test | Input | Expected Output |
|---|---|---|
| `test_all_chunks_have_required_keys` | All chunks | Every chunk has all 7 required metadata keys |
| `test_source_is_non_empty_string` | `metadata["source"]` | Non-empty `str` |
| `test_filepath_is_non_empty_string` | `metadata["filepath"]` | Non-empty `str` |
| `test_chunk_id_is_non_negative_integer` | `metadata["chunk_id"]` | `int ≥ 0` |
| `test_section_heading_is_string` | `metadata["section_heading"]` | `isinstance(…, str)` |
| `test_heading_level_is_non_negative_integer` | `metadata["heading_level"]` | `int ≥ 0` |
| `test_chunk_type_is_valid` | `metadata["chunk_type"]` | `∈ {"section", "subsection"}` |
| `test_chunk_size_metadata_matches_page_content_length` | `metadata["chunk_size"]` vs. `len(page_content)` | Equal |

### Class `TestChunkIdIntegrity` — 4 tests

| Test | Input | Expected Output |
|---|---|---|
| `test_chunk_ids_start_at_zero` | First chunk's `chunk_id` | `== 0` |
| `test_chunk_ids_are_sequential` | All `chunk_id` values | `== [0, 1, 2, …, n-1]` |
| `test_chunk_ids_are_unique` | All `chunk_id` values | No duplicates |
| `test_total_chunk_count_equals_max_id_plus_one` | `len(chunks)` vs. `max(chunk_id) + 1` | Equal |

### Class `TestSourceMetadataAccuracy` — 3 tests + 14 parametrised

| Test | Input | Expected Output |
|---|---|---|
| `test_all_sources_are_known_filenames` | All `metadata["source"]` values | No unknown filenames appear |
| `test_each_file_contributes_at_least_one_chunk` *(×7)* | Chunks filtered by each filename | At least 1 chunk per file |
| `test_filepath_contains_filename` *(×7)* | Each chunk's `filepath` | Filename appears within the full path string |

### Class `TestChunkSizeConstraints` — 4 tests

`_CHUNK_SIZE = 1000`, `_CHUNK_OVERLAP = 200`, max allowed = 1260 chars.

| Test | Input | Expected Output |
|---|---|---|
| `test_no_subsection_chunk_exceeds_size_plus_heading_overhead` | All `"subsection"` chunks | `chunk_size ≤ 1260` |
| `test_section_chunks_within_chunk_size_limit` | All `"section"` chunks | `chunk_size ≤ 1000` |
| `test_minimum_meaningful_chunk_length` | All chunks | No chunk shorter than 3 characters |
| `test_average_chunk_size_is_reasonable` | Average of `len(page_content)` across all chunks | `50 ≤ avg ≤ 1260` |

### Class `TestPerFileCoverage` — 3 tests × 7 parametrised

| Test | Input | Expected Output |
|---|---|---|
| `test_file_chunk_content_covers_source_text` *(×7)* | Sum of chunk chars vs. original file char count | Total chunk chars `≥ 50%` of original |
| `test_section_headings_are_propagated` *(×7)* | All headings from a file's chunks | `isinstance(headings, list)` — no exception |
| `test_heading_level_zero_only_when_no_heading` *(×7)* | Chunks with `heading_level == 0` | `section_heading == ""` (consistent pairing) |

---

## 10. Performance Tests

**File:** `test_performance.py`  
**Purpose:** Measure latency of pipeline components against defined time budgets. No real model files or API calls.

### Latency Budgets

| Operation | Budget |
|---|---|
| Single `detect_defects()` call | 500 ms |
| Average over 10 `detect_defects()` calls | 500 ms |
| `build_metadata()` | 50 ms |
| 1 000× `_map_to_grade()` | 100 ms |
| `FAISS.from_documents()` on 5 docs | 1 000 ms |
| `retriever.invoke()` | 500 ms |
| `format_context()` | 100 ms |
| `chunk_documents()` on 7 docs | 3 000 ms |

---

### Class `TestDefectDetectionPerformance`

Input fixture: `_orange_frame()` — a solid-colour 224×224 BGR array (no mango model needed).

| Test | Input | Expected Output |
|---|---|---|
| `test_single_frame_within_latency_budget` | One `detect_defects(frame)` call | Elapsed `< 500 ms` |
| `test_average_over_ten_runs_within_budget` | 10 calls after 1 warm-up | Average elapsed `< 500 ms` |
| `test_latency_is_stable_across_runs` | 5 calls after warm-up | `max < 5 × min + 50 ms` (no runaway spikes) |

### Class `TestExportMetadataPerformance`

| Test | Input | Expected Output |
|---|---|---|
| `test_build_metadata_within_budget` | `_DEFECT_DICT` with 6 numeric fields | Elapsed `< 50 ms` |
| `test_grade_derivation_1000_calls_within_budget` | `_map_to_grade(85.0, "minimal", 1.5)` × 1 000 | Total elapsed `< 100 ms` |
| `test_full_defect_to_metadata_chain_within_budget` | OpenCV frame → `detect_defects` → `build_metadata` | Combined elapsed `< 550 ms` |

### Class `TestRAGPipelinePerformance`

Uses `_FakeEmbeddings` (deterministic random, no HuggingFace download).

| Test | Input | Expected Output |
|---|---|---|
| `test_faiss_vectorstore_build_within_budget` | 5 `Document` objects | `FAISS.from_documents` elapsed `< 1 000 ms` |
| `test_faiss_single_query_within_budget` | `"mango export grade requirements"` | `retriever.invoke` elapsed `< 500 ms` |
| `test_context_formatting_within_budget` | 5 `Document` objects | `format_context` elapsed `< 100 ms` |
| `test_chunking_seven_docs_within_budget` | 7 synthetic raw docs | `chunk_documents` elapsed `< 3 000 ms` |
| `test_end_to_end_rag_chain_within_combined_budget` | chunk → embed → FAISS → query | Total elapsed `< 4 500 ms` |

### Class `TestMemoryFootprint`

Skipped if `psutil` is not installed.

| Test | Input | Expected Output |
|---|---|---|
| `test_defect_detector_memory_increase_within_budget` | 5× `MangoDefectDetector()` instantiated | RSS increase `< 100 MB` |
| `test_faiss_index_memory_increase_within_budget` | FAISS index built from 50 documents | RSS increase `< 200 MB` |

---

## 11. Integration Tests — Functional Flows

**File:** `test_integration.py`  
**Purpose:** Wire multiple real modules together (with only heavy ML models mocked) to verify end-to-end data flows.

---

### Class `TestDefectToExportFlow`

Runs real `MangoDefectDetector` + real `ExportAdvisor` on a synthetic image.

| Test | Input | Expected Output |
|---|---|---|
| `test_healthy_mango_grades_as_a` | `dummy_bgr_image` (solid colour, no defects) | `export_grade ∈ {"A", "B"}` — a clean image cannot grade as C |

### Class `TestIntegratedAnalyzerE2E`

| Test | Input | Expected Output |
|---|---|---|
| `test_full_analysis_structure` | `dummy_bgr_image` + synthetic `ml_seg` dict (`disease_percentage=0.5`, zero mask) | Result dict has exactly: `opencv_defects`, `ml_disease`, `combined_analysis`, `export_recommendations`, `performance` |
| `test_analysis_without_ml_seg` | `dummy_bgr_image` only (no `ml_seg`) | `ml_disease == None`; `combined_analysis["total_defect_percentage"] ≥ 0` |

### Class `TestReportJsonStructure`

Simulates the JSON export logic from `streamlit_app.py`.

| Test | Input | Expected Output |
|---|---|---|
| `test_json_roundtrip` | `sample_pipeline_result` + a mock export recommendation dict | After `json.dumps` → `json.loads`: `total_mangoes == 1`, `classification.class_name == "Healthy"`, `export_recommendation.grade == "A"` |

### Class `TestChunkerIntegration`

| Test | Input | Expected Output |
|---|---|---|
| `test_multiple_documents_chunked` | 2 Markdown docs with different filenames | Both `"doc1.md"` and `"doc2.md"` appear in chunk sources |
| `test_chunk_content_not_empty` | Single short doc | All chunks have non-empty `page_content` |

### Class `TestPrePostRoundTrip`

| Test | Input | Expected Output |
|---|---|---|
| `test_combined_pipeline_preserves_shape` | `dummy_bgr_image` → `preprocess_image` → `smooth_image` → `contrast` | `shape == (224, 224, 3)`, `dtype == uint8` |
| `test_pipeline_on_random_image` | Random `(224×224×3) uint8` → `preprocess_image` → `contrast` | All pixels `∈ [0, 255]` |

### Class `TestPromptTemplates`

Skipped if `prompt_templates.json` is not found.

| Test | Input | Expected Output |
|---|---|---|
| `test_prompt_templates_valid_json` | `prompt_templates.json` file on disk | JSON parses successfully; contains `"system_message"`, `"export_recommendation_prompt"`, `"rag_config"` |
| `test_rag_config_has_required_keys` | `rag_config` section | Contains `vector_store`, `embedding_model`, `llm`, `retrieval_chunks`, `max_tokens`, `temperature` |

---

## 12. Integration Tests — Full Pipeline

**File:** `test_integration_pipeline.py`  
**Purpose:** Component boundary tests across the full pipeline. TFLite / PyTorch / HuggingFace are mocked at module level; OpenCV and pure-Python code run for real.

Key test doubles:
- `_FakeEmbeddings` — deterministic 384-dim vectors, no model download
- `_mock_interpreter(class_idx)` — TFLite stub returning specified softmax probabilities
- `loaded_seg_model` — `SegmentationModel` stub returning a fixed mask with 5% disease

---

### Class `TestVisionPipelineIntegration` — Classification Model

| Test | Input | Expected Output |
|---|---|---|
| `test_classification_model_predict_returns_dict` | `dummy_frame` through mocked `ClassificationModel` | Returns dict with `class_name`, `confidence`, `probabilities` |
| `test_classification_model_class_name_is_string` | Same | `class_name` is `str` |
| `test_classification_model_confidence_in_range` | Same | `0 ≤ confidence ≤ 1.0` |
| `test_classification_model_probabilities_sum_to_one` | Same | Sum of probability values ≈ 1.0 (within 1e-3) |
| `test_classification_healthy_class` | Mock set to class index 3 (Healthy) | `class_name == "Healthy"` |

### Class `TestSegmentationModelIntegration`

| Test | Input | Expected Output |
|---|---|---|
| `test_segmentation_returns_three_tuple` | `dummy_frame` through stub `SegmentationModel` | Returns `(mask, overlay, disease_pct)` — 3-tuple |
| `test_segmentation_mask_is_2d` | Same | `mask.ndim == 2` |
| `test_segmentation_overlay_is_3d` | Same | `overlay.ndim == 3` |
| `test_disease_percentage_is_float` | Same | `isinstance(disease_pct, float)` |
| `test_disease_percentage_non_negative` | Same | `disease_pct ≥ 0.0` |

### Class `TestMangoPipelineIntegration`

| Test | Input | Expected Output |
|---|---|---|
| `test_process_single_frame_returns_dict` | `dummy_frame` through full `MangoPipeline` (both models mocked) | Returns dict |
| `test_frame_result_has_class_name` | Same | `"class_name" ∈ result` |
| `test_frame_result_has_confidence` | Same | `"confidence" ∈ result` |
| `test_frame_result_confidence_in_range` | Same | `0 ≤ confidence ≤ 1.0` |

### Class `TestDefectExportGradeIntegration`

| Test | Input | Expected Output |
|---|---|---|
| `test_detect_defects_returns_analysis` | `dummy_frame` through `MangoDefectDetector` | Returns `DefectAnalysis` |
| `test_build_metadata_from_defect_analysis` | `DefectAnalysis` → `ExportAdvisor.build_metadata` | Returns dict with `"export_grade"` key |
| `test_export_grade_is_valid` | Same | `export_grade ∈ {"A", "B", "C"}` |
| `test_grade_a_for_clean_image` | `dummy_frame` (minimal defects) → full chain | `export_grade ∈ {"A", "B"}` |

### Class `TestRAGDocumentPipelineIntegration`

| Test | Input | Expected Output |
|---|---|---|
| `test_chunk_documents_returns_chunks` | 3 synthetic raw docs | `len(chunks) > 0` |
| `test_chunks_have_metadata` | Same | Each chunk has `"source"` and `"chunk_id"` in metadata |
| `test_vectorstore_created_from_documents` | Chunks + `_FakeEmbeddings` | `FAISS.from_documents` returns without error; result is a `FAISS` instance |
| `test_retriever_returns_documents` | FAISS store + query `"mango export grade requirements"` | Returns non-empty list of `Document` objects |
| `test_retrieved_documents_have_page_content` | Same | Each returned doc has non-empty `page_content` |

### Class `TestLLMManagerIntegration`

| Test | Input | Expected Output |
|---|---|---|
| `test_format_context_returns_string` | 4 `Document` objects | `format_context()` returns a `str` |
| `test_format_context_non_empty` | Same | Returned string is non-empty |
| `test_create_export_prompt_returns_string` | Grade-A metadata dict + formatted context | Returns `str` |
| `test_create_rag_prompt_returns_string` | Question + context | Returns `str` |
| `test_export_prompt_contains_grade` | Grade-A metadata | `"A"` appears somewhere in the returned prompt |

### Class `TestVectorStoreManagerIntegration`

| Test | Input | Expected Output |
|---|---|---|
| `test_create_embeddings_and_vectorstore` | 3 docs + `_FakeEmbeddings` | Returns `(vectorstore, embeddings)` tuple |
| `test_load_vectorstore_from_saved` | FAISS index saved to temp dir, then `load_vectorstore` called | Returns `(vectorstore, embeddings)` pair |
| `test_create_retriever_from_vectorstore` | `FAISS` instance | Returns a retriever with an `invoke` method |
| `test_retriever_invoke_returns_docs` | Retriever + query string | Returns a list |

### Class `TestEndToEndFramePipeline`

Full synthetic frame → classification → defect detection → export grade → JSON serialisable output.

| Test | Input | Expected Output |
|---|---|---|
| `test_full_frame_to_grade_pipeline` | `dummy_frame` through MangoPipeline + MangoDefectDetector + ExportAdvisor | `export_grade ∈ {"A", "B", "C"}` |
| `test_pipeline_result_json_serialisable` | Full pipeline result dict | `json.dumps(result)` succeeds without exception |
| `test_pipeline_result_has_all_keys` | Same | Dict contains `class_name`, `confidence`, `defect_analysis`, `export_metadata` |

---

## 13. Reliability Tests

**File:** `test_reliability.py`  
**Purpose:** Verify that all deterministic (non-LLM) components return identical results across 5 repeated calls with the same input.

`RUNS = 5` (all repeated tests run 5 times).

---

### Class `TestDefectDetectionReliability`

Input: same `_orange_frame()` (solid-colour 224×224 BGR) passed 5 times.

| Test | Expected Output |
|---|---|
| `test_surface_quality_score_is_identical` | All 5 scores equal |
| `test_color_uniformity_score_is_identical` | All 5 scores equal |
| `test_total_defect_percentage_is_identical` | All 5 values equal |
| `test_dark_spot_count_is_identical` | All 5 counts equal |
| `test_brown_spot_count_is_identical` | All 5 counts equal |
| `test_export_grade_impact_is_identical` | All 5 strings equal |
| `test_defect_count_is_identical` | All 5 counts equal |
| `test_different_frames_produce_different_results` | Plain frame vs. noisy frame (with random dark patch) → at least one score differs |

### Class `TestGradeDerivationReliability`

Parametrised over 7 `(quality, impact, defect_pct, expected_grade)` combinations; each run 5 times.

| Test | Expected Output |
|---|---|
| `test_grade_is_always_consistent` | All 5 runs return the expected grade |
| `test_export_advisor_metadata_grade_consistent` | `build_metadata` called 5 times → all grades identical |
| `test_export_advisor_metadata_disease_percentage_consistent` | `build_metadata` with `disease_percentage=3.5` × 5 → identical values |

### Class `TestChunkingReliability`

Input: same 2 raw docs run through `chunk_documents()` 5 times.

| Test | Expected Output |
|---|---|
| `test_chunk_count_is_consistent` | All 5 runs produce the same number of chunks |
| `test_chunk_content_is_identical_across_runs` | All runs return identical `page_content` lists |
| `test_chunk_source_metadata_is_identical_across_runs` | All runs return identical `source` lists |
| `test_chunk_ids_are_identical_across_runs` | All runs return identical `chunk_id` lists |
| `test_chunk_ids_are_zero_based_sequential` | `chunk_id` values in run 1 are `[0, 1, …, n-1]` |

### Class `TestFAISSRetrievalReliability`

Uses `_FakeEmbeddings` (seeded determinism).

| Test | Expected Output |
|---|---|
| `test_same_query_returns_same_documents` | Same query × 5 → identical `source` lists in all runs |
| `test_same_query_returns_same_content` | Same query × 5 → identical `page_content` lists |
| `test_different_queries_can_return_different_results` | Two different queries → both return valid sets (no exception) |

### Class `TestContextFormattingReliability`

| Test | Expected Output |
|---|---|
| `test_format_context_output_is_identical_across_runs` | `format_context(docs)` × 5 → all strings equal |
| `test_export_prompt_output_is_identical_across_runs` | `create_export_prompt(metadata, context)` × 5 → all strings equal |
| `test_rag_prompt_output_is_identical_across_runs` | `create_rag_prompt(question, context)` × 5 → all strings equal |

---

## 14. Streamlit UI Tests

**File:** `test_streamlit_app.py`  
**Framework:** `streamlit.testing.v1.AppTest` — renders `streamlit_app.py` headlessly.  
**Heavy dependencies mocked:** TFLite, PyTorch, HuggingFace embeddings, OpenAI, video utilities, segmentation utils, LLM manager, RAG internals, SQLite logger.

Helper `_fresh()` returns a freshly rendered home-page `AppTest`. Helper `_on_chat_page()` forces `session_state["current_page"] = "chat"`.

---

### Class `TestSessionStateInitialisation`

| Test | Expected Output |
|---|---|
| `test_no_exceptions_on_first_render` | `at.exception` is falsy |
| `test_current_page_defaults_to_home` | `session_state["current_page"] == "home"` |
| `test_started_defaults_to_false` | `session_state["started"] is False` |
| `test_results_defaults_to_empty_list` | `session_state["results"] == []` |
| `test_chat_messages_defaults_to_empty_list` | `session_state["chat_messages"] == []` |
| `test_processing_complete_defaults_to_false` | `session_state["processing_complete"] is False` |
| `test_source_type_defaults_to_video_file` | `session_state["source_type"] == "Video File"` |
| `test_chat_rag_ready_defaults_to_false` | `session_state["chat_rag_ready"] is False` |
| `test_active_report_json_defaults_to_none` | `session_state["active_report_json"] is None` |

### Class `TestHomePageWidgets`

| Test | Expected Output |
|---|---|
| `test_start_inspection_button_exists` | A button with label containing `"Start Inspection"` is rendered |
| `test_open_chat_shortcut_button_exists` | A button with label containing `"Open Mango Export Chat"` is rendered |
| `test_source_type_radio_exists` | At least one radio widget rendered |
| `test_source_type_radio_has_video_file_option` | `"Video File" ∈ radio[0].options` |
| `test_source_type_radio_has_camera_feed_option` | `"Camera Feed" ∈ radio[0].options` |
| `test_source_type_radio_defaults_to_video_file` | `radio[0].value == "Video File"` |
| `test_use_default_video_checkbox_exists` | At least one checkbox rendered |
| `test_use_default_video_checkbox_is_checked_by_default` | `checkbox[0].value is True` |
| `test_unchecking_default_video_updates_checkbox_state` | Uncheck → re-render → `checkbox[0].value is False` |
| `test_unchecking_default_video_reveals_file_uploader` | Uncheck → a file uploader widget appears |

### Class `TestHomePageNavigation`

| Test | Input | Expected Output |
|---|---|---|
| `test_start_inspection_without_video_does_not_navigate` | Click "Start Inspection" with no video selected | Page stays on `"home"` |
| `test_start_inspection_without_video_stays_on_home` | Same | `session_state["current_page"] == "home"` |
| `test_chat_button_navigates_to_chat_page` | Click "Open Mango Export Chat" | `session_state["current_page"] == "chat"` |
| `test_chat_page_renders_without_exception` | Force `current_page = "chat"` and re-render | `at.exception` is falsy |

### Class `TestChatPageWidgets`

| Test | Expected Output |
|---|---|
| `test_chat_page_has_chat_input` | At least one `chat_input` widget rendered on chat page |
| `test_chat_page_back_button_exists` | A button with `"Back"` in label is rendered |

### Class `TestChatPageNavigation`

| Test | Input | Expected Output |
|---|---|---|
| `test_back_button_returns_to_home` | On chat page, click "Back" button | `session_state["current_page"] == "home"` |

### Class `TestQuickPrompts`

| Test | Input | Expected Output |
|---|---|---|
| `test_quick_prompt_buttons_exist_on_chat_page` | Chat page rendered | At least 1 quick-prompt button is visible |

### Class `TestResultsPage`

Simulates a completed inspection by injecting `results` and `processing_complete=True` into session state.

| Test | Input | Expected Output |
|---|---|---|
| `test_results_page_renders_without_exception` | Pre-populated `results` list, page forced to `"results"` | `at.exception` is falsy |

### Class `TestChatInteraction`

| Test | Input | Expected Output |
|---|---|---|
| `test_chat_message_appears_after_submission` | Submit `"What markets accept Grade A mangoes?"` via `chat_input` | Message appears in `session_state["chat_messages"]` |
| `test_chat_response_is_generated` | Same | `len(chat_messages) ≥ 2` (user + assistant) |

---
