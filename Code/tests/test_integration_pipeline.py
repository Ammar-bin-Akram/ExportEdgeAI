"""
Integration tests for ExportEdge AI pipeline.

Tests component boundaries:
  1. Vision Pipeline  (MangoPipeline → ClassificationModel → SegmentationModel)
  2. Defect → Export Grade  (MangoDefectDetector → ExportAdvisor)
  3. RAG Document Pipeline  (chunker → FAISS → retriever)
  4. LLM Manager  (context formatting + mocked LLM client)
  5. Full End-to-End  (frame → defect analysis → export metadata → mocked RAG)

External API calls (Groq, OpenAI, Gemini, LM Studio) are fully mocked.
Heavy model loading (TFLite, PyTorch, HuggingFace) is mocked at the method level.

Run from the Code/ directory:
    pytest tests/test_integration_pipeline.py -v
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Block tensorflow import before any project code triggers it ───────────────
# tflite_runtime mock satisfies `import tflite_runtime.interpreter as tflite`
# in models/classification.py, preventing the tf → protobuf import chain.
_mock_tflite = MagicMock()
sys.modules.setdefault("tflite_runtime", _mock_tflite)
sys.modules.setdefault("tflite_runtime.interpreter", _mock_tflite)

# ── Path setup (mirrors conftest.py) ──────────────────────────────────────────
_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

# language/ uses bare `from data_config import …` (not a relative import)
_LANG_DIR = _CODE_DIR / "language"
if str(_LANG_DIR) not in sys.path:
    sys.path.insert(0, str(_LANG_DIR))

# ── Project imports ────────────────────────────────────────────────────────────
from config.settings import Settings
from models.classification import ClassificationModel
from models.segmentation import SegmentationModel
from fruits.mango.pipeline import MangoPipeline
from vision.defect_detector import MangoDefectDetector, DefectAnalysis
from vision.export_advisor import ExportAdvisor, _map_to_grade
from language.llm_manager import LLMManager
from language.chunker import chunk_documents
from language.vector_store_manager import (
    create_embeddings_and_vectorstore,
    create_retriever,
    load_vectorstore,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS


# ─────────────────────────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

class _FakeEmbeddings(Embeddings):
    """Deterministic fake embeddings — no network, no model download."""

    DIM = 384  # matches all-MiniLM-L6-v2 output dimension

    def embed_documents(self, texts):
        rng = np.random.RandomState(42)
        return [rng.rand(self.DIM).tolist() for _ in texts]

    def embed_query(self, text: str):
        return np.random.RandomState(0).rand(self.DIM).tolist()


def _mock_interpreter(predicted_class_idx: int = 3) -> MagicMock:
    """TFLite interpreter mock that predicts the given class index."""
    probs = [0.01] * 5
    probs[predicted_class_idx] = 0.92
    interp = MagicMock()
    interp.get_input_details.return_value = [{"index": 0}]
    interp.get_output_details.return_value = [{"index": 0}]
    interp.get_tensor.return_value = np.array([probs], dtype=np.float32)
    return interp


def _orange_frame(h: int = 224, w: int = 224) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (30, 130, 210)  # BGR orange-ish
    return img


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def settings():
    return Settings()


@pytest.fixture
def dummy_frame():
    return _orange_frame()


@pytest.fixture
def loaded_cls_model():
    """ClassificationModel with a mocked interpreter — no .tflite file needed."""
    model = ClassificationModel()
    model.interpreter = _mock_interpreter(predicted_class_idx=3)  # → Healthy
    model.input_details = [{"index": 0}]
    model.output_details = [{"index": 0}]
    return model


@pytest.fixture
def loaded_seg_model():
    """SegmentationModel whose segment() is stubbed — no .pth file needed."""
    stub = MagicMock(spec=SegmentationModel)
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[80:130, 80:130] = 1  # small diseased patch
    overlay = np.zeros((224, 224, 3), dtype=np.uint8)
    stub.segment.return_value = (mask, overlay, 5.0)
    return stub


@pytest.fixture
def mango_pipeline(settings, loaded_cls_model, loaded_seg_model):
    """MangoPipeline with both models replaced by test doubles."""
    pipeline = MangoPipeline(settings)
    pipeline.classification_model = loaded_cls_model
    pipeline.segmentation_model = loaded_seg_model
    return pipeline


@pytest.fixture
def sample_raw_docs():
    """Plain dicts in the format chunk_documents() expects (content/filename/filepath)."""
    return [
        {
            "content": "# Grade A Requirements\n\nGrade A mangoes must have less than 2% defect area and no disease symptoms.",
            "filename": "mango_standard.md",
            "filepath": "/data/mango_standard.md",
        },
        {
            "content": "# Grade B Requirements\n\nGrade B mangoes are allowed up to 5% defect area for certain regional markets.",
            "filename": "mango_standard.md",
            "filepath": "/data/mango_standard.md",
        },
        {
            "content": "# EU Export Requirements\n\nExport to EU requires a phytosanitary certificate and cold-chain documentation.",
            "filename": "eu_export_reqs.md",
            "filepath": "/data/eu_export_reqs.md",
        },
    ]


@pytest.fixture
def sample_docs():
    """Minimal LangChain Documents simulating chunked regulatory text."""
    return [
        Document(
            page_content="Grade A mangoes must have less than 2% defect area and no disease symptoms.",
            metadata={"source": "mango_standard.md", "section_heading": "Grade A Requirements"},
        ),
        Document(
            page_content="Grade B mangoes are allowed up to 5% defect area for certain regional markets.",
            metadata={"source": "mango_standard.md", "section_heading": "Grade B Requirements"},
        ),
        Document(
            page_content="Export to EU requires a phytosanitary certificate and cold-chain documentation.",
            metadata={"source": "eu_export_reqs.md", "section_heading": "EU Export Requirements"},
        ),
        Document(
            page_content="Middle East markets accept Grade B mangoes with a fumigation certificate.",
            metadata={"source": "middle_east_reqs.md", "section_heading": "Middle East Standards"},
        ),
        Document(
            page_content="Japan imposes strict pesticide residue limits. Only Grade A mangoes are accepted.",
            metadata={"source": "japan_reqs.md", "section_heading": "Japan Import Standards"},
        ),
    ]


@pytest.fixture
def faiss_retriever(sample_docs):
    """Real FAISS retriever built with fake embeddings — no HuggingFace download."""
    embeddings = _FakeEmbeddings()
    vector_store = FAISS.from_documents(sample_docs, embeddings)
    return create_retriever(vector_store, k=3)


@pytest.fixture
def llm_manager_mocked():
    """LLMManager with its OpenAI client fully mocked — no API calls."""
    with patch("language.llm_manager.OpenAI"):
        manager = LLMManager()
    manager.client = MagicMock()
    return manager


# ─────────────────────────────────────────────────────────────────────────────
#  1. Vision Pipeline Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestVisionPipelineIntegration:
    """MangoPipeline wires preprocessing → classification → segmentation correctly."""

    def test_output_contains_required_keys(self, mango_pipeline, dummy_frame):
        result = mango_pipeline.process_single_frame(dummy_frame)
        required = {
            "original", "preprocessed", "class_name", "confidence",
            "segmentation_mask", "segmentation_overlay", "disease_percentage",
        }
        assert required.issubset(result.keys())

    def test_class_name_is_one_of_known_classes(self, mango_pipeline, dummy_frame):
        result = mango_pipeline.process_single_frame(dummy_frame)
        assert result["class_name"] in Settings().CLASS_NAMES

    def test_confidence_is_in_valid_range(self, mango_pipeline, dummy_frame):
        result = mango_pipeline.process_single_frame(dummy_frame)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_mock_predicts_healthy_at_index_3(self, mango_pipeline, dummy_frame):
        result = mango_pipeline.process_single_frame(dummy_frame)
        assert result["class_name"] == "Healthy"

    def test_disease_percentage_matches_segmentation_stub(self, mango_pipeline, dummy_frame):
        result = mango_pipeline.process_single_frame(dummy_frame)
        assert result["disease_percentage"] == pytest.approx(5.0)

    def test_segmentation_disabled_returns_zero_disease(self, settings, loaded_cls_model, dummy_frame):
        settings.ENABLE_SEGMENTATION = False
        pipeline = MangoPipeline(settings)
        pipeline.classification_model = loaded_cls_model
        result = pipeline.process_single_frame(dummy_frame)
        assert result["disease_percentage"] == 0.0
        assert result["segmentation_mask"] is None

    def test_preprocessed_image_matches_input_shape_setting(self, mango_pipeline, dummy_frame):
        result = mango_pipeline.process_single_frame(dummy_frame)
        expected_h, expected_w = Settings().INPUT_SHAPE[:2]
        assert result["preprocessed"].shape[:2] == (expected_h, expected_w)

    def test_predict_top_k_returns_correct_count(self, loaded_cls_model):
        img = np.zeros((150, 150, 3), dtype=np.uint8)
        top_k = loaded_cls_model.predict_top_k(img, k=3)
        assert len(top_k) == 3

    def test_predict_top_k_all_confidences_in_range(self, loaded_cls_model):
        img = np.zeros((150, 150, 3), dtype=np.uint8)
        top_k = loaded_cls_model.predict_top_k(img, k=5)
        assert all(0.0 <= conf <= 1.0 for _, conf in top_k)

    def test_predict_top_k_highest_confidence_is_first(self, loaded_cls_model):
        img = np.zeros((150, 150, 3), dtype=np.uint8)
        top_k = loaded_cls_model.predict_top_k(img, k=3)
        confidences = [c for _, c in top_k]
        assert confidences == sorted(confidences, reverse=True)

    def test_different_class_predictions_propagate(self, settings, dummy_frame):
        """Wiring test: changing the mock prediction changes the pipeline output."""
        for class_idx, class_name in enumerate(Settings().CLASS_NAMES):
            model = ClassificationModel()
            model.interpreter = _mock_interpreter(predicted_class_idx=class_idx)
            model.input_details = [{"index": 0}]
            model.output_details = [{"index": 0}]
            pipeline = MangoPipeline(settings)
            pipeline.classification_model = model
            pipeline.segmentation_model = MagicMock(spec=SegmentationModel)
            pipeline.segmentation_model.segment.return_value = (
                np.zeros((224, 224), dtype=np.uint8),
                np.zeros((224, 224, 3), dtype=np.uint8),
                0.0,
            )
            result = pipeline.process_single_frame(dummy_frame)
            assert result["class_name"] == class_name


# ─────────────────────────────────────────────────────────────────────────────
#  2. Defect Detection → Export Grade Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestDefectExportGradeIntegration:
    """Defect detection → export advisory chain (pure OpenCV + Python, no ML)."""

    def test_detect_defects_returns_defect_analysis_instance(self, dummy_frame):
        result = MangoDefectDetector().detect_defects(dummy_frame)
        assert isinstance(result, DefectAnalysis)

    def test_surface_quality_score_in_range(self, dummy_frame):
        result = MangoDefectDetector().detect_defects(dummy_frame)
        assert 0.0 <= result.surface_quality_score <= 100.0

    def test_color_uniformity_score_in_range(self, dummy_frame):
        result = MangoDefectDetector().detect_defects(dummy_frame)
        assert 0.0 <= result.color_uniformity_score <= 100.0

    def test_defect_percentage_is_non_negative(self, dummy_frame):
        result = MangoDefectDetector().detect_defects(dummy_frame)
        assert result.total_defect_percentage >= 0.0

    def test_grade_impact_is_valid_category(self, dummy_frame):
        result = MangoDefectDetector().detect_defects(dummy_frame)
        assert result.export_grade_impact in ("minimal", "moderate", "significant")

    def test_grade_a_for_minimal_defects_under_2pct(self):
        assert _map_to_grade(95.0, "minimal", 1.0) == "A"

    def test_grade_a_at_exact_2pct_boundary(self):
        assert _map_to_grade(90.0, "minimal", 2.0) == "A"

    def test_grade_b_when_minimal_impact_above_2pct(self):
        assert _map_to_grade(90.0, "minimal", 2.1) == "B"

    def test_grade_b_for_moderate_impact_under_5pct(self):
        assert _map_to_grade(75.0, "moderate", 4.0) == "B"

    def test_grade_c_for_significant_impact(self):
        assert _map_to_grade(50.0, "significant", 10.0) == "C"

    def test_grade_c_when_defect_exceeds_5pct(self):
        assert _map_to_grade(80.0, "moderate", 6.0) == "C"

    def test_build_metadata_contains_all_required_keys(self, sample_defect_analysis_dict):
        metadata = ExportAdvisor().build_metadata(
            defect_analysis=sample_defect_analysis_dict,
            disease_percentage=3.5,
        )
        required = {
            "surface_quality_score", "color_uniformity_score",
            "total_defect_percentage", "dark_spot_count",
            "brown_spot_count", "export_grade_impact",
            "export_grade", "disease_percentage",
        }
        assert required.issubset(metadata.keys())

    def test_build_metadata_disease_percentage_stored(self, sample_defect_analysis_dict):
        metadata = ExportAdvisor().build_metadata(sample_defect_analysis_dict, disease_percentage=7.5)
        assert metadata["disease_percentage"] == pytest.approx(7.5)

    def test_build_metadata_grade_consistent_with_map_to_grade(self, sample_defect_analysis_dict):
        metadata = ExportAdvisor().build_metadata(sample_defect_analysis_dict, disease_percentage=1.0)
        expected = _map_to_grade(
            surface_quality=sample_defect_analysis_dict["surface_quality_score"],
            export_grade_impact=sample_defect_analysis_dict["export_grade_impact"],
            total_defect_pct=sample_defect_analysis_dict["total_defect_percentage"],
        )
        assert metadata["export_grade"] == expected

    def test_full_chain_frame_to_export_metadata(self, dummy_frame):
        """OpenCV frame → DefectAnalysis → ExportAdvisor metadata — no ML models."""
        analysis = MangoDefectDetector().detect_defects(dummy_frame)
        defect_dict = {
            "surface_quality_score": analysis.surface_quality_score,
            "color_uniformity_score": analysis.color_uniformity_score,
            "total_defect_percentage": analysis.total_defect_percentage,
            "dark_spot_count": analysis.dark_spot_count,
            "brown_spot_count": analysis.brown_spot_count,
            "export_grade_impact": analysis.export_grade_impact,
        }
        metadata = ExportAdvisor().build_metadata(defect_dict, disease_percentage=0.5)
        assert metadata["export_grade"] in ("A", "B", "C")


# ─────────────────────────────────────────────────────────────────────────────
#  3. RAG Document Pipeline Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestRAGDocumentPipelineIntegration:
    """chunker → FAISS embedding → retriever, no external network calls."""

    def test_chunk_documents_preserves_source_metadata(self, sample_raw_docs):
        chunks = chunk_documents(sample_raw_docs)
        assert len(chunks) > 0
        for chunk in chunks:
            assert "source" in chunk.metadata

    def test_chunk_documents_assigns_chunk_ids(self, sample_raw_docs):
        chunks = chunk_documents(sample_raw_docs)
        assert all("chunk_id" in c.metadata for c in chunks)

    def test_vectorstore_created_from_documents(self, sample_docs):
        vs = FAISS.from_documents(sample_docs, _FakeEmbeddings())
        assert vs is not None

    def test_retriever_returns_document_objects(self, faiss_retriever):
        results = faiss_retriever.invoke("mango export grade requirements")
        assert len(results) > 0
        assert all(isinstance(d, Document) for d in results)

    def test_retriever_respects_k_parameter(self, sample_docs):
        vs = FAISS.from_documents(sample_docs, _FakeEmbeddings())
        retriever = create_retriever(vs, k=2)
        results = retriever.invoke("mango export")
        assert len(results) <= 2

    def test_retrieved_documents_have_non_empty_content(self, faiss_retriever):
        results = faiss_retriever.invoke("Grade A requirements")
        assert all(len(d.page_content) > 0 for d in results)

    def test_retrieved_documents_contain_source_in_metadata(self, faiss_retriever):
        results = faiss_retriever.invoke("EU export requirements")
        assert all("source" in d.metadata for d in results)

    def test_vectorstore_save_and_load_roundtrip(self, sample_docs):
        """Persist FAISS index to disk, reload it, and query successfully."""
        embeddings = _FakeEmbeddings()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("language.vector_store_manager.get_cached_embeddings", return_value=embeddings):
                create_embeddings_and_vectorstore(sample_docs, tmpdir)
                loaded_vs, _ = load_vectorstore(tmpdir)

            results = loaded_vs.similarity_search("mango export grade", k=2)
            assert len(results) > 0

    def test_chunked_docs_retrievable_from_vectorstore(self, sample_raw_docs):
        """Chunks produced by chunk_documents() can be embedded and retrieved."""
        chunks = chunk_documents(sample_raw_docs)
        vs = FAISS.from_documents(chunks, _FakeEmbeddings())
        retriever = create_retriever(vs, k=3)
        results = retriever.invoke("Grade A mango export")
        assert len(results) > 0


# ─────────────────────────────────────────────────────────────────────────────
#  4. LLM Manager Integration (mocked LLM client — no API calls)
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMManagerIntegration:
    """LLMManager context formatting and query flow with a stubbed OpenAI client."""

    def _set_llm_response(self, manager: LLMManager, text: str) -> None:
        choice = MagicMock()
        choice.message.content = text
        manager.client.chat.completions.create.return_value = MagicMock(choices=[choice])

    # ── format_context ────────────────────────────────────────────────────────

    def test_format_context_includes_source_header_tags(self, llm_manager_mocked, sample_docs):
        context = llm_manager_mocked.format_context(sample_docs)
        assert "[Source 1:" in context
        assert "mango_standard.md" in context

    def test_format_context_non_empty_for_valid_docs(self, llm_manager_mocked, sample_docs):
        context = llm_manager_mocked.format_context(sample_docs)
        assert len(context) > 0

    def test_format_context_empty_string_for_no_docs(self, llm_manager_mocked):
        assert llm_manager_mocked.format_context([]) == ""

    def test_format_context_includes_page_content(self, llm_manager_mocked, sample_docs):
        context = llm_manager_mocked.format_context(sample_docs)
        assert "Grade A" in context

    # ── rag_query ─────────────────────────────────────────────────────────────

    def test_rag_query_returns_success_status(self, llm_manager_mocked, faiss_retriever):
        self._set_llm_response(llm_manager_mocked, "EU and Japan accept Grade A mangoes.")
        result = llm_manager_mocked.rag_query("Which markets accept Grade A?", faiss_retriever)
        assert result["status"] == "success"

    def test_rag_query_answer_contains_llm_response(self, llm_manager_mocked, faiss_retriever):
        self._set_llm_response(llm_manager_mocked, "EU, Japan accept Grade A.")
        result = llm_manager_mocked.rag_query("Grade A markets?", faiss_retriever)
        assert result["answer"] == "EU, Japan accept Grade A."

    def test_rag_query_sources_is_a_list(self, llm_manager_mocked, faiss_retriever):
        self._set_llm_response(llm_manager_mocked, "Some answer.")
        result = llm_manager_mocked.rag_query("export requirements", faiss_retriever)
        assert isinstance(result["sources"], list)

    def test_rag_query_num_retrieved_is_positive(self, llm_manager_mocked, faiss_retriever):
        self._set_llm_response(llm_manager_mocked, "Answer.")
        result = llm_manager_mocked.rag_query("mango quality standards", faiss_retriever)
        assert result["num_retrieved"] > 0

    def test_rag_query_context_length_is_positive(self, llm_manager_mocked, faiss_retriever):
        self._set_llm_response(llm_manager_mocked, "Answer.")
        result = llm_manager_mocked.rag_query("export standards", faiss_retriever)
        assert result["context_length"] > 0

    def test_rag_query_gracefully_handles_retriever_error(self, llm_manager_mocked):
        failing = MagicMock()
        failing.invoke.side_effect = RuntimeError("FAISS index missing")
        result = llm_manager_mocked.rag_query("any query", failing)
        assert result["status"] == "error"
        assert "sources" in result

    # ── export_rag_query ──────────────────────────────────────────────────────

    def test_export_prompt_contains_grade_label(self, llm_manager_mocked, sample_docs):
        metadata = {
            "export_grade": "A", "export_grade_impact": "minimal",
            "surface_quality_score": 90, "color_uniformity_score": 88,
            "total_defect_percentage": 1.5, "dark_spot_count": 0,
            "brown_spot_count": 0, "disease_percentage": 0.5,
        }
        context = llm_manager_mocked.format_context(sample_docs)
        prompt = llm_manager_mocked.create_export_prompt(metadata, context)
        assert "Grade A" in prompt

    def test_export_prompt_contains_quality_scores(self, llm_manager_mocked, sample_docs):
        metadata = {
            "export_grade": "B", "export_grade_impact": "moderate",
            "surface_quality_score": 75, "color_uniformity_score": 70,
            "total_defect_percentage": 4.0, "dark_spot_count": 2,
            "brown_spot_count": 1, "disease_percentage": 3.0,
        }
        context = llm_manager_mocked.format_context(sample_docs)
        prompt = llm_manager_mocked.create_export_prompt(metadata, context)
        assert "75" in prompt
        assert "70" in prompt

    def test_export_rag_query_returns_success_status(self, llm_manager_mocked, faiss_retriever):
        self._set_llm_response(
            llm_manager_mocked,
            "RECOMMENDED COUNTRIES:\n- UAE (Grade B)\nNOT RECOMMENDED:\n- Japan",
        )
        metadata = {
            "export_grade": "B", "export_grade_impact": "moderate",
            "surface_quality_score": 75, "color_uniformity_score": 70,
            "total_defect_percentage": 4.0, "dark_spot_count": 2,
            "brown_spot_count": 1, "disease_percentage": 3.0,
        }
        result = llm_manager_mocked.export_rag_query(metadata, faiss_retriever)
        assert result["status"] == "success"

    def test_export_rag_query_echoes_inspection_metadata(self, llm_manager_mocked, faiss_retriever):
        self._set_llm_response(llm_manager_mocked, "RECOMMENDED COUNTRIES:\n- EU (Grade A)")
        metadata = {
            "export_grade": "A", "export_grade_impact": "minimal",
            "surface_quality_score": 90, "color_uniformity_score": 88,
            "total_defect_percentage": 1.5, "dark_spot_count": 0,
            "brown_spot_count": 0, "disease_percentage": 0.5,
        }
        result = llm_manager_mocked.export_rag_query(metadata, faiss_retriever)
        assert result["inspection_metadata"] == metadata

    def test_export_rag_query_embeds_grade_in_retrieval_query(self, llm_manager_mocked, faiss_retriever):
        self._set_llm_response(llm_manager_mocked, "Suitable markets: EU.")
        metadata = {
            "export_grade": "B", "export_grade_impact": "moderate",
            "surface_quality_score": 75, "color_uniformity_score": 70,
            "total_defect_percentage": 4.0, "dark_spot_count": 2,
            "brown_spot_count": 1, "disease_percentage": 3.0,
        }
        result = llm_manager_mocked.export_rag_query(metadata, faiss_retriever)
        assert "B" in result["query"]

    def test_export_rag_query_handles_retriever_error(self, llm_manager_mocked):
        failing = MagicMock()
        failing.invoke.side_effect = RuntimeError("vector store unavailable")
        metadata = {
            "export_grade": "C", "export_grade_impact": "significant",
            "surface_quality_score": 40, "color_uniformity_score": 50,
            "total_defect_percentage": 12.0, "dark_spot_count": 5,
            "brown_spot_count": 3, "disease_percentage": 8.0,
        }
        result = llm_manager_mocked.export_rag_query(metadata, failing)
        assert result["status"] == "error"
        assert result["inspection_metadata"] == metadata


# ─────────────────────────────────────────────────────────────────────────────
#  5. Full End-to-End Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestFullEndToEndIntegration:
    """Smoke tests: frame → vision → export metadata → RAG (mocked LLM)."""

    def test_opencv_only_chain_frame_to_export_grade(self, dummy_frame):
        """No ML models: raw OpenCV frame → defect analysis → export grade."""
        analysis = MangoDefectDetector().detect_defects(dummy_frame)
        defect_dict = {
            "surface_quality_score": analysis.surface_quality_score,
            "color_uniformity_score": analysis.color_uniformity_score,
            "total_defect_percentage": analysis.total_defect_percentage,
            "dark_spot_count": analysis.dark_spot_count,
            "brown_spot_count": analysis.brown_spot_count,
            "export_grade_impact": analysis.export_grade_impact,
        }
        metadata = ExportAdvisor().build_metadata(defect_dict, disease_percentage=2.0)
        assert metadata["export_grade"] in ("A", "B", "C")
        assert metadata["disease_percentage"] == pytest.approx(2.0)

    def test_pipeline_classification_feeds_export_metadata(self, mango_pipeline, dummy_frame):
        """Mocked ML pipeline: vision frame → classify → defect → export metadata."""
        frame_result = mango_pipeline.process_single_frame(dummy_frame)
        analysis = MangoDefectDetector().detect_defects(dummy_frame)
        defect_dict = {
            "surface_quality_score": analysis.surface_quality_score,
            "color_uniformity_score": analysis.color_uniformity_score,
            "total_defect_percentage": analysis.total_defect_percentage,
            "dark_spot_count": analysis.dark_spot_count,
            "brown_spot_count": analysis.brown_spot_count,
            "export_grade_impact": analysis.export_grade_impact,
        }
        metadata = ExportAdvisor().build_metadata(
            defect_dict,
            disease_percentage=frame_result["disease_percentage"],
        )
        assert frame_result["class_name"] in Settings().CLASS_NAMES
        assert metadata["export_grade"] in ("A", "B", "C")
        assert metadata["disease_percentage"] == pytest.approx(frame_result["disease_percentage"])

    def test_full_pipeline_to_rag_query(
        self, mango_pipeline, dummy_frame, llm_manager_mocked, faiss_retriever
    ):
        """Full chain: vision → defect → export metadata → mocked RAG response."""
        mock_answer = (
            "RECOMMENDED COUNTRIES:\n- UAE (Grade B)\n- Malaysia (Grade B)\n"
            "NOT RECOMMENDED:\n- Japan (strict pesticide limits)\n"
            "ACTIONABLE STEPS:\n1. Sort and grade mangoes\n2. Obtain phytosanitary cert\n"
            "CONDITIONS:\n- Cold-chain required for UAE"
        )
        choice = MagicMock()
        choice.message.content = mock_answer
        llm_manager_mocked.client.chat.completions.create.return_value = MagicMock(
            choices=[choice]
        )

        # Vision stage
        frame_result = mango_pipeline.process_single_frame(dummy_frame)
        analysis = MangoDefectDetector().detect_defects(dummy_frame)
        defect_dict = {
            "surface_quality_score": analysis.surface_quality_score,
            "color_uniformity_score": analysis.color_uniformity_score,
            "total_defect_percentage": analysis.total_defect_percentage,
            "dark_spot_count": analysis.dark_spot_count,
            "brown_spot_count": analysis.brown_spot_count,
            "export_grade_impact": analysis.export_grade_impact,
        }
        metadata = ExportAdvisor().build_metadata(
            defect_dict,
            disease_percentage=frame_result["disease_percentage"],
        )

        # RAG stage
        rag_result = llm_manager_mocked.export_rag_query(metadata, faiss_retriever)

        assert rag_result["status"] == "success"
        assert len(rag_result["answer"]) > 0
        assert rag_result["inspection_metadata"]["export_grade"] in ("A", "B", "C")
        assert rag_result["num_retrieved"] > 0
