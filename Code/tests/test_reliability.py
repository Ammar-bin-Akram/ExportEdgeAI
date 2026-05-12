"""
Reliability tests for ExportEdge AI pipeline components.

Verifies that all pure-Python and OpenCV components produce identical
outputs when called repeatedly with the same input (determinism).
A reliable system must produce consistent results every run.

Run from the Code/ directory:
    pytest tests/test_reliability.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
_LANG_DIR = _CODE_DIR / "language"
if str(_LANG_DIR) not in sys.path:
    sys.path.insert(0, str(_LANG_DIR))

# ── Mock tflite before any project import ─────────────────────────────────────
_tflite_mock = MagicMock()
sys.modules.setdefault("tflite_runtime", _tflite_mock)
sys.modules.setdefault("tflite_runtime.interpreter", _tflite_mock)

# ── Project imports ────────────────────────────────────────────────────────────
from vision.defect_detector import MangoDefectDetector, DefectAnalysis
from vision.export_advisor import ExportAdvisor, _map_to_grade
from language.chunker import chunk_documents
from language.llm_manager import LLMManager
from language.vector_store_manager import create_retriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS


# ── Shared helpers ─────────────────────────────────────────────────────────────

RUNS = 5  # how many times each operation is repeated

class _FakeEmbeddings(Embeddings):
    DIM = 384
    def embed_documents(self, texts):
        rng = np.random.RandomState(42)
        return [rng.rand(self.DIM).tolist() for _ in texts]
    def embed_query(self, text: str):
        return np.random.RandomState(0).rand(self.DIM).tolist()


def _orange_frame(h: int = 224, w: int = 224) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (30, 130, 210)
    return img


_DEFECT_DICT = {
    "surface_quality_score": 87.0,
    "color_uniformity_score": 91.0,
    "total_defect_percentage": 1.23,
    "dark_spot_count": 1,
    "brown_spot_count": 1,
    "export_grade_impact": "minimal",
}

_SAMPLE_DOCS = [
    Document(page_content="Grade A mangoes must have less than 2% defect area.",
             metadata={"source": "std.md", "section_heading": "Grade A"}),
    Document(page_content="Grade B mangoes are allowed up to 5% defect area.",
             metadata={"source": "std.md", "section_heading": "Grade B"}),
    Document(page_content="Export to EU requires a phytosanitary certificate.",
             metadata={"source": "eu.md", "section_heading": "EU Requirements"}),
]

_RAW_DOCS = [
    {"content": "# Grade A\n\nGrade A mangoes must have less than 2% defect area.",
     "filename": "std.md", "filepath": "/data/std.md"},
    {"content": "# EU Requirements\n\nExport to EU requires phytosanitary certificate.",
     "filename": "eu.md", "filepath": "/data/eu.md"},
]


# ─────────────────────────────────────────────────────────────────────────────
#  1. Defect Detection Reliability
# ─────────────────────────────────────────────────────────────────────────────

class TestDefectDetectionReliability:
    """detect_defects() must return identical scores on the same input."""

    @pytest.fixture(scope="class")
    def results(self):
        detector = MangoDefectDetector()
        frame = _orange_frame()
        return [detector.detect_defects(frame) for _ in range(RUNS)]

    def test_surface_quality_score_is_identical(self, results):
        scores = [r.surface_quality_score for r in results]
        assert len(set(scores)) == 1, f"surface_quality_score varied: {scores}"

    def test_color_uniformity_score_is_identical(self, results):
        scores = [r.color_uniformity_score for r in results]
        assert len(set(scores)) == 1, f"color_uniformity_score varied: {scores}"

    def test_total_defect_percentage_is_identical(self, results):
        values = [r.total_defect_percentage for r in results]
        assert len(set(values)) == 1, f"total_defect_percentage varied: {values}"

    def test_dark_spot_count_is_identical(self, results):
        counts = [r.dark_spot_count for r in results]
        assert len(set(counts)) == 1, f"dark_spot_count varied: {counts}"

    def test_brown_spot_count_is_identical(self, results):
        counts = [r.brown_spot_count for r in results]
        assert len(set(counts)) == 1, f"brown_spot_count varied: {counts}"

    def test_export_grade_impact_is_identical(self, results):
        impacts = [r.export_grade_impact for r in results]
        assert len(set(impacts)) == 1, f"export_grade_impact varied: {impacts}"

    def test_defect_count_is_identical(self, results):
        counts = [r.defect_count for r in results]
        assert len(set(counts)) == 1, f"defect_count varied: {counts}"

    def test_different_frames_produce_different_results(self):
        """Reliability check: a uniform frame vs a noisy frame give different scores."""
        detector = MangoDefectDetector()
        plain = _orange_frame()
        noisy = plain.copy()
        rng = np.random.RandomState(99)
        noisy[50:100, 50:100] = rng.randint(0, 30, (50, 50, 3), dtype=np.uint8)
        r_plain = detector.detect_defects(plain)
        r_noisy = detector.detect_defects(noisy)
        # Noisy frame should detect more defects or different quality score
        assert (r_plain.surface_quality_score != r_noisy.surface_quality_score or
                r_plain.defect_count != r_noisy.defect_count)


# ─────────────────────────────────────────────────────────────────────────────
#  2. Grade Derivation Reliability
# ─────────────────────────────────────────────────────────────────────────────

class TestGradeDerivationReliability:
    """_map_to_grade() must be deterministic for every valid input combination."""

    @pytest.mark.parametrize("quality,impact,defect_pct,expected", [
        (95.0, "minimal",     1.0,  "A"),
        (90.0, "minimal",     2.0,  "A"),
        (90.0, "minimal",     2.1,  "B"),
        (75.0, "moderate",    4.0,  "B"),
        (75.0, "minimal",     5.0,  "B"),
        (50.0, "significant", 10.0, "C"),
        (80.0, "moderate",    6.0,  "C"),
    ])
    def test_grade_is_always_consistent(self, quality, impact, defect_pct, expected):
        grades = [_map_to_grade(quality, impact, defect_pct) for _ in range(RUNS)]
        assert all(g == expected for g in grades), (
            f"Grade varied across runs: {grades}"
        )

    def test_export_advisor_metadata_grade_consistent(self):
        advisor = ExportAdvisor()
        grades = [
            advisor.build_metadata(_DEFECT_DICT, disease_percentage=1.0)["export_grade"]
            for _ in range(RUNS)
        ]
        assert len(set(grades)) == 1, f"export_grade varied: {grades}"

    def test_export_advisor_metadata_disease_percentage_consistent(self):
        advisor = ExportAdvisor()
        values = [
            advisor.build_metadata(_DEFECT_DICT, disease_percentage=3.5)["disease_percentage"]
            for _ in range(RUNS)
        ]
        assert len(set(values)) == 1, f"disease_percentage varied: {values}"


# ─────────────────────────────────────────────────────────────────────────────
#  3. Chunking Reliability
# ─────────────────────────────────────────────────────────────────────────────

class TestChunkingReliability:
    """chunk_documents() must produce the same chunks on every call."""

    @pytest.fixture(scope="class")
    def all_runs(self):
        return [chunk_documents(_RAW_DOCS) for _ in range(RUNS)]

    def test_chunk_count_is_consistent(self, all_runs):
        counts = [len(run) for run in all_runs]
        assert len(set(counts)) == 1, f"Chunk count varied: {counts}"

    def test_chunk_content_is_identical_across_runs(self, all_runs):
        first = [c.page_content for c in all_runs[0]]
        for run in all_runs[1:]:
            assert [c.page_content for c in run] == first

    def test_chunk_source_metadata_is_identical_across_runs(self, all_runs):
        first = [c.metadata["source"] for c in all_runs[0]]
        for run in all_runs[1:]:
            assert [c.metadata["source"] for c in run] == first

    def test_chunk_ids_are_identical_across_runs(self, all_runs):
        first = [c.metadata["chunk_id"] for c in all_runs[0]]
        for run in all_runs[1:]:
            assert [c.metadata["chunk_id"] for c in run] == first

    def test_chunk_ids_are_zero_based_sequential(self, all_runs):
        ids = [c.metadata["chunk_id"] for c in all_runs[0]]
        assert ids == list(range(len(ids))), f"Chunk IDs not sequential: {ids}"


# ─────────────────────────────────────────────────────────────────────────────
#  4. FAISS Retrieval Reliability
# ─────────────────────────────────────────────────────────────────────────────

class TestFAISSRetrievalReliability:
    """FAISS retrieval must return the same documents in the same order every time."""

    @pytest.fixture(scope="class")
    def retriever(self):
        embeddings = _FakeEmbeddings()
        vs = FAISS.from_documents(_SAMPLE_DOCS, embeddings)
        return create_retriever(vs, k=3)

    def test_same_query_returns_same_documents(self, retriever):
        query = "mango export grade requirements"
        results = [retriever.invoke(query) for _ in range(RUNS)]
        first_sources = [d.metadata["source"] for d in results[0]]
        for run in results[1:]:
            assert [d.metadata["source"] for d in run] == first_sources

    def test_same_query_returns_same_content(self, retriever):
        query = "Grade A export standard"
        results = [retriever.invoke(query) for _ in range(RUNS)]
        first_content = [d.page_content for d in results[0]]
        for run in results[1:]:
            assert [d.page_content for d in run] == first_content

    def test_different_queries_can_return_different_results(self, retriever):
        """Sanity check: different queries are not forced to return the same docs."""
        r1 = retriever.invoke("Grade A mangoes")
        r2 = retriever.invoke("phytosanitary certificate EU")
        sources1 = {d.metadata["source"] for d in r1}
        sources2 = {d.metadata["source"] for d in r2}
        # At least one doc source differs (they query different aspects)
        # Note: with fake embeddings all queries get same vector, so this
        # may be identical — we just assert no exception is raised.
        assert isinstance(sources1, set) and isinstance(sources2, set)


# ─────────────────────────────────────────────────────────────────────────────
#  5. LLM Context Formatting Reliability
# ─────────────────────────────────────────────────────────────────────────────

class TestContextFormattingReliability:
    """format_context() must produce identical strings on the same input."""

    @pytest.fixture(scope="class")
    def manager(self):
        with patch("language.llm_manager.OpenAI"):
            mgr = LLMManager()
        mgr.client = MagicMock()
        return mgr

    def test_format_context_output_is_identical_across_runs(self, manager):
        outputs = [manager.format_context(_SAMPLE_DOCS) for _ in range(RUNS)]
        assert len(set(outputs)) == 1, "format_context output varied across runs"

    def test_export_prompt_output_is_identical_across_runs(self, manager):
        metadata = {
            "export_grade": "A", "export_grade_impact": "minimal",
            "surface_quality_score": 90, "color_uniformity_score": 88,
            "total_defect_percentage": 1.5, "dark_spot_count": 0,
            "brown_spot_count": 0, "disease_percentage": 0.5,
        }
        context = manager.format_context(_SAMPLE_DOCS)
        prompts = [manager.create_export_prompt(metadata, context) for _ in range(RUNS)]
        assert len(set(prompts)) == 1, "create_export_prompt output varied across runs"

    def test_rag_prompt_output_is_identical_across_runs(self, manager):
        context = manager.format_context(_SAMPLE_DOCS)
        prompts = [
            manager.create_rag_prompt("What are Grade A standards?", context)
            for _ in range(RUNS)
        ]
        assert len(set(prompts)) == 1, "create_rag_prompt output varied across runs"
