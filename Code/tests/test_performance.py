"""
Performance tests for ExportEdge AI pipeline components.

Measures inference latency and memory footprint of:
  - Defect detection  (OpenCV — no model file needed)
  - Export metadata construction  (pure Python)
  - RAG document pipeline  (chunking, FAISS build, FAISS retrieval)
  - LLM context formatting  (pure string operations)

Thresholds are intentionally generous so they pass on both
development machines and edge devices (Raspberry Pi 5).

Note: Real ML model latency (TFLite / PyTorch) is NOT tested here
because it requires model files and a working TFLite/PyTorch runtime.
Measure those directly on the target hardware.

Run from the Code/ directory:
    pytest tests/test_performance.py -v
"""

import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ── Path setup ────────────────────────────────────────────────────────────────
_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
_LANG_DIR = _CODE_DIR / "language"
if str(_LANG_DIR) not in sys.path:
    sys.path.insert(0, str(_LANG_DIR))

# ── Mock tflite before any project import triggers TF ─────────────────────────
_tflite_mock = MagicMock()
sys.modules.setdefault("tflite_runtime", _tflite_mock)
sys.modules.setdefault("tflite_runtime.interpreter", _tflite_mock)

# ── Project imports ────────────────────────────────────────────────────────────
from vision.defect_detector import MangoDefectDetector
from vision.export_advisor import ExportAdvisor, _map_to_grade
from language.chunker import chunk_documents
from language.llm_manager import LLMManager
from language.vector_store_manager import create_retriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS


# ── Shared helpers ─────────────────────────────────────────────────────────────

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


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


_SAMPLE_DOCS = [
    Document(page_content="Grade A mangoes must have less than 2% defect area.",
             metadata={"source": "std.md", "section_heading": "Grade A"}),
    Document(page_content="Grade B mangoes are allowed up to 5% defect area for certain markets.",
             metadata={"source": "std.md", "section_heading": "Grade B"}),
    Document(page_content="Export to EU requires a phytosanitary certificate.",
             metadata={"source": "eu.md", "section_heading": "EU Requirements"}),
    Document(page_content="Middle East markets accept Grade B with fumigation certificate.",
             metadata={"source": "me.md", "section_heading": "Middle East"}),
    Document(page_content="Japan imposes strict pesticide limits — Grade A only.",
             metadata={"source": "jp.md", "section_heading": "Japan"}),
]

_SAMPLE_RAW_DOCS = [
    {"content": f"# Section {i}\n\nContent for document {i} covering export requirements.",
     "filename": f"doc{i}.md", "filepath": f"/data/doc{i}.md"}
    for i in range(7)
]

_DEFECT_DICT = {
    "surface_quality_score": 85.0,
    "color_uniformity_score": 90.0,
    "total_defect_percentage": 1.5,
    "dark_spot_count": 1,
    "brown_spot_count": 0,
    "export_grade_impact": "minimal",
}

# ── Latency budgets (milliseconds) ────────────────────────────────────────────
# Generous to pass on Raspberry Pi 5 (4 GB) as well as dev machines.
BUDGET = {
    "defect_single":  500,    # one detect_defects() call
    "defect_average": 500,    # average over 10 calls
    "metadata":        50,    # build_metadata()
    "grade_1k":       100,    # 1 000 × _map_to_grade()
    "faiss_build":   1000,    # FAISS.from_documents() on 5 docs
    "faiss_query":    500,    # retriever.invoke()
    "format_ctx":     100,    # format_context()
    "chunking":      3000,    # chunk_documents() on 7 docs
}


# ─────────────────────────────────────────────────────────────────────────────
#  1. Defect Detection Latency
# ─────────────────────────────────────────────────────────────────────────────

class TestDefectDetectionPerformance:
    """MangoDefectDetector (pure OpenCV) must meet latency budgets."""

    @pytest.fixture(scope="class")
    def detector(self):
        return MangoDefectDetector()

    @pytest.fixture(scope="class")
    def frame(self):
        return _orange_frame()

    def test_single_frame_within_latency_budget(self, detector, frame):
        t = time.perf_counter()
        detector.detect_defects(frame)
        elapsed = _ms(t)
        assert elapsed < BUDGET["defect_single"], (
            f"detect_defects took {elapsed:.1f} ms — budget {BUDGET['defect_single']} ms"
        )

    def test_average_over_ten_runs_within_budget(self, detector, frame):
        detector.detect_defects(frame)  # warm-up
        times = []
        for _ in range(10):
            t = time.perf_counter()
            detector.detect_defects(frame)
            times.append(_ms(t))
        avg = sum(times) / len(times)
        assert avg < BUDGET["defect_average"], (
            f"Average detect_defects {avg:.1f} ms over 10 runs — budget {BUDGET['defect_average']} ms"
        )

    def test_latency_is_stable_across_runs(self, detector, frame):
        """Peak latency must not exceed 5× warm minimum (allows cold-start spike)."""
        detector.detect_defects(frame)  # warm-up
        times = []
        for _ in range(5):
            t = time.perf_counter()
            detector.detect_defects(frame)
            times.append(_ms(t))
        assert max(times) < 5 * min(times) + 50, (
            f"Latency unstable: min={min(times):.1f} ms, max={max(times):.1f} ms"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  2. Export Metadata Latency
# ─────────────────────────────────────────────────────────────────────────────

class TestExportMetadataPerformance:

    def test_build_metadata_within_budget(self):
        advisor = ExportAdvisor()
        t = time.perf_counter()
        advisor.build_metadata(_DEFECT_DICT, disease_percentage=0.5)
        elapsed = _ms(t)
        assert elapsed < BUDGET["metadata"], (
            f"build_metadata took {elapsed:.1f} ms — budget {BUDGET['metadata']} ms"
        )

    def test_grade_derivation_1000_calls_within_budget(self):
        t = time.perf_counter()
        for _ in range(1000):
            _map_to_grade(85.0, "minimal", 1.5)
        elapsed = _ms(t)
        assert elapsed < BUDGET["grade_1k"], (
            f"1 000× _map_to_grade took {elapsed:.1f} ms — budget {BUDGET['grade_1k']} ms"
        )

    def test_full_defect_to_metadata_chain_within_budget(self):
        """OpenCV detection + metadata build combined must stay within budget."""
        detector = MangoDefectDetector()
        advisor = ExportAdvisor()
        frame = _orange_frame()
        t = time.perf_counter()
        analysis = detector.detect_defects(frame)
        defect_dict = {
            "surface_quality_score": analysis.surface_quality_score,
            "color_uniformity_score": analysis.color_uniformity_score,
            "total_defect_percentage": analysis.total_defect_percentage,
            "dark_spot_count": analysis.dark_spot_count,
            "brown_spot_count": analysis.brown_spot_count,
            "export_grade_impact": analysis.export_grade_impact,
        }
        advisor.build_metadata(defect_dict, disease_percentage=0.5)
        elapsed = _ms(t)
        assert elapsed < BUDGET["defect_single"] + BUDGET["metadata"], (
            f"Full defect→metadata chain took {elapsed:.1f} ms"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  3. RAG Pipeline Latency
# ─────────────────────────────────────────────────────────────────────────────

class TestRAGPipelinePerformance:

    def test_faiss_vectorstore_build_within_budget(self):
        embeddings = _FakeEmbeddings()
        t = time.perf_counter()
        FAISS.from_documents(_SAMPLE_DOCS, embeddings)
        elapsed = _ms(t)
        assert elapsed < BUDGET["faiss_build"], (
            f"FAISS.from_documents took {elapsed:.1f} ms — budget {BUDGET['faiss_build']} ms"
        )

    def test_faiss_single_query_within_budget(self):
        embeddings = _FakeEmbeddings()
        vs = FAISS.from_documents(_SAMPLE_DOCS, embeddings)
        retriever = create_retriever(vs, k=3)
        retriever.invoke("mango export grade")  # warm-up
        t = time.perf_counter()
        retriever.invoke("mango export grade requirements")
        elapsed = _ms(t)
        assert elapsed < BUDGET["faiss_query"], (
            f"retriever.invoke took {elapsed:.1f} ms — budget {BUDGET['faiss_query']} ms"
        )

    def test_context_formatting_within_budget(self):
        with patch("language.llm_manager.OpenAI"):
            manager = LLMManager()
        manager.client = MagicMock()
        t = time.perf_counter()
        manager.format_context(_SAMPLE_DOCS)
        elapsed = _ms(t)
        assert elapsed < BUDGET["format_ctx"], (
            f"format_context took {elapsed:.1f} ms — budget {BUDGET['format_ctx']} ms"
        )

    def test_chunking_seven_docs_within_budget(self):
        t = time.perf_counter()
        chunk_documents(_SAMPLE_RAW_DOCS)
        elapsed = _ms(t)
        assert elapsed < BUDGET["chunking"], (
            f"chunk_documents (7 docs) took {elapsed:.1f} ms — budget {BUDGET['chunking']} ms"
        )

    def test_end_to_end_rag_chain_within_combined_budget(self):
        """chunk → embed → build FAISS → query must fit within combined budget."""
        combined_budget = BUDGET["faiss_build"] + BUDGET["faiss_query"] + BUDGET["chunking"]
        embeddings = _FakeEmbeddings()
        t = time.perf_counter()
        chunks = chunk_documents(_SAMPLE_RAW_DOCS)
        vs = FAISS.from_documents(chunks, embeddings)
        retriever = create_retriever(vs, k=3)
        retriever.invoke("mango export grade A requirements")
        elapsed = _ms(t)
        assert elapsed < combined_budget, (
            f"End-to-end RAG chain took {elapsed:.1f} ms — budget {combined_budget} ms"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  4. Memory Footprint
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not installed — pip install psutil")
class TestMemoryFootprint:
    """RSS memory increase must stay within budget after instantiating components."""

    def _rss_mb(self) -> float:
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

    def test_defect_detector_memory_increase_within_budget(self):
        before = self._rss_mb()
        detectors = [MangoDefectDetector() for _ in range(5)]
        after = self._rss_mb()
        increase = after - before
        assert increase < 100, (
            f"5× MangoDefectDetector increased RSS by {increase:.1f} MB (budget: 100 MB)"
        )
        _ = detectors  # keep alive until measured

    def test_faiss_index_memory_increase_within_budget(self):
        embeddings = _FakeEmbeddings()
        before = self._rss_mb()
        vs = FAISS.from_documents(_SAMPLE_DOCS * 10, embeddings)
        after = self._rss_mb()
        increase = after - before
        assert increase < 200, (
            f"FAISS index (50 docs) increased RSS by {increase:.1f} MB (budget: 200 MB)"
        )
        _ = vs
