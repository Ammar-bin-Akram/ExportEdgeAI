"""
Export Advisor — bridge between the vision pipeline and the RAG language module.

Takes the outputs of:
  - MangoDefectDetector  (defect counts, quality scores, grade impact)
  - Segmentation model   (disease percentage)

No variety classification is required. Export-country prediction is driven
purely by physical inspection results (defect area, surface quality, colour
uniformity, grade impact).

Produces:
  1. A structured metadata dict  (build_metadata)
  2. An export-country recommendation via the RAG pipeline  (get_recommendation)

Usage
-----
    from vision.export_advisor import ExportAdvisor

    advisor = ExportAdvisor()          # lazy-loads RAG pipeline on first call

    metadata = advisor.build_metadata(
        defect_analysis={...},         # dict stored in result_data['defect_analysis']
        disease_percentage=0.5,
    )

    recommendation = advisor.get_recommendation(metadata)
    # → {"answer": "...", "sources": [...], "status": "success", ...}
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _map_to_grade(surface_quality: float, export_grade_impact: str,
                  total_defect_pct: float) -> str:
    """Derive a simple A/B/C export grade from defect analysis outputs."""
    if export_grade_impact == "minimal" and total_defect_pct <= 2.0:
        return "A"
    if export_grade_impact in ("minimal", "moderate") and total_defect_pct <= 5.0:
        return "B"
    return "C"


class ExportAdvisor:
    """
    Singleton-friendly helper that:
      - Converts raw vision results into a structured inspection dict
      - Loads the RAG pipeline lazily (only when get_recommendation is first called)
      - Calls LLMManager.export_rag_query() for country-specific advice
    """

    def __init__(self, vector_store_path: Optional[str] = None):
        """
        Args:
            vector_store_path: Override the default FAISS vector store path.
                               If None, uses the path from data_config.py.
        """
        self._vector_store_path = vector_store_path
        self._retriever = None
        self._llm_manager = None

    # ──────────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────────

    def build_metadata(
        self,
        defect_analysis: Dict[str, Any],
        disease_percentage: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Assemble a flat inspection metadata dict from vision pipeline outputs.

        Args:
            defect_analysis:   The 'defect_analysis' sub-dict stored in result_data.
                               Expected keys: dark_spot_count, brown_spot_count,
                               total_defect_percentage, color_uniformity_score,
                               surface_quality_score, export_grade_impact.
            disease_percentage: Segmentation disease coverage percentage (0–100).

        Returns:
            Dict ready to be passed to get_recommendation() or stored with results.
        """
        da = defect_analysis or {}
        surface_quality   = float(da.get("surface_quality_score", 0))
        color_uniformity  = float(da.get("color_uniformity_score", 0))
        defect_pct        = float(da.get("total_defect_percentage", 0.0))
        dark_count        = int(da.get("dark_spot_count", 0))
        brown_count       = int(da.get("brown_spot_count", 0))
        grade_impact      = str(da.get("export_grade_impact", "unknown"))

        grade = _map_to_grade(surface_quality, grade_impact, defect_pct)

        return {
            "surface_quality_score": surface_quality,
            "color_uniformity_score": color_uniformity,
            "total_defect_percentage": defect_pct,
            "dark_spot_count": dark_count,
            "brown_spot_count": brown_count,
            "export_grade_impact": grade_impact,
            "export_grade": grade,
            "disease_percentage": float(disease_percentage),
        }

    def get_recommendation(
        self,
        metadata: Dict[str, Any],
        retrieval_query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Query the RAG pipeline for export-country advice given inspection metadata.

        Lazy-loads the vector store and LLMManager on the first call.

        Args:
            metadata:          Dict from build_metadata().
            retrieval_query:   Optional override for the FAISS embedding query.

        Returns:
            Dict with keys: status, answer, sources, query, context_length,
                            num_retrieved, inspection_metadata.
        """
        try:
            self._ensure_rag_loaded()
        except Exception as e:
            logger.error("Failed to load RAG pipeline: %s", e)
            return {
                "status": "error",
                "error": str(e),
                "answer": f"RAG pipeline could not be loaded: {e}",
                "sources": [],
                "inspection_metadata": metadata,
            }

        return self._llm_manager.export_rag_query(
            inspection_metadata=metadata,
            retriever=self._retriever,
            retrieval_query=retrieval_query,
        )

    # ──────────────────────────────────────────────────────────────────────
    #  Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _ensure_rag_loaded(self) -> None:
        """Load the FAISS vector store + LLMManager once."""
        if self._retriever is not None and self._llm_manager is not None:
            return  # already loaded

        # Add the language module to sys.path so its imports resolve
        language_dir = Path(__file__).parent.parent / "language"
        if str(language_dir) not in sys.path:
            sys.path.insert(0, str(language_dir))

        from vector_store_manager import load_vectorstore, create_retriever
        from llm_manager import LLMManager
        from data_config import VECTOR_STORE_PATH, SEARCH_TYPE, RETRIEVAL_K

        vs_path = self._vector_store_path or VECTOR_STORE_PATH
        logger.info("Loading FAISS vector store from %s", vs_path)
        vector_store, _ = load_vectorstore(vs_path)
        self._retriever = create_retriever(
            vector_store, search_type=SEARCH_TYPE, k=RETRIEVAL_K
        )
        self._llm_manager = LLMManager()
        logger.info("ExportAdvisor RAG pipeline ready.")


def parse_structured_recommendation(answer: str) -> Dict[str, list]:
    """
    Parse the structured LLM response into discrete sections.

    Expected section headers (case-insensitive):
        RECOMMENDED COUNTRIES:, NOT RECOMMENDED:,
        ACTIONABLE STEPS:, CONDITIONS:

    Returns:
        Dict with keys 'recommended_countries', 'not_recommended',
        'actionable_steps', 'conditions'.  Each value is a list of
        stripped bullet strings.  Missing sections yield empty lists.
    """
    import re

    sections = {
        "recommended_countries": [],
        "not_recommended": [],
        "actionable_steps": [],
        "conditions": [],
    }

    # Map header patterns → dict keys
    header_map = [
        (r"recommended\s+countries", "recommended_countries"),
        (r"not\s+recommended", "not_recommended"),
        (r"actionable\s+steps", "actionable_steps"),
        (r"conditions", "conditions"),
    ]

    # Split text into blocks by known headers
    # Build a combined pattern that captures any header
    combined = "|".join(f"(?P<h{i}>{pat})" for i, (pat, _) in enumerate(header_map))
    header_re = re.compile(rf"^\s*(?:{combined})\s*[:：]", re.IGNORECASE | re.MULTILINE)

    matches = list(header_re.finditer(answer))
    for mi, m in enumerate(matches):
        # Determine which header matched
        key = None
        for i, (_, k) in enumerate(header_map):
            if m.group(f"h{i}") is not None:
                key = k
                break
        if key is None:
            continue

        start = m.end()
        end = matches[mi + 1].start() if mi + 1 < len(matches) else len(answer)
        block = answer[start:end].strip()

        # Extract bullet lines (starting with - or numbered like 1. 2.)
        for line in block.splitlines():
            line = line.strip()
            cleaned = re.sub(r"^[-•*]\s*|^\d+[.)\s]+", "", line).strip()
            if cleaned and cleaned.lower() != "none":
                sections[key].append(cleaned)

    return sections


__all__ = ["ExportAdvisor", "parse_structured_recommendation"]
