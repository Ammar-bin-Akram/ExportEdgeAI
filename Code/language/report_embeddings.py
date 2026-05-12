"""Generate and manage embeddings for inspection reports."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

try:
    from .embeddings_cache import get_cached_embeddings
except ImportError:
    from embeddings_cache import get_cached_embeddings


class ReportEmbeddingService:
    """Create and manage embeddings for inspection reports for RAG."""

    def __init__(self) -> None:
        self.embeddings = get_cached_embeddings()
        self.vector_store: Optional[FAISS] = None
        self.retriever = None

    def create_report_embeddings(self, json_data: Dict[str, Any]) -> None:
        """
        Chunk and embed a report JSON into a FAISS vector store.
        Stores embeddings in memory (session-level).
        
        Args:
            json_data: Report JSON dict with structure:
                {
                    "report_name": "...",
                    "generated_at": "...",
                    "total_mangoes": N,
                    "mangoes": [
                        {
                            "mango_id": 1,
                            "classification": {...},
                            "defect_analysis": {...},
                            "export_recommendation": {...},
                            "segmentation": {...},
                        },
                        ...
                    ]
                }
        """
        chunks = self._chunk_report(json_data)
        if not chunks:
            return

        # Create FAISS vector store from chunks
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3},
        )

    def _chunk_report(self, json_data: Dict[str, Any]) -> List[Document]:
        """Split report into semantic chunks with metadata."""
        chunks: List[Document] = []
        report_name = json_data.get("report_name", "unknown_report")
        generated_at = json_data.get("generated_at", "")
        
        # Chunk 1: Overview
        overview_text = (
            f"Report: {report_name}\n"
            f"Generated: {generated_at}\n"
            f"Total mangoes inspected: {json_data.get('total_mangoes', 0)}\n\n"
            f"This is a mango inspection report containing quality assessment, "
            f"defect analysis, disease detection, and export recommendations for each mango."
        )
        chunks.append(
            Document(
                page_content=overview_text,
                metadata={
                    "source": report_name,
                    "section": "overview",
                    "mango_id": None,
                },
            )
        )

        # Chunk 2-N: Per-mango analysis
        for mango in json_data.get("mangoes", []):
            mango_id = mango.get("mango_id", "unknown")
            
            # Classification
            classification = mango.get("classification", {})
            if classification:
                class_text = (
                    f"Mango #{mango_id} Classification:\n"
                    f"Disease: {classification.get('class_name', 'N/A')}\n"
                    f"Confidence: {classification.get('confidence', 0):.1%}\n\n"
                    f"This mango was classified as {classification.get('class_name', 'unknown')} "
                    f"with {classification.get('confidence', 0):.1%} confidence."
                )
                chunks.append(
                    Document(
                        page_content=class_text,
                        metadata={
                            "source": report_name,
                            "section": "classification",
                            "mango_id": mango_id,
                        },
                    )
                )
            
            # Defect Analysis
            defects = mango.get("defect_analysis", {})
            if defects:
                defect_text = (
                    f"Mango #{mango_id} Defect Analysis:\n"
                    f"Surface Quality: {defects.get('surface_quality_score', 0):.0f}/100\n"
                    f"Colour Uniformity: {defects.get('color_uniformity_score', 0):.0f}/100\n"
                    f"Defect Area: {defects.get('total_defect_percentage', 0):.2f}%\n"
                    f"Dark Spots: {defects.get('dark_spot_count', 0)}\n"
                    f"Brown Spots: {defects.get('brown_spot_count', 0)}\n"
                    f"Export Grade Impact: {defects.get('export_grade_impact', 'N/A')}\n\n"
                    f"Defect assessment shows quality score {defects.get('surface_quality_score', 0):.0f}/100, "
                    f"with {defects.get('total_defect_percentage', 0):.2f}% defect coverage. "
                    f"Export impact is {defects.get('export_grade_impact', 'unknown')}."
                )
                chunks.append(
                    Document(
                        page_content=defect_text,
                        metadata={
                            "source": report_name,
                            "section": "defects",
                            "mango_id": mango_id,
                        },
                    )
                )
            
            # Segmentation (disease)
            segmentation = mango.get("segmentation")
            if segmentation:
                seg_text = (
                    f"Mango #{mango_id} Disease Segmentation:\n"
                    f"Disease Coverage: {segmentation.get('disease_percentage', 0):.2f}%\n\n"
                    f"Segmentation analysis detected {segmentation.get('disease_percentage', 0):.2f}% "
                    f"of the mango surface affected by disease."
                )
                chunks.append(
                    Document(
                        page_content=seg_text,
                        metadata={
                            "source": report_name,
                            "section": "segmentation",
                            "mango_id": mango_id,
                        },
                    )
                )
            
            # Export Recommendation
            recommendation = mango.get("export_recommendation")
            if recommendation and recommendation.get("status") == "success":
                rec_text = (
                    f"Mango #{mango_id} Export Recommendation:\n\n"
                    f"{recommendation.get('answer', 'No recommendation available')}\n\n"
                    f"This recommendation is based on regulatory standards and the mango's quality metrics."
                )
                chunks.append(
                    Document(
                        page_content=rec_text,
                        metadata={
                            "source": report_name,
                            "section": "export_recommendation",
                            "mango_id": mango_id,
                        },
                    )
                )
            
            # Price estimates
            prices = recommendation.get("price_estimates", []) if recommendation else []
            if prices:
                price_text = "Mango #{} Estimated Export Prices (UN Comtrade):\n".format(mango_id)
                for p in prices:
                    if p.get("status") == "success":
                        price_text += (
                            f"  {p.get('country', 'N/A')}: "
                            f"${p.get('price_usd_per_kg', 0):.4f}/kg ({p.get('period', 'N/A')})\n"
                        )
                    else:
                        price_text += (
                            f"  {p.get('country', 'N/A')}: "
                            f"Price data unavailable ({p.get('status', 'unknown')})\n"
                        )
                chunks.append(
                    Document(
                        page_content=price_text,
                        metadata={
                            "source": report_name,
                            "section": "prices",
                            "mango_id": mango_id,
                        },
                    )
                )

        return chunks

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """
        Retrieve relevant report chunks for a query.
        Falls back to empty list if no embeddings are loaded.
        """
        if not self.retriever:
            return []
        try:
            return self.retriever.invoke(query)
        except Exception:
            return []


__all__ = ["ReportEmbeddingService"]
