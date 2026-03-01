"""
Report Generator — produces a downloadable PDF inspection report.

Converts vision pipeline results (images, defect metrics, classification,
segmentation, and optional RAG export recommendation) into a styled HTML
document rendered to PDF via WeasyPrint.

Usage
-----
    from vision.report_generator import ReportGenerator

    gen = ReportGenerator()
    pdf_bytes = gen.generate(results, recommendations={1: rec_dict, ...})
    # → bytes ready for st.download_button or disk write
"""

from __future__ import annotations

import base64
import io
import logging
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

# Path to the Jinja2 templates directory (sibling of this file)
_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def _img_to_b64(image: np.ndarray) -> str:
    """Encode a BGR OpenCV image to a base64 PNG string."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    success, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    # imencode wants BGR, but we already have BGR as input — just encode directly
    success, buf = cv2.imencode(".png", image)
    if not success:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _ns(d: dict, keep_dict_keys: set | None = None) -> SimpleNamespace:
    """Recursively convert a dict to SimpleNamespace for dot-access in Jinja2.
    
    Args:
        keep_dict_keys: Set of key names whose values should remain as plain
                        dicts (e.g. 'probabilities') so that .items() works
                        in templates.
    """
    keep_dict_keys = keep_dict_keys or set()
    for k, v in d.items():
        if isinstance(v, dict) and k not in keep_dict_keys:
            d[k] = _ns(v, keep_dict_keys)
    return SimpleNamespace(**d)


class ReportGenerator:
    """Renders inspection results into an HTML string and optionally a PDF."""

    def __init__(self, template_name: str = "inspection_report.html"):
        self._env = Environment(
            loader=FileSystemLoader(str(_TEMPLATE_DIR)),
            autoescape=False,  # we trust our own template
        )
        self._template = self._env.get_template(template_name)

    # ──────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────

    def generate(
        self,
        results: List[Dict[str, Any]],
        recommendations: Optional[Dict[int, Dict[str, Any]]] = None,
        video_name: Optional[str] = None,
        as_html: bool = False,
    ) -> bytes:
        """
        Produce an inspection report from pipeline results.

        Args:
            results:          List of result_data dicts (one per detected mango).
            recommendations:  {1-based-index: recommendation_dict, ...} for any
                              mangoes that had the RAG export query run.
            video_name:       Original video filename (shown in header).
            as_html:          If True, return UTF-8 HTML bytes instead of PDF.

        Returns:
            PDF bytes (or HTML bytes if as_html=True).
        """
        recommendations = recommendations or {}
        template_data = self._prepare_template_data(results, recommendations, video_name)
        html = self._template.render(**template_data)

        if as_html:
            return html.encode("utf-8")

        return self._html_to_pdf(html)

    # ──────────────────────────────────────────────────────────────────
    #  Internal helpers
    # ──────────────────────────────────────────────────────────────────

    def _prepare_template_data(
        self,
        results: List[Dict[str, Any]],
        recommendations: Dict[int, Dict[str, Any]],
        video_name: Optional[str],
    ) -> dict:
        """Build the context dict consumed by the Jinja2 template."""
        from vision.export_advisor import _map_to_grade, parse_structured_recommendation

        rendered_results = []

        for idx, r in enumerate(results, start=1):
            # ── Images (only original + segmentation) ────────────
            images: Dict[str, str] = {}
            if r.get("original_roi") is not None:
                images["original"] = _img_to_b64(r["original_roi"])
            seg = r.get("segmentation")
            if seg and seg.get("overlay") is not None:
                images["segmentation"] = _img_to_b64(seg["overlay"])

            # ── Defect analysis ──────────────────────────────────
            da = r.get("defect_analysis", {})

            # ── Derived grade ────────────────────────────────────
            grade = _map_to_grade(
                float(da.get("surface_quality_score", 0)),
                str(da.get("export_grade_impact", "unknown")),
                float(da.get("total_defect_percentage", 0)),
            )

            # ── Disease percentage (from segmentation) ───────────
            disease_pct = None
            if seg:
                disease_pct = seg.get("disease_percentage", 0)

            # ── Recommendation (if available) ────────────────────
            rec = recommendations.get(idx)
            rec_parsed = {
                "recommended_countries": [],
                "not_recommended": [],
                "actionable_steps": [],
                "conditions": [],
            }
            if rec and rec.get("status") == "success":
                rec_parsed = parse_structured_recommendation(rec.get("answer", ""))

            # ── Assemble entry ───────────────────────────────────
            entry = {
                "prediction_name": r.get("prediction", {}).get("class_name", "N/A"),
                "defect_analysis": {
                    "dark_spot_count": da.get("dark_spot_count", 0),
                    "brown_spot_count": da.get("brown_spot_count", 0),
                    "total_defect_percentage": da.get("total_defect_percentage", 0),
                    "color_uniformity_score": da.get("color_uniformity_score", 0),
                    "surface_quality_score": da.get("surface_quality_score", 0),
                    "export_grade_impact": da.get("export_grade_impact", "N/A"),
                },
                "images": _ns(images),
                "grade": grade,
                "disease_pct": disease_pct,
                "recommendation": None,
                "rec_parsed": rec_parsed,
            }

            if rec:
                entry["recommendation"] = _ns({
                    "status": rec.get("status", "error"),
                    "answer": rec.get("answer", ""),
                    "sources": rec.get("sources", []),
                })

            entry["defect_analysis"] = _ns(entry["defect_analysis"])
            rendered_results.append(_ns(entry))

        return {
            "results": rendered_results,
            "report_date": datetime.now().strftime("%d %B %Y, %H:%M"),
            "video_name": video_name,
        }

    @staticmethod
    def _html_to_pdf(html: str) -> bytes:
        """Render HTML string to PDF bytes.
        
        Tries WeasyPrint first (best quality) then falls back to xhtml2pdf
        (pure-Python, no system deps required).
        """
        # Attempt 1 — WeasyPrint (needs GTK3/Pango runtime on Windows)
        try:
            from weasyprint import HTML as WeasyprintHTML
            pdf_bytes = WeasyprintHTML(string=html).write_pdf()
            logger.info("PDF generated via WeasyPrint — %d bytes", len(pdf_bytes))
            return pdf_bytes
        except (ImportError, OSError) as e:
            logger.warning("WeasyPrint unavailable (%s), falling back to xhtml2pdf", e)

        # Attempt 2 — xhtml2pdf (pure Python)
        try:
            from xhtml2pdf import pisa
            buf = io.BytesIO()
            pisa_status = pisa.CreatePDF(html, dest=buf)
            if pisa_status.err:
                raise RuntimeError(f"xhtml2pdf reported {pisa_status.err} error(s)")
            pdf_bytes = buf.getvalue()
            logger.info("PDF generated via xhtml2pdf — %d bytes", len(pdf_bytes))
            return pdf_bytes
        except ImportError:
            raise ImportError(
                "No PDF backend available. Install one of:\n"
                "  pip install weasyprint   (needs GTK3 runtime)\n"
                "  pip install xhtml2pdf    (pure Python)"
            )
