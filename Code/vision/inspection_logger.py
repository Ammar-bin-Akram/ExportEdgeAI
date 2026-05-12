"""
Inspection Logger
=================
Appends a human-readable summary of every pipeline run to a persistent
text log file (``logs/inspection_log.txt``).

Usage
-----
    from vision.inspection_logger import log_inspection

    log_inspection(
        results=results,            # list[dict] – per-mango pipeline dicts
        recommendations=recs,       # dict[int, dict] – keyed by 1-based index
        video_name="batch_01.mp4",
    )

Each call appends a timestamped block; the file is never overwritten.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Default log location: <project_root>/logs/inspection_log.txt
_DEFAULT_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"


def log_inspection(
    results: List[Dict[str, Any]],
    recommendations: Optional[Dict[int, Dict[str, Any]]] = None,
    video_name: str = "",
    log_dir: Optional[Path] = None,
) -> Path:
    """Append an inspection summary to the log file.

    Parameters
    ----------
    results : list[dict]
        The per-mango result dicts produced by the pipeline.
    recommendations : dict[int, dict] | None
        RAG/LLM recommendations keyed by 1-based mango index.
    video_name : str
        Source video filename (for reference).
    log_dir : Path | None
        Override the default ``logs/`` directory.

    Returns
    -------
    Path
        Absolute path to the log file that was written.
    """
    if recommendations is None:
        recommendations = {}

    log_dir = log_dir or _DEFAULT_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "inspection_log.txt"

    now = datetime.now()
    lines: List[str] = []

    # ── Header ────────────────────────────────────────────────────────
    lines.append("=" * 72)
    lines.append(f"  INSPECTION LOG  —  {now.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 72)
    if video_name:
        lines.append(f"Video source : {video_name}")
    lines.append(f"Mangoes analysed : {len(results)}")
    lines.append("")

    # ── Per-mango details ─────────────────────────────────────────────
    for idx, result in enumerate(results, 1):
        lines.append("-" * 60)
        lines.append(f"  Mango #{idx}")
        lines.append("-" * 60)

        # Classification
        pred = result.get("prediction", {})
        lines.append(f"  Classification : {pred.get('class_name', 'N/A')}")
        lines.append(f"  Confidence     : {pred.get('confidence', 0):.2%}")

        # Segmentation
        seg = result.get("segmentation")
        if seg:
            lines.append(f"  Disease %      : {seg.get('disease_percentage', 0):.2f}%")
            lines.append(f"  Seg. time      : {seg.get('segmentation_time', 0):.3f}s")
        else:
            lines.append("  Segmentation   : not available")

        # Defect analysis
        da = result.get("defect_analysis", {})
        if da:
            lines.append(f"  Defect count   : {da.get('defect_count', 0)}"
                         f"  (dark: {da.get('dark_spot_count', 0)},"
                         f" brown: {da.get('brown_spot_count', 0)})")
            lines.append(f"  Defect area    : {da.get('total_defect_percentage', 0):.2f}%")
            lines.append(f"  Surface quality: {da.get('surface_quality_score', 0):.0f}/100")
            lines.append(f"  Colour uniform.: {da.get('color_uniformity_score', 0):.0f}/100")
            lines.append(f"  Export impact  : {da.get('export_grade_impact', 'N/A')}")

        # LLM / RAG recommendation
        rec = recommendations.get(idx)
        if rec:
            status = rec.get("status", "unknown")
            lines.append(f"  LLM status     : {status}")
            answer = rec.get("answer", "")
            if answer:
                # Indent every line of the LLM response
                lines.append("  LLM response   :")
                for al in answer.strip().splitlines():
                    lines.append(f"    {al}")
            sources = rec.get("sources", [])
            if sources:
                lines.append(f"  Sources cited  : {len(sources)}")
                for s in sources:
                    lines.append(f"    - {s.get('source', '')} / {s.get('section', '')}")
        else:
            lines.append("  LLM recommendation : not requested")

        lines.append("")

    # ── Footer ────────────────────────────────────────────────────────
    lines.append(f"[end of inspection — {now.strftime('%H:%M:%S')}]")
    lines.append("")
    lines.append("")

    # Append to file (never overwrite)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return log_path


__all__ = ["log_inspection"]
