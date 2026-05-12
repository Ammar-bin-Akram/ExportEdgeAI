"""SQLite persistence for export reports and mango-level report items."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from vision.export_advisor import parse_structured_recommendation


class ExportReportRepository:
    """Stores report-level and item-level export analysis records."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        default_path = Path(__file__).resolve().parents[2] / "logs" / "export_reports.db"
        self.db_path = Path(db_path) if db_path else default_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS ExportReport (
                    report_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT NOT NULL,
                    generated_timestamp TEXT NOT NULL,
                    overall_summary TEXT
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS ExportReportItem (
                    item_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_id INTEGER NOT NULL,
                    image_snapshot BLOB,
                    health_status TEXT,
                    predicted_disease TEXT,
                    recommended_export_country TEXT,
                    estimated_price_range TEXT,
                    regulatory_citations TEXT,
                    FOREIGN KEY (report_id) REFERENCES ExportReport(report_id)
                )
                """
            )
            conn.commit()

    @staticmethod
    def _encode_snapshot(result: Dict[str, Any]) -> Optional[bytes]:
        image = result.get("original_roi")
        if image is None:
            return None
        ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
        if not ok:
            return None
        return bytes(buf)

    @staticmethod
    def _health_status(defect_analysis: Dict[str, Any]) -> str:
        impact = str(defect_analysis.get("export_grade_impact", "")).lower()
        if impact == "minimal":
            return "healthy"
        if impact == "moderate":
            return "monitor"
        if impact:
            return "diseased"
        return "unknown"

    @staticmethod
    def _predicted_disease(result: Dict[str, Any]) -> str:
        pred = result.get("prediction", {})
        name = str(pred.get("class_name", "")).strip()
        return name or "unknown"

    @staticmethod
    def _recommended_country_text(rec: Optional[Dict[str, Any]]) -> str:
        if not rec or rec.get("status") != "success":
            return ""
        parsed = parse_structured_recommendation(rec.get("answer", ""))
        countries = parsed.get("recommended_countries", [])
        return ", ".join(countries)

    @staticmethod
    def _estimated_price_range(rec: Optional[Dict[str, Any]]) -> str:
        if not rec:
            return ""
        prices = rec.get("price_estimates", []) or []
        vals = []
        for p in prices:
            if p.get("status") == "success":
                try:
                    vals.append(float(p.get("price_usd_per_kg", 0)))
                except (TypeError, ValueError):
                    continue
        if not vals:
            return ""
        lo = min(vals)
        hi = max(vals)
        if abs(hi - lo) < 1e-9:
            return f"{lo:.4f} USD/kg"
        return f"{lo:.4f}-{hi:.4f} USD/kg"

    @staticmethod
    def _regulatory_citations(rec: Optional[Dict[str, Any]]) -> str:
        if not rec:
            return ""
        srcs = rec.get("sources", []) or []
        compact = []
        for s in srcs:
            source = str(s.get("source", "")).strip()
            section = str(s.get("section", "")).strip()
            if source or section:
                compact.append({"source": source, "section": section})
        return json.dumps(compact, ensure_ascii=False)

    @staticmethod
    def _build_overall_summary(results: List[Dict[str, Any]], recommendations: Dict[int, Dict[str, Any]]) -> str:
        total = len(results)
        with_rec = sum(1 for i in range(1, total + 1) if recommendations.get(i, {}).get("status") == "success")
        diseased = 0
        for r in results:
            impact = str((r.get("defect_analysis") or {}).get("export_grade_impact", "")).lower()
            if impact in {"moderate", "significant"}:
                diseased += 1
        return f"Batch processed: {total} mangoes, recommendations: {with_rec}, higher-risk: {diseased}."

    def save_report(
        self,
        batch_id: str,
        results: List[Dict[str, Any]],
        recommendations: Dict[int, Dict[str, Any]],
        generated_timestamp: Optional[str] = None,
        overall_summary: Optional[str] = None,
    ) -> int:
        generated_timestamp = generated_timestamp or datetime.utcnow().isoformat()
        overall_summary = overall_summary or self._build_overall_summary(results, recommendations)

        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO ExportReport (batch_id, generated_timestamp, overall_summary)
                VALUES (?, ?, ?)
                """,
                (batch_id, generated_timestamp, overall_summary),
            )
            report_id = int(cur.lastrowid)

            for idx, result in enumerate(results, start=1):
                rec = recommendations.get(idx)
                da = result.get("defect_analysis", {}) or {}
                cur.execute(
                    """
                    INSERT INTO ExportReportItem (
                        report_id,
                        image_snapshot,
                        health_status,
                        predicted_disease,
                        recommended_export_country,
                        estimated_price_range,
                        regulatory_citations
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        report_id,
                        self._encode_snapshot(result),
                        self._health_status(da),
                        self._predicted_disease(result),
                        self._recommended_country_text(rec),
                        self._estimated_price_range(rec),
                        self._regulatory_citations(rec),
                    ),
                )

            conn.commit()
            return report_id


__all__ = ["ExportReportRepository"]
