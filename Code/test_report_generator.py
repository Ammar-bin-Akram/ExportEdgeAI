"""Quick smoke test for ReportGenerator — no video or models needed."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np

def main():
    from vision.report_generator import ReportGenerator

    gen = ReportGenerator()

    # Fake images (small orange squares to simulate mango)
    fake_img = np.full((224, 224, 3), [0, 140, 255], dtype=np.uint8)  # BGR orange
    fake_vis = fake_img.copy()

    results = [
        {
            "frame_idx": 120,
            "timestamp_ms": 4000,
            "motion_area": 15432,
            "prediction": {
                "class_name": "Anthracnose",
                "confidence": 0.87,
                "probabilities": {
                    "Healthy": 0.05,
                    "Anthracnose": 0.87,
                    "Alternaria": 0.04,
                    "Black Mould Rot": 0.02,
                    "Stem End Rot": 0.02,
                },
            },
            "original_roi": fake_img,
            "processed_roi": fake_img,
            "defect_analysis": {
                "defect_count": 3,
                "dark_spot_count": 1,
                "brown_spot_count": 2,
                "total_defect_percentage": 3.5,
                "color_uniformity_score": 72,
                "surface_quality_score": 65,
                "export_grade_impact": "moderate",
                "defect_visualization": fake_vis,
            },
            "segmentation": {
                "disease_percentage": 4.2,
                "overlay": fake_img,
                "mask": fake_img[:, :, 0],
                "segmentation_time": 0.45,
            },
        },
    ]

    # Fake recommendation (structured format matching the updated LLM prompt)
    recommendations = {
        1: {
            "status": "success",
            "answer": (
                "RECOMMENDED COUNTRIES:\n"
                "- UAE / Gulf States (Grade B acceptable under GCC Standardization Org, GSO 1016)\n"
                "- Saudi Arabia (Moderate defects within SASO tolerance for Class II)\n"
                "- Malaysia (acceptable under MS 490:2005 for general market)\n"
                "\n"
                "NOT RECOMMENDED:\n"
                "- Japan (zero tolerance on visible defects, MAFF Plant Protection Act)\n"
                "- EU markets (defect area exceeds 2% EC Regulation 543/2011 Class I limit)\n"
                "\n"
                "ACTIONABLE STEPS:\n"
                "1. Apply hot water treatment (HWT) at 48°C for 60 minutes for fruit fly disinfestation.\n"
                "2. Sort and grade under CODEX STAN 184 Class II criteria.\n"
                "3. Ensure phytosanitary certificate is issued by NPPO before shipment.\n"
                "4. Label packaging with lot number, grade, and country of origin.\n"
                "\n"
                "CONDITIONS:\n"
                "- Phytosanitary certificate required (ISPM 12 compliant)\n"
                "- Maximum residue limits (MRLs) must be within destination-country thresholds\n"
                "- Cold chain must be maintained at 10-13°C during transit\n"
            ),
            "sources": [
                {
                    "index": 1,
                    "source": "Mango_Inspection_Instructions.md",
                    "section": "Grade Standards",
                    "content_preview": "Grade B mangoes may have superficial blemishes...",
                },
                {
                    "index": 2,
                    "source": "Manual_For_Export_of_Mangoes.md",
                    "section": "Pre-export Requirements",
                    "content_preview": "All mangoes must pass phytosanitary inspection...",
                },
            ],
        }
    }

    # Test HTML generation
    html_bytes = gen.generate(results, recommendations=recommendations,
                              video_name="test_video.mp4", as_html=True)
    print(f"[OK] HTML generated — {len(html_bytes):,} bytes")

    # Test PDF generation
    try:
        pdf_bytes = gen.generate(results, recommendations=recommendations,
                                 video_name="test_video.mp4")
        print(f"[OK] PDF generated — {len(pdf_bytes):,} bytes")

        # Write PDF to disk for manual inspection
        out_path = os.path.join(os.path.dirname(__file__), "test_report.pdf")
        with open(out_path, "wb") as f:
            f.write(pdf_bytes)
        print(f"[OK] Saved to {out_path}")
    except ImportError as e:
        print(f"[SKIP] PDF generation skipped (missing system deps): {e}")
    except Exception as e:
        print(f"[WARN] PDF generation failed: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
