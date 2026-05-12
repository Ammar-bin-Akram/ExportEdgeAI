"""Quick test: export recommendation via LM Studio."""
import sys, time
sys.path.insert(0, ".")

from vision.export_advisor import ExportAdvisor

# Mock defect analysis (Grade B mango with anthracnose)
mock_defect = {
    "dark_spot_count": 3,
    "brown_spot_count": 2,
    "total_defect_percentage": 3.5,
    "color_uniformity_score": 72,
    "surface_quality_score": 65,
    "export_grade_impact": "moderate",
}

print("1. Building metadata...")
advisor = ExportAdvisor()
metadata = advisor.build_metadata(mock_defect, disease_percentage=2.1)
print(f"   Grade: {metadata['export_grade']}")
print()

print("2. Getting recommendation from LM Studio...")
print("   (This may take a few minutes with the 1.5B model...)")
start = time.time()
rec = advisor.get_recommendation(metadata)
elapsed = time.time() - start
print(f"   Status: {rec['status']}  ({elapsed:.0f}s)")
print(f"   Sources: {rec.get('num_retrieved', 'N/A')}")
print()
print("=== ANSWER (first 800 chars) ===")
print(rec["answer"][:800])
