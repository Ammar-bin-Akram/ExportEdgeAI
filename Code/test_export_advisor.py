"""Quick test for ExportAdvisor — no video needed."""
import sys
sys.path.insert(0, 'd:/FYP/Code')

from vision.export_advisor import ExportAdvisor

advisor = ExportAdvisor()

# Simulate a defect analysis result (as produced by MangoDefectDetector)
fake_defect = {
    'surface_quality_score': 82,
    'color_uniformity_score': 70,
    'total_defect_percentage': 1.5,
    'dark_spot_count': 1,
    'brown_spot_count': 0,
    'export_grade_impact': 'minimal',
}

metadata = advisor.build_metadata(
    defect_analysis=fake_defect,
    disease_percentage=0.4,
)
print("Metadata:", metadata)

print("\nQuerying RAG pipeline (this may take a few seconds)...")
rec = advisor.get_recommendation(metadata)

print("\nStatus:", rec['status'])
if rec['status'] == 'success':
    print("\nAnswer:\n", rec['answer'])
    print("\nSources used:")
    for s in rec.get('sources', []):
        print(f"  {s['index']}. {s['source']} — {s['section']}")
else:
    print("Error:", rec.get('answer'))
