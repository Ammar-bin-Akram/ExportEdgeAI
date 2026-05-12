"""Test refined detector on conveyor-belt ROIs, preprocessed ROIs, and dataset images."""
import sys, cv2
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from vision.defect_detector import MangoDefectDetector
from vision.image_processor import ImageProcessor

det = MangoDefectDetector()                     # skip_clahe=True (default for preprocessed)
det_raw = MangoDefectDetector({'skip_clahe': False})  # for raw images
proc = ImageProcessor()
out = Path(__file__).parent / 'defect_test_results'
out.mkdir(exist_ok=True)

root = Path(__file__).parent.parent

# Test images
tests = {
    'conveyor_roi': root / 'relevant_frames_test' / 'peak_000074_3340ms_area47101.jpg',
    'conveyor_fullframe': root / 'relevant_frames1' / 'peak_000156_7750_fullframe.jpg',
    'dataset_class_I': root / 'Mango Variety and Grading Dataset' / 'Dataset' / 'Grading_dataset' / 'Class_I' / 'IMG_20210703_151539.jpg',
    'dataset_class_II': root / 'Mango Variety and Grading Dataset' / 'Dataset' / 'Grading_dataset' / 'Class_II' / 'IMG_20210703_155544.jpg',
}

def run_test(name, img, detector, tag=''):
    r = detector.detect_defects(img, save_debug=True, debug_path=out / f"debug_{name}{tag}")
    label = f"{name}{tag}"
    print(f"\n[{label}] {img.shape}")
    print(f"  Defects: {r.defect_count} (dark={r.dark_spot_count}, brown={r.brown_spot_count})")
    print(f"  Defect %: {r.total_defect_percentage:.2f}%  |  Mango area: {r.mango_area:.0f}px")
    print(f"  Color uniformity: {r.color_uniformity_score:.1f}  |  Surface quality: {r.surface_quality_score:.1f}")
    print(f"  Export impact: {r.export_grade_impact}")
    for i, d in enumerate(r.defect_regions, 1):
        print(f"    {i}. {d.type}: {d.area:.0f}px @ {d.center} ({d.severity}, {d.area_pct:.2f}%)")
    vis = detector.visualize_defects(img, r)
    cv2.imwrite(str(out / f"{label}_result.jpg"), vis)

for name, path in tests.items():
    if not path.exists():
        print(f"[SKIP] {name}: {path}")
        continue
    img = cv2.imread(str(path))
    if img is None:
        print(f"[FAIL] {name}: could not load")
        continue

    # Test 1: Raw image (skip_clahe=False)
    run_test(name, img, det_raw, '_raw')

    # Test 2: Preprocessed 224×224 (like the real pipeline)
    preprocessed = proc.preprocess_image(img)
    run_test(name, preprocessed, det, '_preprocessed')

print(f"\nResults saved to {out}")
