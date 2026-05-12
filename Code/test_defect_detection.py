#!/usr/bin/env python3
"""
Simple test script for OpenCV mango defect detection

This script tests the MangoDefectDetector to verify it's working correctly.
Place a mango image in the Code folder or update the image_path below.
"""

import cv2
import os
import sys
from pathlib import Path

# Add the Code directory to Python path
code_dir = Path(__file__).parent
sys.path.insert(0, str(code_dir))

# Import our defect detector
try:
    from vision.defect_detector import MangoDefectDetector
    print("✅ Successfully imported MangoDefectDetector")
except ImportError as e:
    print(f"❌ Error importing MangoDefectDetector: {e}")
    sys.exit(1)


def test_defect_detection(image_path=None):
    """
    Test the defect detection on a mango image
    
    Args:
        image_path: Path to test image. If None, will look for common image files
    """
    print("🍋 Mango Defect Detection Test")
    print("=" * 50)
    
    # Try to find a test image if none provided
    if image_path is None:
        # Look for common image files in current directory
        common_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in common_extensions:
            image_files.extend(list(Path('.').glob(f'*{ext}')))
            image_files.extend(list(Path('.').glob(f'*{ext.upper()}')))
        
        if image_files:
            image_path = str(image_files[0])
            print(f"📸 Using found image: {image_path}")
        else:
            print("❌ No test image found!")
            print("Please:")
            print("1. Place a mango image in the Code folder, or")
            print("2. Update the image_path in this script")
            return False
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return False
    
    # Load the image
    print(f"📂 Loading image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print("❌ Failed to load image")
        return False
    
    print(f"✅ Image loaded successfully: {image.shape}")
    
    # Initialize the defect detector
    print("\n🔧 Initializing MangoDefectDetector...")
    detector = MangoDefectDetector()
    print("✅ Detector initialized")
    
    # Run defect detection
    print("\n🔍 Running defect detection...")
    try:
        results = detector.detect_defects(image)
        print("✅ Defect detection completed successfully!")
    except Exception as e:
        print(f"❌ Error during defect detection: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Print results
    print("\n📊 DETECTION RESULTS:")
    print("-" * 30)
    print(f"🔢 Total defects found: {results.defect_count}")
    print(f"   ├── Dark/black spots: {results.dark_spot_count}")
    print(f"   └── Brown spots:      {results.brown_spot_count}")
    print(f"🍋 Mango area: {results.mango_area:.0f} pixels")
    print(f"📐 Total defect area: {results.total_defect_area:.1f} pixels")
    print(f"📈 Defect percentage: {results.total_defect_percentage:.2f}% of mango surface")
    print(f"🎨 Color uniformity score: {results.color_uniformity_score:.1f}/100")
    print(f"✨ Surface quality score: {results.surface_quality_score:.1f}/100")
    print(f"📤 Export grade impact: {results.export_grade_impact}")
    print(f"⏱️  Processing time: {results.processing_time:.3f} seconds")
    
    # Show defect details
    if results.defect_regions:
        print(f"\n🔍 DEFECT DETAILS:")
        for i, defect in enumerate(results.defect_regions, 1):
            print(f"  {i}. {defect.type.replace('_', ' ').title()}: "
                  f"{defect.area:.1f} px² "
                  f"({defect.severity}, conf={defect.confidence:.2f}) "
                  f"at {defect.center}")
    else:
        print("\n✨ No defects detected — clean mango!")
    
    # Create visualization
    print("\n🎨 Creating visualization...")
    try:
        visualization = detector.visualize_defects(image, results)
        
        # Save results
        output_dir = Path("defect_test_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save original image
        original_path = output_dir / "01_original.jpg"
        cv2.imwrite(str(original_path), image)
        
        # Save visualization
        viz_path = output_dir / "02_defect_visualization.jpg"
        cv2.imwrite(str(viz_path), visualization)
        
        # Save mango mask for debugging
        mango_mask = detector.create_mango_mask(image)
        mask_path = output_dir / "03_mango_mask.jpg"
        cv2.imwrite(str(mask_path), mango_mask)
        
        # Save structured JSON
        import json
        from datetime import datetime
        json_data = {
            "timestamp": datetime.now().isoformat(),
            "image_path": str(image_path),
            "image_shape": list(image.shape),
            "classification": None,
            "defect_analysis": {
                "surface_quality_score": round(results.surface_quality_score, 2),
                "color_uniformity_score": round(results.color_uniformity_score, 2),
                "total_defect_percentage": round(results.total_defect_percentage, 4),
                "defect_count": results.defect_count,
                "dark_spot_count": results.dark_spot_count,
                "brown_spot_count": results.brown_spot_count,
                "mango_area": round(results.mango_area, 1),
                "total_defect_area": round(results.total_defect_area, 1),
                "export_grade_impact": results.export_grade_impact,
                "processing_time": round(results.processing_time, 4),
                "defect_regions": [
                    {
                        "type": d.type,
                        "area": round(d.area, 1),
                        "severity": d.severity,
                        "confidence": round(d.confidence, 4),
                        "center": list(d.center),
                    }
                    for d in (results.defect_regions or [])
                ],
            },
        }
        json_path = output_dir / "defect_results.json"
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(json_data, jf, indent=2, ensure_ascii=False)
        
        print(f"💾 Original image saved: {original_path}")
        print(f"💾 Visualization saved: {viz_path}")
        print(f"💾 Mango mask saved: {mask_path}")
        print(f"💾 JSON results saved: {json_path}")
        
        # Try to display if possible (works in some environments)
        try:
            cv2.imshow('Original Image', cv2.resize(image, (600, 400)))
            cv2.imshow('Defect Detection', cv2.resize(visualization, (600, 400)))
            print("\n👀 Images displayed in windows")
            print("Press any key in the image windows to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            print("\n📺 Display not available, but images saved to disk")
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        return False
    
    # Generate quality assessment
    print(f"\n🏆 QUALITY ASSESSMENT:")
    print("-" * 25)
    
    if results.total_defect_percentage <= 2.0 and results.color_uniformity_score >= 85:
        grade = "Grade A (Premium Export)"
        markets = "USA, EU, Japan, Australia"
    elif results.total_defect_percentage <= 5.0 and results.color_uniformity_score >= 70:
        grade = "Grade B (Standard Export)" 
        markets = "Middle East, Asia, Canada"
    elif results.total_defect_percentage <= 10.0:
        grade = "Grade C (Local Market)"
        markets = "Local, Regional"
    else:
        grade = "Processing Grade"
        markets = "Processing only"
    
    print(f"🥇 Estimated grade: {grade}")
    print(f"🌍 Suitable markets: {markets}")
    
    # Generate RAG description
    print(f"\n🤖 RAG SYSTEM INPUT:")
    print("-" * 20)
    rag_description = f"""
Mango Quality Analysis:
- Defect Coverage: {results.total_defect_percentage:.2f}%
- Color Uniformity: {results.color_uniformity_score:.1f}/100
- Surface Quality: {results.surface_quality_score:.1f}/100
- Export Impact: {results.export_grade_impact}
- Estimated Grade: {grade}

Based on this visual analysis, which export countries would be most suitable?
    """.strip()
    
    print(rag_description)
    
    print(f"\n✅ Test completed successfully!")
    return True


def test_with_config():
    """Test with custom configuration"""
    print("\n🔧 Testing with custom configuration...")
    
    # Custom config for more sensitive detection
    custom_config = {
        'dark_spot_threshold': 60,    # Stricter dark spot threshold
        'dark_spot_min_area': 50,     # Detect slightly smaller spots
        'brown_min_area': 80,         # Detect slightly smaller brown spots
        'color_variance_threshold': 1200,  # Slightly stricter uniformity
    }
    
    detector = MangoDefectDetector(custom_config)
    print("✅ Custom detector created")
    print(f"   - Dark spot threshold: {custom_config['dark_spot_threshold']}")
    print(f"   - Dark spot min area: {custom_config['dark_spot_min_area']} px")
    print(f"   - Brown spot min area: {custom_config['brown_min_area']} px")
    print(f"   - Color variance threshold: {custom_config['color_variance_threshold']}")


if __name__ == "__main__":
    print("🚀 Starting OpenCV Mango Defect Detection Test")
    print("=" * 60)
    
    # Test 1: Basic detection
    # success = test_defect_detection(str(Path(__file__).parent.parent / 'Mango Variety and Grading Dataset' / 'Dataset' / 'Grading_dataset' / 'Class_I' / 'IMG_20210703_151539.jpg'))
    # success = test_defect_detection(str(Path(__file__).parent.parent / 'Mango Variety and Grading Dataset' / 'Dataset' / 'Grading_dataset' / 'Class_II' / 'IMG_20210703_155515.jpg'))
    # success = test_defect_detection(str(Path(__file__).parent.parent / 'Mango Variety and Grading Dataset' / 'Dataset' / 'Grading_dataset' / 'Extra_Class' / 'IMG_20210703_142244.jpg'))
    success = test_defect_detection('Mango Variety and Grading Dataset\\Dataset\\Grading_dataset\\Class_II\\IMG_20210703_155750.jpg')
    if success:
        # Test 2: Custom configuration
        test_with_config()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("\nNext steps:")
        print("1. ✅ OpenCV defect detection is working!")
        print("2. 🔗 Ready to integrate with your main pipeline")
        print("3. 🤖 Ready to connect with RAG system")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ TESTS FAILED!")
        print("Please check the error messages above.")
        print("=" * 60)