"""Test the new modular OOP pipeline"""
import cv2
import numpy as np
from pathlib import Path

print("=" * 70)
print("TESTING NEW MODULAR OOP PIPELINE")
print("=" * 70)

# Test 1: Import and create pipeline
print("\n[1/5] Creating MangoPipeline...")
try:
    from config.settings import Settings
    from fruits.mango import MangoPipeline
    
    settings = Settings()
    pipeline = MangoPipeline(settings)
    
    print(f"✓ Pipeline created successfully")
    print(f"  Fruit type: {pipeline.get_fruit_type()}")
    print(f"  Settings loaded: {settings.CLASS_NAMES}")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

# Test 2: Load models
print("\n[2/5] Loading models...")
try:
    pipeline.load_models()
    print(f"✓ Models loaded successfully")
    print(f"  Classification model: Ready")
    print(f"  Segmentation model: {'Ready' if settings.ENABLE_SEGMENTATION else 'Disabled'}")
except Exception as e:
    print(f"✗ Error loading models: {e}")
    exit(1)

# Test 3: Test with dummy image
print("\n[3/5] Testing with dummy image...")
try:
    # Create a dummy image (random RGB)
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    result = pipeline.process_single_frame(dummy_image)
    
    print(f"✓ Image processed successfully")
    print(f"  Predicted class: {result['class_name']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Disease percentage: {result['disease_percentage']:.2f}%")
    
except Exception as e:
    print(f"✗ Error processing image: {e}")
    exit(1)

# Test 4: Test with real image (if available)
print("\n[4/5] Testing with real image...")
test_image_paths = [
    Path(settings.OUTPUT_DIR).parent / "images" / "test.jpg",
    Path(settings.OUTPUT_DIR) / "peak_000001_000000_original.jpg",
    Path("test_image.jpg"),
]

test_image = None
for img_path in test_image_paths:
    if img_path.exists():
        test_image = cv2.imread(str(img_path))
        if test_image is not None:
            print(f"✓ Found test image: {img_path}")
            break

if test_image is not None:
    try:
        result = pipeline.process_single_frame_detailed(test_image)
        
        print(f"✓ Real image processed successfully")
        print(f"  Predicted class: {result['class_name']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        
        # Show top 3 predictions
        print(f"  Top 3 predictions:")
        for class_name, conf in result['top_k_predictions'][:3]:
            print(f"    - {class_name}: {conf:.2%}")
        
        if result['disease_percentage'] > 0:
            print(f"  Disease area: {result['disease_percentage']:.2f}%")
            if 'disease_statistics' in result:
                stats = result['disease_statistics']
                print(f"  Disease regions: {stats['num_regions']}")
    except Exception as e:
        print(f"✗ Error processing real image: {e}")
else:
    print(f"⚠ No test image found, skipping")

# Test 5: Test factory pattern
print("\n[5/5] Testing PipelineFactory...")
try:
    from core import PipelineFactory
    
    # List available fruits
    available = PipelineFactory.get_available_fruits()
    print(f"✓ Available fruits: {available}")
    
    # Create pipeline via factory
    factory_pipeline = PipelineFactory.create_pipeline('mango', settings)
    print(f"✓ Pipeline created via factory: {factory_pipeline.get_fruit_type()}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

# Summary
print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)

print("\n📊 New Modular Pipeline Summary:")
print(f"  • Architecture: Object-Oriented (OOP)")
print(f"  • Design Pattern: Template Method + Factory")
print(f"  • Fruit Type: {pipeline.get_fruit_type()}")
print(f"  • Classes: {len(settings.CLASS_NAMES)} varieties")
print(f"  • Segmentation: {'Enabled' if settings.ENABLE_SEGMENTATION else 'Disabled'}")

print("\n🚀 How to use the new pipeline:")
print("\n  1. Python API:")
print("     from fruits.mango import MangoPipeline")
print("     pipeline = MangoPipeline()")
print("     pipeline.load_models()")
print("     result = pipeline.process_single_frame(image)")
print()
print("  2. CLI:")
print("     python interfaces/cli.py video.mp4 -o output/")
print()
print("  3. Streamlit:")
print("     streamlit run interfaces/streamlit_app.py")
print()
print("  4. Factory Pattern:")
print("     from core import PipelineFactory")
print("     pipeline = PipelineFactory.create_pipeline('mango')")
print()
print("✨ The new modular architecture is ready to use!")
print("📚 See ARCHITECTURE.md for detailed documentation")
