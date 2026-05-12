# Modular OOP Architecture

This project has been refactored to use a modular Object-Oriented Programming (OOP) architecture to support future expansion to multiple fruit types.

## Directory Structure

```
Code/
├── config/                 # Configuration management
│   ├── __init__.py
│   └── settings.py        # Settings class with all configuration
│
├── core/                  # Core pipeline architecture
│   ├── __init__.py
│   ├── base_pipeline.py   # Abstract base class for fruit pipelines
│   └── pipeline_factory.py # Factory pattern for creating pipelines
│
├── vision/                # Computer vision utilities
│   ├── __init__.py
│   ├── frame_extractor.py # Frame extraction with motion detection
│   ├── motion_detector.py # Motion detection algorithms
│   ├── roi_extractor.py   # Region of Interest extraction
│   └── image_processor.py # Image preprocessing
│
├── models/                # ML model wrappers
│   ├── __init__.py
│   ├── classification.py  # Classification model
│   └── segmentation.py    # Segmentation model
│
├── fruits/                # Fruit-specific implementations
│   ├── __init__.py
│   └── mango/
│       ├── __init__.py
│       └── pipeline.py    # Mango pipeline implementation
│
├── interfaces/            # User interfaces
│   ├── __init__.py
│   ├── cli.py            # Command-line interface
│   └── streamlit_app.py  # Web interface
│
├── main.py               # Backward compatible CLI (uses new modules)
└── streamlit_app.py      # Backward compatible web UI (uses new modules)
```

## Design Patterns

### 1. **Template Method Pattern**
`BaseFruitPipeline` defines the skeleton of the processing algorithm:
- `load_models()` - Load required models
- `preprocess_image()` - Fruit-specific preprocessing
- `classify()` - Classification step
- `segment_if_needed()` - Conditional segmentation
- `process_single_frame()` - Complete frame processing

Subclasses like `MangoPipeline` implement fruit-specific behavior.

### 2. **Factory Pattern**
`PipelineFactory` creates fruit-specific pipelines:
```python
from core import PipelineFactory

# Create mango pipeline
pipeline = PipelineFactory.create_pipeline('mango')
```

### 3. **Strategy Pattern**
Different preprocessing strategies can be swapped:
- `ImageProcessor.preprocess_image()` - Full pipeline
- `ImageProcessor.preprocess_image_minimal()` - Lightweight processing

## Usage

### New Modular Interface

```python
from config.settings import Settings
from fruits.mango import MangoPipeline

# Create and configure pipeline
settings = Settings()
pipeline = MangoPipeline(settings)
pipeline.load_models()

# Process single frame
result = pipeline.process_single_frame(frame)
print(f"Class: {result['class_name']}")
print(f"Confidence: {result['confidence']}")

# Process video
results = pipeline.process_video('video.mp4', 'output/')
```

### Command Line

```bash
# New modular CLI
python interfaces/cli.py video.mp4 -o output/

# Old CLI (backward compatible)
python main.py  # Uses new modules internally
```

### Web Interface

```bash
# New modular interface
streamlit run interfaces/streamlit_app.py

# Old interface (backward compatible)
streamlit run streamlit_app.py  # Uses new modules internally
```

## Backward Compatibility

All old procedural functions still work:
```python
# Old style still works
from video_utils import extract_peak_frames
from preprocessing import preprocess_image
from model_utils import predict_disease

frames = extract_peak_frames('video.mp4')
# ... etc
```

These functions now internally use the new OOP classes.

## Adding New Fruits

To add support for a new fruit (e.g., apple):

1. **Create fruit directory:**
   ```
   fruits/apple/
   ├── __init__.py
   └── pipeline.py
   ```

2. **Implement pipeline:**
   ```python
   from core.base_pipeline import BaseFruitPipeline
   
   class ApplePipeline(BaseFruitPipeline):
       def load_models(self):
           # Load apple-specific models
           pass
       
       def preprocess_image(self, image):
           # Apple-specific preprocessing
           pass
       
       def classify(self, preprocessed_image):
           # Apple classification
           pass
       
       def get_fruit_type(self):
           return "apple"
   ```

3. **Register pipeline:**
   ```python
   from core import PipelineFactory
   PipelineFactory.register_pipeline('apple', ApplePipeline)
   ```

4. **Use it:**
   ```python
   pipeline = PipelineFactory.create_pipeline('apple')
   ```

## Key Classes

### Settings (config/settings.py)
Centralized configuration with backward compatibility:
```python
settings = Settings()
print(settings.MODEL_PATH)
print(settings.ENABLE_SEGMENTATION)
```

### MangoPipeline (fruits/mango/pipeline.py)
Complete mango processing pipeline:
```python
pipeline = MangoPipeline()
pipeline.load_models()
result = pipeline.process_single_frame(frame)
```

### ClassificationModel (models/classification.py)
TFLite classification wrapper:
```python
model = ClassificationModel()
model.load()
class_name, confidence = model.predict(image)
```

### SegmentationModel (models/segmentation.py)
PyTorch segmentation wrapper:
```python
model = SegmentationModel()
model.load()
mask, overlay, percentage = model.segment(image)
```

### ImageProcessor (vision/image_processor.py)
Preprocessing utilities:
```python
processor = ImageProcessor()
processed = processor.preprocess_image(image)
```

## Benefits

1. **Scalability** - Easy to add new fruits
2. **Maintainability** - Modular, organized code
3. **Reusability** - Shared components across fruits
4. **Testability** - Easy to unit test individual components
5. **Backward Compatibility** - Old code still works
6. **Type Safety** - Clear interfaces and contracts

## Migration Path

Current code can gradually migrate from procedural to OOP:

**Phase 1** (Current): Backward compatible wrapper
- Old files use new modules internally
- Both interfaces work

**Phase 2** (Future): Full migration
- Update all code to use new interfaces
- Remove backward compatibility wrappers
- Clean up old procedural files
