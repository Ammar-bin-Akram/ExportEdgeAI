"""Mango-specific processing pipeline"""
from core.base_pipeline import BaseFruitPipeline
from vision import FrameExtractor, ImageProcessor
from models import ClassificationModel, SegmentationModel
from config.settings import Settings


class MangoPipeline(BaseFruitPipeline):
    """
    Complete processing pipeline for mango fruit
    
    Handles:
    - Video frame extraction with motion detection
    - Image preprocessing
    - Variety classification
    - Disease segmentation (if diseased)
    """
    
    def __init__(self, settings=None):
        """
        Initialize mango pipeline
        
        Args:
            settings: Configuration settings (uses default if None)
        """
        super().__init__(settings or Settings())
        
        # Initialize components
        self.frame_extractor = FrameExtractor(self.settings)
        self.image_processor = ImageProcessor(self.settings)
        self.classification_model = ClassificationModel(settings=self.settings)
        self.segmentation_model = SegmentationModel(settings=self.settings)
    
    def load_models(self):
        """Load classification and segmentation models"""
        print("Loading mango processing models...")
        
        # Load classification model
        self.classification_model.load()
        
        # Load segmentation model if enabled
        if self.settings.ENABLE_SEGMENTATION:
            self.segmentation_model.load()
        else:
            print("Segmentation disabled in settings")
        
        print("All models loaded successfully")
        return self
    
    def preprocess_image(self, image):
        """
        Preprocess image for mango classification
        
        Uses full preprocessing pipeline with:
        - Light denoising
        - Color correction
        - Unsharp masking
        - Contrast enhancement
        
        Args:
            image: Input BGR image
        
        Returns:
            Preprocessed image matching model input shape
        """
        # Get target size from settings (INPUT_SHAPE is (height, width, channels))
        target_size = (self.settings.INPUT_SHAPE[0], self.settings.INPUT_SHAPE[1])
        return self.image_processor.preprocess_image(
            image,
            target_size=target_size
        )
    
    def classify(self, preprocessed_image):
        """
        Classify mango variety
        
        Args:
            preprocessed_image: Preprocessed image (224x224)
        
        Returns:
            Tuple of (class_name, confidence)
        """
        if self.classification_model is None:
            raise RuntimeError("Classification model not loaded")
        
        return self.classification_model.predict(preprocessed_image)
    
    def get_fruit_type(self):
        """Return fruit type"""
        return "mango"
    
    def process_single_frame_detailed(self, frame):
        """
        Process frame with detailed output for debugging
        
        Args:
            frame: Input BGR frame
        
        Returns:
            Dictionary with detailed results
        """
        # Get basic results
        results = self.process_single_frame(frame)
        
        # Add top-k predictions
        preprocessed = results['preprocessed']
        top_k = self.classification_model.predict_top_k(preprocessed, k=3)
        results['top_k_predictions'] = top_k
        
        # Add disease statistics if segmented
        if results['segmentation_mask'] is not None:
            stats = self.segmentation_model.get_disease_statistics(
                results['segmentation_mask']
            )
            results['disease_statistics'] = stats
        
        return results


# Register mango pipeline with factory
from core import PipelineFactory
PipelineFactory.register_pipeline('mango', MangoPipeline)
