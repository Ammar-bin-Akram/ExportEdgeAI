"""Base pipeline architecture for fruit processing"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Tuple, List
import cv2
import numpy as np


class BaseFruitPipeline(ABC):
    """
    Abstract base class for fruit processing pipelines
    
    This class defines the common interface that all fruit-specific
    pipelines must implement. It follows the Template Method pattern.
    """
    
    def __init__(self, settings=None):
        """
        Initialize pipeline
        
        Args:
            settings: Configuration settings object
        """
        self.settings = settings
        self.classification_model = None
        self.segmentation_model = None
        self.frame_extractor = None
        self.image_processor = None
    
    @abstractmethod
    def load_models(self):
        """Load all required models for this fruit"""
        pass
    
    @abstractmethod
    def preprocess_image(self, image):
        """
        Fruit-specific preprocessing
        
        Args:
            image: Input BGR image
        
        Returns:
            Preprocessed image
        """
        pass
    
    @abstractmethod
    def classify(self, preprocessed_image):
        """
        Classify fruit variety/condition
        
        Args:
            preprocessed_image: Preprocessed image
        
        Returns:
            Tuple of (class_name, confidence)
        """
        pass
    
    def segment_if_needed(self, image, class_name, confidence):
        """
        Perform segmentation on all classified mangoes
        
        Args:
            image: Original image
            class_name: Predicted class
            confidence: Prediction confidence
        
        Returns:
            Tuple of (mask, overlay, disease_percentage) or (None, None, 0)
        """
        # Default implementation - can be overridden
        if not self.settings.ENABLE_SEGMENTATION:
            return None, None, 0
        
        if self.segmentation_model is None:
            return None, None, 0
        
        # Perform segmentation for all mangoes (healthy and diseased)
        return self.segmentation_model.segment(image)
    
    def process_single_frame(self, frame):
        """
        Process a single frame through the pipeline
        
        Args:
            frame: Input BGR frame
        
        Returns:
            Dictionary with processing results
        """
        # Preprocess
        preprocessed = self.preprocess_image(frame)
        
        # Classify
        class_name, confidence = self.classify(preprocessed)
        
        # Segment if needed
        mask, overlay, disease_percentage = self.segment_if_needed(
            frame, class_name, confidence
        )
        
        results = {
            'original': frame,
            'preprocessed': preprocessed,
            'class_name': class_name,
            'confidence': confidence,
            'segmentation_mask': mask,
            'segmentation_overlay': overlay,
            'disease_percentage': disease_percentage
        }
        
        return results
    
    def process_video(self, video_path, output_dir=None):
        """
        Process entire video through the pipeline
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save results
        
        Returns:
            List of processing results for each frame
        """
        if self.frame_extractor is None:
            raise RuntimeError("Frame extractor not initialized")
        
        # Extract peak frames
        peak_frames = self.frame_extractor.extract_peak_frames(video_path)
        
        # Process each frame
        results = []
        for idx, (frame, metadata) in enumerate(peak_frames):
            frame_result = self.process_single_frame(frame)
            frame_result['metadata'] = metadata
            frame_result['frame_index'] = idx
            
            results.append(frame_result)
            
            # Save if output directory provided
            if output_dir:
                self._save_frame_results(frame_result, output_dir, idx)
        
        return results
    
    def _save_frame_results(self, result, output_dir, frame_idx):
        """Save frame processing results to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save original
        cv2.imwrite(
            str(output_path / f"frame_{frame_idx:03d}_original.jpg"),
            result['original']
        )
        
        # Save preprocessed
        cv2.imwrite(
            str(output_path / f"frame_{frame_idx:03d}_preprocessed.jpg"),
            result['preprocessed']
        )
        
        # Save segmentation if available
        if result['segmentation_overlay'] is not None:
            cv2.imwrite(
                str(output_path / f"frame_{frame_idx:03d}_segmented.jpg"),
                result['segmentation_overlay']
            )
        
        # Save metadata
        metadata_file = output_path / f"frame_{frame_idx:03d}_metadata.txt"
        with open(metadata_file, 'w') as f:
            f.write(f"Class: {result['class_name']}\n")
            f.write(f"Confidence: {result['confidence']:.4f}\n")
            if result['disease_percentage'] > 0:
                f.write(f"Disease Area: {result['disease_percentage']:.2f}%\n")
    
    @abstractmethod
    def get_fruit_type(self):
        """Return the type of fruit this pipeline handles"""
        pass
