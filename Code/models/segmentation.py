"""Segmentation model utilities"""
import torch
import torchvision
import cv2
import numpy as np
from config.settings import Settings


class SegmentationModel:
    """Handles disease segmentation using DeepLabV3"""
    
    def __init__(self, model_path=None, settings=None):
        self.settings = settings or Settings()
        self.model_path = model_path or self.settings.SEGMENTATION_MODEL_PATH
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load(self):
        """Load segmentation model"""
        print(f"Loading segmentation model from: {self.model_path}")
        print(f"Using device: {self.device}")
        
        # Load checkpoint (weights_only=False needed for models saved with full model object)
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Check if checkpoint is a full model or just state_dict
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            # Checkpoint contains state_dict key
            state_dict = checkpoint['state_dict']
            self.model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
                weights=None,
                num_classes=2
            )
            self.model.load_state_dict(state_dict)
        elif isinstance(checkpoint, dict) and not any(key.startswith('backbone') or key.startswith('classifier') for key in checkpoint.keys()):
            # It's a state_dict directly
            self.model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
                weights=None,
                num_classes=2
            )
            self.model.load_state_dict(checkpoint)
        elif hasattr(checkpoint, 'eval'):
            # Full model object was saved
            self.model = checkpoint
        else:
            # Assume it's a state_dict
            self.model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
                weights=None,
                num_classes=2
            )
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        print("Segmentation model loaded successfully")
        return self
    
    def segment(self, image):
        """
        Segment diseased regions from image
        
        Args:
            image: Input BGR image
        
        Returns:
            Tuple of (mask, overlay, disease_percentage)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Prepare input
        original_size = image.shape[:2]
        input_tensor = self._preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)['out']
            probabilities = torch.softmax(output, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        
        # Process output
        mask = predictions[0].cpu().numpy().astype(np.uint8)
        mask = cv2.resize(mask, (original_size[1], original_size[0]), 
                         interpolation=cv2.INTER_NEAREST)
        
        # Create overlay
        overlay = self._create_overlay(image, mask)
        
        # Calculate disease percentage
        disease_percentage = self._calculate_disease_percentage(mask)
        
        return mask, overlay, disease_percentage
    
    def _preprocess_image(self, image):
        """Preprocess image for segmentation model"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        resized = cv2.resize(rgb_image, (224, 224))
        
        # Convert to tensor and normalize
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float()
        tensor = tensor / 255.0
        
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0).to(self.device)
        
        return tensor
    
    def _create_overlay(self, image, mask):
        """Create red overlay on diseased regions"""
        overlay = image.copy()
        red_mask = np.zeros_like(image)
        red_mask[mask == 1] = [0, 0, 255]  # Red for diseased areas
        
        overlay = cv2.addWeighted(overlay, 0.7, red_mask, 0.3, 0)
        
        return overlay
    
    def create_colored_mask(self, mask):
        """
        Create a colored visualization of the segmentation mask
        
        Args:
            mask: Binary segmentation mask (0=background, 1=disease)
        
        Returns:
            Colored mask image (BGR)
        """
        # Create a colored visualization
        colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        
        # Background - blue
        colored[mask == 0] = [180, 120, 60]  # Dark blue for background
        
        # Disease region - light cyan/turquoise  
        colored[mask == 1] = [200, 180, 120]  # Light blue/cyan for detected regions
        
        return colored
    
    def _calculate_disease_percentage(self, mask):
        """Calculate percentage of diseased area"""
        total_pixels = mask.size
        diseased_pixels = np.sum(mask == 1)
        percentage = (diseased_pixels / total_pixels) * 100
        
        return percentage
    
    def get_disease_statistics(self, mask):
        """
        Get detailed disease statistics
        
        Args:
            mask: Binary segmentation mask
        
        Returns:
            Dictionary with statistics
        """
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        total_area = mask.size
        diseased_area = np.sum(mask == 1)
        disease_percentage = (diseased_area / total_area) * 100
        
        # Calculate contour statistics
        num_regions = len(contours)
        region_areas = [cv2.contourArea(c) for c in contours] if contours else [0]
        
        stats = {
            'total_pixels': total_area,
            'diseased_pixels': diseased_area,
            'disease_percentage': disease_percentage,
            'num_regions': num_regions,
            'largest_region_area': max(region_areas),
            'smallest_region_area': min(region_areas),
            'average_region_area': np.mean(region_areas) if region_areas else 0
        }
        
        return stats


# Backward compatibility functions
def load_segmentation_model(model_path=None):
    """Load segmentation model - procedural wrapper"""
    settings = Settings()
    model_path = model_path or settings.SEGMENTATION_MODEL_PATH
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
        weights=None,
        num_classes=2
    )
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model

def segment_disease(model, image):
    """Segment disease - procedural wrapper"""
    seg_model = SegmentationModel()
    seg_model.model = model
    seg_model.device = next(model.parameters()).device
    
    return seg_model.segment(image)

def create_segmentation_overlay(image, mask):
    """Create overlay - procedural wrapper"""
    seg_model = SegmentationModel()
    return seg_model._create_overlay(image, mask)

def get_disease_statistics(mask):
    """Get statistics - procedural wrapper"""
    seg_model = SegmentationModel()
    return seg_model.get_disease_statistics(mask)
