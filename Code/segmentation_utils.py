"""Segmentation utilities for disease region detection"""
import cv2
import numpy as np
import torch
import torchvision
import config


def load_segmentation_model(model_path=None):
    """
    Load the segmentation model
    
    Args:
        model_path: Path to segmentation model (if None, uses config)
    
    Returns:
        Loaded PyTorch segmentation model
    """
    if model_path is None:
        model_path = config.SEGMENTATION_MODEL_PATH
    
    print(f"Loading segmentation model from: {model_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Check if checkpoint is a full model or just state_dict
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
            weights=None,
            num_classes=2
        )
        model.load_state_dict(state_dict)
    elif isinstance(checkpoint, dict) and not any(key.startswith('backbone') or key.startswith('classifier') for key in checkpoint.keys()):
        # It's a state_dict directly
        model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
            weights=None,
            num_classes=2
        )
        model.load_state_dict(checkpoint)
    elif hasattr(checkpoint, 'eval'):
        # Full model object was saved
        model = checkpoint
    else:
        # Assume it's a state_dict
        model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
            weights=None,
            num_classes=2
        )
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print("Segmentation model loaded successfully")
    return model


def segment_disease(model, image):
    """
    Segment diseased regions from mango image
    
    Args:
        model: Loaded segmentation model
        image: Input BGR image
    
    Returns:
        Dictionary with segmentation results
    """
    import time
    start_time = time.time()
    
    # Get device
    device = next(model.parameters()).device
    
    # Prepare image for model
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    resized = cv2.resize(image_rgb, (224, 224))
    input_tensor = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    input_tensor = (input_tensor - mean) / std
    input_tensor = np.transpose(input_tensor, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0)
    input_tensor = torch.from_numpy(input_tensor).to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(input_tensor)['out']
        probabilities = torch.softmax(output, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
    
    output_predictions = predictions[0].cpu().numpy().astype(np.uint8)
    
    # Resize mask back to original image size
    original_h, original_w = image.shape[:2]
    mask = cv2.resize(output_predictions, (original_w, original_h), 
                     interpolation=cv2.INTER_NEAREST)
    
    # Create colored overlay
    overlay = create_segmentation_overlay(image, mask)
    
    # Calculate disease area percentage
    disease_pixels = np.sum(mask > 0)
    total_pixels = mask.shape[0] * mask.shape[1]
    disease_percentage = (disease_pixels / total_pixels) * 100
    
    end_time = time.time()
    
    return {
        'mask': mask,
        'overlay': overlay,
        'disease_percentage': disease_percentage,
        'segmentation_time': end_time - start_time
    }


def create_segmentation_overlay(image, mask, alpha=0.5):
    """
    Create visualization overlay of segmentation mask on original image
    
    Args:
        image: Original BGR image
        mask: Segmentation mask
        alpha: Transparency factor for overlay
    
    Returns:
        Overlaid image
    """
    overlay = image.copy()
    
    # Create colored mask (red for diseased regions)
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = [0, 0, 255]  # Red in BGR
    
    # Blend original image with colored mask
    result = cv2.addWeighted(overlay, 1 - alpha, colored_mask, alpha, 0)
    
    # Draw contours around diseased regions
    contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                   cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)  # Green contours
    
    return result


def extract_diseased_region(image, mask):
    """
    Extract only the diseased region from image
    
    Args:
        image: Original BGR image
        mask: Segmentation mask
    
    Returns:
        Image with only diseased regions (rest is black)
    """
    result = image.copy()
    result[mask == 0] = 0
    return result


def get_disease_statistics(mask):
    """
    Calculate statistics about diseased regions
    
    Args:
        mask: Segmentation mask
    
    Returns:
        Dictionary with statistics
    """
    # Find contours
    contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                   cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return {
            'num_regions': 0,
            'total_area': 0,
            'largest_region_area': 0,
            'bounding_boxes': []
        }
    
    # Calculate areas
    areas = [cv2.contourArea(c) for c in contours]
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    
    return {
        'num_regions': len(contours),
        'total_area': sum(areas),
        'largest_region_area': max(areas),
        'average_region_area': np.mean(areas),
        'bounding_boxes': bounding_boxes
    }
