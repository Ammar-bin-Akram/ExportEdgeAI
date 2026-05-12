"""Image preprocessing for classification"""
import cv2
import numpy as np
from config.settings import Settings


class ImageProcessor:
    """Handles image preprocessing for classification"""
    
    def __init__(self, settings=None):
        self.settings = settings or Settings()
    
    def preprocess_image(self, img, target_size=(224, 224)):
        """
        Full preprocessing pipeline for classification
        
        Args:
            img: Input BGR image
            target_size: Target size for model input
        
        Returns:
            Preprocessed image
        """
        img_resized = cv2.resize(img, target_size)
        img_denoised = self.denoise_light(img_resized)
        img_corrected = self.color_correct(img_denoised)
        img_sharp = self.unsharp_mask(img_corrected)
        img_contrast = self.enhance_contrast(img_sharp)
        
        return img_contrast
    
    def preprocess_image_minimal(self, img, target_size=(224, 224)):
        """
        Minimal preprocessing for faster processing
        
        Args:
            img: Input BGR image
            target_size: Target size for model input
        
        Returns:
            Minimally preprocessed image
        """
        img_resized = cv2.resize(img, target_size)
        img_corrected = self.color_correct(img_resized)
        img_contrast = self.enhance_contrast(img_corrected)
        
        return img_contrast
    
    def denoise_light(self, img):
        """Light bilateral filtering to reduce noise while preserving edges"""
        return cv2.bilateralFilter(img, d=9, sigmaColor=50, sigmaSpace=50)
    
    def color_correct(self, img):
        """Enhance color using histogram stretching"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        l_min, l_max = np.percentile(l, (1, 99))
        l = np.clip((l - l_min) / (l_max - l_min) * 255, 0, 255).astype(np.uint8)
        
        corrected = cv2.merge([l, a, b])
        corrected = cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)
        
        return corrected
    
    def unsharp_mask(self, img, kernel_size=(5, 5), sigma=1.5, amount=1.2, threshold=0):
        """Apply unsharp masking for better detail preservation"""
        blurred = cv2.GaussianBlur(img, kernel_size, sigma)
        sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
        
        if threshold > 0:
            low_contrast_mask = np.abs(img - blurred) < threshold
            sharpened = np.where(low_contrast_mask, img, sharpened)
        
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def enhance_contrast(self, img):
        """Apply CLAHE for adaptive contrast enhancement"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        l_clahe = clahe.apply(l)
        
        enhanced = cv2.merge([l_clahe, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def normalize_for_model(self, img):
        """
        Normalize image for model input (0-1 range)
        
        Args:
            img: Preprocessed BGR image
        
        Returns:
            Normalized float32 array
        """
        return img.astype(np.float32) / 255.0
    
    def prepare_batch(self, img):
        """
        Prepare image batch for model inference
        
        Args:
            img: Single image
        
        Returns:
            Batch with shape (1, H, W, C)
        """
        return np.expand_dims(img, axis=0)


# Backward compatibility functions
def denoise_light(img):
    """Denoise - procedural wrapper"""
    processor = ImageProcessor()
    return processor.denoise_light(img)

def color_correct(img):
    """Color correction - procedural wrapper"""
    processor = ImageProcessor()
    return processor.color_correct(img)

def unsharp_mask(img, kernel_size=(5, 5), sigma=1.5, amount=1.2, threshold=0):
    """Unsharp mask - procedural wrapper"""
    processor = ImageProcessor()
    return processor.unsharp_mask(img, kernel_size, sigma, amount, threshold)

def enhance_contrast(img):
    """Enhance contrast - procedural wrapper"""
    processor = ImageProcessor()
    return processor.enhance_contrast(img)

def preprocess_image(img, target_size=(224, 224)):
    """Full preprocessing - procedural wrapper"""
    processor = ImageProcessor()
    return processor.preprocess_image(img, target_size)

def preprocess_image_minimal(img, target_size=(224, 224)):
    """Minimal preprocessing - procedural wrapper"""
    processor = ImageProcessor()
    return processor.preprocess_image_minimal(img, target_size)
