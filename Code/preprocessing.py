"""Image preprocessing functions for mango disease detection"""
import cv2
import numpy as np

def sharpen(frame):
    """Gentle sharpening to enhance edges without artifacts"""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(frame, -1, kernel)

def enhance_contrast(frame):
    """Adaptive contrast enhancement - preserves natural appearance"""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Use adaptive CLAHE with larger tiles to avoid block artifacts
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    cl = clahe.apply(l)
    
    enhanced = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def remove_shadow(frame):
    """Gentle shadow removal without over-correction"""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Use smaller kernel to preserve details
    dilated = cv2.dilate(l, np.ones((3, 3), np.uint8))
    bg = cv2.medianBlur(dilated, 21)
    
    diff = cv2.subtract(bg, l)
    norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    corrected_l = cv2.subtract(255, norm)
    
    # More conservative blending to preserve original appearance
    blended_l = cv2.addWeighted(l, 0.85, corrected_l, 0.15, 0)
    corrected_lab = cv2.merge((blended_l, a, b))
    
    return cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)

def denoise_light(frame):
    """Light denoising to reduce noise without blurring"""
    return cv2.fastNlMeansDenoisingColored(frame, None, h=6, hColor=6, 
                                          templateWindowSize=7, searchWindowSize=21)

def color_correct(frame):
    """Improved white balance using percentile-based approach"""
    result = frame.copy().astype(np.float32)
    
    # Use 95th percentile instead of mean for more robust white balance
    for i in range(3):
        percentile_95 = np.percentile(result[:, :, i], 95)
        if percentile_95 > 0:
            result[:, :, i] = np.clip(result[:, :, i] * (255 / percentile_95), 0, 255)
    
    return result.astype(np.uint8)

def unsharp_mask(frame, sigma=1.0, strength=0.3):
    """Unsharp masking for better detail preservation"""
    blurred = cv2.GaussianBlur(frame, (0, 0), sigma)
    sharpened = cv2.addWeighted(frame, 1.0 + strength, blurred, -strength, 0)
    return sharpened

def resize_high_quality(frame, target_size=(150, 150)):
    """High-quality resize using LANCZOS interpolation"""
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4)

def preprocess_image(frame):
    """
    Improved preprocessing pipeline with better detail preservation
    
    Args:
        frame: Input BGR image
    
    Returns:
        Preprocessed image
    """
    # 1. Light denoising first (minimal)
    processed = denoise_light(frame)
    
    # 2. Gentle shadow removal
    processed = remove_shadow(processed)
    
    # 3. White balance correction
    processed = color_correct(processed)
    
    # 4. Adaptive contrast enhancement
    processed = enhance_contrast(processed)
    
    # 5. Unsharp mask for detail enhancement (instead of aggressive sharpening)
    processed = unsharp_mask(processed, sigma=1.0, strength=0.4)
    
    return processed

def preprocess_image_minimal(frame):
    """
    Minimal preprocessing - use if over-processing is an issue
    
    Args:
        frame: Input BGR image
    
    Returns:
        Lightly preprocessed image
    """
    # Just normalize and enhance contrast
    processed = enhance_contrast(frame)
    processed = unsharp_mask(processed, sigma=0.8, strength=0.3)
    
    return processed
