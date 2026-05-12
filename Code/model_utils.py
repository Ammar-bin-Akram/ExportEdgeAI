"""Model loading and prediction functions"""
import tensorflow as tf
import numpy as np
import time
import config

def load_tflite_model(model_path=None):
    """Load TFLite model
    
    Args:
        model_path: Path to TFLite model file (if None, uses config)
    
    Returns:
        Tuple of (interpreter, input_details, output_details)
    """
    if model_path is None:
        model_path = config.MODEL_PATH
    
    print("-" * 50)
    print(f"Loading TFLite model from: {model_path}")
    start_time = time.time()
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    end_time = time.time()
    print(f"TFLite model loaded in {end_time - start_time:.2f} seconds")
    print("-" * 50)
    
    return interpreter, input_details, output_details

def load_model_weights(model_path=None):
    """Load TFLite model (kept for backward compatibility)
    
    Args:
        model_path: Path to TFLite model file (if None, uses config)
    
    Returns:
        Tuple of (interpreter, input_details, output_details)
    """
    return load_tflite_model(model_path)

def predict_disease(model, image, return_probabilities=False):
    """Predict disease class for given image
    
    Args:
        model: Tuple of (interpreter, input_details, output_details)
        image: BGR image (already preprocessed)
        return_probabilities: If True, return all class probabilities
    
    Returns:
        dict with prediction results
    """
    interpreter, input_details, output_details = model
    
    # Prepare image
    img_resized = tf.image.resize(image, config.INPUT_SHAPE[:2])
    img_normalized = img_resized / 255.0
    input_data = np.expand_dims(img_normalized, axis=0).astype(np.float32)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Predict
    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()
    
    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    pred_class = int(np.argmax(output_data, axis=1)[0])
    confidence = float(np.max(output_data))
    
    result = {
        'class_idx': pred_class,
        'class_name': config.CLASS_NAMES[pred_class],
        'confidence': confidence,
        'inference_time': end_time - start_time
    }
    
    if return_probabilities:
        result['probabilities'] = {
            name: float(prob)
            for name, prob in zip(config.CLASS_NAMES, output_data[0])
        }
    
    return result
