"""Classification model utilities"""
import tensorflow as tf
import numpy as np
from config.settings import Settings


class ClassificationModel:
    """Handles mango variety classification using TFLite"""
    
    def __init__(self, model_path=None, settings=None):
        self.settings = settings or Settings()
        self.model_path = model_path or self.settings.MODEL_PATH
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.class_names = self.settings.CLASS_NAMES
    
    def load(self):
        """Load TFLite model"""
        print(f"Loading classification model from: {self.model_path}")
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print("Classification model loaded successfully")
        return self
    
    def predict(self, preprocessed_img):
        """
        Predict mango variety
        
        Args:
            preprocessed_img: Preprocessed image (224x224)
        
        Returns:
            Tuple of (class_name, confidence)
        """
        if self.interpreter is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Normalize to 0-1 range
        input_data = preprocessed_img.astype(np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Get predictions
        predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        # Get class with highest probability
        predicted_class_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_class_idx])
        class_name = self.class_names[predicted_class_idx]
        
        return class_name, confidence
    
    def predict_top_k(self, preprocessed_img, k=3):
        """
        Get top-k predictions
        
        Args:
            preprocessed_img: Preprocessed image (224x224)
            k: Number of top predictions to return
        
        Returns:
            List of tuples (class_name, confidence)
        """
        if self.interpreter is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Normalize to 0-1 range
        input_data = preprocessed_img.astype(np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Get predictions
        predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        # Get top-k classes
        top_k_indices = np.argsort(predictions)[-k:][::-1]
        top_k_results = [
            (self.class_names[idx], float(predictions[idx]))
            for idx in top_k_indices
        ]
        
        return top_k_results


# Backward compatibility functions
def load_model_weights(model_path=None):
    """Load model - procedural wrapper"""
    settings = Settings()
    model_path = model_path or settings.MODEL_PATH
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    return interpreter, input_details, output_details

def predict_disease(interpreter, input_details, output_details, preprocessed_img):
    """Predict disease - procedural wrapper"""
    settings = Settings()
    
    # Normalize to 0-1 range
    input_data = preprocessed_img.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Get predictions
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Get class with highest probability
    predicted_class_idx = np.argmax(predictions)
    confidence = float(predictions[predicted_class_idx])
    class_name = settings.CLASS_NAMES[predicted_class_idx]
    
    return class_name, confidence
