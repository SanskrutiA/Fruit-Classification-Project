import tensorflow as tf
from keras.models import load_model
import numpy as np
from pathlib import Path
import os
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Verify TensorFlow version
logger.info(f"Using TensorFlow version: {tf.__version__}")

class FruitClassifier:
    def __init__(self, model_path: str = None, class_indices_path: str = None):
        if model_path is None:
            # Use absolute path to the model file
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            model_path = os.path.join(base_dir, "model", "fruit_classifier.h5")
            class_indices_path = os.path.join(base_dir, "model", "class_indices.json")
        
        self.model = self._load_model(model_path)
        self.class_names = self._load_class_names(class_indices_path)
        
    def _load_model(self, model_path: str):
        """
        Load the pre-trained model with TensorFlow 2.13.0 specific settings
        """
        model_path = Path(model_path)
        logger.info(f"Looking for model at: {model_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        try:
            logger.info("Loading model...")
            # Set memory growth to prevent GPU memory issues
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            
            # Load model with custom_objects to handle any custom layers
            model = load_model(model_path, compile=False)
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _load_class_names(self, class_indices_path: str = None):
        """
        Load class names from JSON file or use default
        """
        # Default class names if no file is found
        default_class_names = {0: 'bad', 1: 'diseased', 2: 'good', 3: 'raw'}
        
        if class_indices_path and Path(class_indices_path).exists():
            try:
                logger.info(f"Loading class names from: {class_indices_path}")
                with open(class_indices_path, 'r') as f:
                    class_indices = json.load(f)
                    if not class_indices:  # If file is empty
                        logger.warning("Class indices file is empty, using defaults")
                        return default_class_names
                    
                    # Invert the dictionary to get index -> class_name mapping
                    inverted_indices = {str(v): k for k, v in class_indices.items()}
                    return inverted_indices
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding class indices file: {str(e)}")
                return default_class_names
            except Exception as e:
                logger.error(f"Error loading class names: {str(e)}")
                return default_class_names
        
        logger.warning("Using default class names")
        return default_class_names
    
    def predict(self, image_array: np.ndarray) -> dict:
        """
        Make prediction on preprocessed image
        """
        try:
            # Ensure image is in the correct format
            if len(image_array.shape) == 3:
                image_array = np.expand_dims(image_array, axis=0)
            
            # Make prediction
            predictions = self.model.predict(image_array, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            # Get class name
            class_name = self.class_names.get(str(predicted_class), f"class_{predicted_class}")
            
            # Create all probabilities dictionary
            all_probabilities = {}
            for i, prob in enumerate(predictions[0]):
                class_key = self.class_names.get(str(i), f"class_{i}")
                all_probabilities[class_key] = float(prob)
            
            # Return in the format expected by the schema
            return {
                "class": class_name,
                "confidence": confidence,
                "all_probabilities": all_probabilities
            }
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise 