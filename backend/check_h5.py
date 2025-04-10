import os
import h5py
import tensorflow as tf
from tensorflow import keras
import json
from pathlib import Path

def check_h5_file():
    # Define the path to the model file
    model_path = 'model/fruit_classifier.h5'
    class_indices_path = 'model/class_indices.json'
    
    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    print(f"Model file found: {model_path}")
    print(f"File size: {os.path.getsize(model_path) / (1024 * 1024):.2f} MB")
    
    # Check the model file using h5py
    try:
        with h5py.File(model_path, "r") as f:
            keras_version = f.attrs.get("keras_version")
            backend = f.attrs.get("backend")
            print(f"Keras version used to save the model: {keras_version}")
            print(f"Backend used to save the model: {backend}")
            
            # Show model layers if available
            if "model_weights" in f:
                print("\nModel layers:")
                for layer in f["model_weights"].keys():
                    print(f"  - {layer}")
    except Exception as e:
        print(f"Error reading H5 file with h5py: {e}")
    
    # Try to load the model with Keras
    try:
        model = keras.models.load_model(model_path, compile=False)
        print("\nSuccessfully loaded model with Keras")
        print(f"TensorFlow version: {tf.__version__}")
        print(f"Keras version: {keras.__version__}")
        print("\nModel summary:")
        model.summary()
        
        # Check if class indices file exists
        if os.path.exists(class_indices_path):
            with open(class_indices_path, 'r') as f:
                class_indices = json.load(f)
                print("\nClass indices:")
                for class_name, idx in class_indices.items():
                    print(f"  - {class_name}: {idx}")
        
    except Exception as e:
        print(f"Error loading model with Keras: {e}")

if __name__ == "__main__":
    check_h5_file()
