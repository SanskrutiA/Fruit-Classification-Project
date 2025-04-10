import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import json
import glob
from tkinter import Tk, filedialog

# Load the trained model
MODEL_PATH = 'models/fruit_classifier.h5'
model = load_model(MODEL_PATH)

# Load class indices
with open('models/class_indices.json', 'r') as f:
    class_indices = json.load(f)
    
# Invert class indices to map prediction index to class name
indices_class = {v: k for k, v in class_indices.items()}

def get_condition_and_fruit(class_name):
    # Assuming class names format is like "good_apple", "bad_banana", etc.
    parts = class_name.split('_')
    if len(parts) >= 2:
        condition = parts[0]
        fruit = '_'.join(parts[1:])  # Join the rest as the fruit name (handles multi-word fruits)
        return condition, fruit
    return class_name, ""  # Fallback if format is different

def get_recommendation(condition):
    recommendations = {
        "good": "Pluck",
        "raw": "Keep",
        "bad": "Discard",
        "diseased": "Remove to prevent infection"
    }
    return recommendations.get(condition.lower(), "Unknown recommendation")

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    return img_array

def predict_image(img_path):
    # Preprocess the image
    processed_img = preprocess_image(img_path)
    
    # Make prediction
    predictions = model.predict(processed_img)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = indices_class[predicted_class_idx]
    confidence = float(predictions[0][predicted_class_idx])
    
    # Extract condition and fruit type
    condition, fruit = get_condition_and_fruit(predicted_class)
    recommendation = get_recommendation(condition)
    
    return {
        "fruit": fruit.capitalize(),
        "condition": condition.capitalize(),
        "recommendation": recommendation,
        "confidence": f"{confidence:.2%}"
    }

def predict_multiple_images():
    root = Tk()
    root.withdraw()  # Hide the main window
    
    # Ask user to select multiple image files
    file_paths = filedialog.askopenfilenames(
        title="Select fruit images to classify",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    
    if not file_paths:
        print("No files selected.")
        return
    
    print(f"Selected {len(file_paths)} images for classification.")
    
    # Create figure for displaying results
    n_images = len(file_paths)
    cols = 3
    rows = (n_images + cols - 1) // cols
    
    plt.figure(figsize=(15, 5 * rows))
    
    for i, img_path in enumerate(file_paths):
        # Make prediction
        result = predict_image(img_path)
        
        # Display image with results
        plt.subplot(rows, cols, i + 1)
        img = image.load_img(img_path)
        plt.imshow(img)
        plt.title(f"Fruit: {result['fruit']}\nCondition: {result['condition']}\n"
                 f"Recommendation: {result['recommendation']}\n"
                 f"Confidence: {result['confidence']}")
        plt.axis('off')
        
        # Print results to console
        print(f"\nResults for {os.path.basename(img_path)}:")
        print(f"Fruit: {result['fruit']}")
        print(f"Condition: {result['condition']}")
        print(f"Recommendation: {result['recommendation']}")
        print(f"Confidence: {result['confidence']}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    predict_multiple_images()
