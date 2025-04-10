import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

def evaluate_model():
    # Load the model
    model = load_model('models/fruit_classifier.h5')
    
    # Load class indices
    with open('models/class_indices.json', 'r') as f:
        class_indices = json.load(f)
    
    # Invert class indices
    indices_to_classes = {v: k for k, v in class_indices.items()}
    
    # Initialize lists to store true and predicted labels
    true_labels = []
    predicted_labels = []
    
    # Test directory path
    test_dir = 'dataset'
    
    # Supported image extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    # Evaluate each image
    for class_name in os.listdir(test_dir):
        class_dir = os.path.join(test_dir, class_name)
        if os.path.isdir(class_dir):
            print(f"Processing class: {class_name}")
            
            # Walk through all subdirectories
            for root, dirs, files in os.walk(class_dir):
                for file in files:
                    # Check if file is an image
                    if any(file.lower().endswith(ext) for ext in valid_extensions):
                        img_path = os.path.join(root, file)
                        try:
                            # Load and preprocess image
                            img_array = load_and_preprocess_image(img_path)
                            
                            # Make prediction
                            prediction = model.predict(img_array, verbose=0)  # Set verbose=0 to reduce output
                            predicted_class_idx = np.argmax(prediction[0])
                            predicted_class = indices_to_classes[predicted_class_idx]
                            confidence = prediction[0][predicted_class_idx]
                            
                            # Store true and predicted labels
                            true_labels.append(class_name)
                            predicted_labels.append(predicted_class)
                            
                            # Print individual prediction results
                            print(f"\nImage: {file}")
                            print(f"True class: {class_name}")
                            print(f"Predicted class: {predicted_class}")
                            print(f"Confidence: {confidence:.2%}")
                            print("-" * 50)
                            
                        except Exception as e:
                            print(f"Error processing {img_path}: {str(e)}")
                            continue
    
    if len(true_labels) == 0:
        print("No valid images found for testing!")
        return
    
    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels))
    
    # Generate confusion matrix
    classes = list(set(true_labels))
    cm = confusion_matrix(true_labels, predicted_labels, labels=classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png')
    plt.close()
    
    # Calculate overall accuracy
    accuracy = sum(t == p for t, p in zip(true_labels, predicted_labels)) / len(true_labels)
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    
    # Save detailed results to a file
    with open('models/evaluation_results.txt', 'w') as f:
        f.write("Detailed Classification Results\n")
        f.write("============================\n\n")
        f.write(f"Overall Accuracy: {accuracy:.2%}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(true_labels, predicted_labels))

if __name__ == "__main__":
    evaluate_model() 