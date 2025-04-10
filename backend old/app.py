import os
import json
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB upload

# Load the trained model
MODEL_PATH = 'models/fruit_classifier.h5'
model = load_model(MODEL_PATH)

# Load class indices
with open('models/class_indices.json', 'r') as f:
    class_indices = json.load(f)
    
# Invert class indices to map prediction index to class name
indices_class = {v: k for k, v in class_indices.items()}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_condition_and_fruit(class_name):
    # Assuming class names format is like "good_apple", "bad_banana", etc.
    parts = class_name.split('_')
    if len(parts) >= 2:
        condition = parts[0]
        fruit = '_'.join(parts[1:])
        return condition, fruit
    return class_name, ""

def get_recommendation(condition):
    recommendations = {
        "good": "Pluck",
        "raw": "Keep",
        "bad": "Discard",
        "diseased": "Remove to prevent infection" 
    }
    return recommendations.get(condition.lower(), "Unknown recommendation")

def predict_image(img_path):
    try:
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Make prediction
        predictions = model.predict(img_array)
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
    except Exception as e:
        return {"error": f"Error processing image: {str(e)}"}

@app.route('/classify', methods=['POST'])
def classify_fruits():
    # Check if any file was sent
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    files = request.files.getlist('file')  # Get all files
    
    if not files or files[0].filename == '':
        return jsonify({"error": "No selected files"}), 400
    
    results = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            try:
                file.save(filepath)
                result = predict_image(filepath)
                result['filename'] = filename  # Add filename to results
                results.append(result)
                
                # Optionally remove the file after processing
                os.remove(filepath)
            except Exception as e:
                results.append({
                    "filename": filename,
                    "error": str(e)
                })
    
    # Return single result if only one file, otherwise return array
    if len(results) == 1:
        return jsonify(results[0])
    return jsonify(results)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "OK", "message": "Server is running"})

@app.route('/', methods=['GET'])
def home():
    return '''
    <html>
        <body>
            <h2>Fruit Classification API</h2>
            <form action="/classify" method="post" enctype="multipart/form-data">
                <input type="file" name="file">
                <input type="submit" value="Classify">
            </form>
        </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(debug=True, port=5000) 