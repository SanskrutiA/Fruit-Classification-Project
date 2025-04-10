# Fruit Classification Project

A web application that uses Convolutional Neural Networks (CNN) to classify fruits based on their type and condition.

## Features

- Classifies fruits into different types (apple, banana, lime, etc.)
- Detects fruit condition (Good, Bad, Raw, Diseased)
- Provides recommendations based on fruit condition
- Interactive web interface for uploading and classifying fruit images
- Supports multiple image classifications for testing

## Project Structure 

📂 Fruit-Classification-Project/
│
├── 📂 backend (Flask API + CNN Model)
│ ├── 📂 dataset (Images of fruits)
│ │ ├── good_fruits/
│ │ ├── bad_fruits/
│ │ ├── raw_fruits/
│ │ ├── diseased_fruits/
│ ├── train_model.py (Train CNN)
│ ├── test_model.py (Test CNN Model)
│ ├── app.py (Flask API for classification)
│ ├── models/
│ │ ├── fruit_classifier.h5 (Saved CNN model)
│ ├── uploads/ (User-uploaded images)
│
├── 📂 frontend (React UI)
│ ├── src/components/
│ │ ├── FileUpload.js (Upload fruit images)
│ │ ├── ResultDisplay.js (Show result)
│ ├── App.js (Frontend logic)
│
└── README.md (Project setup & instructions)

## Setup Instructions

### Prerequisites

- Python 3.8+
- Node.js 14+
- TensorFlow 2.x
- Flask
- React

### Backend Setup

1. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install required Python packages:
   ```
   pip install tensorflow numpy flask flask-cors pillow matplotlib
   ```

3. Prepare your dataset:
   - Place fruit images in appropriate folders under `backend/dataset/`
   - Organize by condition and fruit type (e.g., `good_apple`, `bad_banana`)

4. Train the model:
   ```
   cd backend
   python train_model.py
   ```

5. Run the Flask API:
   ```
   python app.py
   ```

### Frontend Setup

1. Install dependencies:
   ```
   cd frontend
   npm install
   ```

2. Start the development server:
   ```
   npm start
   ```

3. Access the application at `http://localhost:3000`

## Usage

1. Upload an image of a fruit using the interface
2. The system will analyze the image and provide:
   - Fruit type (Apple, Banana, etc.)
   - Condition (Good, Bad, Raw, Diseased)
   - Recommendation (Pluck, Keep, Discard, or Remove)

## API Response Format

When an image is classified, the API returns a JSON response in this format:

```json
{
    "fruit": "Apple",
    "condition": "Diseased",
    "recommendation": "Remove to prevent infection",
    "confidence": "92.7%"
}
```

## Testing

You can test the model with multiple images using the test script:

```
cd backend
python test_model.py
```

This will open a file dialog allowing you to select multiple images for batch classification.

## License

[MIT License](LICENSE)