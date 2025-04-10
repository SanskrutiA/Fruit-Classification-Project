import tensorflow as tf

model_path = "C:\\Users\\Avhale\\Downloads\\Fruit-Classification-Project\\backend\\model\\fruit_classifier.h5"

try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
    print(model.summary())  # Display model architecture
except Exception as e:
    print(f"Error loading model: {e}")
