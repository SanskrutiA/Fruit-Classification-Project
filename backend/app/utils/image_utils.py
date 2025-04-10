from PIL import Image
import numpy as np
import io

def preprocess_image(image: Image.Image, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Preprocess the image for model input
    """
    # Resize image
    image = image.resize(target_size)
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array and normalize
    img_array = np.array(image)
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def decode_image(image_bytes: bytes) -> Image.Image:
    """
    Decode image bytes to PIL Image
    """
    return Image.open(io.BytesIO(image_bytes)) 