from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.ml_model import FruitClassifier
from app.utils.image_utils import decode_image, preprocess_image
from app.schemas.schemas import PredictionResponse, ErrorResponse
import logging

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter()
classifier = FruitClassifier()

@router.post("/predict", response_model=PredictionResponse, responses={400: {"model": ErrorResponse}})
async def predict_image(file: UploadFile = File(...)):
    """
    Upload an image and get fruit classification predictions
    """
    try:
        # Check file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"File type not supported. Expected image, got {file.content_type}"
            )
        
        # Read and decode image
        logger.info(f"Processing image: {file.filename}")
        contents = await file.read()
        image = decode_image(contents)
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = classifier.predict(processed_image)
        logger.info(f"Prediction result: {prediction}")
        
        return prediction
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Error processing image: {str(e)}"
        ) 