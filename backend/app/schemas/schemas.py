from pydantic import BaseModel, Field
from typing import Dict, List

class PredictionRequest(BaseModel):
    image_path: str

class PredictionResponse(BaseModel):
    class_: str = Field(alias="class")
    confidence: float
    all_probabilities: Dict[str, float]

class ErrorResponse(BaseModel):
    error: str
    detail: str 