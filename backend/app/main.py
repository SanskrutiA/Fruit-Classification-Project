from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import prediction

app = FastAPI(
    title="Fruit Classification API",
    description="API for classifying fruit images using a CNN model",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(prediction.router, prefix="/api/v1", tags=["predictions"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Fruit Classification API"} 