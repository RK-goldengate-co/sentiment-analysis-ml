from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os

app = FastAPI(
    title="Sentiment Analysis API",
    description="RESTful API for sentiment analysis using machine learning",
    version="1.0.0"
)

# Load model and vectorizer
MODEL_PATH = "models/sentiment_model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

model = None
vectorizer = None

@app.on_event("startup")
async def load_model():
    """Load model and vectorizer on startup"""
    global model, vectorizer
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
        if os.path.exists(VECTORIZER_PATH):
            with open(VECTORIZER_PATH, 'rb') as f:
                vectorizer = pickle.load(f)
        print("Model and vectorizer loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")

class TextInput(BaseModel):
    text: str
    
class PredictionResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probabilities: dict

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Sentiment Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Predict sentiment for a single text",
            "/health": "GET - Check API health status",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(input_data: TextInput):
    """
    Predict sentiment for input text
    
    Args:
        input_data: TextInput object containing the text to analyze
    
    Returns:
        PredictionResponse with sentiment label and confidence scores
    """
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check model files."
        )
    
    try:
        # Preprocess and vectorize text
        text_vectorized = vectorizer.transform([input_data.text])
        
        # Get prediction
        prediction = model.predict(text_vectorized)[0]
        
        # Get probability scores
        probabilities = model.predict_proba(text_vectorized)[0]
        
        # Map to sentiment labels (assuming binary classification: negative=0, positive=1)
        sentiment_labels = ['negative', 'positive']
        sentiment = sentiment_labels[prediction]
        confidence = float(probabilities[prediction])
        
        # Create probability dictionary
        prob_dict = {label: float(prob) for label, prob in zip(sentiment_labels, probabilities)}
        
        return PredictionResponse(
            text=input_data.text,
            sentiment=sentiment,
            confidence=confidence,
            probabilities=prob_dict
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during prediction: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
