# API Deployment Guide

## Sentiment Analysis RESTful API

This guide provides instructions for deploying and integrating the Sentiment Analysis API.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Trained sentiment model files (sentiment_model.pkl and vectorizer.pkl)

## Installation

### 1. Install Dependencies

```bash
pip install fastapi uvicorn pydantic scikit-learn numpy
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

### 2. Prepare Model Files

Ensure your trained model and vectorizer are in the `models/` directory:
```
models/
  ├── sentiment_model.pkl
  └── vectorizer.pkl
```

## Running the API

### Development Mode

Run the API locally for development:

```bash
python app.py
```

Or using uvicorn directly:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

### Production Mode

For production deployment:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### 1. Root Endpoint

**GET /**

Returns API information and available endpoints.

```bash
curl http://localhost:8000/
```

Response:
```json
{
  "message": "Sentiment Analysis API",
  "version": "1.0.0",
  "endpoints": {
    "/predict": "POST - Predict sentiment for a single text",
    "/health": "GET - Check API health status",
    "/docs": "GET - API documentation"
  }
}
```

### 2. Health Check

**GET /health**

Check API health and model status.

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "vectorizer_loaded": true
}
```

### 3. Predict Sentiment

**POST /predict**

Predict sentiment for input text.

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie is absolutely amazing!"}'
```

Response:
```json
{
  "text": "This movie is absolutely amazing!",
  "sentiment": "positive",
  "confidence": 0.95,
  "probabilities": {
    "negative": 0.05,
    "positive": 0.95
  }
}
```

## Integration Examples

### Python Integration

```python
import requests

url = "http://localhost:8000/predict"
data = {"text": "I love this product!"}

response = requests.post(url, json=data)
result = response.json()

print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### JavaScript Integration

```javascript
const analyzeSentiment = async (text) => {
  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ text: text })
  });
  
  const result = await response.json();
  console.log(`Sentiment: ${result.sentiment}`);
  console.log(`Confidence: ${(result.confidence * 100).toFixed(2)}%`);
  return result;
};

// Usage
analyzeSentiment('This is great!');
```

### cURL Integration

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here"}'
```

## Deployment Options

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t sentiment-api .
docker run -p 8000:8000 sentiment-api
```

### Cloud Deployment (Heroku)

1. Create `Procfile`:
```
web: uvicorn app:app --host 0.0.0.0 --port $PORT
```

2. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

### Cloud Deployment (AWS Lambda + API Gateway)

Use Mangum adapter:

```python
from mangum import Mangum

handler = Mangum(app)
```

## API Documentation

Once the API is running, access interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `422`: Validation error (invalid input)
- `500`: Internal server error
- `503`: Service unavailable (model not loaded)

Example error response:
```json
{
  "detail": "Model not loaded. Please check model files."
}
```

## Performance Considerations

- Use multiple workers for production: `--workers 4`
- Consider caching for frequently analyzed texts
- Monitor memory usage with large models
- Use load balancing for high traffic

## Security Best Practices

1. Use HTTPS in production
2. Implement rate limiting
3. Add authentication for sensitive deployments
4. Validate and sanitize all inputs
5. Keep dependencies updated

## Monitoring

Add logging to track API usage:

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/predict")
async def predict_sentiment(input_data: TextInput):
    logger.info(f"Prediction request received: {input_data.text[:50]}...")
    # ... rest of code
```

## Troubleshooting

### Model Not Loading
- Check model files exist in `models/` directory
- Verify pickle file compatibility
- Check file permissions

### Port Already in Use
```bash
# Use different port
uvicorn app:app --port 8001
```

### CORS Issues
Add CORS middleware for browser access:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Support

For issues or questions, please open an issue on the GitHub repository.
