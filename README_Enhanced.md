# Enhanced Waste Detection Prediction API v2.0.0

An improved FastAPI-based machine learning service for waste detection predictions using Random Forest algorithms.

## 🚀 What's New in v2.0.0

### ✨ Enhanced Features
- **Robust Error Handling**: Comprehensive error handling and validation for all endpoints
- **Input Validation**: Strict Pydantic validation for all request models
- **Health Monitoring**: Advanced health checks and status monitoring
- **Better Logging**: Structured logging for debugging and monitoring
- **CORS Support**: Cross-origin requests enabled for frontend integration
- **API Documentation**: Auto-generated Swagger/OpenAPI documentation
- **Model Management**: Safe model loading with dependency injection
- **Enhanced Responses**: Richer response models with timestamps and metadata

### 🔧 New API Endpoints

#### Status & Monitoring
- `GET /` - Enhanced root endpoint with health status
- `GET /health` - Detailed health check with uptime and model status
- `GET /model/info` - Comprehensive model information
- `GET /credentials/check` - Check API credential configuration
- `POST /model/reload` - Safely reload the ML model

#### Enhanced Prediction Endpoints
- `POST /predict/` - Custom feature predictions (improved validation)
- `POST /predictFuture` - Future waste detection predictions (enhanced)
- `POST /predict_trash_hotspots/` - Hotspot location predictions (robust error handling)

## 🛠 Setup & Installation

### 1. Quick Start
```bash
# Navigate to your FastAPI directory
cd "C:\Users\yazan\OneDrive\Bureaublad\1.4\Fastapi1.4"

# Run the enhanced startup script
start_enhanced.bat
```

### 2. Manual Setup
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirementsgoat.txt

# Configure environment variables
copy .env.template .env
# Edit .env with your credentials

# Run the API
uvicorn application:app --reload --host 0.0.0.0 --port 8000
```

### 3. Environment Configuration
Create a `.env` file with your credentials:
```bash
API_USERNAME=your_username
API_PASSWORD=your_password
API_PASSWORD1=your_alternative_password
```

## 📖 API Usage

### Health Check
```python
import requests

# Basic health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Model information
response = requests.get("http://localhost:8000/model/info")
print(response.json())
```

### Enhanced Predictions
```python
# Future prediction with validation
response = requests.post("http://localhost:8000/predictFuture", json={
    "confidence": 0.85,        # Validated: 0.0 - 1.0
    "celcius": 22.5,          # Temperature in Celsius
    "time": "2025-06-23T14:30:00Z",  # ISO timestamp
    "location": "centrum",     # Optional
    "trashType": "plastic"     # Optional
})

result = response.json()
print(f"Prediction: {result['prediction'][0]:.3f}")
print(f"Confidence interval: {result['confidence_interval']}")
```

### Hotspot Prediction with Error Handling
```python
# Enhanced hotspot prediction
response = requests.post("http://localhost:8000/predict_trash_hotspots/", json={
    "days": 7  # Validated: 1-365 days
})

if response.status_code == 200:
    result = response.json()
    print(f"Hotspots: {result['hotspots']}")
    print(f"Total locations analyzed: {result['total_locations']}")
    print(f"Predictions made: {result['predictions_made']}")
else:
    print(f"Error: {response.json()['detail']}")
```

### Custom Features with Validation
```python
# Direct feature prediction with validation
response = requests.post("http://localhost:8000/predict/", json={
    "features": [0.8, 20.0, 14]  # [confidence, temperature, hour]
})

result = response.json()
print(f"Custom prediction: {result['prediction'][0]:.3f}")
```

## 🔍 API Response Examples

### Enhanced Prediction Response
```json
{
  "prediction": [0.742],
  "timestamp": "2025-06-23T14:30:00.123456",
  "model_version": "2.0.0",
  "confidence_interval": {
    "lower": 0.592,
    "upper": 0.892
  }
}
```

### Health Check Response
```json
{
  "status": "healthy",
  "message": "Service operational",
  "timestamp": "2025-06-23T14:30:00.123456",
  "model_loaded": true,
  "uptime_seconds": 3600.5
}
```

### Enhanced Hotspot Response
```json
{
  "hotspots": ["centrum", "station"],
  "location_predictions": {
    "centrum": 4.25,
    "station": 3.87,
    "park": 2.14
  },
  "total_locations": 3,
  "predictions_made": 15,
  "data_points_processed": 25,
  "timestamp": "2025-06-23T14:30:00.123456"
}
```

## 🐳 Docker Deployment

### Build and Run
```bash
# Build the enhanced Docker image
docker build -t waste-detection-api-v2 .

# Run with environment variables
docker run -d \
  --name waste-api-enhanced \
  -p 8000:80 \
  -e API_USERNAME=your_username \
  -e API_PASSWORD=your_password \
  -e API_PASSWORD1=your_alt_password \
  waste-detection-api-v2
```

### Health Check
The Docker container includes health checks:
```bash
# Check container health
docker ps
# Look for "healthy" status
```

## 🔧 Development Features

### API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Monitoring Endpoints
- **Health**: http://localhost:8000/health
- **Model Info**: http://localhost:8000/model/info
- **Credentials Check**: http://localhost:8000/credentials/check

### Enhanced Error Handling
The API now provides detailed error messages:
```json
{
  "detail": "Prediction failed: Invalid input format",
  "status_code": 400,
  "timestamp": "2025-06-23T14:30:00"
}
```

## 🧪 Validation Features

### Input Validation
- **Confidence**: Must be between 0.0 and 1.0
- **Temperature**: Must be between -50°C and 60°C
- **Hour**: Must be between 0 and 23
- **Days**: Must be between 1 and 365
- **Time Format**: Must be valid ISO timestamp

### Model Validation
- Automatic model loading validation
- Model availability checks
- Safe prediction with error handling

## 🚨 Troubleshooting

### Common Issues

1. **Model Not Loading**
   ```bash
   # Check if model file exists
   ls random_forest_model.pkl
   
   # Check health endpoint
   curl http://localhost:8000/health
   ```

2. **Authentication Issues**
   ```bash
   # Check credentials configuration
   curl http://localhost:8000/credentials/check
   ```

3. **Validation Errors**
   - Check API documentation: http://localhost:8000/docs
   - Ensure input values are within valid ranges
   - Verify JSON format

### Debugging
```bash
# View detailed logs
# The enhanced API provides structured logging

# Check API status
curl http://localhost:8000/health

# Reload model if needed
curl -X POST http://localhost:8000/model/reload
```

## 🔐 Security Improvements

- **Input Validation**: All inputs are strictly validated
- **Error Handling**: Sensitive information is not exposed
- **Environment Variables**: Credentials stored securely
- **Non-root Container**: Docker runs as non-privileged user
- **CORS Configuration**: Configurable for production

## 📊 Performance & Monitoring

- **Health Checks**: Built-in monitoring endpoints
- **Uptime Tracking**: Service uptime monitoring
- **Request Logging**: All API requests are logged
- **Error Tracking**: Comprehensive error logging
- **Response Times**: Optimized for fast responses

## 🎯 Integration with Blazor Frontend

The enhanced API is fully compatible with your Blazor frontend:

1. **Status Checks**: The frontend can monitor API health
2. **Error Handling**: Better error messages for user feedback
3. **Response Format**: Consistent JSON responses
4. **CORS Support**: Cross-origin requests enabled

### Example Integration
```csharp
// Enhanced API client usage in Blazor
var health = await PredictionApiClient.GetHealthAsync();
if (health?.ModelLoaded == true) {
    var prediction = await PredictionApiClient.MakeFuturePredictionAsync(request);
    // Handle enhanced response with confidence intervals
}
```

## 🚀 Deployment to Azure

Your existing Azure deployment will work with the enhanced version:

1. **GitHub Actions**: Your existing CI/CD pipeline will deploy the enhanced version
2. **Environment Variables**: Add the new variables to your Azure App Service
3. **Health Checks**: Azure can use the `/health` endpoint for monitoring

Enjoy the enhanced Waste Detection API v2.0.0! 🎉
