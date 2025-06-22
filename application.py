import pandas as pd
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import requests
import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv() # laad de .env file

# Global variables
model = None
app_start_time = datetime.utcnow()

# Enhanced Pydantic models with validation
class FutureFeatures(BaseModel):
    """Enhanced model for future predictions with validation"""
    id: Optional[str] = Field(default_factory=lambda: str(datetime.utcnow().timestamp()))
    time: str = Field(..., description="ISO timestamp")
    trashType: str = Field(default="unknown", description="Type of trash")
    location: str = Field(default="unknown", description="Location identifier")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence between 0 and 1")
    celcius: float = Field(..., description="Temperature in Celsius")
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Confidence must be between 0 and 1')
        return v
    
    @validator('time')
    def validate_time(cls, v):
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
            return v
        except ValueError:
            raise ValueError("Invalid time format. Use ISO format.")

class Features(BaseModel):
    """Enhanced features model with validation"""
    features: List[float] = Field(..., min_items=3, max_items=3, 
                                 description="Exactly 3 features: [confidence, temperature, hour]")
    
    @validator('features')
    def validate_features(cls, v):
        if len(v) != 3:
            raise ValueError("Exactly 3 features required")
        if not (0 <= v[0] <= 1):
            raise ValueError("First feature (confidence) must be between 0 and 1")
        if not (-50 <= v[1] <= 60):
            raise ValueError("Second feature (temperature) must be between -50 and 60")
        if not (0 <= v[2] <= 23):
            raise ValueError("Third feature (hour) must be between 0 and 23")
        return v

class TimeFrameRequest(BaseModel):
    """Enhanced timeframe request with validation"""
    days: int = Field(..., ge=1, le=365, description="Number of days to look back (1-365)")

class PredictionResponse(BaseModel):
    """Enhanced prediction response"""
    prediction: List[float]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_version: str = "2.0.0"
    confidence_interval: Optional[Dict[str, float]] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_loaded: bool
    uptime_seconds: float

class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_loaded: bool
    model_version: str = "2.0.0"
    features_count: int = 3
    endpoints_available: List[str]

# Model loading with error handling
def load_model_safely():
    """Safely load the ML model with proper error handling"""
    global model
    try:
        thisfile = Path(__file__).parent
        modelfile = (thisfile / "random_forest_model.pkl").resolve()
        
        if not modelfile.exists():
            logger.error(f"Model file not found at {modelfile}")
            raise FileNotFoundError(f"Model file not found at {modelfile}")
        
        model = joblib.load(modelfile)
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None
        return False

# Dependency to check if model is loaded
def get_model():
    """Dependency to ensure model is loaded"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model

# Initialize FastAPI with enhanced configuration
app = FastAPI(
    title="Enhanced Waste Detection Predictor",
    description="Improved ML-powered waste detection predictions using Random Forest",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
@app.on_event("startup")
async def startup_event():
    """Load model on application startup"""
    logger.info("Starting up application...")
    success = load_model_safely()
    if not success:
        logger.warning("Application started but model failed to load")

# Enhanced endpoints with better error handling

@app.get("/", response_model=HealthResponse)
async def read_root():
    """Enhanced root endpoint with health information"""
    uptime = (datetime.utcnow() - app_start_time).total_seconds()
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        message="Python AI FastAPI model is live! Enhanced version 2.0.0 🚀",
        model_loaded=model is not None,
        uptime_seconds=uptime
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check endpoint"""
    uptime = (datetime.utcnow() - app_start_time).total_seconds()
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        message="Service operational" if model is not None else "Model not loaded",
        model_loaded=model is not None,
        uptime_seconds=uptime
    )

@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get detailed model information"""
    return ModelInfoResponse(
        model_loaded=model is not None,
        endpoints_available=[
            "/",
            "/health",
            "/model/info",
            "/predict/",
            "/predictFuture",
            "/predict_trash_hotspots/",
            "/model/reload"
        ]
    )

@app.get("/credentials/check")
async def check_credentials():
    """Check if API credentials are configured"""
    credentials_set = bool(
        os.environ.get("API_USERNAME") and 
        os.environ.get("API_PASSWORD") and 
        os.environ.get("API_PASSWORD1")
    )
    return {
        "status": "Credentials are set" if credentials_set else "Credentials are missing",
        "has_username": bool(os.environ.get("API_USERNAME")),
        "has_password": bool(os.environ.get("API_PASSWORD")),
        "has_password1": bool(os.environ.get("API_PASSWORD1"))
    }

@app.post("/predict/", response_model=PredictionResponse)
async def predict(input: Features, model_instance=Depends(get_model)):
    """Enhanced custom prediction with better error handling"""
    try:
        prediction = model_instance.predict(np.array([input.features]))
        logger.info(f"Custom prediction made: features={input.features}, result={prediction}")
        
        return PredictionResponse(
            prediction=prediction.tolist(),
            confidence_interval={
                "lower": max(0, prediction[0] - 0.1),
                "upper": min(1, prediction[0] + 0.1)
            }
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predictFuture", response_model=PredictionResponse)
async def predict_future(input: FutureFeatures, model_instance=Depends(get_model)):
    """Enhanced future prediction with validation and error handling"""
    try:
        # Extract hour from timestamp
        dt = datetime.fromisoformat(input.time.replace("Z", "+00:00"))
        hour = dt.hour
        
        # Prepare features: [confidence, temperature, hour]
        features = [input.confidence, input.celcius, hour]
        prediction = model_instance.predict([features])
        
        logger.info(f"Future prediction: location={input.location}, features={features}, result={prediction}")
        
        return PredictionResponse(
            prediction=prediction.tolist(),
            confidence_interval={
                "lower": max(0, prediction[0] - 0.15),
                "upper": min(1, prediction[0] + 0.15)
            }
        )
    except ValueError as e:
        logger.error(f"Validation error in future prediction: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Future prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Future prediction failed: {str(e)}")

@app.post("/predict_trash_hotspots/")
async def predict_trash_hotspots(request: TimeFrameRequest, model_instance=Depends(get_model)):
    """Enhanced hotspot prediction with better error handling and validation"""
    try:
        # Check credentials
        username = os.environ.get("API_USERNAME")
        password1 = os.environ.get("API_PASSWORD1")
        
        if not username or not password1:
            raise HTTPException(
                status_code=500, 
                detail="API credentials not configured. Please set API_USERNAME and API_PASSWORD1."
            )
        
        # Authenticate with external API
        login_url = "https://bitbybit-api.orangecliff-c30465b7.northeurope.azurecontainerapps.io/account/login"
        login_data = {"username": username, "password": password1}
        
        logger.info("Authenticating with external API...")
        login_response = requests.post(login_url, json=login_data, timeout=10)
        login_response.raise_for_status()
        
        response_data = login_response.json()
        token = response_data.get("access_token")
        
        if not token:
            raise HTTPException(status_code=401, detail="Failed to authenticate with external API")
        
        # Fetch litter data
        headers = {"Authorization": f"Bearer {token}"}
        api_url = "https://bitbybit-api.orangecliff-c30465b7.northeurope.azurecontainerapps.io/litter"
        
        logger.info("Fetching litter data...")
        data_response = requests.get(api_url, headers=headers, timeout=10)
        data_response.raise_for_status()
        data = data_response.json()
        
        if not data:
            logger.warning("No data received from external API")
            return {
                "hotspots": [],
                "location_predictions": {},
                "total_locations": 0,
                "message": "No data available"
            }
        
        # Filter data by time frame
        cutoff = datetime.utcnow() - timedelta(days=request.days)
        filtered = []
        
        for item in data:
            try:
                item_time = datetime.fromisoformat(item["time"].replace("Z", "+00:00"))
                if item_time >= cutoff:
                    # Validate required fields
                    if all(key in item for key in ["confidence", "celcius", "location"]):
                        filtered.append(item)
                    else:
                        logger.warning(f"Skipping item with missing required fields: {item}")
            except (KeyError, ValueError) as e:
                logger.warning(f"Skipping invalid data item: {e}")
                continue
        
        if not filtered:
            return {
                "hotspots": [],
                "location_predictions": {},
                "total_locations": 0,
                "message": f"No valid data found for the last {request.days} days"
            }
        
        # Make predictions for each filtered item
        predictions = []
        for item in filtered:
            try:
                hour = datetime.fromisoformat(item["time"].replace("Z", "+00:00")).hour
                features = [item["confidence"], item["celcius"], hour]
                pred = model_instance.predict([features])[0]
                predictions.append({
                    "location": item["location"], 
                    "prediction": pred
                })
            except Exception as e:
                logger.warning(f"Skipping prediction for item due to error: {e}")
                continue
        
        # Aggregate predictions by location
        location_counts = {}
        for p in predictions:
            loc = p["location"]
            location_counts[loc] = location_counts.get(loc, 0) + p["prediction"]
        
        # Find location(s) with highest predicted trash
        hotspots = []
        if location_counts:
            max_pred = max(location_counts.values())
            hotspots = [loc for loc, val in location_counts.items() if val == max_pred]
        
        logger.info(f"Hotspot prediction completed: {len(hotspots)} hotspots found from {len(predictions)} predictions")
        
        return {
            "hotspots": hotspots,
            "location_predictions": location_counts,
            "total_locations": len(location_counts),
            "predictions_made": len(predictions),
            "data_points_processed": len(filtered),
            "timestamp": datetime.utcnow()
        }
        
    except requests.RequestException as e:
        logger.error(f"External API request failed: {e}")
        raise HTTPException(status_code=502, detail=f"External API request failed: {str(e)}")
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Hotspot prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Hotspot prediction failed: {str(e)}")

@app.post("/model/reload")
async def reload_model():
    """Reload the machine learning model"""
    try:
        success = load_model_safely()
        if success:
            return {
                "message": "Model reloaded successfully", 
                "timestamp": datetime.utcnow(),
                "status": "success"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to reload model")
    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")

# Development server runner
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "application:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )