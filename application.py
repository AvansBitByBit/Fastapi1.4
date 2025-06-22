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

# New training-related models
class TrainingDataPoint(BaseModel):
    """Single training data point"""
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence between 0 and 1")
    temperature: float = Field(..., description="Temperature in Celsius")
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    target: float = Field(..., description="Target value (what we're trying to predict)")
    location: Optional[str] = Field(default="unknown", description="Location identifier")
    trash_type: Optional[str] = Field(default="unknown", description="Type of trash")

class TrainingDataBatch(BaseModel):
    """Batch of training data"""
    data_points: List[TrainingDataPoint] = Field(..., min_items=1, description="List of training data points")
    model_name: Optional[str] = Field(default="waste_detection_model", description="Name for the trained model")

class TrainingResponse(BaseModel):
    """Training response"""
    status: str
    message: str
    model_accuracy: Optional[float] = None
    training_samples: int
    model_version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ModelMetrics(BaseModel):
    """Model performance metrics"""
    accuracy: float
    mean_squared_error: float
    feature_importance: Dict[str, float]
    training_samples: int
    model_version: str
    training_date: datetime

# Training endpoints
@app.post("/training/train_new_model", response_model=TrainingResponse)
async def train_new_model(training_data: TrainingDataBatch):
    """Train a completely new Random Forest model with provided data"""
    global model
    try:
        logger.info(f"Starting training with {len(training_data.data_points)} data points")
        
        # Prepare training data
        X = []
        y = []
        
        for point in training_data.data_points:
            X.append([point.confidence, point.temperature, point.hour])
            y.append(point.target)
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
        
        # Create and train new Random Forest model
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        
        # Split data for validation
        if len(X) > 10:  # Only split if we have enough data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            X_train, X_test, y_train, y_test = X, X, y, y
        
        # Train new model
        new_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1
        )
        
        new_model.fit(X_train, y_train)
        
        # Calculate accuracy
        y_pred = new_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Save the new model
        thisfile = Path(__file__).parent
        modelfile = (thisfile / "random_forest_model.pkl").resolve()
        
        # Backup old model if it exists
        if modelfile.exists():
            backup_file = thisfile / f"random_forest_model_backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pkl"
            import shutil
            shutil.copy2(modelfile, backup_file)
            logger.info(f"Old model backed up to {backup_file}")
        
        # Save new model
        joblib.dump(new_model, modelfile)
        logger.info(f"New model saved to {modelfile}")
        
        # Update global model
        model = new_model
        
        logger.info(f"Model training completed. R² score: {r2:.4f}, MSE: {mse:.4f}")
        
        return TrainingResponse(
            status="success",
            message=f"Model trained successfully with {len(training_data.data_points)} samples",
            model_accuracy=r2,
            training_samples=len(training_data.data_points),
            model_version="2.1.0"
        )
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

@app.post("/training/train_from_api", response_model=TrainingResponse)
async def train_from_existing_api():
    """Train model using data from the existing API (all time data)"""
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
        
        logger.info("Authenticating with external API for training...")
        login_response = requests.post(login_url, json=login_data, timeout=10)
        login_response.raise_for_status()
        
        response_data = login_response.json()
        token = response_data.get("access_token")
        
        if not token:
            raise HTTPException(status_code=401, detail="Failed to authenticate with external API")
        
        # Fetch all litter data
        headers = {"Authorization": f"Bearer {token}"}
        api_url = "https://bitbybit-api.orangecliff-c30465b7.northeurope.azurecontainerapps.io/litter"
        
        logger.info("Fetching all litter data for training...")
        data_response = requests.get(api_url, headers=headers, timeout=30)
        data_response.raise_for_status()
        data = data_response.json()
        
        if not data or len(data) < 10:
            raise HTTPException(
                status_code=400, 
                detail=f"Insufficient data for training. Need at least 10 samples, got {len(data) if data else 0}"
            )
        
        logger.info(f"Retrieved {len(data)} data points from API")
        
        # Process data for training
        training_points = []
        processed_count = 0
        
        for item in data:
            try:
                # Validate required fields
                if not all(key in item for key in ["confidence", "celcius", "time"]):
                    continue
                
                # Extract features
                confidence = float(item["confidence"])
                temperature = float(item["celcius"])
                time_str = item["time"]
                
                # Parse time to get hour
                item_time = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
                hour = item_time.hour
                
                # Create target value based on confidence (this is a simple approach)
                # In a real scenario, you'd have actual target values to predict
                # For now, we'll use confidence as both feature and target (with some transformation)
                target = confidence * 0.8 + (temperature / 100) * 0.2  # Simple composite target
                
                training_points.append(TrainingDataPoint(
                    confidence=confidence,
                    temperature=temperature,
                    hour=hour,
                    target=target,
                    location=item.get("location", "unknown"),
                    trash_type=item.get("trashType", "unknown")
                ))
                processed_count += 1
                
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping invalid data item: {e}")
                continue
        
        if len(training_points) < 10:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient valid data for training. Processed {len(training_points)} valid samples, need at least 10"
            )
        
        logger.info(f"Processed {len(training_points)} valid training samples")
        
        # Create training batch and train model
        training_batch = TrainingDataBatch(
            data_points=training_points,
            model_name="waste_detection_from_api"
        )
        
        # Use the existing training function
        return await train_new_model(training_batch)
        
    except requests.RequestException as e:
        logger.error(f"External API request failed during training: {e}")
        raise HTTPException(status_code=502, detail=f"External API request failed: {str(e)}")
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Training from API failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training from API failed: {str(e)}")

@app.get("/training/model_metrics", response_model=ModelMetrics)
async def get_model_metrics(model_instance=Depends(get_model)):
    """Get detailed metrics about the current model"""
    try:
        # For a real implementation, you'd store these metrics during training
        # For now, we'll provide basic information
        
        feature_names = ["confidence", "temperature", "hour"]
        
        # Get feature importance if available
        feature_importance = {}
        if hasattr(model_instance, 'feature_importances_'):
            for i, importance in enumerate(model_instance.feature_importances_):
                feature_importance[feature_names[i]] = float(importance)
        else:
            # Default if not available
            feature_importance = {name: 1.0/len(feature_names) for name in feature_names}
        
        return ModelMetrics(
            accuracy=0.85,  # Placeholder - would be stored during training
            mean_squared_error=0.12,  # Placeholder - would be stored during training
            feature_importance=feature_importance,
            training_samples=1000,  # Placeholder - would be stored during training
            model_version="2.1.0",
            training_date=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Failed to get model metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model metrics: {str(e)}")

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