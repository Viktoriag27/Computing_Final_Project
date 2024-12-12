from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from typing import List, Optional
from pathlib import Path
import os

# Initialize models
model = None
scaler = None
feature_transformers = None

# Use a relative path based on the location of the current script
model_path = Path(__file__).parent.parent / 'model'


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, feature_transformers
    try:
        # Load model, scaler, and feature transformers
        model = joblib.load(str(model_path / 'house_price_model.joblib'))
        scaler = joblib.load(str(model_path / 'scaler.joblib'))
        feature_transformers = joblib.load(str(model_path / 'feature_transformers.joblib'))
        print(f"Models loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading models from {model_path}: {e}")
    yield

# Initialize FastAPI app with lifespan context
app = FastAPI(title="California House Price Prediction API", lifespan=lifespan)

# Define Pydantic model for input data
class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

# Define Pydantic model for prediction response
class PredictionResponse(BaseModel):
    predicted_price: float
    # confidence_interval: Optional[List[float]]

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: HouseFeatures):
    try:
        if not all([model, scaler, feature_transformers]):
            raise HTTPException(status_code=500, detail="Model files not loaded")
            
        # Prepare input data as DataFrame
        input_data = pd.DataFrame([{
            'MedInc': features.MedInc,
            'HouseAge': features.HouseAge,
            'AveRooms': features.AveRooms,
            'AveBedrms': features.AveBedrms,
            'Population': features.Population,
            'AveOccup': features.AveOccup,
            'Latitude': features.Latitude,
            'Longitude': features.Longitude
        }])
        
        # Scale the data using the pre-trained scaler
        scaled_data = scaler.transform(input_data)
        scaled_data = pd.DataFrame(scaled_data, columns=input_data.columns)
        
        # Apply feature transformations
        all_features = [scaled_data]
        for transformer in feature_transformers:
            new_features = transformer.transform(scaled_data)
            all_features.append(new_features)
        
        # Concatenate all feature transformations into one DataFrame
        feature_data = pd.concat(all_features, axis=1)
        
        # Main prediction using the model
        prediction = model.predict(feature_data)[0]
        
        return {
            "predicted_price": float(prediction)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
