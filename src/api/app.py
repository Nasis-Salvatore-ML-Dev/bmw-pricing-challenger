"""
BMW Pricing API - FastAPI Application

Production REST API for BMW vehicle price predictions
Serves trained Random Forest model with <50ms latency target

Endpoints:
- POST /predict - Single prediction
- POST /predict/batch - Batch predictions
- GET /health - Health check
- GET /metrics - Model performance metrics
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import joblib
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import time
import warnings

# Suppress sklearn version warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model, encoders, and target encodings
MODEL = None
ENCODERS = None
TARGET_ENCODINGS = None
FEATURE_NAMES = None


def load_model():
    """Load trained model, encoders, and target encodings"""
    global MODEL, ENCODERS, TARGET_ENCODINGS, FEATURE_NAMES

    try:
        model_path = Path("data/models/checkpoints/rand_forest_v1.pkl")

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        logger.info(f"Loading model from {model_path}")
        package = joblib.load(model_path)

        if isinstance(package, dict):
            MODEL = package['model']
            ENCODERS = package.get('encoders', {})
            TARGET_ENCODINGS = package.get('target_encodings', {})
            logger.info(f"✅ Model loaded with {len(ENCODERS)} encoders and target encodings")
        else:
            MODEL = package
            ENCODERS = {}
            TARGET_ENCODINGS = {}
            logger.warning("⚠️  Old model format - no encoders or target encodings found")

        if hasattr(MODEL, 'feature_names_in_'):
            FEATURE_NAMES = MODEL.feature_names_in_.tolist()
        else:
            FEATURE_NAMES = None

        logger.info("✅ Model loaded successfully")

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler - replaces on_event"""
    # Startup
    load_model()
    yield
    # Shutdown (cleanup if needed)
    logger.info("API shutting down")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="BMW Pricing API",
    description="ML-powered BMW vehicle price prediction service",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CarFeatures(BaseModel):
    """Input schema for single car prediction"""
    model_key: str = Field(..., example="320d")
    mileage: float = Field(..., ge=0, example=120000, description="Mileage in kilometers")
    engine_power: float = Field(..., gt=0, example=184, description="Engine power in HP")
    registration_date: str = Field(..., example="2015-03-01", description="YYYY-MM-DD format")
    fuel: str = Field(..., example="diesel")
    paint_color: str = Field(..., example="black")
    car_type: str = Field(..., example="sedan")
    sold_at: str = Field(..., example="2020-06-15", description="Sale date YYYY-MM-DD")

    # Optional binary features (feature_1 to feature_8)
    feature_1: bool = Field(default=False, description="Binary feature 1")
    feature_2: bool = Field(default=False, description="Binary feature 2")
    feature_3: bool = Field(default=False, description="Binary feature 3")
    feature_4: bool = Field(default=False, description="Binary feature 4")
    feature_5: bool = Field(default=False, description="Binary feature 5")
    feature_6: bool = Field(default=False, description="Binary feature 6")
    feature_7: bool = Field(default=False, description="Binary feature 7")
    feature_8: bool = Field(default=False, description="Binary feature 8")

    @field_validator('registration_date', 'sold_at')
    @classmethod
    def validate_date(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')

    @field_validator('fuel')
    @classmethod
    def validate_fuel(cls, v):
        allowed = ['diesel', 'petrol', 'hybrid_petrol', 'electro']
        if v.lower() not in allowed:
            raise ValueError(f'Fuel must be one of: {allowed}')
        return v.lower()

    @field_validator('car_type')
    @classmethod
    def validate_car_type(cls, v):
        allowed = ['sedan', 'suv', 'coupe', 'convertible', 'estate', 'hatchback', 'van', 'subcompact']
        if v.lower() not in allowed:
            raise ValueError(f'Car type must be one of: {allowed}')
        return v.lower()


class BatchPredictionRequest(BaseModel):
    """Input schema for batch predictions"""
    cars: List[CarFeatures] = Field(..., max_items=100, description="Max 100 cars per batch")


class PredictionResponse(BaseModel):
    """Output schema for single prediction"""
    predicted_price: float = Field(..., description="Predicted price in EUR")
    confidence_interval: Dict[str, float] = Field(..., description="95% confidence interval")
    model_version: str = Field(..., description="Model version used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class BatchPredictionResponse(BaseModel):
    """Output schema for batch predictions"""
    predictions: List[PredictionResponse]
    total_cars: int
    processing_time_ms: float


def preprocess_input(car: CarFeatures) -> pd.DataFrame:
    """
    Preprocess input features to match EXACT training format

    Model expects 38 features in this EXACT order:
    1-17: Raw features (maker_key, model_key, mileage, ... sold_at)
    18-35: Engineered features (bmw_series, luxury_tier, ... is_old_car)
    36-38: Target-encoded features (fuel_encoded, color_encoded, car_type_encoded)

    CRITICAL: Model wants BOTH raw categoricals (label-encoded) AND target-encoded versions!
    """

    # Convert to dict (Pydantic V2)
    data = car.model_dump()

    # Calculate car age
    reg_date = pd.to_datetime(data['registration_date'])
    sold_date = pd.to_datetime(data['sold_at'])
    data['car_age_years'] = (sold_date - reg_date).days / 365.25

    # Extract BMW series and luxury indicators
    model_key_upper = data['model_key'].upper()
    if 'X5' in model_key_upper:
        bmw_series = 'X5'
        luxury_tier = 3
        is_luxury = 1
    elif 'X3' in model_key_upper:
        bmw_series = 'X3'
        luxury_tier = 2
        is_luxury = 0
    elif 'X1' in model_key_upper:
        bmw_series = 'X1'
        luxury_tier = 1
        is_luxury = 0
    elif '7' in model_key_upper:
        bmw_series = '7_series'
        luxury_tier = 4
        is_luxury = 1
    elif '5' in model_key_upper:
        bmw_series = '5_series'
        luxury_tier = 3
        is_luxury = 1
    elif '3' in model_key_upper:
        bmw_series = '3_series'
        luxury_tier = 2
        is_luxury = 0
    elif '1' in model_key_upper:
        bmw_series = '1_series'
        luxury_tier = 1
        is_luxury = 0
    else:
        bmw_series = 'other_series'
        luxury_tier = 2
        is_luxury = 0

    data['bmw_series'] = bmw_series
    data['luxury_tier'] = luxury_tier
    data['is_luxury'] = is_luxury
    data['is_performance'] = 1 if 'M' in model_key_upper else 0

    # Temporal features
    data['registration_year'] = reg_date.year
    data['registration_month'] = reg_date.month
    data['registration_quarter'] = reg_date.quarter
    data['is_summer_registration'] = 1 if reg_date.month in [6, 7, 8] else 0
    data['is_year_end_registration'] = 1 if reg_date.month in [11, 12] else 0

    # Interaction features
    data['age_mileage_interaction'] = data['car_age_years'] * data['mileage'] / 10000
    data['mileage_per_power'] = data['mileage'] / (data['engine_power'] + 1)
    data['annual_mileage'] = data['mileage'] / (data['car_age_years'] + 1)
    data['power_age_ratio'] = data['engine_power'] / (data['car_age_years'] + 1)
    data['luxury_mileage_interaction'] = is_luxury * data['mileage'] / 10000
    data['luxury_age_interaction'] = is_luxury * data['car_age_years']
    data['is_high_mileage'] = 1 if (data['mileage'] / (data['car_age_years'] + 1)) > 25000 else 0
    data['is_old_car'] = 1 if data['car_age_years'] > 8 else 0

    # Add maker_key (always BMW)
    data['maker_key'] = 'BMW'

    # Convert binary features to integers
    for i in range(1, 9):
        feature_name = f'feature_{i}'
        data[feature_name] = int(data.get(feature_name, False))

    # Create DataFrame
    df = pd.DataFrame([data])

    # ------------------------------------------------------------------
    # Apply label encoders to raw categorical columns
    # This includes: maker_key, model_key, fuel, paint_color, car_type
    # ------------------------------------------------------------------
    if ENCODERS:
        for col, encoder in ENCODERS.items():
            if col in df.columns:
                try:
                    # Map each value to its integer label; unseen categories get -1
                    df[col] = df[col].astype(str).apply(
                        lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                    )
                except Exception as e:
                    logger.warning(f"Encoding error for column {col}: {e}")
                    df[col] = -1
    else:
        # Fallback if no encoders (should not happen)
        logger.warning("No encoders found; using default mapping.")
        for col in ['maker_key', 'model_key', 'fuel', 'paint_color', 'car_type']:
            if col in df.columns:
                df[col] = 0

    # ------------------------------------------------------------------
    # Create target-encoded columns using saved mappings
    # These are: fuel_encoded, color_encoded, car_type_encoded
    # ------------------------------------------------------------------
    if TARGET_ENCODINGS:
        # Fuel
        fuel_map = TARGET_ENCODINGS.get('fuel', {})
        df['fuel_encoded'] = df['fuel'].map(fuel_map).fillna(np.mean(list(fuel_map.values())) if fuel_map else 0)

        # Color (paint_color)
        color_map = TARGET_ENCODINGS.get('color', {})
        df['color_encoded'] = df['paint_color'].map(color_map).fillna(np.mean(list(color_map.values())) if color_map else 0)

        # Car type
        car_type_map = TARGET_ENCODINGS.get('car_type', {})
        df['car_type_encoded'] = df['car_type'].map(car_type_map).fillna(np.mean(list(car_type_map.values())) if car_type_map else 0)
    else:
        # Fallback (should not happen)
        logger.warning("Target encodings missing; using zeros.")
        df['fuel_encoded'] = 0
        df['color_encoded'] = 0
        df['car_type_encoded'] = 0

    # Ensure numeric columns are float
    numeric_cols = ['mileage', 'engine_power', 'car_age_years', 'luxury_tier', 'is_luxury',
                    'is_performance', 'registration_year', 'registration_month', 'registration_quarter',
                    'is_summer_registration', 'is_year_end_registration', 'age_mileage_interaction',
                    'mileage_per_power', 'annual_mileage', 'power_age_ratio', 'luxury_mileage_interaction',
                    'luxury_age_interaction', 'is_high_mileage', 'is_old_car',
                    'fuel_encoded', 'color_encoded', 'car_type_encoded'] + \
                   [f'feature_{i}' for i in range(1, 9)]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)

    # CRITICAL: Model expects features in specific order
    # Reorder columns to match model's feature_names_in_
    expected_order = [
        'maker_key', 'model_key', 'mileage', 'engine_power', 'registration_date',
        'fuel', 'paint_color', 'car_type',
        'feature_1', 'feature_2', 'feature_3', 'feature_4',
        'feature_5', 'feature_6', 'feature_7', 'feature_8',
        'sold_at', 'bmw_series', 'luxury_tier', 'is_luxury', 'is_performance',
        'registration_year', 'registration_month', 'registration_quarter',
        'is_summer_registration', 'is_year_end_registration', 'car_age_years',
        'age_mileage_interaction', 'mileage_per_power', 'annual_mileage',
        'power_age_ratio', 'luxury_mileage_interaction', 'luxury_age_interaction',
        'is_high_mileage', 'is_old_car',
        'fuel_encoded', 'color_encoded', 'car_type_encoded'
    ]

    # Reorder DataFrame to match expected feature order
    df = df[expected_order]

    return df


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "BMW Pricing API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "predict": "/predict",
            "batch": "/predict/batch",
            "health": "/health",
            "metrics": "/metrics"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "status": "healthy",
        "model_loaded": True,
        "encoders_count": len(ENCODERS),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(car: CarFeatures, request: Request):
    """
    Predict price for a single BMW vehicle

    Returns predicted price with 95% confidence interval
    """
    start_time = time.time()

    try:
        # Preprocess input
        features_df = preprocess_input(car)

        # Make prediction (in log scale)
        prediction_log = MODEL.predict(features_df)[0]

        # Convert back to Euro
        predicted_price = np.exp(prediction_log)

        # Calculate confidence interval (±1.96 * std error)
        # Using RMSE from training as proxy for std error
        rmse_euro = 2875  # From your training metrics
        confidence_margin = 1.96 * rmse_euro

        processing_time = (time.time() - start_time) * 1000

        logger.info(f"Prediction: €{predicted_price:.0f} | {processing_time:.1f}ms")

        return PredictionResponse(
            predicted_price=float(predicted_price),
            confidence_interval={
                "lower": float(max(0, predicted_price - confidence_margin)),
                "upper": float(predicted_price + confidence_margin)
            },
            model_version="rand_forest_v1",
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict prices for multiple BMW vehicles

    Maximum 100 cars per batch
    """
    start_time = time.time()

    try:
        predictions = []

        for car in request.cars:
            # Reuse single prediction logic
            features_df = preprocess_input(car)
            prediction_log = MODEL.predict(features_df)[0]
            predicted_price = np.exp(prediction_log)

            rmse_euro = 2875
            confidence_margin = 1.96 * rmse_euro

            predictions.append(PredictionResponse(
                predicted_price=float(predicted_price),
                confidence_interval={
                    "lower": float(max(0, predicted_price - confidence_margin)),
                    "upper": float(predicted_price + confidence_margin)
                },
                model_version="rand_forest_v1",
                processing_time_ms=0.0  # Individual timing not tracked in batch
            ))

        processing_time = (time.time() - start_time) * 1000

        logger.info(f"Batch prediction: {len(predictions)} cars | {processing_time:.1f}ms")

        return BatchPredictionResponse(
            predictions=predictions,
            total_cars=len(predictions),
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/metrics")
async def get_metrics():
    """
    Get model performance metrics

    Returns metrics from last training
    """
    try:
        # Load latest metrics report
        metrics_dir = Path("src/evaluation/reports/metrics")
        metrics_files = sorted(metrics_dir.glob("rand_forest_*.json"))

        if not metrics_files:
            return {"error": "No metrics available"}

        latest_metrics = metrics_files[-1]

        import json
        with open(latest_metrics, 'r') as f:
            metrics_data = json.load(f)

        return {
            "model_metrics": metrics_data.get('metrics', {}),
            "targets_met": metrics_data.get('targets_met', {}),
            "last_updated": metrics_data.get('timestamp', 'unknown')
        }

    except Exception as e:
        logger.error(f"Error loading metrics: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)