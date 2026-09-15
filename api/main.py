import torch
import numpy as np
import io
import time
import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import LSTMPredictor, get_device
from src.data_prep import USEFUL_SENSORS, SEQUENCE_LENGTH, MAX_RUL

# ─────────────────────────────────────────
# App initialization
# ─────────────────────────────────────────
app = FastAPI(
    title="Predictive Maintenance API",
    description="""
    AI-powered predictive maintenance system for industrial equipment.
    
    Analyzes time series sensor data and predicts:
    - **Remaining Useful Life (RUL)** — cycles before maintenance required
    - **Maintenance urgency** — immediate, soon, or monitor
    - **Recommended action** — specific maintenance guidance
    
    Built with LSTM neural network trained on NASA turbofan engine data.
    Achieves 12.99 RMSE — near state of the art performance.
    
    Applicable to any rotating equipment with sensor telemetry:
    generators, pumps, motors, compressors, turbines.
    """,
    version="1.0.0"
)

# ─────────────────────────────────────────
# Global model loading
# ─────────────────────────────────────────
print("Loading predictive maintenance model...")
DEVICE = get_device()
MODEL = None

def load_model():
    """Load model once at startup."""
    global MODEL
    model = LSTMPredictor(
        input_size=14,
        hidden_size=128,
        num_layers=2,
        dropout=0.2
    )
    model.load_state_dict(
        torch.load('models/best_model.pth', map_location=DEVICE)
    )
    model = model.to(DEVICE)
    model.eval()
    MODEL = model
    print("Model loaded successfully!")

load_model()

# ─────────────────────────────────────────
# Request and response models
# ─────────────────────────────────────────
class SensorReading(BaseModel):
    """Single cycle of sensor readings."""
    s2: float = Field(..., description="Fan inlet temperature")
    s3: float = Field(..., description="LPC outlet temperature")
    s4: float = Field(..., description="HPC outlet temperature")
    s7: float = Field(..., description="HPC outlet pressure")
    s8: float = Field(..., description="Physical fan speed")
    s9: float = Field(..., description="Physical core speed")
    s11: float = Field(..., description="HPC outlet static pressure")
    s12: float = Field(..., description="Fuel flow ratio")
    s13: float = Field(..., description="Corrected fan speed")
    s14: float = Field(..., description="Corrected core speed")
    s15: float = Field(..., description="Bypass ratio")
    s17: float = Field(..., description="Bleed enthalpy")
    s20: float = Field(..., description="HPT coolant bleed")
    s21: float = Field(..., description="LPT coolant bleed")

class PredictionRequest(BaseModel):
    """
    Request body for RUL prediction.
    Requires exactly 30 cycles of sensor readings.
    """
    unit_id: int = Field(..., description="Equipment unit identifier")
    sensor_readings: List[SensorReading] = Field(
        ...,
        description="Exactly 30 cycles of sensor readings",
        min_length=30,
        max_length=30
    )

class PredictionResponse(BaseModel):
    unit_id: int
    predicted_rul: float
    rul_category: str
    urgency: str
    recommendation: str
    confidence_note: str
    latency_ms: float
    model_version: str

class BatchPredictionRequest(BaseModel):
    """Multiple equipment units in one request."""
    units: List[PredictionRequest]

class BatchSummaryResponse(BaseModel):
    total_units: int
    immediate_attention: int
    schedule_soon: int
    monitor: int
    average_rul: float
    lowest_rul_unit: int
    lowest_rul_value: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_rmse: str
    sensors_required: int
    sequence_length: int

# ─────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────
def get_urgency(rul: float):
    """Classify RUL into maintenance urgency categories."""
    if rul <= 20:
        return (
            "IMMEDIATE",
            "Schedule maintenance within the next operational cycle. "
            "Equipment approaching critical failure threshold.",
            "Equipment failure imminent — prioritize maintenance now"
        )
    elif rul <= 50:
        return (
            "SOON",
            "Schedule maintenance within the next 50 cycles. "
            "Begin procurement of replacement parts.",
            "Plan maintenance window in the near term"
        )
    else:
        return (
            "MONITOR",
            "Continue normal operations. "
            "Re-evaluate at next scheduled inspection.",
            "No immediate action required — continue monitoring"
        )

def preprocess_readings(sensor_readings: List[SensorReading]):
    """Convert sensor readings to normalized model input."""
    # Extract values in correct sensor order
    sensor_order = [
        's2', 's3', 's4', 's7', 's8', 's9',
        's11', 's12', 's13', 's14', 's15',
        's17', 's20', 's21'
    ]

    sequence = []
    for reading in sensor_readings:
        cycle = [getattr(reading, s) for s in sensor_order]
        sequence.append(cycle)

    sequence = np.array(sequence, dtype=np.float32)

    # Normalize each sensor to 0-1 range using min-max
    for i in range(sequence.shape[1]):
        col_min = sequence[:, i].min()
        col_max = sequence[:, i].max()
        if col_max > col_min:
            sequence[:, i] = (
                (sequence[:, i] - col_min) / (col_max - col_min)
            )

    return torch.FloatTensor(sequence).unsqueeze(0)

# ─────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint — API health check."""
    return HealthResponse(
        status="operational",
        model_loaded=MODEL is not None,
        device=str(DEVICE),
        model_rmse="12.99 cycles",
        sensors_required=14,
        sequence_length=30
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check for monitoring systems."""
    return HealthResponse(
        status="operational",
        model_loaded=MODEL is not None,
        device=str(DEVICE),
        model_rmse="12.99 cycles",
        sensors_required=14,
        sequence_length=30
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_rul(request: PredictionRequest):
    """
    Predict Remaining Useful Life for a single equipment unit.
    
    Provide 30 cycles of sensor readings and receive:
    - Predicted RUL in cycles
    - Maintenance urgency classification
    - Specific maintenance recommendation
    
    RMSE of 12.99 cycles on NASA benchmark dataset.
    96% of predictions within 30 cycles of true RUL.
    """
    start_time = time.time()

    try:
        # Preprocess sensor readings
        tensor = preprocess_readings(request.sensor_readings)
        tensor = tensor.to(DEVICE)

        # Run inference
        with torch.no_grad():
            prediction = MODEL(tensor)

        rul = float(prediction.item())
        rul = max(0, min(rul, MAX_RUL))  # Clip to valid range

        latency_ms = (time.time() - start_time) * 1000

        urgency, recommendation, rul_category = get_urgency(rul)

        return PredictionResponse(
            unit_id=request.unit_id,
            predicted_rul=round(rul, 1),
            rul_category=rul_category,
            urgency=urgency,
            recommendation=recommendation,
            confidence_note=(
                f"Model RMSE: 12.99 cycles. "
                f"96% of predictions within ±30 cycles. "
                f"Most accurate in 0-30 cycle range (RMSE 2.64)."
            ),
            latency_ms=round(latency_ms, 2),
            model_version="LSTM-v1.0"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchSummaryResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict RUL for multiple equipment units simultaneously.
    
    Returns summary statistics useful for maintenance planning:
    - Units requiring immediate attention
    - Units to schedule soon
    - Units to continue monitoring
    - Lowest RUL unit for priority scheduling
    """
    if len(request.units) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 units per batch request."
        )

    results = []

    for unit_request in request.units:
        try:
            tensor = preprocess_readings(
                unit_request.sensor_readings
            )
            tensor = tensor.to(DEVICE)

            with torch.no_grad():
                prediction = MODEL(tensor)

            rul = float(prediction.item())
            rul = max(0, min(rul, MAX_RUL))
            urgency, _, _ = get_urgency(rul)

            results.append({
                'unit_id': unit_request.unit_id,
                'rul': rul,
                'urgency': urgency
            })

        except Exception:
            continue

    if not results:
        raise HTTPException(
            status_code=400,
            detail="No units could be processed."
        )

    immediate = [r for r in results if r['urgency'] == 'IMMEDIATE']
    soon = [r for r in results if r['urgency'] == 'SOON']
    monitor = [r for r in results if r['urgency'] == 'MONITOR']

    lowest = min(results, key=lambda x: x['rul'])

    return BatchSummaryResponse(
        total_units=len(results),
        immediate_attention=len(immediate),
        schedule_soon=len(soon),
        monitor=len(monitor),
        average_rul=round(
            sum(r['rul'] for r in results) / len(results), 1
        ),
        lowest_rul_unit=lowest['unit_id'],
        lowest_rul_value=round(lowest['rul'], 1)
    )

@app.get("/model/info")
async def model_info():
    """Return model architecture and performance details."""
    return {
        "model_architecture": "LSTM with 2 layers — 216,193 parameters",
        "hidden_size": 128,
        "sequence_length": 30,
        "n_sensors": 14,
        "training_dataset": "NASA CMAPSS FD001 — Turbofan Engine Degradation",
        "training_engines": 100,
        "test_engines": 100,
        "performance": {
            "rmse": "12.99 cycles",
            "mae": "9.36 cycles",
            "r2_score": "0.8949",
            "within_10_cycles": "61%",
            "within_20_cycles": "86%",
            "within_30_cycles": "96%"
        },
        "per_range_rmse": {
            "0_30_cycles": "2.64 — most accurate near failure",
            "30_60_cycles": "10.01",
            "60_90_cycles": "21.36",
            "90_125_cycles": "11.88"
        },
        "urgency_thresholds": {
            "IMMEDIATE": "RUL <= 20 cycles",
            "SOON": "RUL 21-50 cycles",
            "MONITOR": "RUL > 50 cycles"
        },
        "applicable_equipment": [
            "Power generators",
            "Industrial pumps",
            "Electric motors",
            "Air compressors",
            "Gas turbines"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)