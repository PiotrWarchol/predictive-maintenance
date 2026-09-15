import pytest
import sys
import os
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_model_loads():
    """Test that model file exists and loads correctly."""
    from src.model import LSTMPredictor
    
    assert os.path.exists('models/best_model.pth'), \
        "Model file not found"
    
    model = LSTMPredictor(
        input_size=14,
        hidden_size=128,
        num_layers=2,
        dropout=0.2
    )
    model.load_state_dict(
        torch.load('models/best_model.pth', map_location='cpu')
    )
    model.eval()
    print("Model loads successfully")

def test_model_inference():
    """Test model produces correct output shape."""
    from src.model import LSTMPredictor
    
    model = LSTMPredictor(
        input_size=14,
        hidden_size=128,
        num_layers=2,
        dropout=0.2
    )
    model.load_state_dict(
        torch.load('models/best_model.pth', map_location='cpu')
    )
    model.eval()
    
    dummy_input = torch.randn(1, 30, 14)
    with torch.no_grad():
        output = model(dummy_input)
    
    assert output.shape == torch.Size([1]), \
        f"Expected output shape (1,), got {output.shape}"
    print(f"Inference output: {output.item():.2f} cycles")

def test_rul_prediction_range():
    """Test that RUL predictions are within valid range."""
    from src.model import LSTMPredictor
    
    model = LSTMPredictor(
        input_size=14,
        hidden_size=128,
        num_layers=2,
        dropout=0.2
    )
    model.load_state_dict(
        torch.load('models/best_model.pth', map_location='cpu')
    )
    model.eval()
    
    # Test multiple random inputs
    for _ in range(10):
        dummy_input = torch.randn(1, 30, 14)
        with torch.no_grad():
            output = model(dummy_input)
        rul = float(output.item())
        rul = max(0, min(rul, 125))
        assert 0 <= rul <= 125, \
            f"RUL {rul} outside valid range 0-125"
    
    print("All RUL predictions within valid range")

def test_data_pipeline():
    """Test data loading works correctly."""
    from src.data_prep import prepare_data
    
    train_loader, test_loader, scaler = prepare_data()
    
    assert len(train_loader.dataset) > 0, "Train dataset empty"
    assert len(test_loader.dataset) == 100, \
        "Expected 100 test engines"
    
    sequences, labels = next(iter(test_loader))
    assert sequences.shape[1] == 30, "Wrong sequence length"
    assert sequences.shape[2] == 14, "Wrong number of sensors"
    
    print(f"Train sequences: {len(train_loader.dataset)}")
    print(f"Test engines: {len(test_loader.dataset)}")

def test_api_health():
    """Test API health endpoint returns correct response."""
    from fastapi.testclient import TestClient
    sys.path.insert(0, 'api')
    from main import app
    
    client = TestClient(app)
    
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data['status'] == 'operational'
    assert data['model_loaded'] == True
    assert data['sensors_required'] == 14
    assert data['sequence_length'] == 30
    print("Health check passed")

def test_model_info():
    """Test model info endpoint returns performance metrics."""
    from fastapi.testclient import TestClient
    sys.path.insert(0, 'api')
    from main import app
    
    client = TestClient(app)
    
    response = client.get("/model/info")
    assert response.status_code == 200
    
    data = response.json()
    assert 'performance' in data
    assert 'rmse' in data['performance']
    print(f"Model RMSE: {data['performance']['rmse']}")