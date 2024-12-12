# tests/test_model.py
"""Tests for model implementation."""

import pytest
import pandas as pd
import numpy as np
from DSDM_CaliHousePredict.models.model import HousePriceModel

@pytest.fixture
def sample_data():
    """
    Provide sample data for model testing.
    
    Returns:
    - X: A DataFrame containing feature data for testing.
    - y: A Series containing target values for testing.
    """
    X = pd.DataFrame({
        'MedInc': [1.0] * 100,
        'HouseAge': [10] * 100,
        'AveRooms': [5.0] * 100,
        'AveBedrms': [1.0] * 100,
        'Population': [300] * 100,
        'AveOccup': [3.0] * 100,
        'Latitude': [37.5] * 100,
        'Longitude': [-120.0] * 100
    })
    y = pd.Series(range(100))
    return X, y

def test_model_training(sample_data):
    """
    Test that the model can successfully train on the provided data.
    
    Verifies:
    - The model's internal representation (`model.model`) is initialized after training.
    """
    X, y = sample_data
    model = HousePriceModel()  # Initialize the model
    model.train(X, y)  # Train the model
    assert model.model is not None  # Ensure the model attribute is set

def test_model_prediction(sample_data):
    """
    Test that the model can make predictions after training.
    
    Verifies:
    - The length of predictions matches the number of input rows.
    - Predictions are returned as a NumPy array.
    """
    X, y = sample_data
    model = HousePriceModel()  # Initialize the model
    model.train(X, y)  # Train the model
    predictions = model.predict(X)  # Make predictions
    assert len(predictions) == len(X)  # Ensure predictions cover all rows
    assert isinstance(predictions, np.ndarray)  # Check output type