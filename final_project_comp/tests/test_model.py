# tests/test_model.py
"""Tests for model implementation."""

import pytest
import pandas as pd
import numpy as np
from final_project_comp.models.model import HousePriceModel

@pytest.fixture
def sample_data():
    """
    Provide sample data for model testing.
    
    Returns:
    - X: A DataFrame containing feature data for testing.
    - y: A Series containing target values for testing.
    """
    X = pd.DataFrame({
        'feature1': range(100),  # Simulated feature data
        'feature2': range(100)  # Another simulated feature
    })
    y = pd.Series(range(100))  # Simulated target values
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

def test_hyperparameter_tuning(sample_data):
    """
    Test that the model's hyperparameter tuning functionality works as expected.
    
    Verifies:
    - The best hyperparameters (`best_params`) are set after tuning.
    - Cross-validation results (`cv_results`) are available after tuning.
    """
    X, y = sample_data
    model = HousePriceModel()  # Initialize the model
    model.tune_hyperparameters(X, y)  # Perform hyperparameter tuning
    assert model.best_params is not None  # Ensure best parameters are set
    assert model.cv_results is not None  # Ensure CV results are available
