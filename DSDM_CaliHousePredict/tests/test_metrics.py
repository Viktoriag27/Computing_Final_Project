# tests/test_metrics.py
"""Unit tests for evaluation metrics."""

import pytest
import numpy as np
from DSDM_CaliHousePredict.evaluation.metrics import calculate_metrics

def test_calculate_metrics():
    """Ensure calculate_metrics computes expected regression metrics."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
    
    metrics = calculate_metrics(y_true, y_pred)
    
    # Check that all expected metric keys are in the results
    assert 'mse' in metrics
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert 'r2' in metrics
    assert 'mape' in metrics

    # Validate metric ranges and properties
    assert metrics['mse'] >= 0  # MSE should always be non-negative
    assert metrics['r2'] <= 1  # R2 should not exceed 1
