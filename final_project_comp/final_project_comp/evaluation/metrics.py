# final_project_comp/evaluation/metrics.py
"""
Evaluation metrics for regression models.
"""

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics.
    
    Args:
        y_true (array-like): True target values
        y_pred (array-like): Predicted values
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }
    
    return metrics