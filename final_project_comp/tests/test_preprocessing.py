# tests/test_preprocessing.py
"""Tests for preprocessing components."""

import pytest
import pandas as pd
import numpy as np
from final_project_comp.preprocessing.preprocessors import MissingValueImputer, Scaler

def test_missing_value_imputer():
    """
    Test that MissingValueImputer correctly fills missing values.
    
    Verifies:
    - Missing values in the DataFrame are filled after transformation.
    - No missing values remain after applying the imputer.
    """
    df = pd.DataFrame({
        'MedInc': [5.0, np.nan, 6.0],  # Column with missing value
        'HouseAge': [30, 20, np.nan]  # Another column with missing value
    })
    imputer = MissingValueImputer()  # Initialize the imputer
    imputer.fit(df)  # Fit the imputer to the data
    transformed = imputer.transform(df)  # Transform the data
    assert transformed.isna().sum().sum() == 0  # Ensure no missing values remain

def test_scaler():
    """
    Test that Scaler standardizes data correctly.
    
    Verifies:
    - The mean of the transformed DataFrame is approximately zero.
    - The standard deviation of the transformed DataFrame is approximately one.
    """
    df = pd.DataFrame({
        'MedInc': [5.0, 6.0, 7.0],  # Example feature
        'HouseAge': [30, 20, 25]  # Another example feature
    })
    scaler = Scaler()  # Initialize the scaler
    scaler.fit(df)  # Fit the scaler to the data
    transformed = scaler.transform(df)  # Transform the data
    assert np.allclose(transformed.mean(), 0, atol=1e-10)  # Check mean is ~0
    assert np.allclose(transformed.std(), 1, atol=1e-10)  # Check std is ~1
