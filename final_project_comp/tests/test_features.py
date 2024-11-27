# tests/test_features.py
"""Tests for feature engineering components."""

import pytest
import pandas as pd
import numpy as np
from final_project_comp.features.feature_engineering import (
    DensityFeatures, LocationFeatures, IncomeFeatures,
    OccupancyFeatures, AgeInteractions
)

@pytest.fixture
def sample_data():
    """Provide sample California housing data for testing."""
    return pd.DataFrame({
        'MedInc': [5.0, 6.0],
        'HouseAge': [30, 20],
        'AveRooms': [6, 7],
        'AveBedrms': [2, 3],
        'Population': [1000, 2000],
        'AveOccup': [3, 4],
        'Latitude': [37.88, 37.86],
        'Longitude': [-122.23, -122.22]
    })

def test_density_features(sample_data):
    """Ensure DensityFeatures generates correct feature columns."""
    transformer = DensityFeatures()
    result = transformer.fit_transform(sample_data)
    assert 'ROOM_DENSITY' in result.columns
    assert 'BEDROOM_RATIO' in result.columns
    assert len(result) == len(sample_data)

def test_location_features(sample_data):
    """Ensure LocationFeatures computes distance to San Francisco."""
    transformer = LocationFeatures()
    result = transformer.fit_transform(sample_data)
    assert 'DIST_TO_SF' in result.columns
    assert len(result) == len(sample_data)

def test_income_features(sample_data):
    """Ensure IncomeFeatures calculates income per person."""
    transformer = IncomeFeatures()
    result = transformer.fit_transform(sample_data)
    assert 'INCOME_PER_PERSON' in result.columns
    assert len(result) == len(sample_data)

def test_occupancy_features(sample_data):
    """Ensure OccupancyFeatures computes population density."""
    transformer = OccupancyFeatures()
    result = transformer.fit_transform(sample_data)
    assert 'POP_DENSITY' in result.columns
    assert len(result) == len(sample_data)

def test_age_interactions(sample_data):
    """Ensure AgeInteractions generates interaction features."""
    transformer = AgeInteractions()
    result = transformer.fit_transform(sample_data)
    assert 'AGE_INCOME' in result.columns
    assert 'AGE_ROOMS' in result.columns
    assert len(result) == len(sample_data)
