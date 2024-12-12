# DSDM_CaliHousePredict/features/feature_engineering.py
"""Feature engineering for California Housing dataset."""

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class DensityFeatures(BaseEstimator, TransformerMixin):
    """
    Custom transformer to compute density-related features for a dataset.
    This includes:
    - ROOM_DENSITY: The average number of rooms per occupant (AveRooms / AveOccup).
    - BEDROOM_RATIO: The ratio of bedrooms to total rooms (AveBedrms / AveRooms).
    Validates input data to ensure required columns ('AveRooms', 'AveOccup') are present.
    """
    def _validate_data(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        required = ['AveRooms', 'AveOccup']
        if not all(col in X.columns for col in required):
            raise ValueError(f"Missing required columns: {required}")

    def fit(self, X, y=None):
        self._validate_data(X)
        return self
        
    def transform(self, X):
        self._validate_data(X)
        features = pd.DataFrame()
        features['ROOM_DENSITY'] = X['AveRooms'] / X['AveOccup']
        features['BEDROOM_RATIO'] = X['AveBedrms'] / X['AveRooms']
        return features

class LocationFeatures(BaseEstimator, TransformerMixin):
    """
    Custom transformer to compute location-related features for a dataset.
    This includes:
    - DIST_TO_SF: The distance of each record's coordinates (Latitude, Longitude) 
                  from San Francisco's coordinates (37.7749, -122.4194) using Euclidean distance.
    Validates input data to ensure required columns ('Latitude', 'Longitude') are present.
    """
    def _validate_data(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        required = ['Latitude', 'Longitude']
        if not all(col in X.columns for col in required):
            raise ValueError(f"Missing required columns: {required}")

    def fit(self, X, y=None):
        self._validate_data(X)
        return self
        
    def transform(self, X):
        self._validate_data(X)
        features = pd.DataFrame()
        sf_lat, sf_long = 37.7749, -122.4194
        features['DIST_TO_SF'] = np.sqrt(
            (X['Latitude'] - sf_lat)**2 + 
            (X['Longitude'] - sf_long)**2
        )
        return features

class IncomeFeatures(BaseEstimator, TransformerMixin):
    """
    Custom transformer to compute income-related features for a dataset.
    This includes:
    - INCOME_PER_PERSON: The median income (MedInc) divided by the average occupancy (AveOccup).
    Validates input data to ensure required columns ('MedInc', 'AveOccup') are present.
    """
    def _validate_data(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        required = ['MedInc', 'AveOccup']
        if not all(col in X.columns for col in required):
            raise ValueError(f"Missing required columns: {required}")

    def fit(self, X, y=None):
        self._validate_data(X)
        return self
        
    def transform(self, X):
        self._validate_data(X)
        features = pd.DataFrame()
        features['INCOME_PER_PERSON'] = X['MedInc'] / X['AveOccup']
        return features

class OccupancyFeatures(BaseEstimator, TransformerMixin):
    """
    Custom transformer to compute occupancy-related features for a dataset.
    This includes:
    - POP_DENSITY: Population density calculated as the total population (Population) 
                   divided by the product of average rooms (AveRooms) and average occupancy (AveOccup).
    Validates input data to ensure required columns ('Population', 'AveRooms', 'AveOccup') are present.
    """
    def _validate_data(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        required = ['Population', 'AveRooms', 'AveOccup']
        if not all(col in X.columns for col in required):
            raise ValueError(f"Missing required columns: {required}")

    def fit(self, X, y=None):
        self._validate_data(X)
        return self
        
    def transform(self, X):
        self._validate_data(X)
        features = pd.DataFrame()
        features['POP_DENSITY'] = X['Population'] / (X['AveRooms'] * X['AveOccup'])
        return features

class AgeInteractions(BaseEstimator, TransformerMixin):
    """
    Custom transformer to compute interaction features related to the age of houses.
    This includes:
    - AGE_INCOME: Interaction term between house age (HouseAge) and median income (MedInc).
    - AGE_ROOMS: Interaction term between house age (HouseAge) and average rooms (AveRooms).
    Validates input data to ensure required columns ('HouseAge', 'MedInc', 'AveRooms') are present.
    """
    def _validate_data(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        required = ['HouseAge', 'MedInc', 'AveRooms']
        if not all(col in X.columns for col in required):
            raise ValueError(f"Missing required columns: {required}")

    def fit(self, X, y=None):
        self._validate_data(X)
        return self
        
    def transform(self, X):
        self._validate_data(X)
        features = pd.DataFrame()
        features['AGE_INCOME'] = X['HouseAge'] * X['MedInc']
        features['AGE_ROOMS'] = X['HouseAge'] * X['AveRooms']
        return features
