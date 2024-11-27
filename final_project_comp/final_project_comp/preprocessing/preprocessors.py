# final_project_comp/preprocessing/preprocessors.py
"""Module containing preprocessing transformers."""

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class MissingValueImputer(BaseEstimator, TransformerMixin):
    """
    A custom transformer to handle missing values in a DataFrame.

    Attributes:
    - strategy: The imputation strategy ('mean' or 'median'). Default is 'mean'.
    - fill_values: Stores the calculated values (mean or median) to be used for imputation.

    Methods:
    - fit(X): Calculates the fill values (mean or median) based on the strategy for each column in the DataFrame.
    - transform(X): Imputes missing values in the DataFrame using the precomputed fill values.
    """

    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.fill_values = None

    def _validate_data(self, X):
        """
        Validates that the input is a pandas DataFrame and the strategy is valid.
        Raises a ValueError if validation fails.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        if self.strategy not in ['mean', 'median']:
            raise ValueError("Strategy must be 'mean' or 'median'") 

    def fit(self, X, y=None):
        """
        Computes the fill values (mean or median) for each column in the DataFrame.
        """
        self._validate_data(X)
        if self.strategy == 'mean':
            self.fill_values = X.mean()
        elif self.strategy == 'median':
            self.fill_values = X.median()
        return self

    def transform(self, X):
        """
        Fills missing values in the DataFrame using the precomputed fill values.
        """
        self._validate_data(X)
        if self.fill_values is None:
            raise ValueError("Must fit imputer before transform")
        return X.fillna(self.fill_values)

class Scaler(BaseEstimator, TransformerMixin):
    """
    A custom transformer to standardize features in a DataFrame by removing the mean and scaling to unit variance.

    Attributes:
    - mean: Stores the mean of each column in the DataFrame.
    - std: Stores the standard deviation of each column in the DataFrame.

    Methods:
    - fit(X): Calculates and stores the mean and standard deviation for each column in the DataFrame.
    - transform(X): Scales the DataFrame by subtracting the mean and dividing by the standard deviation.
    """

    def __init__(self):
        self.mean = None
        self.std = None
   
    def _validate_data(self, X):
        """
        Validates that the input is a pandas DataFrame.
        Raises a ValueError if validation fails.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

    def fit(self, X, y=None):
        """
        Computes the mean and standard deviation for each column in the DataFrame.
        """
        self._validate_data(X)
        self.mean = X.mean()
        self.std = X.std()
        return self

    def transform(self, X):
        """
        Scales the DataFrame by subtracting the mean and dividing by the standard deviation.
        """
        self._validate_data(X)
        if self.mean is None or self.std is None:
            raise ValueError("Must fit scaler before transform")
        return (X - self.mean) / self.std
