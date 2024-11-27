# final_project_comp/data/loader.py
"""
Module for loading and splitting data for machine learning pipeline.
Contains DataLoader class that handles data import and train/test splitting.
Columns: ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
"""

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd

class DataLoader:
    """Class to load and split California Housing dataset.
    
    Attributes:
        data (pd.DataFrame): Feature data
        target (pd.Series): Target variable (house prices)
    """
    
    def __init__(self):
        self.data = None
        self.target = None
        
    def load_data(self):
        """Load California housing dataset into pandas DataFrame/Series.
        
        Returns:
            tuple: (pd.DataFrame, pd.Series) containing features and target
        """
        california = fetch_california_housing()
        self.data = pd.DataFrame(california.data, columns=california.feature_names)
        self.target = pd.Series(california.target, name='PRICE')
        return self.data, self.target
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into training and test sets.
        
        Args:
            test_size (float): Proportion of data for test set (default: 0.2)
            random_state (int): Random seed for reproducibility (default: 42)
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test) splits
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.data, self.target, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test