# Import necessary libraries and modules
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import numpy as np
import pandas as pd

# Define the HousePriceModel class for training, hyperparameter tuning, and predictions
class HousePriceModel:
    
    # Constructor to initialize the model and other attributes
    def __init__(self):
        self.model = None          # To store the trained model
        self.best_params = None    # To store the best hyperparameters found during tuning
        self.cv_results = None     # To store the results of cross-validation from grid search
        
    # Private method to validate the input data
    def _validate_data(self, X, y=None, is_training=False):
        # Ensure that X is a pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        
        # Define the expected feature columns required for the model
        expected_features = [
            'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
            'Population', 'AveOccup', 'Latitude', 'Longitude'
        ]
        
        # Check if any required features are missing in the input data
        missing_cols = set(expected_features) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing required features: {missing_cols}")
        
        # Check if there are any missing (NaN) values in the data
        if X.isna().any().any():
            raise ValueError("Input data contains missing values")
        
        # If we are training the model, validate the target data (y)
        if is_training:
            # Ensure that y is provided for training
            if y is None:
                raise ValueError("Target values required for training")
            # Ensure y is either a pandas Series or numpy array
            if not isinstance(y, (pd.Series, np.ndarray)):
                raise ValueError("y must be a pandas Series or numpy array")
            # Ensure that the length of X and y match
            if len(X) != len(y):
                raise ValueError("X and y must have same length")
    
    # Method to train the model on the provided data
    def train(self, X_train, y_train):
        # Validate the input data
        self._validate_data(X_train, y_train, is_training=True)
        # Initialize a RandomForestRegressor model
        self.model = RandomForestRegressor(random_state=42)
        # Train the model on the data
        self.model.fit(X_train, y_train)
    
    # Method to perform hyperparameter tuning using GridSearchCV
    def tune_hyperparameters(self, X_train, y_train):
        # Validate the input data
        self._validate_data(X_train, y_train, is_training=True)
        
        # Define parameter grids for RandomForest, XGBoost, and CatBoost
        param_grids = {
            'RandomForest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'XGBoost': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.1, 0.05],
                'subsample': [0.8, 1.0]
            },
            'CatBoost': {
                'iterations': [100, 200],
                'depth': [6, 8],
                'learning_rate': [0.1, 0.05],
                'l2_leaf_reg': [3, 5]
            }
        }
        
        # Initialize models
        models = {
            'RandomForest': RandomForestRegressor(random_state=42),
            'XGBoost': XGBRegressor(random_state=42, use_label_encoder=False, verbosity=0),
            'CatBoost': CatBoostRegressor(random_state=42, verbose=0)
        }
        
        best_model = None
        best_score = float('-inf')
        best_params = None
        best_cv_results = None
        
        # Perform grid search for each model
        for model_name, model in models.items():
            grid_search = GridSearchCV(
                model,
                param_grids[model_name],
                cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            # Update best model if current model is better
            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                best_cv_results = grid_search.cv_results_
        
        # Store the best model, parameters, and results
        self.model = best_model
        self.best_params = best_params
        self.cv_results = best_cv_results
    
    # Method to make predictions using the trained model
    def predict(self, X):
        # Validate the input data
        self._validate_data(X)
        
        # Ensure that the model has been trained before making predictions
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Return the predictions for the input data
        return self.model.predict(X)
