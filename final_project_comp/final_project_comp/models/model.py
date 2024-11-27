# Import necessary libraries and modules
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
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
        
        # Define a grid of hyperparameters to search over
        param_grid = {
            'n_estimators': [100],      # Number of trees in the forest
            'max_depth': [10],           # Maximum depth of the trees
            'min_samples_split': [2],    # Minimum number of samples required to split a node
            'min_samples_leaf': [1]      # Minimum number of samples required to be at a leaf node
        }
        
        # Set up GridSearchCV to perform cross-validation with the hyperparameter grid
        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=42),  # Model to tune
            param_grid,                              # Hyperparameter grid
            cv=3,                                    # Number of cross-validation folds
            scoring='neg_mean_squared_error',        # Scoring metric for evaluation (negative MSE)
            n_jobs=-1                                 # Use all CPU cores for parallel computation
        )
        
        # Perform grid search on the training data
        grid_search.fit(X_train, y_train)
        # Store the best model found during the grid search
        self.model = grid_search.best_estimator_
        # Store the best hyperparameters
        self.best_params = grid_search.best_params_
        # Store the cross-validation results
        self.cv_results = grid_search.cv_results_
    
    # Method to make predictions using the trained model
    def predict(self, X):
        # Validate the input data
        self._validate_data(X)
        
        # Ensure that the model has been trained before making predictions
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Return the predictions for the input data
        return self.model.predict(X)
