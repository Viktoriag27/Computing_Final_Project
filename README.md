# House Price Prediction Library

A scalable machine learning library for predicting house prices.

## Installation

```bash
pip install -e .
```

## Used Dataset 
Calfornia housing dataset from scikit-learn. The data pertains to the houses found in a given California district and some summary stats about them based on the 1990 census data. The target variable is the median house value for California districts,
expressed in hundreds of thousands of dollars ($100,000).
URL: https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset


## Project Structure

- `data/`: Data loading and splitting utilities
- `preprocessing/`: Data preprocessing transformers
- `features/`: Feature engineering classes
- `models/`: Model training and tuning
- `evaluation/`: Metrics and evaluation tools
- `api/`: FastAPI implementation
- `tests/`: Unit tests
- `notebooks/`: Example notebooks

## Features
- **Preprocessing**: Handles missing values, scales numerical features, and encodes categorical variables.
- **Feature Engineering**: Implements custom transformers to add meaningful features for better predictive power.
- **Modeling**:
  - Random Forest
  - XGBoost
  - CatBoost
- **Unit Testing**: Ensures functionality and reliability of key components through automated tests.

---
## Scaling Features and Model Selection
Feature preprocessing
- Additional methods for handling missing values and outliers can be incorporated within the architecture
- New features can be added by including additional functions to compute the needed features and evaluated
- Further feature evaluation functions can be added with EDA and Frequency distribution checks once the number of feature quantum scales.

New models can be incorporated by the following steps in model.py file
-- Loading relevant library
-- Adding the model optimisation details within model_param_grids dictionary in the function tune_hyperparameters 

Additional Evaluation metrics can be incorporated in the evaluate function by including the additional evaluation formula


## API Usage

Start the API:
```bash
uvicorn api.main:app --reload
```

Make predictions:
```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d @sample_input.json
```

## Testing

Run tests:
```bash
pytest tests/


```
