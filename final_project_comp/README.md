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

## Adding New Components

### New Preprocessors
1. Create new class in `preprocessing/preprocessors.py`
2. Inherit from `BaseEstimator` and `TransformerMixin`
3. Implement `fit` and `transform` methods
4. Add unit tests

### New Features
1. Add new feature class in `features/feature_engineering.py`
2. Follow sklearn transformer interface
3. Document feature logic
4. Add tests in `tests/test_features.py`

### New Models
1. Create model class in `models/`
2. Implement training and prediction methods
3. Add hyperparameter tuning support
4. Include tests

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