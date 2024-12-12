# tests/test_preprocessing.py
"""Tests for preprocessing components."""

import pytest
import pandas as pd
import numpy as np
from DSDM_CaliHousePredict.preprocessing.preprocessors import MissingValueImputer, Scaler

# Creating the data frame for testing,fixture ensures the df can be reutilized

@pytest.fixture
def df_with_missing():
    return pd.DataFrame({
        'MedInc': [5.0, np.nan, 6.0], # Column with missing values
        'HouseAge': [30, 20, np.nan] #Another column with missing values
    })

@pytest.fixture
def df_no_missing():
    return pd.DataFrame({
        'MedInc': [5.0, 6.0, 7.0], # Example feature
        'HouseAge': [30, 20, 25] # Another example feature
    })

@pytest.fixture
def df_empty():
    return pd.DataFrame()

# Starting the tests for MissingValueImputer

def test_missing_value_imputer(df_with_missing):
    """
    Test that MissingValueImputer correctly fills missing values.
    
    Verifies:
    - Missing values in the DataFrame are filled after transformation.
    - No missing values remain after applying the imputer.
    """
    imputer = MissingValueImputer()  # Initialize the imputer
    imputer.fit(df_with_missing)  # Fit the imputer to the data
    transformed = imputer.transform(df_with_missing)  # Transform the data
    assert transformed.isna().sum().sum() == 0  # Ensure no missing values remain

def test_fill_values_mean (df_with_missing):
    """
    Test that the imputer calculates the correct mean values.

    """
    imputer = MissingValueImputer() # Initialize the imputer
    imputer.fit(df_with_missing) # Fit the imputer to the data
    expected_fill_values = {
        'MedInc': 5.5,  # Mean of [5.0, 6.0]
        'HouseAge': 25.0  # Mean of [30, 20]
    }
    pd.testing.assert_series_equal( # Checks if the mean valeus are correct
        imputer.fill_values,
        pd.Series(expected_fill_values)
    )

def test_empty_dataframe(df_empty):
    """Test behavior when fitting and transforming an empty DataFrame."""
    imputer = MissingValueImputer() # Initialize the imputer
    imputer.fit(df_empty) # Fit the imputer to the empty data
    transformed = imputer.transform(df_empty)
    pd.testing.assert_frame_equal(df_empty, transformed) # checks if df empty

def test_no_missing_values(df_no_missing):
    """Test behavior when no missing values are present in the DataFrame."""
    imputer = MissingValueImputer() # Initialize the imputer
    imputer.fit(df_no_missing) # Fit the imputer to the data with no missing
    transformed = imputer.transform(df_no_missing)
    pd.testing.assert_frame_equal(df_no_missing, transformed)  # Should remain unchanged

def test_validate_data_invalid_type():
    """Test that _validate_data raises an error for invalid input types."""
    imputer = MissingValueImputer() # Initialize the imputer
    X_invalid = [1, 2, 3]  # Not a DataFrame
    with pytest.raises(ValueError, match="X must be a pandas DataFrame"):
        imputer._validate_data(X_invalid)

def test_validate_data_invalid_strategy(df_with_missing):
    """Test that _validate_data raises an error for invalid strategies."""
    imputer = MissingValueImputer(strategy="invalid_strategy")
    with pytest.raises(ValueError, match="Strategy must be 'mean' or 'median'"):
        imputer._validate_data(df_with_missing)

def test_fit_ignores_non_numeric_columns():
    """Test that fit ignores non-numeric columns in the DataFrame."""
    df = pd.DataFrame({
        'NumericCol': [1.0, 2.0, None],
        'NonNumericCol': ['a', 'b', 'c']
    })
    imputer = MissingValueImputer()
    imputer.fit(df)
    expected_fill_values = {'NumericCol': 1.5}  # Mean of [1.0, 2.0]
    pd.testing.assert_series_equal(
        imputer.fill_values,
        pd.Series(expected_fill_values)
    )

def test_fill_values_median(df_with_missing):
    """Test that the imputer calculates the correct median values."""
    imputer = MissingValueImputer(strategy='median')
    imputer.fit(df_with_missing)
    expected_fill_values = {
        'MedInc': 5.5,  # Median of [5.0, 6.0]
        'HouseAge': 25.0  # Median of [30, 20]
    }
    pd.testing.assert_series_equal(
        imputer.fill_values,
        pd.Series(expected_fill_values)
    )

def test_transform_before_fit(df_with_missing):
    """Test that transform raises an error if called before fit."""
    imputer = MissingValueImputer()
    with pytest.raises(ValueError, match="Must fit imputer before transform"):
        imputer.transform(df_with_missing)

def test_fit_transform_ignores_non_numeric_columns():
    """Test that non-numeric columns are ignored in both fit and transform."""
    df = pd.DataFrame({
        'NumericCol': [1.0, None, 2.0],
        'NonNumericCol': ['a', 'b', 'c']
    })
    imputer = MissingValueImputer()
    imputer.fit(df)
    transformed = imputer.transform(df)
    expected = pd.DataFrame({
        'NumericCol': [1.0, 1.5, 2.0],  # Missing value filled with mean
        'NonNumericCol': ['a', 'b', 'c']  # Unchanged
    })
    pd.testing.assert_frame_equal(transformed, expected)


# Starting the tests for Scaler

def test_scaler_transform_before_fit(df_no_missing):
    """Test that transform raises an error if called before fit."""
    scaler = Scaler()
    with pytest.raises(ValueError, match="Must fit scaler before transform"):
        scaler.transform(df_no_missing)

def test_scaler_empty_dataframe(df_empty):
    """Test that scaler correctly handles an empty DataFrame."""
    scaler = Scaler()
    scaler.fit(df_empty)
    transformed = scaler.transform(df_empty)
    pd.testing.assert_frame_equal(df_empty, transformed)


def test_scaler(df_no_missing):
    """
    Test that Scaler standardizes data correctly.
    
    Verifies:
    - The mean of the transformed DataFrame is approximately zero.
    - The standard deviation of the transformed DataFrame is approximately one.
    """
    scaler = Scaler()  # Initialize the scaler
    scaler.fit(df_no_missing)  # Fit the scaler to the data
    transformed = scaler.transform(df_no_missing)  # Transform the data
    assert np.allclose(transformed.mean(), 0, atol=1e-10)  # Check mean is ~0
    assert np.allclose(transformed.std(), 1, atol=1e-10)  # Check std is ~1

def test_fit_transform_on_empty_dataframe(df_empty):
    """Test fit and transform on an empty DataFrame."""
    scaler = Scaler()
    scaler.fit(df_empty) # Fitting scaler on the empty df
    transformed = scaler.transform(df_empty) # Transforming empty df
    pd.testing.assert_frame_equal(df_empty, transformed) # Verifying result
