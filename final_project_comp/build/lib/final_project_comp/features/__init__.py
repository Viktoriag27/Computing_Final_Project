# final_project_comp/features/__init__.py
"""Feature engineering module."""
from .feature_engineering import (
   DensityFeatures,
   LocationFeatures, 
   IncomeFeatures,
   OccupancyFeatures, 
   AgeInteractions
)

__all__ = [
   'DensityFeatures',
   'LocationFeatures',
   'IncomeFeatures',
   'OccupancyFeatures',
   'AgeInteractions'
]