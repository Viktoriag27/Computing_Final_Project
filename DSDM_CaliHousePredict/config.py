"""Configuration settings for the project."""
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "model"

# Model parameters
MODEL_PARAMS = {
    'random_state': 42,
    'test_size': 0.2,
    'cv_folds': 5
}

# API settings
API_HOST = os.getenv('API_HOST', 'localhost')
API_PORT = int(os.getenv('API_PORT', 8000))