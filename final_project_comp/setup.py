from setuptools import setup, find_packages

setup(
    name="final_project_comp",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.2",
        "pandas>=1.2.0",
        "scikit-learn>=0.24.0",
        "fastapi>=0.65.0",
        "uvicorn>=0.13.0",
        "pytest>=6.2.0",
        "joblib>=1.0.0",
        "pydantic>=1.8.0",
        "xgboost>=1.5.0",
        "catboost>=1.0.0"
    ],
    author="Viktoria Gagua, Angad Sahota, Alejandro Delgado",
    author_email="viktoria.gagua@bse.eu",
    description="A scalable machine learning library for house price prediction",
    keywords="machine learning, housing prices, prediction",
    python_requires=">=3.7",
)