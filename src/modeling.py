# src/modeling.py

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib

# Ensure we can import config
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import PROCESSED_DATA_PATH

# FLAML AutoML
from flaml import AutoML

# Path to features CSV
FEATURES_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "voting_features.csv"
)
# Where to dump the trained model
MODEL_OUTPUT_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "xgb_automl_model.pkl"
)

def load_features(path: str = FEATURES_PATH) -> pd.DataFrame:
    """Load the feature set (no geometry)."""
    return pd.read_csv(path)

def train_model(df: pd.DataFrame, target: str = "new_pct_dem"):
    """Train an XGBoost model via FLAML AutoML."""
    # Separate X and y
    X = df.drop(columns=[target, "fips", "GEOID"])
    y = df[target]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize AutoML with only XGBoost
    automl = AutoML()
    automl_settings = {
        "time_budget": 300,              # in seconds
        "metric": "r2",
        "task": "regression",
        "estimator_list": ["xgboost"],   # only XGBoost
        "seed": 42,
    }

    # Fit
    automl.fit(X_train=X_train, y_train=y_train, **automl_settings)

    # Predict & evaluate
    y_pred = automl.predict(X_test)
    score = r2_score(y_test, y_pred)
    print(f"Best learner: {automl.best_estimator}")
    print(f"Validation R² on test set: {score:.4f}")

    # Return the fitted automl object and test R2
    return automl, score, (X_test, y_test, y_pred)

def save_model(automl, path: str = MODEL_OUTPUT_PATH):
    """Persist the trained AutoML model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(automl, path)
    print(f"Model saved to {path}")

if __name__ == "__main__":
    # 1. Load data
    df = load_features()

    # 2. Train
    automl, score, (X_test, y_test, y_pred) = train_model(df)

    # 3. Save
    save_model(automl)
