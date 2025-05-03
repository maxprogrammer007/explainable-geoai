# src/bootstrap_uncertainty.py

import os
import sys
import joblib
import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from flaml import AutoML
from xgboost import XGBRegressor

# Project root on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import PROCESSED_DATA_PATH

# Paths
FEATURES_PATH       = os.path.join(PROJECT_ROOT, "data", "processed", "voting_features.csv")
OUTPUT_STATS_PATH   = os.path.join(PROJECT_ROOT, "data", "processed", "bootstrap_shap_stats.csv")

# Bootstrap parameters
B                   = 100    # Number of bootstrap samples
TEST_SIZE           = 0.2    # Test split fraction

def load_data():
    df = pd.read_csv(FEATURES_PATH)
    # **Only** keep numeric columns (int or float)
    numeric = df.select_dtypes(include=["int64", "float64"])
    # Ensure target column is in numeric set
    if "new_pct_dem" not in numeric.columns:
        numeric["new_pct_dem"] = df["new_pct_dem"]
    # X: all numeric except target
    X = numeric.drop(columns=["new_pct_dem"])
    y = numeric["new_pct_dem"]
    return X, y

def extract_sklearn_xgb(automl):
    wrapped = automl.model
    if hasattr(wrapped, "model") and isinstance(wrapped.model, XGBRegressor):
        return wrapped.model
    if hasattr(wrapped, "estimator") and isinstance(wrapped.estimator, XGBRegressor):
        return wrapped.estimator
    return wrapped

def bootstrap_shap_stats(X, y, B=100):
    feature_names = X.columns.tolist()
    all_shap = []

    for b in range(B):
        # Sample with replacement
        idx = np.random.choice(len(X), size=len(X), replace=True)
        Xb, yb = X.iloc[idx], y.iloc[idx]

        # Train/test split for consistency
        X_train, _, y_train, _ = train_test_split(Xb, yb, test_size=TEST_SIZE, random_state=42)

        # AutoML XGB on bootstrap sample
        automl = AutoML()
        automl.fit(
            X_train=X_train,
            y_train=y_train,
            task="regression",
            metric="r2",
            time_budget=60,
            estimator_list=["xgboost"],
            seed=42,
        )
        xgb_model = extract_sklearn_xgb(automl)

        # SHAP TreeExplainer on numeric bootstrap sample
        explainer = shap.TreeExplainer(xgb_model)
        shap_vals = explainer.shap_values(Xb)  # shape (n_samples, n_features)
        all_shap.append(shap_vals)

        print(f"Bootstrap {b+1}/{B} done")

    # Stack to shape (B, n_samples, n_features)
    arr = np.stack(all_shap, axis=0)
    stats = []
    for i, feat in enumerate(feature_names):
        vals = arr[:, :, i].ravel()
        stats.append({
            "feature": feat,
            "mean_phi": np.mean(vals),
            "std_phi": np.std(vals),
            "ci_lower": np.percentile(vals, 2.5),
            "ci_upper": np.percentile(vals, 97.5),
        })

    return pd.DataFrame(stats)

def main():
    X, y = load_data()
    stats_df = bootstrap_shap_stats(X, y, B=B)
    os.makedirs(os.path.dirname(OUTPUT_STATS_PATH), exist_ok=True)
    stats_df.to_csv(OUTPUT_STATS_PATH, index=False)
    print(f"Bootstrap SHAP stats saved to {OUTPUT_STATS_PATH}")

if __name__ == "__main__":
    main()
