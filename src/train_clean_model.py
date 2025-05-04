# src/train_clean_model.py

import os, sys, joblib
import pandas as pd
from flaml import AutoML

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, PROJECT_ROOT)

# Paths
FEATURES_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "voting_features.csv")
CLEAN_MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "xgb_automl_model_clean.pkl")

# 1) Load the cleaned features CSV
df = pd.read_csv(FEATURES_CSV, dtype={"GEOID": str})

# 2) Define your exact feature list (engineered demographics + centroids)
FEATURE_LIST = [
    "proj_x", "proj_y", "total_pop", "sex_ratio",
    "pct_black", "pct_hisp", "pct_bach", "median_income",
    "pct_65_over", "pct_age_18_29", "gini", "pct_manuf",
    "ln_pop_den", "pct_3rd_party", "turn_out", "pct_fb",
    "pct_uninsured"
]

X = df[FEATURE_LIST]
y = df["new_pct_dem"]

# 3) Train FLAML AutoML for XGBoost only
automl = AutoML()
automl_settings = {
    "task": "regression",
    "metric": "r2",
    "estimator_list": ["xgboost"],
    "time_budget": 300,  # adjust as needed
    "seed": 42,
}
automl.fit(X_train=X, y_train=y, **automl_settings)

# 4) Save the new clean model
joblib.dump(automl, CLEAN_MODEL_PATH)
print(f"Clean model saved to {CLEAN_MODEL_PATH}")
