# src/shap_explainer.py

import os
import sys
import joblib
import pandas as pd
import shap

# Make project root importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Paths
FEATURES_CSV     = os.path.join(PROJECT_ROOT, "data", "processed", "voting_features.csv")
CLEAN_MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "xgb_automl_model_clean.pkl")
OUTPUT_CSV       = os.path.join(PROJECT_ROOT, "data", "processed", "shap_explanations.csv")

# The exact features used by the clean model
FEATURE_LIST = [
    "proj_x", "proj_y", "total_pop", "sex_ratio",
    "pct_black", "pct_hisp", "pct_bach", "median_income",
    "pct_65_over", "pct_age_18_29", "gini", "pct_manuf",
    "ln_pop_den", "pct_3rd_party", "turn_out", "pct_fb",
    "pct_uninsured"
]

def main():
    # 1) Load the tabular features + GEOID
    df = pd.read_csv(FEATURES_CSV, dtype={"GEOID": str})
    geoids = df["GEOID"].copy()
    
    # 2) Build X for SHAP
    X = df[FEATURE_LIST]
    
    # 3) Load clean model and extract XGBRegressor
    automl = joblib.load(CLEAN_MODEL_PATH)
    wrapped = automl.model
    xgb_model = wrapped.model if hasattr(wrapped, "model") else wrapped
    
    # 4) Initialize TreeExplainer
    explainer = shap.TreeExplainer(xgb_model)
    
    # 5) Compute SHAP values
    shap_vals = explainer.shap_values(X)  # returns (n_samples, n_features)
    expected_value = explainer.expected_value
    
    # 6) Build output DataFrame
    out = pd.DataFrame(
        shap_vals,
        columns=[f"phi_{feat}" for feat in FEATURE_LIST]
    )
    out.insert(0, "expected_value", expected_value)
    out.insert(0, "GEOID", geoids.values)
    
    # 7) Save to the processed folder
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"SHAP explanations saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
