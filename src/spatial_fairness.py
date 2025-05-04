# src/spatial_fairness.py

import os
import sys
import joblib
import pandas as pd
import geopandas as gpd
import numpy as np

# Make project root importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Paths
FEATURES_CSV        = os.path.join(PROJECT_ROOT, "data", "processed", "voting_features.csv")
CLEAN_MODEL_PATH    = os.path.join(PROJECT_ROOT, "data", "processed", "xgb_automl_model_clean.pkl")
SHAPE_PATH          = os.path.join(
    PROJECT_ROOT, "data", "raw", "shapefiles", "cb_2018_us_county_500k.shp"
)
OUTPUT_PATH         = os.path.join(PROJECT_ROOT, "data", "processed", "fairness_metrics.csv")

# Sensitive attributes
SENSITIVE_ATTRS = ["pct_black", "pct_hisp", "median_income"]

# Exactly the features the clean model was trained on:
FEATURE_LIST = [
    "proj_x", "proj_y", "total_pop", "sex_ratio",
    "pct_black", "pct_hisp", "pct_bach", "median_income",
    "pct_65_over", "pct_age_18_29", "gini", "pct_manuf",
    "ln_pop_den", "pct_3rd_party", "turn_out", "pct_fb",
    "pct_uninsured"
]


def predict_residuals():
    # Load only the cleaned feature table
    df = pd.read_csv(FEATURES_CSV, dtype={"GEOID": str})
    # True target
    y_true = df["new_pct_dem"]
    # Subset to exactly the model features
    X = df[FEATURE_LIST]

    # Load clean FLAML model
    automl = joblib.load(CLEAN_MODEL_PATH)
    wrapped = automl.model
    xgb_model = wrapped.model if hasattr(wrapped, "model") else wrapped

    # Predict and compute residual
    preds = xgb_model.predict(X)
    res = pd.DataFrame({
        "GEOID": df["GEOID"],
        "residual": preds - y_true
    })
    # Carry along sensitive attrs for stratification
    for attr in SENSITIVE_ATTRS:
        res[attr] = df[attr]
    return res


def compute_fairness(res_df):
    df = res_df.copy()
    global_gaps = {}
    for attr in SENSITIVE_ATTRS:
        # quantile‐based strata
        bins = np.nanquantile(df[attr], np.linspace(0, 1, 5))
        strata = pd.cut(df[attr], bins=bins, labels=False, include_lowest=True)
        df[f"{attr}_stratum"] = strata

        # mean residual by stratum and fairness score
        means = df.groupby(f"{attr}_stratum")["residual"].mean()
        df[f"{attr}_mean_residual"] = strata.map(means)
        df[f"{attr}_fairness_score"] = (df["residual"] - df[f"{attr}_mean_residual"]).abs()

        # global gap
        global_gaps[attr] = means.max() - means.min()

    print("=== Global Fairness Gaps ===")
    for attr, gap in global_gaps.items():
        print(f"{attr}: {gap:.4f}")

    return df


def main():
    # 1. Predict residuals on the clean tabular data
    res_df = predict_residuals()
    # 2. Compute fairness metrics
    fairness_df = compute_fairness(res_df)
    # 3. Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fairness_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Fairness metrics saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
