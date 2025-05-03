# src/shap_explainer.py

import os
import sys
import joblib
import pandas as pd
import geopandas as gpd
import shap
import numpy as np

# Make project root importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import PROCESSED_DATA_PATH

# Paths
FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "voting_features.csv")
MODEL_PATH    = os.path.join(PROJECT_ROOT, "data", "processed", "xgb_automl_model.pkl")
SHAPE_PATH    = os.path.join(
    PROJECT_ROOT, "data", "raw", "shapefiles", "cb_2018_us_county_500k.shp"
)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "shap")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_merge():
    df = pd.read_csv(FEATURES_PATH, dtype={"fips": str, "GEOID": str})
    gdf = gpd.read_file(SHAPE_PATH).to_crs("EPSG:5070")
    gdf["GEOID"] = gdf["GEOID"].str.zfill(5)
    return gdf.merge(df, on="GEOID", how="inner")


def prepare_X_y(merged, target="new_pct_dem"):
    merged["centroid_x"] = merged.geometry.centroid.x
    merged["centroid_y"] = merged.geometry.centroid.y
    df = merged.drop(columns="geometry")
    numeric = df.select_dtypes(include=["int64", "float64"])
    y = numeric[target]
    geoids = df["GEOID"]
    X = numeric.drop(columns=[target])
    return X, y, geoids


def main():
    # 1. Load & merge
    merged = load_and_merge()
    X_df, y, geoids = prepare_X_y(merged)

    # 2. Load FLAML AutoML
    automl = joblib.load(MODEL_PATH)

    # 3. Create a plain-function wrapper around automl.predict
    #    that accepts numpy arrays or DataFrames
    feature_cols = X_df.columns.tolist()
    def predict_fn(X):
        # X may be ndarray or DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=feature_cols)
        return automl.predict(X)

    # 4. Sample a small background from X_df
    background = X_df.sample(n=100, random_state=42)

    # 5. Initialize KernelExplainer on our plain predict_fn
    explainer = shap.KernelExplainer(predict_fn, background)

    # 6. Compute SHAP values for all samples
    shap_vals = explainer.shap_values(X_df, nsamples=200)

    # 7. Build output DataFrame
    df_out = pd.DataFrame(shap_vals, columns=[f"phi_{c}" for c in feature_cols])
    # Baseline is explainer.expected_value
    df_out.insert(0, "expected_value", explainer.expected_value)
    df_out.insert(0, "GEOID", geoids.values)

    # 8. Save CSV
    out_path = os.path.join(OUTPUT_DIR, "shap_explanations.csv")
    df_out.to_csv(out_path, index=False)
    print(f"SHAP explanations saved to {out_path}")


if __name__ == "__main__":
    main()
