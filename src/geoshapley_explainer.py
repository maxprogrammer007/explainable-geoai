# src/geoshapley_explainer.py

import os, sys, joblib, pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import PROCESSED_DATA_PATH
from geoshapley import GeoShapleyExplainer

# Paths
FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "voting_features.csv")
MODEL_PATH    = os.path.join(PROJECT_ROOT, "data", "processed", "xgb_automl_model.pkl")
OUTPUT_DIR    = os.path.join(PROJECT_ROOT, "data", "processed", "geoshapley")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(path=FEATURES_PATH):
    return pd.read_csv(path)

def load_model(path=MODEL_PATH):
    return joblib.load(path)

def explain_with_geoshapley(df, model, location_col="GEOID", target_col="new_pct_dem"):
    explainer = GeoShapleyExplainer(
        model=model.predict,
        data=df.drop(columns=[location_col, target_col]),
        locations=df[location_col],
        target=df[target_col],
        feature_names=[c for c in df.columns if c not in (location_col, target_col)],
        location_names=df[location_col].astype(str),
    )
    return explainer.explain()

def save_explanations(df, output_dir=OUTPUT_DIR):
    out_path = os.path.join(output_dir, "geoshapley_explanations.csv")
    df.to_csv(out_path, index=False)
    print(f"GeoShapley explanations saved to {out_path}")

if __name__ == "__main__":
    df    = load_data()
    model = load_model()
    geoshap_df = explain_with_geoshapley(df, model)
    save_explanations(geoshap_df)
