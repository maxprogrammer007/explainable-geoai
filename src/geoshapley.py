# src/geoshapley.py

import os
import sys
import joblib
import pandas as pd

# Ensure project root on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import PROCESSED_DATA_PATH
from geoshapley import GeoShapley  # install via `pip install geoshapley`

# Paths
FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "voting_features.csv")
MODEL_PATH    = os.path.join(PROJECT_ROOT, "data", "processed", "xgb_automl_model.pkl")
OUTPUT_DIR    = os.path.join(PROJECT_ROOT, "data", "processed", "geoshapley")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(path: str = FEATURES_PATH) -> pd.DataFrame:
    """Load the feature table (with 'GEOID' and covariates, no geometry)."""
    return pd.read_csv(path)

def load_model(path: str = MODEL_PATH):
    """Load the AutoML-wrapped XGBoost model."""
    return joblib.load(path)

def explain_with_geoshapley(
    df: pd.DataFrame,
    model,
    location_col: str = "GEOID",
    target_col:   str = "new_pct_dem"
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      - location_col,
      - phi_GEO (intrinsic location),
      - phi_<feature> for each covariate,
      - phi_int_<feature> for interaction terms,
      - plus the original target.
    """
    # Initialize explainer
    explainer = GeoShapley(
        model           = model.predict,  # model.predict(X) => array of predictions
        data            = df.drop(columns=[location_col, target_col]),
        locations       = df[location_col],
        target          = df[target_col],
        feature_names   = [c for c in df.columns if c not in (location_col, target_col)],
        location_names  = df[location_col].astype(str),
    )

    # Compute explanations
    # This can take a while depending on data size & bootstrapping settings
    geoshap = explainer.explain()  # returns a pandas DataFrame

    # geoshap will include your phi_GEO, phi_j, and phi_{GEO,j} terms
    return geoshap

def save_explanations(df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """Persist geoshapley results to CSV."""
    out_path = os.path.join(output_dir, "geoshapley_explanations.csv")
    df.to_csv(out_path, index=False)
    print(f"GeoShapley explanations saved to {out_path}")

if __name__ == "__main__":
    # 1. Load
    df    = load_data()
    model = load_model()

    # 2. Explain
    geoshap_df = explain_with_geoshapley(df, model)

    # 3. Save
    save_explanations(geoshap_df)
