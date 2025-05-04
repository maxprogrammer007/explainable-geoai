import os
import sys
import joblib
import pandas as pd
from geoshapley import GeoShapleyExplainer

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, PROJECT_ROOT)

# Paths
FEATURES_CSV = os.path.join(PROJECT_ROOT, "data/processed/voting_features.csv")
MODEL_PATH   = os.path.join(PROJECT_ROOT, "data/processed/xgb_automl_model_clean.pkl")
OUTPUT_CSV   = os.path.join(PROJECT_ROOT, "data/processed/geoshapley_explanations.csv")

# Exact features (in order) for X_geo
FEATURE_LIST = [
    "proj_x","proj_y","total_pop","sex_ratio","pct_black","pct_hisp","pct_bach",
    "median_income","pct_65_over","pct_age_18_29","gini","pct_manuf",
    "ln_pop_den","pct_3rd_party","turn_out","pct_fb","pct_uninsured"
]

def main():
    # 1) Load full table
    df = pd.read_csv(FEATURES_CSV, dtype={"GEOID": str})
    geoids = df["GEOID"]

    # 2) Build X_geo DataFrame
    X_geo = df[FEATURE_LIST]

    # 3) Load the clean XGBoost model
    automl = joblib.load(MODEL_PATH)
    wrapped = automl.model
    xgb_model = wrapped.model if hasattr(wrapped, "model") else wrapped

    # 4) Build a small NumPy‐array background sample
    background = X_geo.sample(n=100, random_state=42).values

    # 5) Initialize GeoShapleyExplainer
    explainer = GeoShapleyExplainer(xgb_model.predict, background)

    # 6) Explain using the DataFrame (it’ll extract .values internally)
    result = explainer.explain(X_geo)

    # … unpack and save as before …


    # 7) Unpack components
    phi0       = result.expected_value
    phi_geo    = result.location        # array length n
    phi_primary= result.primary         # n × K
    phi_int    = result.interaction     # n × K

    # 8) Build output DataFrame
    out = pd.DataFrame({"GEOID": geoids})
    out["phi_base"] = phi0
    out["phi_GEO"]  = phi_geo

    for i, feat in enumerate(FEATURE_LIST):
        out[f"phi_{feat}"]     = phi_primary[:, i]
        out[f"phi_int_{feat}"] = phi_int[:, i]

    # 9) Save CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"GeoShapley explanations saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
