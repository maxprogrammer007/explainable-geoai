import os
import sys
import time
import joblib
import pandas as pd
from geoshapley import GeoShapleyExplainer

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, PROJECT_ROOT)

# Paths
FEATURES_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "voting_features.csv")
MODEL_PATH   = os.path.join(PROJECT_ROOT, "data", "processed", "xgb_automl_model_clean.pkl")
OUTPUT_CSV   = os.path.join(PROJECT_ROOT, "data", "processed", "geoshapley_explanations.csv")

# Features in the exact order the model expects
FEATURE_LIST = [
    "proj_x","proj_y","total_pop","sex_ratio","pct_black","pct_hisp","pct_bach",
    "median_income","pct_65_over","pct_age_18_29","gini","pct_manuf",
    "ln_pop_den","pct_3rd_party","turn_out","pct_fb","pct_uninsured"
]

BG_SIZE = 50
N_JOBS  = 2

def main():
    # 1) Load the feature table
    df = pd.read_csv(FEATURES_CSV, dtype={"GEOID": str})
    geoids = df["GEOID"]
    X_geo  = df[FEATURE_LIST]

    # 2) Load the FLAML‐trained XGBoost model
    automl = joblib.load(MODEL_PATH)
    wrapped = automl.model
    xgb_model = wrapped.model if hasattr(wrapped, "model") else wrapped

    # 3) Build a numpy background sample
    background = X_geo.sample(n=BG_SIZE, random_state=42).values

    # 4) Initialize the GeoShapley explainer
    explainer = GeoShapleyExplainer(xgb_model.predict, background)

    # 5) Compute explanations in parallel
    print(f"Running GeoShapley on {len(X_geo)} points with BG={BG_SIZE}, n_jobs={N_JOBS}")
    start = time.time()
    try:
        result = explainer.explain(X_geo, n_jobs=N_JOBS)
    except Exception as e:
        print("Parallel explain failed:", e)
        print("Retrying with n_jobs=1...")
        result = explainer.explain(X_geo, n_jobs=1)
    print(f"GeoShapley completed in {time.time()-start:.1f}s")

    # 6) Unpack the four components
    phi0       = result.expected_value     # scalar
    phi_geo    = result.location           # shape (n,)
    phi_primary= result.primary            # shape (n, K)
    phi_int    = result.interaction        # shape (n, K)

    # 7) Build output DataFrame
    out = pd.DataFrame({
        "GEOID":   geoids,
        "phi_base": phi0,
        "phi_GEO":  phi_geo
    })
    for i, feat in enumerate(FEATURE_LIST):
        out[f"phi_{feat}"]     = phi_primary[:, i]
        out[f"phi_int_{feat}"] = phi_int[:, i]

    # 8) Save CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print("GeoShapley explanations saved to", OUTPUT_CSV)

if __name__ == "__main__":
    main()
