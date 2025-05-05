# src/geoshapley_explainer.py

import os, sys, time, math, joblib, multiprocessing, pandas as pd
from geoshapley import GeoShapleyExplainer

# ── Project root setup ───────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, PROJECT_ROOT)

# ── Paths ────────────────────────────────────────────────────────────────────
FEATURES_CSV = os.path.join(PROJECT_ROOT, "data/processed/voting_features.csv")
MODEL_PATH   = os.path.join(PROJECT_ROOT, "data/processed/xgb_automl_model_clean.pkl")
OUTPUT_CSV   = os.path.join(PROJECT_ROOT, "data/processed/geoshapley_explanations.csv")

# ── Define features ──────────────────────────────────────────────────────────
geo_features = ["proj_x", "proj_y"]
feat_list    = [
    "total_pop","sex_ratio","pct_black","pct_hisp","pct_bach",
    "median_income","pct_65_over","pct_age_18_29","gini","pct_manuf",
    "ln_pop_den","pct_3rd_party","turn_out","pct_fb","pct_uninsured"
]
ALL_FEATURES = geo_features + feat_list

# ── Tuning params ────────────────────────────────────────────────────────────
BG_SIZE  = 20
N_JOBS   = max(1, multiprocessing.cpu_count() - 1)
CHUNK_SZ = 500

def main():
    # 1) Load data
    df     = pd.read_csv(FEATURES_CSV, dtype={"GEOID": str})
    geoids = df["GEOID"]
    X_geo  = df[ALL_FEATURES]

    # 2) Load model
    automl   = joblib.load(MODEL_PATH)
    wrapped  = automl.model
    xgb_model = wrapped.model if hasattr(wrapped, "model") else wrapped

    # 3) Background sample
    background = X_geo.sample(n=BG_SIZE, random_state=42).values

    # 4) Init explainer
    explainer = GeoShapleyExplainer(xgb_model.predict, background)

    # 5) Chunked explain
    n      = len(X_geo)
    chunks = math.ceil(n / CHUNK_SZ)
    all_chunks = []

    print(f"GeoShapley: {n} pts in {chunks} chunks (BG={BG_SIZE}, jobs={N_JOBS})")
    t0_all = time.time()

    for i in range(chunks):
        lo, hi = i*CHUNK_SZ, min((i+1)*CHUNK_SZ, n)
        Xc = X_geo.iloc[lo:hi]
        ids= geoids.iloc[lo:hi].reset_index(drop=True)

        print(f" Chunk {i+1}/{chunks} [{lo}:{hi}]")
        t0 = time.time()
        try:
            res = explainer.explain(Xc, n_jobs=N_JOBS)
        except Exception as e:
            print("  parallel failed:", e, "; retry single-thread")
            res = explainer.explain(Xc, n_jobs=1)
        print(f"  done in {time.time()-t0:.1f}s")

        # 6) Unpack correct attrs
        phi_base    = res.base_value         # φ₀
        phi_geo     = res.geo                # intrinsic location
        phi_primary = res.primary            # shape (m, 15)
        phi_int     = res.geo_intera         # shape (m, 15)

        # 7) Build chunk DataFrame
        chunk_df = pd.DataFrame({
            "GEOID":    ids,
            "phi_base": phi_base,
            "phi_GEO":  phi_geo
        })
        # only loop feat_list!
        for j, feat in enumerate(feat_list):
            chunk_df[f"phi_{feat}"]     = phi_primary[:, j]
            chunk_df[f"phi_int_{feat}"] = phi_int[:, j]

        all_chunks.append(chunk_df)

    print(f"All chunks done in {time.time()-t0_all:.1f}s")

    # 8) Concat & save
    final_df = pd.concat(all_chunks, ignore_index=True)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    final_df.to_csv(OUTPUT_CSV, index=False)
    print("Saved geoshapley_explanations.csv")

if __name__ == "__main__":
    main()
