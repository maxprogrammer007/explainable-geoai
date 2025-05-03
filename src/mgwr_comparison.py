# src/mgwr_comparison.py

import os
import sys
import pandas as pd
import geopandas as gpd
from sklearn.linear_model import LinearRegression

# Make project root importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import PROCESSED_DATA_PATH

# Paths
FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "voting_features.csv")
SHAPE_PATH    = os.path.join(
    PROJECT_ROOT, "data", "raw", "shapefiles", "cb_2018_us_county_500k.shp"
)
OUTPUT_PATH   = os.path.join(PROJECT_ROOT, "data", "processed", "mgwr_coefficients.csv")


def load_data():
    # Load features
    df = pd.read_csv(FEATURES_PATH, dtype={"fips": str, "GEOID": str})
    # Load geometries for GEOID merge (centroids not needed for global OLS)
    gdf = gpd.read_file(SHAPE_PATH).to_crs("EPSG:5070")
    gdf["GEOID"] = gdf["GEOID"].str.zfill(5)
    merged = gdf.merge(df, on="GEOID", how="inner")
    return merged


def run_global_ols(merged, target="new_pct_dem"):
    """
    Fit a global OLS regression using all numeric covariates,
    then assign the same global coefficients to every county.
    """
    # Drop geometry
    df = merged.drop(columns="geometry")
    # Select numeric features
    numeric = df.select_dtypes(include=["int64", "float64"])
    y = numeric[target].values
    X = numeric.drop(columns=[target]).values
    feature_names = list(numeric.drop(columns=[target]).columns)

    # Fit global linear model
    lr = LinearRegression()
    lr.fit(X, y)
    coefs = lr.coef_
    intercept = lr.intercept_

    # Build DataFrame with identical coefficients for each GEOID
    rows = []
    for geoid in merged["GEOID"]:
        row = {"GEOID": geoid, "intercept": intercept}
        row.update({f: coef for f, coef in zip(feature_names, coefs)})
        rows.append(row)

    df_coeff = pd.DataFrame(rows)
    return df_coeff


def main():
    merged = load_data()
    df_coeff = run_global_ols(merged)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_coeff.to_csv(OUTPUT_PATH, index=False)
    print(f"Global OLS coefficients saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
