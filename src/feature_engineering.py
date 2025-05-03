import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import geopandas as gpd
import fiona
from libpysal.weights import KNN
import libpysal

from config import PROCESSED_DATA_PATH

SHAPEFILE_PATH = os.path.join(
    PROJECT_ROOT, "data", "raw", "shapefiles", "cb_2018_us_county_500k.shp"
)
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "voting_features.csv")


def load_clean_data(path: str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"county_id": str})


def load_county_shapefile(path: str = SHAPEFILE_PATH) -> gpd.GeoDataFrame:
    # Auto-rebuild .shx if missing
    with fiona.Env(SHAPE_RESTORE_SHX="YES"):
        gdf = gpd.read_file(path)

    # Ensure CRS is projected for accurate distance/centroid
    # Albers Equal Area for CONUS: EPSG 5070
    if gdf.crs != "EPSG:5070":
        gdf = gdf.to_crs("EPSG:5070")

    gdf["GEOID"] = gdf["GEOID"].astype(str).str.zfill(5)
    return gdf


def merge_voting_with_geometries(
    voting_df: pd.DataFrame, counties_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    voting_df = voting_df.rename(columns={"county_id": "fips"})
    voting_df["fips"] = voting_df["fips"].str.zfill(5)
    merged = counties_gdf.merge(
        voting_df, left_on="GEOID", right_on="fips", how="inner"
    )
    return merged


def add_spatial_lag(
    gdf: gpd.GeoDataFrame, var: str, k: int = 5
) -> gpd.GeoDataFrame:
    # Use projected centroids for neighbor distances
    centroids = gdf.geometry.centroid
    coords = [(pt.x, pt.y) for pt in centroids]

    # Build k-NN weights
    w = KNN.from_array(coords, k=k)
    # Properly row-standardize
    w.transform='R'

    # Compute spatial lag
    lag_vals = libpysal.weights.lag_spatial(w, gdf[var].values)
    gdf[f"{var}_lag{k}"] = lag_vals
    return gdf


def save_features(gdf: gpd.GeoDataFrame, output_path: str = OUTPUT_PATH) -> None:
    df = gdf.drop(columns="geometry")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Features saved to {output_path}")


if __name__ == "__main__":
    voting_df = load_clean_data()
    counties_gdf = load_county_shapefile()
    geo_df = merge_voting_with_geometries(voting_df, counties_gdf)
    geo_df = add_spatial_lag(geo_df, var="new_pct_dem", k=5)
    save_features(geo_df)
