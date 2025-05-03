import pandas as pd
import os
import sys

# Ensure project root is on sys.path
top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if top_dir not in sys.path:
    sys.path.insert(0, top_dir)

from config import PROCESSED_DATA_PATH


def load_clean_data(path: str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load the cleaned voting data."""
    return pd.read_csv(path)


def create_geodataframe(df: pd.DataFrame, lon: str = 'longitude', lat: str = 'latitude'):
    """Convert DataFrame to GeoDataFrame using longitude & latitude."""
    import geopandas as gpd
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df[lon], df[lat]), crs='EPSG:4326'
    )
    return gdf


def add_spatial_lag(
    gdf, var: str, k: int = 5
) -> 'GeoDataFrame':
    """
    Compute k-nearest neighbors spatial lag for variable `var`.
    Adds a new column `{var}_lag{k}` to the GeoDataFrame.
    """
    import libpysal
    from libpysal.weights import KNN

    coords = [(pt.x, pt.y) for pt in gdf.geometry]
    w = KNN.from_array(coords, k=k)
    w.transform = 'row_standardized'

    lag_vals = libpysal.weights.lag_spatial(w, gdf[var].values)
    gdf[f"{var}_lag{k}"] = lag_vals
    return gdf


def save_features(
    gdf, output_path: str
) -> None:
    """Drop geometry and save features to CSV."""
    df = gdf.drop(columns='geometry')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == '__main__':
    df = load_clean_data()
    gdf = create_geodataframe(df, lon='longitude', lat='latitude')
    gdf = add_spatial_lag(gdf, var='dem_vote_share', k=5)
    features_path = os.path.join(
        os.path.dirname(PROCESSED_DATA_PATH), 'voting_features.csv'
    )
    save_features(gdf, features_path)
    print(f"Features saved to {features_path}.")