import pandas as pd
import os
import sys

# Ensure project root is on sys.path so we can import config.py directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import RAW_DATA_PATH, PROCESSED_DATA_PATH

def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw voting CSV into a DataFrame."""
    return pd.read_csv(path)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: rename columns, drop duplicates/nulls, ensure numeric types."""
    df_clean = df.copy()

    # Standardize column names
    df_clean.rename(columns=lambda x: x.strip().lower().replace(' ', '_'), inplace=True)

    # Drop exact duplicates
    df_clean.drop_duplicates(inplace=True)

    # Drop rows with any null values
    df_clean.dropna(inplace=True)

    # Convert target column to numeric if present
    target = 'dem_vote_share'
    if target in df_clean.columns:
        df_clean[target] = pd.to_numeric(df_clean[target], errors='coerce')
        df_clean.dropna(subset=[target], inplace=True)

    return df_clean

def save_processed_data(df: pd.DataFrame, path: str = PROCESSED_DATA_PATH):
    """Save the cleaned DataFrame to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

if __name__ == '__main__':
    # Run full pipeline
    df_raw = load_raw_data()
    df_clean = preprocess_data(df_raw)
    save_processed_data(df_clean)
    print(f"Processed data saved to {PROCESSED_DATA_PATH}.")
