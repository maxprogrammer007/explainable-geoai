# src/temporal_loader.py

import os
import pandas as pd
import glob

# Base paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
RAW_DIR       = os.path.join(PROJECT_ROOT, "data", "raw")
OUTPUT_PATH   = os.path.join(PROJECT_ROOT, "data", "processed", "voting_panel.csv")

def load_yearly_voting(years, pattern="voting_{year}.csv"):
    """
    Load voting CSVs for each year into a single DataFrame with a 'year' column.
    Expects files in data/raw named like voting_2016.csv, voting_2020.csv, etc.
    """
    dfs = []
    for yr in years:
        filepath = os.path.join(RAW_DIR, pattern.format(year=yr))
        df = pd.read_csv(filepath, dtype={"county_id": str})
        df["year"] = yr
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def load_yearly_acs(years, pattern="acs_{year}.csv"):
    """
    Load ACS demographic CSVs for each year into a DataFrame with 'year' column.
    Expects data/raw/acs_2016.csv, etc.
    """
    dfs = []
    for yr in years:
        filepath = os.path.join(RAW_DIR, pattern.format(year=yr))
        df = pd.read_csv(filepath, dtype={"county_id": str})
        df["year"] = yr
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def merge_panel(voting_df, acs_df):
    """
    Merge voting and ACS panels on ['county_id', 'year'], standardize columns.
    """
    df = voting_df.merge(acs_df, on=["county_id", "year"], how="inner")
    # Clean column names
    df.rename(columns=lambda x: x.strip().lower().replace(" ", "_"), inplace=True)
    return df

def save_panel(df, path=OUTPUT_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Panel data saved to {path}")

if __name__ == "__main__":
    # Define the years you have data for
    years = [2016, 2020, 2024]

    # Load panels
    voting = load_yearly_voting(years, pattern="voting_{year}.csv")
    acs    = load_yearly_acs(years, pattern="acs_{year}.csv")

    # Merge into a panel
    panel = merge_panel(voting, acs)

    # Save out
    save_panel(panel)
    print(f"Panel data saved to {OUTPUT_PATH}")