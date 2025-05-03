# scripts/inspect_columns.py
import pandas as pd

df = pd.read_csv("data/processed/voting_clean.csv")
print(df.columns.tolist())
