import os
import pandas as pd
from census import Census

API_KEY = "d4746e2c918f281e4d8479b9a30c857c4c67d77a"
c = Census(API_KEY)

years = [2016, 2021]  # ACS 5-year endpoints that Census currently supports

variables = ["B01003_001E", "B02001_002E", "B19013_001E"]  # pop, white pop, median income

for yr in years:
    df = pd.DataFrame(
        c.acs5.get(variables + ["NAME"], {'for': 'county:*'}, year=yr)
    )
    df.rename(columns={
        "B01003_001E": "total_pop",
        "B02001_002E": "pop_white",
        "B19013_001E": "median_income"
    }, inplace=True)
    df["county_id"] = df["state"] + df["county"]
    df.to_csv(f"data/raw/acs_{yr}.csv", index=False)
