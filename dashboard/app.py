# dashboard/app.py

import os
import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import streamlit.components.v1 as components
import folium

# ─── Paths & Config ─────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "processed")
SHAPE_PATH   = os.path.join(PROJECT_ROOT, "data", "raw", "shapefiles",
                            "cb_2018_us_county_500k.shp")

SHAP_CSV     = os.path.join(DATA_DIR, "shap_explanations.csv")
GEOSHAP_CSV  = os.path.join(DATA_DIR, "geoshapley_explanations.csv")
MGWR_CSV     = os.path.join(DATA_DIR, "mgwr_coefficients.csv")
BOOT_CSV     = os.path.join(DATA_DIR, "bootstrap_shap_stats.csv")
FAIR_CSV     = os.path.join(DATA_DIR, "fairness_metrics.csv")

SENSITIVE_ATTRS = ["pct_black", "pct_hisp", "median_income"]

st.set_page_config(layout="wide", page_title="🗺️ Explainable GeoAI Dashboard")

# ─── Sidebar Help ───────────────────────────────────────────────────────────
st.sidebar.title("Controls")
st.sidebar.markdown("""
**Mode Descriptions**  
- **SHAP:** Exact `phi_…` columns from your SHAP output + bootstrap uncertainty  
- **GeoShapley:** Decomposed spatial–feature effects  
- **MGWR/OLS:** Local regression coefficients  
- **Fairness:** Residual-based fairness gaps  
""")
with st.expander("❓ How to use"):
    st.write("""
      1. Pick a **Mode**.  
      2. Pick **View** (Point vs Uncertainty).  
      3. For SHAP, choose exactly one `phi_…` column from your CSV.  
      4. Hover on the map or download any CSV below.
    """)

# ─── Data Loader ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def load_data():
    gdf = gpd.read_file(SHAPE_PATH).to_crs("EPSG:4326")
    gdf["GEOID"] = gdf["GEOID"].astype(str).str.zfill(5)

    shap_df    = pd.read_csv(SHAP_CSV,    dtype={"GEOID": str})
    geoshap_df = pd.read_csv(GEOSHAP_CSV, dtype={"GEOID": str})
    mgwr_df    = pd.read_csv(MGWR_CSV,    dtype={"GEOID": str})
    boot_df    = pd.read_csv(BOOT_CSV)
    fair_df    = pd.read_csv(FAIR_CSV,    dtype={"GEOID": str})

    merged = (
        gdf
        .merge(shap_df,    on="GEOID", how="left")
        .merge(geoshap_df, on="GEOID", how="left", suffixes=("_shap","_geoshap"))
        .merge(mgwr_df,    on="GEOID", how="left", suffixes=("", "_mgwr"))
    )
    return merged, shap_df, geoshap_df, mgwr_df, boot_df, fair_df

map_df, shap_df, geoshap_df, mgwr_df, boot_df, fair_df = load_data()

# ─── Mode & View ─────────────────────────────────────────────────────────────
mode = st.sidebar.radio("Select Mode:", ["SHAP", "GeoShapley", "MGWR/OLS", "Fairness"])
view = st.sidebar.radio("View:", ["Point Estimate", "Uncertainty"])

# ─── Sidebar selectors & titles ──────────────────────────────────────────────
if mode == "SHAP":
    # list all existing phi_ columns
    phi_cols = [c for c in shap_df.columns if c.startswith("phi_")]
    feature  = st.sidebar.selectbox("SHAP Column:", sorted(phi_cols))
    col_point   = feature
    col_uncert  = "std_phi"
    title_point = feature
    title_unc   = f"Bootstrap std of {feature}"

elif mode == "GeoShapley":
    geosh_cols = [c for c in geoshap_df.columns if c.startswith("phi_")]
    comp       = st.sidebar.selectbox("GeoShapley Column:", sorted(geosh_cols))
    col_point   = comp
    col_uncert  = None
    title_point = comp
    title_unc   = ""

elif mode == "MGWR/OLS":
    mgwr_cols = [c for c in mgwr_df.columns if c != "GEOID"]
    coef      = st.sidebar.selectbox("MGWR/OLS Column:", sorted(mgwr_cols))
    col_point   = coef
    col_uncert  = None
    title_point = coef
    title_unc   = ""

else:  # Fairness
    fair_labels = {"pct_black":"Black %","pct_hisp":"Hispanic %","median_income":"Median Income"}
    attr      = st.sidebar.selectbox("Attribute:", SENSITIVE_ATTRS,
                                     format_func=lambda x: fair_labels[x])
    col_point   = f"{attr}_fairness_score"
    col_uncert  = None
    title_point = col_point
    title_unc   = ""

# ─── Handle SHAP Uncertainty ────────────────────────────────────────────────
col_to_map, title = col_point, title_point
if mode=="SHAP" and view=="Uncertainty":
    row = boot_df.loc[boot_df["feature"]==feature.removeprefix("phi_")]
    if not row.empty:
        map_df["uncertainty"] = float(row["std_phi"])
        col_to_map, title = "uncertainty", title_unc
    else:
        st.sidebar.warning("No bootstrap std available.")

# ─── Render Map ──────────────────────────────────────────────────────────────
st.subheader(title)
plot_df = map_df.copy()
if mode=="Fairness":
    plot_df = plot_df.merge(fair_df, on="GEOID", how="left")

if col_to_map not in plot_df.columns:
    st.error(f"Column '{col_to_map}' not found. Available: {plot_df.columns.tolist()}")
else:
    m = folium.Map(location=[37.8,-96], zoom_start=4, tiles="cartodbpositron")
    chor = folium.Choropleth(
        geo_data=plot_df,
        data=plot_df,
        columns=["GEOID", col_to_map],
        key_on="feature.properties.GEOID",
        fill_color=("YlGnBu" if mode!="Fairness" else "RdYlBu_r"),
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=title,
        nan_fill_color="white"
    ).add_to(m)

    chor.geojson.add_child(
        folium.features.GeoJsonTooltip(
            fields=["GEOID", col_to_map],
            aliases=["GEOID", title],
            localize=True
        )
    )

    components.html(m._repr_html_(), height=550)

# ─── SHAP Global Importance ─────────────────────────────────────────────────
if mode=="SHAP" and view=="Point Estimate":
    st.subheader("Global SHAP Importance")
    imp = boot_df.copy()
    imp["abs_mean"] = imp["mean_phi"].abs()
    top10 = imp.nlargest(10, "abs_mean")
    fig = px.bar(
        top10, x="feature", y="mean_phi", error_y="std_phi",
        labels={"mean_phi":"Mean SHAP"},
        title="Top 10 SHAP Features"
    )
    st.plotly_chart(fig, use_container_width=True)

# ─── Downloads ───────────────────────────────────────────────────────────────
st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
c1.download_button("Download SHAP",       shap_df.to_csv(index=False),    "shap_explanations.csv")
c2.download_button("Download GeoShapley", geoshap_df.to_csv(index=False), "geoshapley_explanations.csv")
c3.download_button("Download MGWR",       mgwr_df.to_csv(index=False),    "mgwr_coefficients.csv")
c4.download_button("Download Fairness",   fair_df.to_csv(index=False),    "fairness_metrics.csv")
