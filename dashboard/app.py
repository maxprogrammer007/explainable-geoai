import os
import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import streamlit.components.v1 as components
import folium

# --- Configuration ---
PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR      = os.path.join(PROJECT_ROOT, 'data', 'processed')
SHAPE_PATH    = os.path.join(PROJECT_ROOT, 'data', 'raw', 'shapefiles', 'cb_2018_us_county_500k.shp')

SHAP_CSV      = os.path.join(DATA_DIR, 'shap_explanations.csv')
MGWR_CSV      = os.path.join(DATA_DIR, 'mgwr_coefficients.csv')
BOOT_CSV      = os.path.join(DATA_DIR, 'bootstrap_shap_stats.csv')
FAIR_CSV      = os.path.join(DATA_DIR, 'fairness_metrics.csv')

SENSITIVE_ATTRS = ["pct_black", "pct_hisp", "median_income"]

# Must be first Streamlit call
st.set_page_config(layout="wide", page_title="🗺️ Explainable GeoAI Dashboard")

# --- Load Data ---
@st.cache_data
def load_data():
    gdf = gpd.read_file(SHAPE_PATH).to_crs("EPSG:4326")
    gdf['GEOID'] = gdf['GEOID'].astype(str).str.zfill(5)

    shap_df = pd.read_csv(SHAP_CSV, dtype={'GEOID': str})
    mgwr_df = pd.read_csv(MGWR_CSV, dtype={'GEOID': str})
    boot_df = pd.read_csv(BOOT_CSV)
    fair_df = pd.read_csv(FAIR_CSV, dtype={'GEOID': str})

    # Merge shap & mgwr into GeoDataFrame
    map_df = gdf.merge(shap_df, on='GEOID', how='left')
    map_df = map_df.merge(mgwr_df, on='GEOID', how='left', suffixes=('_shap','_mgwr'))

    return map_df, shap_df, mgwr_df, boot_df, fair_df

map_df, shap_df, mgwr_df, boot_df, fair_df = load_data()

# --- Sidebar Controls ---
st.sidebar.title("Controls")

mode = st.sidebar.radio(
    "Select Mode:",
    ["SHAP", "MGWR/OLS", "Fairness"]
)

view = st.sidebar.radio(
    "View:",
    ["Point Estimate", "Uncertainty"]
)

# Determine feature & columns
if mode == "SHAP":
    features = [c.replace('phi_', '') for c in shap_df.columns if c.startswith('phi_')]
    feature = st.sidebar.selectbox("Feature:", features)
    col_point = f"phi_{feature}"
    col_uncert = 'std_phi'
    title_point  = f"SHAP Attribution: {feature}"
    title_uncert = f"SHAP Uncertainty (Std Dev): {feature}"

elif mode == "MGWR/OLS":
    features = [c for c in mgwr_df.columns if c != 'GEOID']
    feature = st.sidebar.selectbox("Coefficient:", features)
    col_point  = feature
    col_uncert = None
    title_point  = f"MGWR/OLS Coefficient: {feature}"
    title_uncert = ""

else:  # Fairness
    fair_labels = {
        "pct_black":"Black %",
        "pct_hisp":"Hispanic %",
        "median_income":"Median Income"
    }
    attr = st.sidebar.selectbox(
        "Sensitive Attribute:",
        SENSITIVE_ATTRS,
        format_func=lambda x: fair_labels[x]
    )
    feature = attr
    col_point  = f"{attr}_fairness_score"
    col_uncert = None
    title_point  = f"Fairness Score – {fair_labels[attr]}"
    title_uncert = ""

# Default to point estimate
col_to_map = col_point
title = title_point

# Handle SHAP uncertainty view
if mode == "SHAP" and view == "Uncertainty":
    # Get std_phi for selected feature
    std_row = boot_df.loc[boot_df["feature"] == feature]
    if not std_row.empty:
        std_val = float(std_row["std_phi"])
        # inject into map_df copy
        map_df["uncertainty"] = std_val
        col_to_map = "uncertainty"
        title = title_uncert
    else:
        st.sidebar.warning("No bootstrap std available for this feature.")
        col_to_map = col_point
        title = title_point

# --- Build Map ---
st.subheader(title)

# Prepare DataFrame to map
plot_df = map_df.copy()
if mode == "Fairness":
    plot_df = plot_df.merge(fair_df, on="GEOID", how="left")

# Verify column exists
if col_to_map not in plot_df.columns:
    st.error(f"🛑 Column '{col_to_map}' not found. Available columns: {plot_df.columns.tolist()}")
else:
    m = folium.Map(location=[37.8, -96], zoom_start=4, tiles='cartodbpositron')

    choropleth = folium.Choropleth(
        geo_data=plot_df,
        data=plot_df,
        columns=["GEOID", col_to_map],
        key_on="feature.properties.GEOID",
        fill_color='YlGnBu' if mode!="Fairness" else 'RdYlBu_r',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=title,
        nan_fill_color="white"
    ).add_to(m)

    # Add tooltip
    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(
            fields=["GEOID", col_to_map],
            aliases=["GEOID", title],
            localize=True
        )
    )

    map_html = m._repr_html_()
    components.html(map_html, height=500, scrolling=True)


# --- SHAP Feature Importance ---
if mode == "SHAP" and view == "Point Estimate":
    st.subheader("Global SHAP Feature Importance")
    imp_df = boot_df.copy()
    imp_df['abs_mean'] = imp_df['mean_phi'].abs()
    top10 = imp_df.sort_values('abs_mean', ascending=False).head(10)
    fig = px.bar(
        top10,
        x='feature',
        y='mean_phi',
        error_y='std_phi',
        labels={'mean_phi':'Mean SHAP'},
        title='Top 10 SHAP Feature Importances'
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Download Buttons ---
st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
c1.download_button("Download SHAP CSV", shap_df.to_csv(index=False), 'shap_explanations.csv')
c2.download_button("Download MGWR CSV", mgwr_df.to_csv(index=False), 'mgwr_coefficients.csv')
c3.download_button("Download Bootstrap Stats", boot_df.to_csv(index=False), 'bootstrap_shap_stats.csv')
c4.download_button("Download Fairness CSV", fair_df.to_csv(index=False), 'fairness_metrics.csv')
