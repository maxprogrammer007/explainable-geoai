import os
import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
from streamlit_folium import st_folium
import folium

# --- Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
SHAPE_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'shapefiles', 'cb_2018_us_county_500k.shp')
BOOTSTRAP_PATH = os.path.join(DATA_DIR, 'bootstrap_shap_stats.csv')

# --- Load Data ---
@st.cache_data
def load_data():
    # GeoDataFrame with geometries
    gdf = gpd.read_file(SHAPE_PATH).to_crs("EPSG:4326")
    gdf['GEOID'] = gdf['GEOID'].astype(str).str.zfill(5)
    # SHAP explanations
    shap_df = pd.read_csv(os.path.join(DATA_DIR, 'shap_explanations.csv'), dtype={'GEOID': str})
    # MGWR/OLS coefficients
    mgwr_df = pd.read_csv(os.path.join(DATA_DIR, 'mgwr_coefficients.csv'), dtype={'GEOID': str})
    # Bootstrap stats
    boot_df = pd.read_csv(BOOTSTRAP_PATH)
    # Merge shap & mgwr onto geodata
    map_df = gdf.merge(shap_df, on='GEOID', how='left')
    map_df = map_df.merge(mgwr_df, on='GEOID', how='left', suffixes=('_shap', '_mgwr'))
    return map_df, shap_df, mgwr_df, boot_df

map_df, shap_df, mgwr_df, boot_df = load_data()

# --- Sidebar Controls ---
st.set_page_config(layout="wide", page_title="Explainable GeoAI Dashboard")
st.sidebar.title("Controls")
mode = st.sidebar.radio("Select Mode:", ["SHAP", "MGWR/OLS"])
view = st.sidebar.radio("View:", ["Point Estimate", "Uncertainty"])

if mode == "SHAP":
    features = [c.replace('phi_', '') for c in shap_df.columns if c.startswith('phi_')]
    feature = st.sidebar.selectbox("SHAP Feature:", features)
    col_point = f"phi_{feature}"
    # Uncertainty column: use std_phi from bootstrap
    boot_row = boot_df[boot_df['feature'] == feature]
    if not boot_row.empty:
        std_val = boot_row['std_phi'].values[0]
    col_uncert = None  # placeholder
    title_point = f"SHAP Attribution: {feature}"
    title_uncert = f"SHAP Uncertainty (Std Dev): {feature}"
else:
    features = [c for c in mgwr_df.columns if c != 'GEOID']
    feature = st.sidebar.selectbox("MGWR Coefficient:", features)
    col_point = feature
    # Uncertainty not available for MGWR; fallback to NaN
    title_point = f"MGWR/OLS Coefficient: {feature}"
    title_uncert = ""  # no uncertainty layer

# --- Determine choropleth column and title ---
if view == "Point Estimate":
    col = col_point
    title = title_point
else:
    if mode == "SHAP":
        # Add uncertainty values into the map_df
        map_df['uncertainty'] = map_df[col_point].map(
            lambda x: boot_df.loc[boot_df['feature']==feature, 'std_phi'].values[0]
        )
        col = 'uncertainty'
        title = title_uncert
    else:
        # No uncertainty data for MGWR
        st.sidebar.warning("Uncertainty not available for MGWR mode.")
        col = col_point
        title = title_point

# --- Main Title ---
st.title("🗺️ Explainable GeoAI Interactive Dashboard")

# --- Choropleth Map ---
m = folium.Map(location=[37.8, -96], zoom_start=4, tiles='cartodbpositron')
folium.Choropleth(
    geo_data=map_df,
    name='choropleth',
    data=map_df,
    columns=['GEOID', col],
    key_on='feature.properties.GEOID',
    fill_color='YlGnBu',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name=title
).add_to(m)
for _, r in map_df.iterrows():
    val = r[col] if pd.notnull(r[col]) else 0
    folium.GeoJson(
        r['geometry'],
        style_function=lambda x, v=val: {'fillColor': folium.utilities.color_brewer_scale(v, 'YlGnBu')[0], 'color':'#444', 'weight':0.5},
        tooltip=folium.Tooltip(f"GEOID: {r['GEOID']}<br>{title}: {val:.3f}" if pd.notnull(r[col]) else "Data unavailable")
    ).add_to(m)

st.subheader(title)
st_folium(m, width=800, height=500)

# --- Feature Importance Bar Chart (SHAP only) ---
if mode == "SHAP" and view == "Point Estimate":
    st.subheader("Global Feature Importance")
    imp_df = boot_df.copy()
    imp_df['abs_mean'] = imp_df['mean_phi'].abs()
    imp_df = imp_df.sort_values('abs_mean', ascending=False).head(10)
    fig = px.bar(
        imp_df, x='feature', y='mean_phi', error_y='std_phi',
        labels={'mean_phi':'Mean SHAP'}, title='Top 10 SHAP Feature Importances'
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Download Buttons ---
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
col1.download_button("Download SHAP CSV", data=shap_df.to_csv(index=False), file_name='shap_explanations.csv')
col2.download_button("Download MGWR CSV", data=mgwr_df.to_csv(index=False), file_name='mgwr_coefficients.csv')
col3.download_button("Download Bootstrap Stats", data=boot_df.to_csv(index=False), file_name='bootstrap_shap_stats.csv')
col4.markdown("&nbsp;")