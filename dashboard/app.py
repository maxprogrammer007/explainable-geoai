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
SHAPE_PATH    = os.path.join(PROJECT_ROOT, 'data', 'raw', 'shapefiles',
                             'cb_2018_us_county_500k.shp')

SHAP_CSV      = os.path.join(DATA_DIR, 'shap_explanations.csv')
GEOSHAP_CSV   = os.path.join(DATA_DIR, 'geoshapley_explanations.csv')
MGWR_CSV      = os.path.join(DATA_DIR, 'mgwr_coefficients.csv')
BOOT_CSV      = os.path.join(DATA_DIR, 'bootstrap_shap_stats.csv')
FAIR_CSV      = os.path.join(DATA_DIR, 'fairness_metrics.csv')

SENSITIVE_ATTRS = ["pct_black", "pct_hisp", "median_income"]

# Must be the first Streamlit call
st.set_page_config(layout="wide", page_title="🗺️ Explainable GeoAI Dashboard")

# --- Load Data ---
@st.cache_data
def load_data():
    # Base GeoDataFrame
    gdf = gpd.read_file(SHAPE_PATH).to_crs("EPSG:4326")
    gdf['GEOID'] = gdf['GEOID'].astype(str).str.zfill(5)

    # Tabular outputs
    shap_df    = pd.read_csv(SHAP_CSV,    dtype={'GEOID': str})
    geoshap_df = pd.read_csv(GEOSHAP_CSV, dtype={'GEOID': str})
    mgwr_df    = pd.read_csv(MGWR_CSV,    dtype={'GEOID': str})
    boot_df    = pd.read_csv(BOOT_CSV)
    fair_df    = pd.read_csv(FAIR_CSV,    dtype={'GEOID': str})

    # Merge all into the GeoDataFrame
    df = (
        gdf
        .merge(shap_df,    on='GEOID', how='left')
        .merge(geoshap_df, on='GEOID', how='left', suffixes=('_shap','_geoshap'))
        .merge(mgwr_df,    on='GEOID', how='left', suffixes=('', '_mgwr'))
    )
    return df, shap_df, geoshap_df, mgwr_df, boot_df, fair_df

map_df, shap_df, geoshap_df, mgwr_df, boot_df, fair_df = load_data()


# --- Sidebar Controls ---
st.sidebar.title("Controls")

mode = st.sidebar.radio(
    "Select Mode:",
    ["SHAP", "GeoShapley", "MGWR/OLS", "Fairness"]
)

view = st.sidebar.radio(
    "View:",
    ["Point Estimate", "Uncertainty"]
)

# --- Determine which column to map ---
if mode == "SHAP":
    features = [c.replace('phi_', '') for c in shap_df.columns if c.startswith('phi_')]
    feature  = st.sidebar.selectbox("SHAP Feature:", features)
    col_point  = f"phi_{feature}"
    col_uncert = 'std_phi'
    title_point  = f"SHAP Attribution: {feature}"
    title_uncert = f"SHAP Uncertainty (Std Dev): {feature}"

elif mode == "GeoShapley":
    # GeoShapley components
    base_opts = ["phi_GEO"]                            # intrinsic location
    feat_opts = [c for c in geoshap_df.columns if c.startswith('phi_') and not c.startswith(('phi_int','phi_GEO'))]
    int_opts  = [c for c in geoshap_df.columns if c.startswith('phi_int_')]
    options   = ["phi_GEO"] + feat_opts + int_opts
    comp      = st.sidebar.selectbox("GeoShapley Component:", options)
    col_point   = comp
    col_uncert  = None
    title_point = comp.replace('phi_','').replace('_',' ').title()
    title_uncert= ""

elif mode == "MGWR/OLS":
    features = [c for c in mgwr_df.columns if c != 'GEOID']
    feature  = st.sidebar.selectbox("Coefficient:", features)
    col_point   = feature
    col_uncert  = None
    title_point = f"MGWR/OLS Coefficient: {feature}"
    title_uncert= ""

else:  # Fairness
    labels = {"pct_black":"Black %","pct_hisp":"Hispanic %","median_income":"Median Income"}
    attr    = st.sidebar.selectbox("Sensitive Attribute:", SENSITIVE_ATTRS,
                                   format_func=lambda x: labels[x])
    col_point   = f"{attr}_fairness_score"
    col_uncert  = None
    title_point = f"Fairness Score – {labels[attr]}"
    title_uncert= ""

# Default to point estimate
col_to_map, title = col_point, title_point

# Handle SHAP uncertainty
if mode=="SHAP" and view=="Uncertainty":
    row = boot_df[boot_df['feature']==feature]
    if not row.empty:
        stdv = float(row['std_phi'])
        map_df['uncertainty'] = stdv
        col_to_map, title = 'uncertainty', title_uncert
    else:
        st.sidebar.warning("No bootstrap std available for this feature.")

# --- Render Map ---
st.subheader(title)

plot_df = map_df.copy()
if mode=="Fairness":
    plot_df = plot_df.merge(fair_df, on="GEOID", how="left")

if col_to_map not in plot_df.columns:
    st.error(f"Column '{col_to_map}' not found. Available: {plot_df.columns.tolist()}")
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

    # Correctly attach tooltips to the GeoJson sub‐layer
        choropleth.geojson.add_child(
            folium.features.GeoJsonTooltip(
            fields=["GEOID", col_to_map],
            aliases=["GEOID", title],
            localize=True
            )
         )

        components.html(m._repr_html_(), height=550)


# --- SHAP Feature Importance Bar (SHAP only) ---
if mode=="SHAP" and view=="Point Estimate":
    st.subheader("Global SHAP Feature Importance")
    imp_df = boot_df.copy()
    imp_df['abs_mean'] = imp_df['mean_phi'].abs()
    top10 = imp_df.nlargest(10, 'abs_mean')
    fig = px.bar(
        top10,
        x='feature', y='mean_phi', error_y='std_phi',
        labels={'mean_phi':'Mean SHAP'},
        title='Top 10 SHAP Feature Importances'
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Downloads ---
st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
c1.download_button("Download SHAP CSV",    shap_df.to_csv(index=False),    'shap_explanations.csv')
c2.download_button("Download GeoShapley",  geoshap_df.to_csv(index=False), 'geoshapley_explanations.csv')
c3.download_button("Download MGWR CSV",    mgwr_df.to_csv(index=False),    'mgwr_coefficients.csv')
c4.download_button("Download Fairness CSV", fair_df.to_csv(index=False),   'fairness_metrics.csv')
