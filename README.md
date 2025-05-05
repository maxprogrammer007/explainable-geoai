
# Explainable GeoAI: Interpreting Socio-Spatial Patterns

An end-to-end spatial XAI pipeline combining XGBoost, SHAP, GeoShapley, and MGWR to uncover and visualize interpretable spatial effects in U.S. county-level voting data.

---

## 📂 Repository Structure

```

explainable-geoai/
├── data/
│   ├── raw/
│   │   ├── census/                   # raw ACS downloads
│   │   ├── shapefiles/              # county geometries
│   │   └── voting\_2021.csv          # raw vote share
│   └── processed/
│       ├── voting\_clean.csv         # cleaned tabular data
│       ├── voting\_features.csv      # with engineered features & spatial lags
│       ├── xgb\_automl\_model.pkl     # trained FLAML+XGBoost model
│       ├── shap\_explanations.csv    # SHAP outputs
│       ├── geoshapley\_explanations.csv # GeoShapley outputs
│       ├── mgwr\_coefficients.csv    # MGWR baseline
│       ├── bootstrap\_shap\_stats.csv # SHAP uncertainty stats
│       └── fairness\_metrics.csv     # spatial fairness gaps
├── src/
│   ├── data\_loader.py               # load & clean
│   ├── feature\_engineering.py       # spatial lags, GeoDataFrame
│   ├── model\_training.py            # FLAML + XGBoost training
│   ├── shap\_explainer.py            # Kernel SHAP wrapper
│   ├── geoshapley\_explainer.py      # GeoShapley computations
│   ├── mgwr\_comparison.py           # MGWR baseline scripts
│   ├── bootstrap\_uncertainty.py     # bootstrap SHAP stats
│   ├── spatial\_fairness.py          # compute residual‐fairness
│   └── config.py                    # paths & constants
├── dashboard/
│   └── app.py                       # Streamlit + Folium dashboard
├── docs/
│   ├── implementation\_notes.md      # detailed pipeline doc
│   └── paper\_summary.pdf            # summary of Li (2025) chapter
├── README.md                        # this file
└── requirements.txt                 # pip dependencies

````

---

## ⚙️ Installation

1. **Clone repo**  
   ```bash
   git clone https://github.com/yourusername/explainable-geoai.git
   cd explainable-geoai
````

2. **Create & activate** a virtual environment

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download raw data**

   * Place `voting_2021.csv` in `data/raw/`
   * Download ACS and shapefiles via `src/download_census.py` or manually.

---

## 🚀 Quick Start

1. **Data & features**

   ```bash
   python src/data_loader.py
   python src/feature_engineering.py
   ```

2. **Train model**

   ```bash
   python src/model_training.py
   ```

3. **Generate explanations**

   ```bash
   python src/shap_explainer.py
   python src/geoshapley_explainer.py
   python src/mgwr_comparison.py
   python src/bootstrap_uncertainty.py
   python src/spatial_fairness.py
   ```

4. **Launch dashboard**

   ```bash
   cd dashboard
   streamlit run app.py
   ```

---

## 📝 Scripts & Modules

* **`data_loader.py`**: cleans raw vote + ACS, saves `voting_clean.csv`.
* **`feature_engineering.py`**: builds spatial lags, exports `voting_features.csv`.
* **`model_training.py`**: uses FLAML to find best XGBoost; saves model.
* **`shap_explainer.py`**: Kernel SHAP over FLAML model → `shap_explanations.csv`.
* **`geoshapley_explainer.py`**: computes GeoShapley components → `geoshapley_explanations.csv`.
* **`mgwr_comparison.py`**: fits MGWR baseline → `mgwr_coefficients.csv`.
* **`bootstrap_uncertainty.py`**: bootstraps SHAP → `bootstrap_shap_stats.csv`.
* **`spatial_fairness.py`**: calculates fairness gaps → `fairness_metrics.csv`.
* **`dashboard/app.py`**: interactive Streamlit + Folium map.

---

## 📊 Dashboard Overview

* **SHAP**: county-level attributions, with uncertainty.
* **GeoShapley**: decomposed intrinsic (GEO), main, and interaction effects.
* **MGWR/OLS**: local regression coefficients for comparison.
* **Fairness**: residual differences across demographic groups.
* **Download** any CSV for offline analysis.

---

## 🧾 Citing

If you use this work, please cite:

> Li, Ziqi (2025). *Explainable AI in Spatial Analysis*. In:
> *Advances in Spatial Data Science*, Springer.

