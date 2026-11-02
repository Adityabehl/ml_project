# COVID-19 Wave Analysis: Growth Modelling and Clustering of Indian States

A complete, reproducible scientific data curation and machine learning pipeline modeling COVID-19 epidemic wave kinetics and state clustering across India.

## Project Overview
- **Domain:** Health & Epidemiology
- **SDGs:** SDG 3 (Good Health and Well-being) & SDG 17 (Partnerships for the Goals)
- **ML Tasks:** Regression (Polynomial & Regularised Ridge) and Clustering (K-Means & Hierarchical Ward Linkage on Normalized Curves + PCA)
- **Evaluation Metrics:** RMSE, $R^2$, and Silhouette Score
- **Article Type:** Data Article / Data in Brief

## Quick Start

### 1. Install requirements
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline
Run the complete pipeline end-to-end to fetch datasets, harmonize panel records, fit regressions, run PCA and clustering, generate 6 publication figures (300 DPI), and export the web dashboard payload:

```bash
cd covid19_wave_analysis
python run_pipeline.py
```

### 3. Launch Interactive Dashboard
Start a local web server from the project folder:

```bash
python -m http.server 8000
```

Then open http://localhost:8000/dashboard/ in your web browser.
```

## Directory Structure
- `data/raw/`: Raw downloaded CSV files from COVID19-India API (`state_wise_daily.csv`, `cowin_vaccine_data_statewise.csv`).
- `data/processed/`: Curated, publication-ready datasets:
  - `cleaned_state_panel.csv`: 22,089 daily panel records across 37 states/UTs with cases, 7-day rolling averages, population, incidence per 100k, and vaccination coverage.
  - `state_wave_metrics.csv`: 9-dimensional state-level feature matrix for PCA and Clustering.
  - `regression_metrics.csv`: Detailed model scorecard (RMSE, $R^2$, polynomial degree, Ridge $\alpha$) for Wave 1 and Wave 2.
  - `clustering_results.csv`: Cluster assignments, PCA coordinates (PC1, PC2, PC3), and silhouette scores.
  - `data_dictionary.json`: Standard metadata and schema documentation.
- `figures/`: 6 high-resolution publication-quality figures (300 DPI):
  - `fig1_epidemic_waves_timeline.png`: National & multi-state epidemic curves.
  - `fig2_regression_fits.png`: Polynomial vs Regularised Ridge regression fits.
  - `fig3_pca_biplot.png`: 2D PCA biplot showing state clusters and feature loadings vectors.
  - `fig4_hierarchical_dendrogram.png`: Hierarchical clustering dendrogram (Ward linkage).
  - `fig5_cluster_characteristics.png`: Comparative profile of state clusters across dimensions.
  - `fig6_sdg_preparedness_index.png`: SDG 3 & 17 Pandemic Preparedness Matrix.
- `dashboard/`: Standalone interactive web dashboard (`index.html`, `style.css`, `app.js`, `data.js`).
- `paper/`: Complete paper draft (`DATA_IN_BRIEF_MANUSCRIPT.md`).


