# COVID-19 Wave Analysis: Growth Modelling and Clustering of Indian States

A complete, reproducible scientific data curation and machine learning pipeline modeling COVID-19 epidemic wave kinetics and state clustering across India.

## Project Overview
- **Domain:** Health & Epidemiology
- **SDGs:** SDG 3 (Good Health and Well-being) & SDG 17 (Partnerships for the Goals)
- **ML Tasks:** Regression (Polynomial & Regularised Ridge) and Clustering (K-Means & Hierarchical Ward Linkage on state-level features extracted from the curves, plus a normalised-curve variant, with PCA)
- **Evaluation Metrics:** RMSE, $R^2$, and Silhouette Score
- **Article Type:** Data Article / Data in Brief

## Quick Start

### 1. Install requirements
```bash
pip install -r requirements.txt
```

### 2. Run the main pipeline
Fetches/loads the datasets, harmonizes the panel, fits the regressions, runs PCA and clustering, generates figures 1-6 (300 DPI) and exports the dashboard payload:

```bash
python run_pipeline.py
```

### 3. Run the extra validation (baselines, hold-out test, cluster checks, EDA)
```bash
python run_validation.py
```
This writes `data/processed/validation_*.csv`, `validation_summary.json` and figures 7-11.

### 4. Launch the interactive dashboard
Serve the project folder locally and open the dashboard in a browser:

```bash
python -m http.server 8000
```
Then visit http://localhost:8000/dashboard/ (a live copy is on GitHub Pages if enabled for this repository).

`covid19_wave_analysis_full.py` is a standalone single-file version of the same analysis; `run_pipeline.py` (with the modules in `src/`) is the main entry point.

## Validation and Limitations
- The pipeline's R2 and RMSE for the regressions are **in-sample** (goodness of fit). `run_validation.py` adds an **out-of-sample test**: models are trained on the first 80% of each wave and scored on the last 20%, against two simple baselines (persistence and training mean). Hyper-parameters (polynomial degree, Ridge alpha) are chosen with time-ordered cross-validation on the training part only.
- Cluster quality is checked against a **shuffled-feature null** (silhouette), **re-sampling stability** (Adjusted Rand Index over 80% subsamples), and agreement between K-Means and Ward clustering. States are also clustered on **normalised epidemic curves** (each curve scaled to peak = 1) and compared with the feature-based clusters.
- Silhouette values near 0.3 mean the clusters overlap; the grouping is exploratory. Mizoram forms a one-state cluster because its Wave 1 peak was very small, which inflates its surge ratio.
- Cluster names in the dashboard are descriptive labels from simple threshold rules, not model outputs. Vaccination coverage is measured at the last date in the data and cannot show that vaccination caused any difference.
- Reported cases depend on testing and reporting practice; populations are projected 2021 estimates; the preparedness matrix uses author-chosen illustrative weights.

## Directory Structure
- `data/raw/`: Raw downloaded CSV files from COVID19-India API (`state_wise_daily.csv`, `cowin_vaccine_data_statewise.csv`).
- `data/processed/`: Curated, publication-ready datasets:
  - `cleaned_state_panel.csv`: 22,089 daily panel records across 37 states/UTs with cases, 7-day rolling averages, population, incidence per 100k, and vaccination coverage.
  - `state_wave_metrics.csv`: 9-dimensional state-level feature matrix for PCA and Clustering.
  - `regression_metrics.csv`: Detailed model scorecard (RMSE, $R^2$, polynomial degree, Ridge $\alpha$) for Wave 1 and Wave 2.
  - `clustering_results.csv`: Cluster assignments, PCA coordinates (PC1, PC2, PC3), and silhouette scores.
  - `data_dictionary.json`: Standard metadata and schema documentation.
- `figures/`: publication-quality figures (300 DPI); figures 7-11 come from `run_validation.py`:
  - `fig1_epidemic_waves_timeline.png`: National & multi-state epidemic curves.
  - `fig2_regression_fits.png`: Polynomial vs Regularised Ridge regression fits.
  - `fig3_pca_biplot.png`: 2D PCA biplot showing state clusters and feature loadings vectors.
  - `fig4_hierarchical_dendrogram.png`: Hierarchical clustering dendrogram (Ward linkage).
  - `fig5_cluster_characteristics.png`: Comparative profile of state clusters across dimensions.
  - `fig6_sdg_preparedness_index.png`: SDG 3 & 17 Pandemic Preparedness Matrix.
- `dashboard/`: Standalone interactive web dashboard (`index.html`, `style.css`, `app.js` and the exported data file).
- `paper/`: Paper draft (`DATA_IN_BRIEF_MANUSCRIPT.md`).
- `report/`: Final assignment report (PDF).

## Submission Deadlines
- **ICAAIC 2027 (Salem):** 09 November 2026
- **ICDAM 2027 (Springer LNNS):** 30 November 2026
- **Data in Brief (Elsevier):** Rolling Submission
- **Scientific Data (Nature Portfolio):** Global Health Collections (March–May 2027)
