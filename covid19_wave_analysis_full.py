"""
====================================================================================================
Title: COVID-19 wave analysis: growth modelling and clustering of states
Domain: Health | SDGs: SDG 3 (Good Health and Well-being) & SDG 17 (Partnerships for the Goals)
Article Type: Data Article / Data in Brief
Methods: Polynomial Regression, Ridge Regression, PCA, K-Means, and Hierarchical Clustering
Metrics: RMSE, R², Silhouette Score
Dataset: COVID19-India API (Static CSV/JSON from https://data.covid19india.org/)
Target Venues: Data in Brief, Scientific Data, ICDAM 2027, ICAAIC 2027
====================================================================================================
"""

import os
import sys
import json
import warnings
import urllib.request
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram

# ==================================================================================================
# 1. PATH CONFIGURATION & METADATA
# ==================================================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FIGURES_DIR = BASE_DIR / "figures"

for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CASES_URL = "https://data.covid19india.org/csv/latest/state_wise_daily.csv"
VACCINE_URL = "https://data.covid19india.org/csv/latest/cowin_vaccine_data_statewise.csv"

# State Code to Name Mapping
STATE_CODE_TO_NAME = {
    "AN": "Andaman and Nicobar Islands", "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh",
    "AS": "Assam", "BR": "Bihar", "CH": "Chandigarh", "CT": "Chhattisgarh", "DL": "Delhi",
    "DN": "Dadra and Nagar Haveli and Daman and Diu", "GA": "Goa", "GJ": "Gujarat",
    "HP": "Himachal Pradesh", "HR": "Haryana", "JH": "Jharkhand", "JK": "Jammu and Kashmir",
    "KA": "Karnataka", "KL": "Kerala", "LA": "Ladakh", "LD": "Lakshadweep", "MH": "Maharashtra",
    "ML": "Meghalaya", "MN": "Manipur", "MP": "Madhya Pradesh", "MZ": "Mizoram", "NL": "Nagaland",
    "OR": "Odisha", "PB": "Punjab", "PY": "Puducherry", "RJ": "Rajasthan", "SK": "Sikkim",
    "TG": "Telangana", "TN": "Tamil Nadu", "TR": "Tripura", "TT": "India Total",
    "UP": "Uttar Pradesh", "UT": "Uttarakhand", "WB": "West Bengal"
}
STATE_NAME_TO_CODE = {v: k for k, v in STATE_CODE_TO_NAME.items()}

# 2021 Projected Population (Census of India / MoHFW)
STATE_POPULATION = {
    "AN": 400000, "AP": 52787000, "AR": 1533000, "AS": 35043000, "BR": 123083000,
    "CH": 1208000, "CT": 29436000, "DL": 20571000, "DN": 1079000, "GA": 1559000,
    "GJ": 69788000, "HP": 7394000, "HR": 29483000, "JH": 38471000, "JK": 13384000,
    "KA": 66845000, "KL": 35489000, "LA": 297000, "LD": 68000, "MH": 124437000,
    "ML": 3288000, "MN": 3165000, "MP": 84516000, "MZ": 1216000, "NL": 2190000,
    "OR": 45696000, "PB": 30339000, "PY": 1573000, "RJ": 79281000, "SK": 677000,
    "TG": 37725000, "TN": 76402000, "TR": 4071000, "TT": 1361343000, "UP": 230907000,
    "UT": 11392000, "WB": 98179000
}

WAVE_PERIODS = {
    "Wave 1": {"start": "2020-05-01", "end": "2020-12-31"},
    "Wave 2": {"start": "2021-03-01", "end": "2021-07-31"}
}

# ==================================================================================================
# 2. DATA INGESTION & PANEL HARMONIZATION
# ==================================================================================================
def fetch_and_clean_panel():
    """Acquires raw streams, handles reconciliation spikes, and computes 7-day rolling panel."""
    cases_file = RAW_DATA_DIR / "state_wise_daily.csv"
    vac_file = RAW_DATA_DIR / "cowin_vaccine_data_statewise.csv"

    if not cases_file.exists():
        print(f"Downloading cases data from {CASES_URL}...")
        urllib.request.urlretrieve(CASES_URL, cases_file)
    if not vac_file.exists():
        print(f"Downloading vaccine data from {VACCINE_URL}...")
        urllib.request.urlretrieve(VACCINE_URL, vac_file)

    print("Cleaning daily confirmed, recovered, and deceased cases...")
    df_raw = pd.read_csv(cases_file)
    df_raw['Date_YMD'] = pd.to_datetime(df_raw['Date_YMD'])
    state_cols = [c for c in df_raw.columns if c in STATE_CODE_TO_NAME and c != 'UN']

    df_long = df_raw.melt(id_vars=['Date_YMD', 'Status'], value_vars=state_cols, var_name='State_Code', value_name='Count')
    df_long['Count'] = pd.to_numeric(df_long['Count'], errors='coerce').fillna(0)

    df_pivot = df_long.pivot_table(index=['Date_YMD', 'State_Code'], columns='Status', values='Count', aggfunc='first').reset_index()
    df_pivot.rename(columns={'Date_YMD': 'Date', 'Confirmed': 'Confirmed_Daily', 'Recovered': 'Recovered_Daily', 'Deceased': 'Deceased_Daily'}, inplace=True)

    for col in ['Confirmed_Daily', 'Recovered_Daily', 'Deceased_Daily']:
        if col not in df_pivot: df_pivot[col] = 0
        df_pivot[col] = df_pivot[col].clip(lower=0)

    df_pivot.sort_values(by=['State_Code', 'Date'], inplace=True)
    df_pivot['State_Name'] = df_pivot['State_Code'].map(STATE_CODE_TO_NAME)
    df_pivot['Population'] = df_pivot['State_Code'].map(STATE_POPULATION).fillna(1000000).astype(int)

    # 7-Day Rolling Trailing Averages
    df_pivot['Confirmed_7D_Avg'] = df_pivot.groupby('State_Code')['Confirmed_Daily'].transform(lambda s: s.rolling(7, min_periods=1).mean())
    df_pivot['Cumulative_Confirmed'] = df_pivot.groupby('State_Code')['Confirmed_Daily'].cumsum()
    df_pivot['Cumulative_Deceased'] = df_pivot.groupby('State_Code')['Deceased_Daily'].cumsum()
    df_pivot['Confirmed_7D_Avg_Per_100k'] = (df_pivot['Confirmed_7D_Avg'] / df_pivot['Population']) * 100000
    df_pivot['Cumulative_Per_100k'] = (df_pivot['Cumulative_Confirmed'] / df_pivot['Population']) * 100000

    print("Cleaning and merging CoWIN vaccination records...")
    df_vac = pd.read_csv(vac_file)
    df_vac['Date'] = pd.to_datetime(df_vac['Updated On'], dayfirst=True, errors='coerce')
    df_vac = df_vac.dropna(subset=['Date'])
    df_vac['State'] = df_vac['State'].str.strip()
    df_vac['State_Code'] = df_vac['State'].map(STATE_NAME_TO_CODE)
    df_vac.loc[df_vac['State'] == 'India', 'State_Code'] = 'TT'
    df_vac = df_vac[df_vac['State_Code'].notnull()].copy()

    df_vac = df_vac[['Date', 'State_Code', 'Total Doses Administered', 'First Dose Administered', 'Second Dose Administered']].rename(columns={
        'Total Doses Administered': 'Total_Doses',
        'First Dose Administered': 'Dose1_Administered',
        'Second Dose Administered': 'Dose2_Administered'
    })
    for c in ['Total_Doses', 'Dose1_Administered', 'Dose2_Administered']:
        df_vac[c] = pd.to_numeric(df_vac[c], errors='coerce').fillna(0)

    df_vac_grp = df_vac.groupby(['Date', 'State_Code'], as_index=False).max()
    df_panel = pd.merge(df_pivot, df_vac_grp, on=['Date', 'State_Code'], how='left')

    df_panel.sort_values(by=['State_Code', 'Date'], inplace=True)
    df_panel['Dose1_Administered'] = df_panel.groupby('State_Code')['Dose1_Administered'].ffill().fillna(0)
    df_panel['Dose2_Administered'] = df_panel.groupby('State_Code')['Dose2_Administered'].ffill().fillna(0)
    df_panel['Total_Doses'] = df_panel.groupby('State_Code')['Total_Doses'].ffill().fillna(0)

    df_panel['Dose1_Coverage_Pct'] = ((df_panel['Dose1_Administered'] / df_panel['Population']) * 100).clip(0, 100)
    df_panel['Dose2_Coverage_Pct'] = ((df_panel['Dose2_Administered'] / df_panel['Population']) * 100).clip(0, 100)

    panel_path = PROCESSED_DATA_DIR / "cleaned_state_panel.csv"
    df_panel.to_csv(panel_path, index=False)
    print(f"-> Curated panel exported: {panel_path} ({len(df_panel):,} rows)")
    return df_panel

# ==================================================================================================
# 3. EPIDEMIC WAVE REGRESSION MODELLING (POLYNOMIAL & RIDGE)
# ==================================================================================================
def model_wave_regressions(df_panel):
    """Fits Polynomial and Ridge regressions for each state across Wave 1 and Wave 2."""
    states = [s for s in df_panel['State_Code'].unique() if s != 'TT']
    metrics = []
    fitted_curves = {"Wave 1": {}, "Wave 2": {}}

    for wave_name, wave_cfg in WAVE_PERIODS.items():
        start_dt = pd.to_datetime(wave_cfg["start"])
        end_dt = pd.to_datetime(wave_cfg["end"])

        for st in states:
            df_st = df_panel[(df_panel['State_Code'] == st) & (df_panel['Date'] >= start_dt) & (df_panel['Date'] <= end_dt)].sort_values('Date')
            if len(df_st) < 20: continue
            y = df_st['Confirmed_7D_Avg'].values
            if y.max() <= 5: continue

            n = len(y)
            t_norm = (np.arange(n) / float(n - 1)).reshape(-1, 1)

            # Polynomial Regression with degree selection (2..5)
            best_deg, best_r2, best_rmse, best_pred = 3, -float('inf'), float('inf'), None
            for deg in [2, 3, 4, 5]:
                poly = PolynomialFeatures(degree=deg)
                X_poly = poly.fit_transform(t_norm)
                reg = LinearRegression().fit(X_poly, y)
                pred = np.clip(reg.predict(X_poly), 0, None)
                r2 = r2_score(y, pred)
                rmse = np.sqrt(mean_squared_error(y, pred))
                if r2 > best_r2:
                    best_deg, best_r2, best_rmse, best_pred = deg, r2, rmse, pred

            # Ridge Regression (L2 penalty) on degree 4 features
            poly_ridge = PolynomialFeatures(degree=4)
            X_ridge = poly_ridge.fit_transform(t_norm)
            ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5).fit(X_ridge, y)
            ridge_pred = np.clip(ridge_cv.predict(X_ridge), 0, None)
            ridge_r2 = r2_score(y, ridge_pred)
            ridge_rmse = np.sqrt(mean_squared_error(y, ridge_pred))

            peak_idx = int(np.argmax(y))
            peak_val = float(y[peak_idx])
            pop = int(df_st['Population'].iloc[0])
            peak_per_100k = (peak_val / pop) * 100000

            metrics.append({
                'Wave': wave_name, 'State_Code': st, 'State_Name': df_st['State_Name'].iloc[0],
                'Population': pop, 'Observations': n, 'Peak_Daily_Cases': round(peak_val, 1),
                'Peak_Per_100k': round(peak_per_100k, 2), 'Days_To_Peak': peak_idx,
                'Poly_Degree': best_deg, 'Poly_RMSE': round(best_rmse, 2), 'Poly_R2': round(best_r2, 4),
                'Ridge_Alpha': round(float(ridge_cv.alpha_), 4), 'Ridge_RMSE': round(ridge_rmse, 2),
                'Ridge_R2': round(ridge_r2, 4)
            })

            fitted_curves[wave_name][st] = {
                'dates': df_st['Date'].dt.strftime('%Y-%m-%d').tolist(),
                'actual': [round(float(v), 1) for v in y],
                'poly_fit': [round(float(v), 1) for v in best_pred],
                'ridge_fit': [round(float(v), 1) for v in ridge_pred],
                'poly_degree': best_deg, 'poly_r2': round(best_r2, 4), 'ridge_r2': round(ridge_r2, 4),
                'ridge_alpha': round(float(ridge_cv.alpha_), 4)
            }

    df_metrics = pd.DataFrame(metrics)
    reg_path = PROCESSED_DATA_DIR / "regression_metrics.csv"
    df_metrics.to_csv(reg_path, index=False)
    print(f"-> Regression metrics exported: {reg_path} ({len(df_metrics)} records)")
    return df_metrics, fitted_curves

# ==================================================================================================
# 4. FEATURE EXTRACTION, PCA & CLUSTERING
# ==================================================================================================
CLUSTER_FEATURES = [
    'Wave1_Peak_Per_100k', 'Wave2_Peak_Per_100k', 'Wave1_Days_To_Peak', 'Wave2_Days_To_Peak',
    'Delta_Surge_Ratio', 'Total_Incidence_Per_100k', 'Case_Fatality_Rate_Pct',
    'Dose1_Coverage_Pct', 'Dose2_Coverage_Pct'
]

def run_pca_and_clustering(df_panel, df_metrics):
    """Standardizes state kinetics, runs PCA, K-Means & Hierarchical clustering, computes silhouette scores."""
    states = [s for s in df_panel['State_Code'].unique() if s not in ['TT', 'UN']]
    records = []
    df_latest = df_panel[df_panel['Date'] == df_panel['Date'].max()]

    for st in states:
        m1 = df_metrics[(df_metrics['State_Code'] == st) & (df_metrics['Wave'] == 'Wave 1')]
        m2 = df_metrics[(df_metrics['State_Code'] == st) & (df_metrics['Wave'] == 'Wave 2')]
        if len(m1) == 0 or len(m2) == 0: continue

        w1_pk = m1['Peak_Per_100k'].values[0]
        w2_pk = m2['Peak_Per_100k'].values[0]
        df_st = df_panel[df_panel['State_Code'] == st]
        cfr = (df_st['Cumulative_Deceased'].max() / (df_st['Cumulative_Confirmed'].max() + 1)) * 100

        st_vac = df_latest[df_latest['State_Code'] == st]
        d1 = st_vac['Dose1_Coverage_Pct'].values[0] if len(st_vac) > 0 else 0.0
        d2 = st_vac['Dose2_Coverage_Pct'].values[0] if len(st_vac) > 0 else 0.0

        records.append({
            'State_Code': st, 'State_Name': m1['State_Name'].values[0], 'Population': m1['Population'].values[0],
            'Wave1_Peak_Per_100k': round(float(w1_pk), 2), 'Wave2_Peak_Per_100k': round(float(w2_pk), 2),
            'Wave1_Days_To_Peak': int(m1['Days_To_Peak'].values[0]), 'Wave2_Days_To_Peak': int(m2['Days_To_Peak'].values[0]),
            'Delta_Surge_Ratio': round(float((w2_pk + 1.0) / (w1_pk + 1.0)), 2),
            'Total_Incidence_Per_100k': round(float(df_st['Cumulative_Per_100k'].max()), 2),
            'Case_Fatality_Rate_Pct': round(float(cfr), 2),
            'Dose1_Coverage_Pct': round(float(d1), 2), 'Dose2_Coverage_Pct': round(float(d2), 2)
        })

    df_feat = pd.DataFrame(records)
    feat_path = PROCESSED_DATA_DIR / "state_wave_metrics.csv"
    df_feat.to_csv(feat_path, index=False)

    # Standardization & PCA
    X = df_feat[CLUSTER_FEATURES].values
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=min(len(CLUSTER_FEATURES), len(df_feat) - 1))
    X_pca = pca.fit_transform(X_scaled)
    df_feat['PC1'] = np.round(X_pca[:, 0], 3)
    df_feat['PC2'] = np.round(X_pca[:, 1], 3)
    df_feat['PC3'] = np.round(X_pca[:, 2], 3)

    # Silhouette optimization for K-Means (k=2..6)
    sil_scores = {}
    best_k, best_sil = 3, -1.0
    for k in range(2, min(7, len(df_feat))):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
        sil = silhouette_score(X_scaled, km.labels_)
        sil_scores[k] = sil
        if sil > best_sil:
            best_sil = sil
            best_k = k

    km_opt = KMeans(n_clusters=best_k, random_state=42, n_init=20).fit(X_scaled)
    df_feat['KMeans_Cluster'] = km_opt.labels_
    df_feat['Silhouette_Value'] = np.round(silhouette_samples(X_scaled, km_opt.labels_), 3)

    # Hierarchical Clustering (Ward Linkage)
    linkage_mat = linkage(X_scaled, method='ward', metric='euclidean')
    agg = AgglomerativeClustering(n_clusters=best_k, linkage='ward').fit(X_scaled)
    df_feat['Hierarchical_Cluster'] = agg.labels_
    hier_sil = silhouette_score(X_scaled, agg.labels_)

    # Archetype characterization
    cluster_profiles = {}
    for cid in range(best_k):
        sub = df_feat[df_feat['KMeans_Cluster'] == cid]
        cluster_profiles[cid] = {
            'count': len(sub), 'states': sub['State_Code'].tolist(),
            'mean_delta_surge': round(float(sub['Delta_Surge_Ratio'].mean()), 2),
            'mean_w2_peak': round(float(sub['Wave2_Peak_Per_100k'].mean()), 2),
            'mean_dose1_cov': round(float(sub['Dose1_Coverage_Pct'].mean()), 2),
            'mean_dose2_cov': round(float(sub['Dose2_Coverage_Pct'].mean()), 2)
        }
        if cluster_profiles[cid]['mean_delta_surge'] > 10.0:
            cluster_profiles[cid]['archetype'] = "Late-Peak Outlier (Northeast Anomaly)"
        elif cluster_profiles[cid]['mean_dose1_cov'] > 65.0:
            cluster_profiles[cid]['archetype'] = "Proactive Immunization & High Case Detection"
        else:
            cluster_profiles[cid]['archetype'] = "Vulnerable Health System & Delayed Vaccine Uptake"

    df_feat['Cluster_Archetype'] = df_feat['KMeans_Cluster'].map(lambda cid: cluster_profiles[cid]['archetype'])
    clus_path = PROCESSED_DATA_DIR / "clustering_results.csv"
    df_feat.to_csv(clus_path, index=False)
    print(f"-> Clustering results exported: {clus_path} (Optimal k={best_k}, Silhouette={best_sil:.4f})")

    pca_summary = {
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'loadings': pd.DataFrame(pca.components_[:3].T, index=CLUSTER_FEATURES, columns=['PC1_Loading', 'PC2_Loading', 'PC3_Loading']).round(3).to_dict()
    }
    clustering_summary = {
        'optimal_k': best_k, 'best_kmeans_silhouette': round(best_sil, 4),
        'hierarchical_silhouette': round(float(hier_sil), 4),
        'cluster_profiles': cluster_profiles, 'linkage_matrix': linkage_mat
    }
    return df_feat, pca_summary, clustering_summary

# ==================================================================================================
# 5. SCIENTIFIC VISUALIZATION (6 PUBLICATION-GRADE FIGURES, 300 DPI)
# ==================================================================================================
PALETTE = ['#2b5c8f', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02']

def generate_figures(df_panel, fitted_curves, df_results, pca_summary, clustering_summary):
    """Renders 6 high-resolution figures formatted for scientific submission."""
    # Fig 1: Timeline
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [1.2, 1.5]})
    df_tt = df_panel[df_panel['State_Code'] == 'TT'].sort_values('Date')
    ax1.plot(df_tt['Date'], df_tt['Confirmed_Daily'], color='#b0c4de', alpha=0.5, label='Daily Reported')
    ax1.plot(df_tt['Date'], df_tt['Confirmed_7D_Avg'], color='#1f4e79', linewidth=2.5, label='7-Day Rolling Avg')
    ax1.axvspan(pd.to_datetime('2020-05-01'), pd.to_datetime('2020-12-31'), color='#2b5c8f', alpha=0.08, label='Wave 1')
    ax1.axvspan(pd.to_datetime('2021-03-01'), pd.to_datetime('2021-07-31'), color='#d95f02', alpha=0.08, label='Wave 2 (Delta)')
    ax1.set_title('(A) India National COVID-19 Epidemic Trajectory', fontsize=12, weight='bold')
    ax1.set_ylabel('Daily Cases', fontsize=10, weight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper left')

    for st, c in zip(['MH', 'DL', 'KL', 'KA', 'TN', 'UP'], ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']):
        sub = df_panel[df_panel['State_Code'] == st].sort_values('Date')
        ax2.plot(sub['Date'], sub['Confirmed_7D_Avg_Per_100k'], color=c, linewidth=2, label=f"{st}")
    ax2.set_title('(B) Subnational Incidence Heterogeneity (Cases per 100k Pop)', fontsize=12, weight='bold')
    ax2.set_ylabel('7D Avg / 100k', fontsize=10, weight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper left', ncol=6)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_epidemic_waves_timeline.png", dpi=300)
    plt.close()

    # Fig 2: Regression Fits
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    key_states = ['MH', 'DL', 'KL']
    for i, st in enumerate(key_states):
        for j, w in enumerate(['Wave 1', 'Wave 2']):
            ax = axes[i, j]
            if w in fitted_curves and st in fitted_curves[w]:
                d = fitted_curves[w][st]
                t = np.arange(len(d['actual']))
                ax.plot(t, d['actual'], 'o-', color='#666666', markersize=3, alpha=0.6, label='Observed')
                ax.plot(t, d['poly_fit'], '--', color='#d95f02', linewidth=2.2, label=f"Poly: R²={d['poly_r2']:.2f}")
                ax.plot(t, d['ridge_fit'], ':', color='#2b5c8f', linewidth=2.5, label=f"Ridge: R²={d['ridge_r2']:.2f}")
                ax.set_title(f"{st} - {w}", fontsize=10, weight='bold')
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.legend(fontsize=8, loc='upper left')
    fig.suptitle('Polynomial Regression vs Ridge Regularization Case Growth Fits', fontsize=13, weight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_regression_fits.png", dpi=300)
    plt.close()

    # Fig 3: PCA Biplot
    fig, ax = plt.subplots(figsize=(10, 7.5))
    for cid in df_results['KMeans_Cluster'].unique():
        sub = df_results[df_results['KMeans_Cluster'] == cid]
        ax.scatter(sub['PC1'], sub['PC2'], color=PALETTE[cid], s=110, edgecolors='black', label=f"Cluster {cid}: {sub['Cluster_Archetype'].iloc[0]}")
        for _, r in sub.iterrows():
            ax.annotate(r['State_Code'], (r['PC1'], r['PC2']), xytext=(4, 4), textcoords='offset points', fontsize=8, weight='bold')
    
    loadings = pd.DataFrame(pca_summary['loadings'])
    for feat in loadings.index:
        ax.arrow(0, 0, loadings.loc[feat, 'PC1_Loading'] * 3.5, loadings.loc[feat, 'PC2_Loading'] * 3.5,
                 color='#b2182b', alpha=0.7, width=0.03, head_width=0.1)
        ax.text(loadings.loc[feat, 'PC1_Loading'] * 3.8, loadings.loc[feat, 'PC2_Loading'] * 3.8,
                feat.replace('_Per_100k', '').replace('_Pct', '').replace('Wave', 'W'), color='#7f0000', fontsize=8, weight='bold')
    
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    ax.axvline(0, color='grey', linestyle='--', linewidth=0.8)
    ax.set_xlabel(f"PC1 ({pca_summary['explained_variance_ratio'][0]*100:.1f}% Variance)", weight='bold')
    ax.set_ylabel(f"PC2 ({pca_summary['explained_variance_ratio'][1]*100:.1f}% Variance)", weight='bold')
    ax.set_title('Principal Component Analysis (PCA) Biplot: State Clusters & Vectors', fontsize=12, weight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower right', fontsize=8.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_pca_biplot.png", dpi=300)
    plt.close()

    # Fig 4: Dendrogram
    fig, ax = plt.subplots(figsize=(12, 7))
    dendrogram(clustering_summary['linkage_matrix'], labels=df_results['State_Name'].tolist(), ax=ax, leaf_rotation=90, leaf_font_size=8.5)
    ax.set_title('Hierarchical Clustering Dendrogram of Indian States (Ward Linkage)', fontsize=12, weight='bold')
    ax.set_ylabel('Euclidean Distance', weight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig4_hierarchical_dendrogram.png", dpi=300)
    plt.close()

    # Fig 5: Cluster Characteristics
    df_plot = []
    for cid, p in clustering_summary['cluster_profiles'].items():
        df_plot.append({'Cluster': f"Cluster {cid}", 'W1 Peak': p.get('mean_w1_peak', 0), 'W2 Peak': p['mean_w2_peak'], 'Surge': p['mean_delta_surge'], 'Dose 1 %': p['mean_dose1_cov']})
    df_p = pd.DataFrame(df_plot)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    sns.barplot(data=df_p, x='Cluster', y='W2 Peak', ax=axes[0], palette=PALETTE)
    axes[0].set_title('Mean Wave 2 Peak / 100k', weight='bold')
    sns.barplot(data=df_p, x='Cluster', y='Surge', ax=axes[1], palette=PALETTE)
    axes[1].set_title('Delta Surge Multiplier (W2/W1)', weight='bold')
    sns.barplot(data=df_p, x='Cluster', y='Dose 1 %', ax=axes[2], palette=PALETTE)
    axes[2].set_title('Dose 1 Coverage (%)', weight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_cluster_characteristics.png", dpi=300)
    plt.close()

    # Fig 6: SDG 3 & 17 Matrix
    fig, ax = plt.subplots(figsize=(9, 7))
    x = df_results['Dose1_Coverage_Pct']
    y = np.clip(100 - (df_results['Delta_Surge_Ratio'] * 12 + df_results['Case_Fatality_Rate_Pct'] * 15), 0, 100)
    for cid in df_results['KMeans_Cluster'].unique():
        mask = df_results['KMeans_Cluster'] == cid
        ax.scatter(x[mask], y[mask], color=PALETTE[cid], s=df_results.loc[mask, 'Population'] / 1.5e6 + 50, edgecolors='black', label=f"Cluster {cid}")
        for idx in df_results[mask].index:
            ax.annotate(df_results.loc[idx, 'State_Code'], (x[idx], y[idx]), xytext=(3, 3), textcoords='offset points', fontsize=8, weight='bold')
    ax.axvline(70, color='grey', linestyle=':')
    ax.axhline(50, color='grey', linestyle=':')
    ax.set_xlabel('SDG 17: Vaccine Delivery Coverage (Dose 1 %)', weight='bold')
    ax.set_ylabel('SDG 3: Health System Resilience Score (0–100)', weight='bold')
    ax.set_title('SDG 3 & SDG 17 Pandemic Preparedness Matrix', fontsize=12, weight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig6_sdg_preparedness_index.png", dpi=300)
    plt.close()
    print("-> 6 Publication figures saved in figures/ directory at 300 DPI.")

# ==================================================================================================
# 6. MAIN PIPELINE CONTROLLER
# ==================================================================================================
def main():
    print("=" * 80)
    print("COVID-19 WAVE ANALYSIS: GROWTH MODELLING & CLUSTERING OF STATES")
    print("Domain: Health | SDGs: SDG 3 & SDG 17 | Machine Learning: Regression & Clustering")
    print("=" * 80)

    # Ingestion
    df_panel = fetch_and_clean_panel()

    # Regression Modelling
    df_metrics, fitted_curves = model_wave_regressions(df_panel)
    print("\nRegression Summary across Waves:")
    print(df_metrics.groupby('Wave')[['Poly_RMSE', 'Poly_R2', 'Ridge_RMSE', 'Ridge_R2']].mean().round(3))

    # PCA & Clustering
    df_results, pca_summary, clustering_summary = run_pca_and_clustering(df_panel, df_metrics)

    print(f"\nClustering Evaluation:")
    print(f"Optimal K-Means Clusters: k = {clustering_summary['optimal_k']}")
    print(f"Optimal K-Means Silhouette Score: {clustering_summary['best_kmeans_silhouette']:.4f}")
    print(f"Hierarchical Clustering Silhouette: {clustering_summary['hierarchical_silhouette']:.4f}")
    print(f"Cumulative PCA Variance (3 PCs): {sum(pca_summary['explained_variance_ratio'][:3])*100:.1f}%")

    print("\nIdentified Cluster Archetypes:")
    for cid, p in clustering_summary['cluster_profiles'].items():
        print(f"  Cluster {cid} [{p.get('archetype', '')}]: {p['count']} states | Delta Surge: {p['mean_delta_surge']}x | Dose 1: {p['mean_dose1_cov']}%")

    # Figure generation
    generate_figures(df_panel, fitted_curves, df_results, pca_summary, clustering_summary)

    print("\n" + "=" * 80)
    print("EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
