"""
COVID-19 Wave Analysis: Growth Modelling and Clustering of Indian States
Main Pipeline Execution Script
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from src.data_loader import build_curated_panel
from src.wave_modeling import fit_wave_regressions, extract_state_wave_features
from src.clustering_pca import run_pca_and_clustering
from src.visualization import generate_all_figures
from src.export_panel import export_metadata_and_dashboard_payload

def main():
    start_time = time.time()
    print("=" * 80)
    print("COVID-19 WAVE ANALYSIS: GROWTH MODELLING & CLUSTERING OF INDIAN STATES")
    print("Domain: Health | SDGs: SDG 3 (Good Health) & SDG 17 (Partnerships)")
    print("=" * 80)
    
    # Step 1: Data Acquisition & Panel Harmonization
    print("\n[STEP 1/6] Downloading, cleaning, and harmonizing state-level panel...")
    df_panel = build_curated_panel()
    print(f"Panel shape: {df_panel.shape} across {df_panel['State_Code'].nunique()} states/UTs.")
    
    # Step 2: Growth Modelling via Polynomial & Ridge Regression
    print("\n[STEP 2/6] Fitting Polynomial & Ridge Regressions on Wave 1 and Wave 2...")
    df_metrics, fitted_curves = fit_wave_regressions(df_panel)
    
    print("\nRegression Performance Summary (RMSE & R²):")
    summary_reg = df_metrics.groupby('Wave')[['Poly_RMSE', 'Poly_R2', 'Ridge_RMSE', 'Ridge_R2']].mean().round(3)
    print(summary_reg)
    
    # Step 3: Feature Extraction
    print("\n[STEP 3/6] Extracting state wave features & vaccination coverage...")
    df_features = extract_state_wave_features(df_panel, df_metrics)
    
    # Step 4: Dimensionality Reduction (PCA) & Clustering
    print("\n[STEP 4/6] Running PCA, K-Means, and Hierarchical Clustering...")
    df_results, pca_summary, clustering_summary = run_pca_and_clustering(df_features)
    
    print(f"\nOptimal K-Means Clusters: k = {clustering_summary['optimal_k']}")
    print(f"Optimal K-Means Silhouette Score: {clustering_summary['best_kmeans_silhouette']:.4f}")
    print(f"Hierarchical Clustering Silhouette Score: {clustering_summary['hierarchical_silhouette']:.4f}")
    print(f"PCA Variance Explained (First 3 PCs): {sum(pca_summary['explained_variance_ratio'][:3])*100:.1f}% "
          f"(PC1: {pca_summary['explained_variance_ratio'][0]*100:.1f}%, "
          f"PC2: {pca_summary['explained_variance_ratio'][1]*100:.1f}%, "
          f"PC3: {pca_summary['explained_variance_ratio'][2]*100:.1f}%)")
    
    print("\nIdentified Cluster Archetypes:")
    for cid, prof in clustering_summary['cluster_profiles'].items():
        print(f"  * Cluster {cid} [{prof.get('archetype', 'Cluster')}]: "
              f"{prof['count']} states | Mean Delta Surge: {prof['mean_delta_surge']}x | "
              f"Dose 1 Coverage: {prof['mean_dose1_cov']}% | Dose 2 Coverage: {prof['mean_dose2_cov']}%")
        print(f"    States: {', '.join(prof['state_codes'][:8])}{'...' if len(prof['state_codes']) > 8 else ''}")
        
    # Step 5: Publication Figures Generation
    print("\n[STEP 5/6] Generating 6 publication-grade scientific figures (300 DPI)...")
    generate_all_figures(df_panel, fitted_curves, df_results, pca_summary, clustering_summary)
    
    # Step 6: Documentation & Dashboard Export
    print("\n[STEP 6/6] Exporting Data in Brief metadata, data dictionary, and dashboard payload...")
    export_metadata_and_dashboard_payload(df_panel, df_metrics, df_results, pca_summary, clustering_summary, fitted_curves)
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
    print("Curated panel files released in data/processed/:")
    print("  - cleaned_state_panel.csv")
    print("  - state_wave_metrics.csv")
    print("  - clustering_results.csv")
    print("  - regression_metrics.csv")
    print("  - data_dictionary.json")
    print("Publication figures generated in figures/:")
    print("  - fig1_epidemic_waves_timeline.png")
    print("  - fig2_regression_fits.png")
    print("  - fig3_pca_biplot.png")
    print("  - fig4_hierarchical_dendrogram.png")
    print("  - fig5_cluster_characteristics.png")
    print("  - fig6_sdg_preparedness_index.png")
    print("=" * 80)

if __name__ == "__main__":
    main()
