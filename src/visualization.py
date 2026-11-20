"""
COVID-19 Wave Analysis: Scientific Publication Visualizations
Generates publication-quality, 300 DPI figures styled for academic journals
(Data in Brief, Scientific Data, Springer LNNS).
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from src.config import FIGURES_DIR
from src.clustering_pca import CLUSTER_FEATURES

# Style configuration for scientific publication
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#e0e0e0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

PALETTE = ['#2b5c8f', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02']

def plot_epidemic_waves_timeline(df_panel):
    """Figure 1: National and State-Level Epidemic Curves across Waves."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True, gridspec_kw={'height_ratios': [1.2, 1.5]})
    
    # 1. National Level (India Total - 'TT')
    df_tt = df_panel[df_panel['State_Code'] == 'TT'].sort_values('Date')
    if len(df_tt) > 0:
        ax1.plot(df_tt['Date'], df_tt['Confirmed_Daily'], color='#b0c4de', alpha=0.5, label='Daily Reported Cases')
        ax1.plot(df_tt['Date'], df_tt['Confirmed_7D_Avg'], color='#1f4e79', linewidth=2.5, label='7-Day Rolling Average')
        
        # Shade Wave periods
        ax1.axvspan(pd.to_datetime('2020-05-01'), pd.to_datetime('2020-12-31'), color='#2b5c8f', alpha=0.08, label='Wave 1 Period')
        ax1.axvspan(pd.to_datetime('2021-03-01'), pd.to_datetime('2021-07-31'), color='#d95f02', alpha=0.08, label='Wave 2 (Delta) Period')
        
        ax1.set_title('(A) India National COVID-19 Epidemic Trajectory (March 2020 – October 2021)', fontsize=13, weight='bold', pad=10)
        ax1.set_ylabel('Daily Confirmed Cases', fontsize=11, weight='bold')
        ax1.grid(True)
        ax1.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
        
        # Annotate peaks
        peak_w1_date = df_tt[(df_tt['Date'] >= '2020-05-01') & (df_tt['Date'] <= '2020-12-31')].sort_values('Confirmed_7D_Avg').iloc[-1]
        peak_w2_date = df_tt[(df_tt['Date'] >= '2021-03-01') & (df_tt['Date'] <= '2021-07-31')].sort_values('Confirmed_7D_Avg').iloc[-1]
        
        ax1.annotate(f"Wave 1 Peak\n{int(peak_w1_date['Confirmed_7D_Avg']):,} cases/day\n({peak_w1_date['Date'].strftime('%d %b %Y')})",
                     xy=(peak_w1_date['Date'], peak_w1_date['Confirmed_7D_Avg']),
                     xytext=(peak_w1_date['Date'] - pd.Timedelta(days=50), peak_w1_date['Confirmed_7D_Avg'] + 40000),
                     arrowprops=dict(facecolor='#1f4e79', shrink=0.05, width=1.5, headwidth=7),
                     fontsize=9, weight='bold')
                     
        ax1.annotate(f"Wave 2 Peak (Delta)\n{int(peak_w2_date['Confirmed_7D_Avg']):,} cases/day\n({peak_w2_date['Date'].strftime('%d %b %Y')})",
                     xy=(peak_w2_date['Date'], peak_w2_date['Confirmed_7D_Avg']),
                     xytext=(peak_w2_date['Date'] - pd.Timedelta(days=60), peak_w2_date['Confirmed_7D_Avg'] + 15000),
                     arrowprops=dict(facecolor='#d95f02', shrink=0.05, width=1.5, headwidth=7),
                     fontsize=9, weight='bold')
    
    # 2. Selected Key States Incidence per 100k
    key_states = ['MH', 'DL', 'KL', 'KA', 'TN', 'UP']
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
    
    for state, color in zip(key_states, colors):
        st_data = df_panel[df_panel['State_Code'] == state].sort_values('Date')
        if len(st_data) > 0:
            name = st_data['State_Name'].iloc[0]
            ax2.plot(st_data['Date'], st_data['Confirmed_7D_Avg_Per_100k'], color=color, linewidth=2.0, label=f"{name} ({state})")
            
    ax2.set_title('(B) State-Level Incidence Heterogeneity (Confirmed Cases per 100,000 Population)', fontsize=13, weight='bold', pad=10)
    ax2.set_xlabel('Timeline (Date)', fontsize=11, weight='bold')
    ax2.set_ylabel('7-Day Avg Cases / 100k Pop', fontsize=11, weight='bold')
    ax2.grid(True)
    ax2.legend(loc='upper left', ncol=3, frameon=True, facecolor='white', framealpha=0.9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    plt.tight_layout()
    output_path = FIGURES_DIR / "fig1_epidemic_waves_timeline.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Figure 1: {output_path}")

def plot_regression_fits(fitted_curves):
    """Figure 2: Polynomial vs Ridge Regression Model Fits for Selected States."""
    states_to_plot = ['MH', 'DL', 'KL', 'KA']
    waves = ['Wave 1', 'Wave 2']
    
    fig, axes = plt.subplots(len(states_to_plot), len(waves), figsize=(14, 11), sharex=False)
    
    for i, state in enumerate(states_to_plot):
        for j, wave in enumerate(waves):
            ax = axes[i, j]
            if wave in fitted_curves and state in fitted_curves[wave]:
                data = fitted_curves[wave][state]
                t = np.arange(len(data['actual']))
                
                ax.plot(t, data['actual'], 'o-', color='#555555', markersize=3, alpha=0.6, label='Observed (7D Avg)')
                ax.plot(t, data['poly_fit'], '--', color='#d95f02', linewidth=2.2, 
                        label=f"Poly (deg {data['poly_degree']}): R²={data['poly_r2']:.2f}")
                ax.plot(t, data['ridge_fit'], ':', color='#2b5c8f', linewidth=2.5, 
                        label=f"Ridge (α={data['ridge_alpha']:.1f}): R²={data['ridge_r2']:.2f}")
                
                ax.set_title(f"{state} - {wave}", fontsize=11, weight='bold')
                ax.set_ylabel('Daily Cases', fontsize=9)
                ax.grid(True)
                ax.legend(fontsize=8, loc='upper left', frameon=True, framealpha=0.85)
                if i == len(states_to_plot) - 1:
                    ax.set_xlabel('Days from Wave Inception', fontsize=9, weight='bold')
            else:
                ax.text(0.5, 0.5, 'Data Unavailable', ha='center', va='center')
                
    fig.suptitle('Polynomial Regression vs Regularised (Ridge) Regression Case Growth Fits', fontsize=14, weight='bold', y=0.995)
    plt.tight_layout()
    output_path = FIGURES_DIR / "fig2_regression_fits.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Figure 2: {output_path}")

def plot_pca_biplot(df_results, pca_summary):
    """Figure 3: PCA Biplot with State Projections and Feature Loadings."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    
    # Scatter points colored by KMeans Cluster
    clusters = df_results['KMeans_Cluster'].unique()
    colors = PALETTE[:len(clusters)]
    
    for clus_id, color in zip(sorted(clusters), colors):
        sub = df_results[df_results['KMeans_Cluster'] == clus_id]
        archetype = sub['Cluster_Archetype'].iloc[0]
        ax.scatter(sub['PC1'], sub['PC2'], color=color, s=120, edgecolors='black', 
                   alpha=0.85, label=f"Cluster {clus_id}: {archetype}")
        
        # Label state codes
        for _, row in sub.iterrows():
            ax.annotate(row['State_Code'], (row['PC1'], row['PC2']),
                        xytext=(4, 4), textcoords='offset points',
                        fontsize=8, weight='bold', color='#222222')
            
    # Draw Feature Loadings Vectors
    loadings = pd.DataFrame(pca_summary['loadings'])
    scale_factor = 3.5  # Scale vectors for visual biplot balance
    
    for feat in loadings.index:
        x_arrow = loadings.loc[feat, 'PC1_Loading'] * scale_factor
        y_arrow = loadings.loc[feat, 'PC2_Loading'] * scale_factor
        
        ax.arrow(0, 0, x_arrow, y_arrow, color='#b2182b', alpha=0.75, width=0.03, head_width=0.12, length_includes_head=True)
        # Shorten feature labels for neat display
        clean_name = feat.replace('_Per_100k', '').replace('_Pct', '').replace('_', ' ')
        ax.text(x_arrow * 1.12, y_arrow * 1.12, clean_name, color='#7f0000', fontsize=8.5, weight='bold', ha='center')
        
    pc1_var = pca_summary['explained_variance_ratio'][0] * 100
    pc2_var = pca_summary['explained_variance_ratio'][1] * 100
    
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axvline(0, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_xlabel(f'Principal Component 1 ({pc1_var:.1f}% Variance Explained)', fontsize=11, weight='bold')
    ax.set_ylabel(f'Principal Component 2 ({pc2_var:.1f}% Variance Explained)', fontsize=11, weight='bold')
    ax.set_title('Principal Component Analysis (PCA) Biplot: State Clusters & Epidemic Vectors', fontsize=13, weight='bold', pad=12)
    ax.grid(True)
    ax.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=9)
    
    plt.tight_layout()
    output_path = FIGURES_DIR / "fig3_pca_biplot.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Figure 3: {output_path}")

def plot_hierarchical_dendrogram(clustering_summary, df_results):
    """Figure 4: Hierarchical Clustering Dendrogram (Ward Linkage)."""
    fig, ax = plt.subplots(figsize=(12, 7.5))
    
    linkage_mat = clustering_summary['linkage_matrix']
    labels = df_results['State_Name'].tolist()
    
    dendro = dendrogram(
        linkage_mat,
        labels=labels,
        ax=ax,
        leaf_rotation=90,
        leaf_font_size=9,
        color_threshold=linkage_mat[-2, 2] # Color branches according to optimal cuts
    )
    
    ax.set_title('Hierarchical Clustering Dendrogram of Indian States (Ward Linkage)', fontsize=13, weight='bold', pad=12)
    ax.set_ylabel('Euclidean Distance (Scaled Feature Space)', fontsize=11, weight='bold')
    ax.grid(axis='y')
    
    plt.tight_layout()
    output_path = FIGURES_DIR / "fig4_hierarchical_dendrogram.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Figure 4: {output_path}")

def plot_cluster_characteristics(clustering_summary):
    """Figure 5: Cluster Archetype Profile Metrics Comparison."""
    profiles = clustering_summary['cluster_profiles']
    
    categories = ['Wave 1 Peak / 100k', 'Wave 2 Peak / 100k', 'Delta Surge Ratio', 'Dose 1 Coverage %', 'Dose 2 Coverage %']
    
    df_plot = []
    for cid, prof in profiles.items():
        df_plot.append({
            'Cluster': f"Cluster {cid}\n({prof.get('archetype', 'Cluster')})",
            'Wave 1 Peak / 100k': prof['mean_w1_peak'],
            'Wave 2 Peak / 100k': prof['mean_w2_peak'],
            'Delta Surge Ratio': prof['mean_delta_surge'],
            'Dose 1 Coverage %': prof['mean_dose1_cov'],
            'Dose 2 Coverage %': prof['mean_dose2_cov']
        })
    df_plot = pd.DataFrame(df_plot)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Peak Cases Per 100k
    df_peaks = df_plot.melt(id_vars=['Cluster'], value_vars=['Wave 1 Peak / 100k', 'Wave 2 Peak / 100k'], 
                            var_name='Wave', value_name='Peak_per_100k')
    sns.barplot(data=df_peaks, x='Cluster', y='Peak_per_100k', hue='Wave', ax=axes[0], palette=['#2b5c8f', '#d95f02'])
    axes[0].set_title('Epidemic Wave Peak Intensities', fontsize=11, weight='bold')
    axes[0].set_ylabel('Peak Daily Cases / 100k', fontsize=10)
    axes[0].tick_params(axis='x', rotation=15)
    axes[0].grid(axis='y')
    
    # 2. Delta Surge Ratio
    sns.barplot(data=df_plot, x='Cluster', y='Delta Surge Ratio', ax=axes[1], palette=PALETTE[:len(df_plot)])
    axes[1].set_title('Wave 2 vs Wave 1 Surge Amplification', fontsize=11, weight='bold')
    axes[1].set_ylabel('Ratio (Wave 2 / Wave 1)', fontsize=10)
    axes[1].tick_params(axis='x', rotation=15)
    axes[1].grid(axis='y')
    
    # 3. Vaccine Coverage
    df_vac = df_plot.melt(id_vars=['Cluster'], value_vars=['Dose 1 Coverage %', 'Dose 2 Coverage %'], 
                           var_name='Vaccine Dose', value_name='Coverage_Pct')
    sns.barplot(data=df_vac, x='Cluster', y='Coverage_Pct', hue='Vaccine Dose', ax=axes[2], palette=['#4daf4a', '#377eb8'])
    axes[2].set_title('Vaccination Rollout Coverage', fontsize=11, weight='bold')
    axes[2].set_ylabel('Population Coverage (%)', fontsize=10)
    axes[2].tick_params(axis='x', rotation=15)
    axes[2].grid(axis='y')
    
    fig.suptitle('Comparative Profile of State Clusters across Epidemic and Vaccination Dimensions', fontsize=13, weight='bold')
    plt.tight_layout()
    output_path = FIGURES_DIR / "fig5_cluster_characteristics.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Figure 5: {output_path}")

def plot_sdg_preparedness_index(df_results):
    """
    Figure 6: SDG 3 (Health Resilience) & SDG 17 (Data/Vaccine Partnerships)
    Pandemic Preparedness Matrix.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # X axis: SDG 17 Vaccine Delivery & Partnership (Dose 1 Coverage %)
    # Y axis: SDG 3 Epidemiological Resilience (Normalized inverse of surge shock & fatality rate)
    x = df_results['Dose1_Coverage_Pct']
    # Resilience score: higher is better containment / lower surge shock
    y = 100 - (df_results['Delta_Surge_Ratio'] * 12 + df_results['Case_Fatality_Rate_Pct'] * 15)
    y = np.clip(y, 0, 100)
    
    clusters = df_results['KMeans_Cluster'].unique()
    colors = PALETTE[:len(clusters)]
    
    for clus_id, color in zip(sorted(clusters), colors):
        mask = df_results['KMeans_Cluster'] == clus_id
        sub = df_results[mask]
        ax.scatter(x[mask], y[mask], color=color, s=sub['Population'] / 1.5e6 + 50,
                   alpha=0.8, edgecolors='black', label=f"Cluster {clus_id}")
        
        for idx in sub.index:
            ax.annotate(df_results.loc[idx, 'State_Code'], (x[idx], y[idx]),
                        xytext=(3, 3), textcoords='offset points', fontsize=8, weight='bold')
            
    # Reference thresholds
    x_mid = 70.0
    y_mid = 50.0
    ax.axvline(x_mid, color='#888888', linestyle=':', linewidth=1.2)
    ax.axhline(y_mid, color='#888888', linestyle=':', linewidth=1.2)
    
    # Quadrant annotations
    ax.text(x_mid + 3, 95, "Quadrant I: High Resilience & Rapid Immunization (Target Model)", 
            fontsize=9.5, weight='bold', color='#1b5e20')
    ax.text(35, 95, "Quadrant II: Moderate Shock, Moderate Vaccine Rollout", 
            fontsize=9.5, weight='bold', color='#e65100')
    ax.text(35, 10, "Quadrant III: Severe Surge Shock & Lagging Vaccine Supply (SDG Priority)", 
            fontsize=9.5, weight='bold', color='#b71c1c')
    ax.text(x_mid + 3, 10, "Quadrant IV: High Surge Shock Mitigated by Accelerated Rollout", 
            fontsize=9.5, weight='bold', color='#0d47a1')
    
    ax.set_xlabel('SDG 17 Partnership Metric: Vaccine Delivery Coverage (Dose 1 % of Population)', fontsize=11, weight='bold')
    ax.set_ylabel('SDG 3 Health Resilience Score (0–100 containment index)', fontsize=11, weight='bold')
    ax.set_title('SDG 3 & SDG 17 Pandemic Preparedness Matrix across Indian States', fontsize=13, weight='bold', pad=12)
    ax.grid(True)
    ax.legend(loc='lower left', frameon=True, framealpha=0.9, fontsize=9)
    
    plt.tight_layout()
    output_path = FIGURES_DIR / "fig6_sdg_preparedness_index.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Figure 6: {output_path}")

def generate_all_figures(df_panel, fitted_curves, df_results, pca_summary, clustering_summary):
    """Executes figure generation pipeline."""
    plot_epidemic_waves_timeline(df_panel)
    plot_regression_fits(fitted_curves)
    plot_pca_biplot(df_results, pca_summary)
    plot_hierarchical_dendrogram(clustering_summary, df_results)
    plot_cluster_characteristics(clustering_summary)
    plot_sdg_preparedness_index(df_results)
    print("All 6 publication figures generated successfully in figures/ directory.")
