"""
COVID-19 Wave Analysis: Dimensionality Reduction (PCA) & Clustering Module
Performs PCA, K-Means clustering, and Hierarchical/Agglomerative clustering
on normalized epidemic curves and vaccination coverage profiles.
Evaluates Silhouette scores, Davies-Bouldin index, and generates cluster profiles.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, dendrogram
from src.config import PROCESSED_DATA_DIR

# Feature columns for clustering states by epidemic dynamics and vaccination preparedness
CLUSTER_FEATURES = [
    'Wave1_Peak_Per_100k',
    'Wave2_Peak_Per_100k',
    'Wave1_Days_To_Peak',
    'Wave2_Days_To_Peak',
    'Delta_Surge_Ratio',
    'Total_Incidence_Per_100k',
    'Case_Fatality_Rate_Pct',
    'Dose1_Coverage_Pct',
    'Dose2_Coverage_Pct'
]

def run_pca_and_clustering(df_features, random_state=42):
    """
    Executes standardization, PCA, optimal K-Means selection, and Hierarchical clustering.
    
    Parameters:
    -----------
    df_features : pd.DataFrame
        DataFrame of state wave metrics and vaccination rates.
    random_state : int
        Seed for reproducibility.
        
    Returns:
    --------
    df_results : pd.DataFrame
        Enriched DataFrame with cluster assignments, PCA coordinates, and silhouette values.
    pca_summary : dict
        PCA explained variance and feature loadings.
    clustering_summary : dict
        Silhouette scores across k, optimal k metrics, linkage matrix for dendrogram.
    """
    df = df_features.copy()
    X = df[CLUSTER_FEATURES].values
    
    # 1. Feature Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Principal Component Analysis (PCA)
    pca = PCA(n_components=min(len(CLUSTER_FEATURES), len(df) - 1))
    X_pca = pca.fit_transform(X_scaled)
    
    df['PC1'] = np.round(X_pca[:, 0], 3)
    df['PC2'] = np.round(X_pca[:, 1], 3)
    df['PC3'] = np.round(X_pca[:, 2], 3)
    
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    pca_loadings = pd.DataFrame(
        pca.components_[:3].T,
        index=CLUSTER_FEATURES,
        columns=['PC1_Loading', 'PC2_Loading', 'PC3_Loading']
    ).round(3)
    
    pca_summary = {
        'explained_variance_ratio': [round(float(v), 4) for v in explained_var[:5]],
        'cumulative_variance_ratio': [round(float(v), 4) for v in cumulative_var[:5]],
        'loadings': pca_loadings.to_dict()
    }
    
    # 3. K-Means Clustering & Silhouette Evaluation for k in [2, 6]
    silhouette_scores = {}
    db_scores = {}
    ch_scores = {}
    
    best_k = 3
    best_sil = -1.0
    
    k_range = range(2, min(7, len(df)))
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
        ch = calinski_harabasz_score(X_scaled, labels)
        
        silhouette_scores[k] = round(float(sil), 4)
        db_scores[k] = round(float(db), 4)
        ch_scores[k] = round(float(ch), 2)
        
        if sil > best_sil:
            best_sil = sil
            best_k = k
            
    # Fit chosen optimal K-Means
    km_optimal = KMeans(n_clusters=best_k, random_state=random_state, n_init=20)
    df['KMeans_Cluster'] = km_optimal.fit_predict(X_scaled)
    sample_sils = silhouette_samples(X_scaled, df['KMeans_Cluster'])
    df['Silhouette_Value'] = np.round(sample_sils, 3)
    
    # 4. Hierarchical / Agglomerative Clustering (Ward Linkage)
    linkage_matrix = linkage(X_scaled, method='ward', metric='euclidean')
    agg = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
    df['Hierarchical_Cluster'] = agg.fit_predict(X_scaled)
    hier_sil = silhouette_score(X_scaled, df['Hierarchical_Cluster'])
    
    # 5. Cluster Interpretation & Characterization
    cluster_profiles = {}
    for clus_id in range(best_k):
        sub = df[df['KMeans_Cluster'] == clus_id]
        cluster_profiles[int(clus_id)] = {
            'states': sub['State_Name'].tolist(),
            'state_codes': sub['State_Code'].tolist(),
            'count': len(sub),
            'mean_w1_peak': round(float(sub['Wave1_Peak_Per_100k'].mean()), 2),
            'mean_w2_peak': round(float(sub['Wave2_Peak_Per_100k'].mean()), 2),
            'mean_delta_surge': round(float(sub['Delta_Surge_Ratio'].mean()), 2),
            'mean_dose1_cov': round(float(sub['Dose1_Coverage_Pct'].mean()), 2),
            'mean_dose2_cov': round(float(sub['Dose2_Coverage_Pct'].mean()), 2),
            'mean_cfr': round(float(sub['Case_Fatality_Rate_Pct'].mean()), 2)
        }
        
    # Generate Descriptive Archetype Names based on profiles
    # Rank clusters by vaccination and surge metrics
    sorted_by_vax = sorted(cluster_profiles.keys(), key=lambda c: cluster_profiles[c]['mean_dose1_cov'])
    sorted_by_surge = sorted(cluster_profiles.keys(), key=lambda c: cluster_profiles[c]['mean_delta_surge'], reverse=True)
    
    for clus_id, prof in cluster_profiles.items():
        if prof['mean_delta_surge'] > 10.0:
            prof['archetype'] = "Late-Peak Outlier (Northeast Anomaly)"
        elif clus_id == sorted_by_vax[-1]:
            prof['archetype'] = "Proactive Immunization & High Case Detection"
        elif clus_id == sorted_by_vax[0]:
            prof['archetype'] = "Vulnerable Health System & Delayed Vaccine Uptake"
        else:
            prof['archetype'] = "Moderate Transmission & Balanced Rollout"
            
    df['Cluster_Archetype'] = df['KMeans_Cluster'].map(lambda cid: cluster_profiles[cid]['archetype'])
    
    # Save results to CSV
    output_path = PROCESSED_DATA_DIR / "clustering_results.csv"
    df.to_csv(output_path, index=False)
    print(f"Clustering results saved: {output_path} (Optimal k={best_k}, Silhouette={best_sil:.4f})")
    
    clustering_summary = {
        'optimal_k': best_k,
        'best_kmeans_silhouette': round(best_sil, 4),
        'hierarchical_silhouette': round(float(hier_sil), 4),
        'silhouette_by_k': silhouette_scores,
        'davies_bouldin_by_k': db_scores,
        'calinski_harabasz_by_k': ch_scores,
        'cluster_profiles': cluster_profiles,
        'linkage_matrix': linkage_matrix
    }
    
    return df, pca_summary, clustering_summary
