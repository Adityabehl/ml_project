"""
COVID-19 Wave Analysis: Panel Dataset Documentation & Dashboard Payload Exporter
Generates standard Data in Brief metadata, data dictionary, and interactive web dashboard JSON.
"""

import json
import pandas as pd
from pathlib import Path
from src.config import PROCESSED_DATA_DIR, DASHBOARD_DIR

DATA_DICTIONARY = {
    "title": "Cleaned State-Level COVID-19 Epidemic Waves and Vaccination Panel for India",
    "temporal_coverage": "2020-03-14 to 2021-10-31",
    "spatial_coverage": "36 States and Union Territories of India + National Total",
    "variables": [
        {"name": "Date", "type": "Date (YYYY-MM-DD)", "description": "Calendar observation date."},
        {"name": "State_Code", "type": "String", "description": "Two-letter state/UT postal abbreviation (e.g. MH, DL, KL, TT)."},
        {"name": "State_Name", "type": "String", "description": "Full official name of Indian State or Union Territory."},
        {"name": "Population", "type": "Integer", "description": "2021 projected population based on Census of India/MoHFW report."},
        {"name": "Confirmed_Daily", "type": "Float", "description": "Daily new lab-confirmed COVID-19 cases (cleaned of negative reconciliations)."},
        {"name": "Confirmed_7D_Avg", "type": "Float", "description": "7-day trailing rolling average of daily confirmed cases."},
        {"name": "Recovered_Daily", "type": "Float", "description": "Daily discharged/recovered COVID-19 cases."},
        {"name": "Deceased_Daily", "type": "Float", "description": "Daily confirmed COVID-19 fatalities."},
        {"name": "Cumulative_Confirmed", "type": "Float", "description": "Running total of cumulative confirmed infections."},
        {"name": "Confirmed_Per_100k", "type": "Float", "description": "Daily new confirmed cases scaled per 100,000 state residents."},
        {"name": "Confirmed_7D_Avg_Per_100k", "type": "Float", "description": "7-day rolling average cases per 100,000 residents."},
        {"name": "Cumulative_Per_100k", "type": "Float", "description": "Cumulative infection attack rate per 100,000 residents."},
        {"name": "Active_Cases_Proxy", "type": "Float", "description": "Proxy for active infectives (Cumulative - Recovered - Deceased)."},
        {"name": "Total_Doses", "type": "Float", "description": "Cumulative vaccine doses administered (CoWIN repository)."},
        {"name": "Dose1_Administered", "type": "Float", "description": "Cumulative first doses administered."},
        {"name": "Dose2_Administered", "type": "Float", "description": "Cumulative second doses administered (full primary course)."},
        {"name": "Dose1_Coverage_Pct", "type": "Float", "description": "Percentage of state population having received at least 1 dose."},
        {"name": "Dose2_Coverage_Pct", "type": "Float", "description": "Percentage of state population fully vaccinated with 2 doses."}
    ],
    "provenance": {
        "epidemiological_source": "COVID19-India volunteer tracker (https://data.covid19india.org/)",
        "vaccination_source": "Ministry of Health and Family Welfare (MoHFW) CoWIN portal via COVID19-India",
        "license": "Creative Commons Attribution 4.0 International (CC-BY 4.0)",
        "intended_uses": "Data in Brief, pandemic growth modeling, epidemiological clustering, health policy benchmark"
    }
}

def export_metadata_and_dashboard_payload(df_panel, df_metrics, df_results, pca_summary, clustering_summary, fitted_curves):
    """
    Exports data dictionary, JSON schema, and web dashboard payload.
    """
    # 1. Save Data Dictionary
    dict_path = PROCESSED_DATA_DIR / "data_dictionary.json"
    with open(dict_path, "w", encoding="utf-8") as f:
        json.dump(DATA_DICTIONARY, f, indent=2)
    print(f"Data dictionary exported: {dict_path}")
    
    # 2. Build Dashboard JSON payload
    # Downsample time series for high performance in browser
    # Extract list of available states
    states = sorted([s for s in df_results['State_Code'].unique()])
    
    # Timeline data for top 10 states and TT
    highlight_states = ['TT', 'MH', 'DL', 'KL', 'KA', 'TN', 'UP', 'GJ', 'WB', 'RJ', 'AP']
    timeline_dict = {}
    
    for st in highlight_states:
        sub = df_panel[df_panel['State_Code'] == st].sort_values('Date')
        timeline_dict[st] = {
            'dates': sub['Date'].dt.strftime('%Y-%m-%d').tolist()[::3], # every 3rd day for crisp fast rendering
            'confirmed_7d': [round(x, 1) for x in sub['Confirmed_7D_Avg'].tolist()[::3]],
            'confirmed_per_100k': [round(x, 2) for x in sub['Confirmed_7D_Avg_Per_100k'].tolist()[::3]],
            'dose1_cov': [round(x, 2) for x in sub['Dose1_Coverage_Pct'].tolist()[::3]],
            'dose2_cov': [round(x, 2) for x in sub['Dose2_Coverage_Pct'].tolist()[::3]]
        }
        
    dashboard_payload = {
        'meta': {
            'title': 'COVID-19 Wave Analysis & State Clustering Dashboard',
            'sdgs': ['SDG 3 (Good Health and Well-being)', 'SDG 17 (Partnerships for the Goals)'],
            'total_states': len(df_results),
            'optimal_k': clustering_summary['optimal_k'],
            'kmeans_silhouette': clustering_summary['best_kmeans_silhouette'],
            'hierarchical_silhouette': clustering_summary['hierarchical_silhouette'],
            'pca_explained_var': pca_summary['explained_variance_ratio']
        },
        'cluster_profiles': clustering_summary['cluster_profiles'],
        'state_results': df_results[[
            'State_Code', 'State_Name', 'Population', 'KMeans_Cluster', 'Cluster_Archetype',
            'PC1', 'PC2', 'PC3', 'Silhouette_Value', 'Wave1_Peak_Per_100k', 'Wave2_Peak_Per_100k',
            'Delta_Surge_Ratio', 'Dose1_Coverage_Pct', 'Dose2_Coverage_Pct', 'Case_Fatality_Rate_Pct'
        ]].to_dict(orient='records'),
        'regression_metrics': df_metrics.to_dict(orient='records'),
        'fitted_curves': fitted_curves,
        'timelines': timeline_dict,
        'pca_loadings': pca_summary['loadings']
    }
    
    dash_json_path = DASHBOARD_DIR / "data.json"
    with open(dash_json_path, "w", encoding="utf-8") as f:
        json.dump(dashboard_payload, f, indent=2)
    print(f"Web Dashboard payload exported: {dash_json_path}")
