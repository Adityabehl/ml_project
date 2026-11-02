"""
COVID-19 Wave Analysis: Data Acquisition & Panel Harmonization
Handles raw download, sanitation, rolling smoothing, and panel dataset curation.
"""

import os
import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    CASES_URL,
    VACCINE_URL,
    STATE_CODE_TO_NAME,
    STATE_NAME_TO_CODE,
    STATE_POPULATION
)

def download_data():
    """Download raw datasets from COVID19-India static repository if not present."""
    cases_path = RAW_DATA_DIR / "state_wise_daily.csv"
    vaccine_path = RAW_DATA_DIR / "cowin_vaccine_data_statewise.csv"

    if not cases_path.exists():
        print(f"Downloading daily cases data from {CASES_URL}...")
        urllib.request.urlretrieve(CASES_URL, cases_path)
        print("Cases data downloaded successfully.")
    else:
        print(f"Daily cases data already cached at {cases_path}")

    if not vaccine_path.exists():
        print(f"Downloading vaccination data from {VACCINE_URL}...")
        urllib.request.urlretrieve(VACCINE_URL, vaccine_path)
        print("Vaccine data downloaded successfully.")
    else:
        print(f"Vaccine data already cached at {vaccine_path}")

    return cases_path, vaccine_path

def load_and_clean_cases(cases_path):
    """
    Load raw cases CSV, reshape from wide to long panel format,
    clean anomalies, and compute rolling averages.
    """
    df_raw = pd.read_csv(cases_path)
    
    # Parse dates
    df_raw['Date_YMD'] = pd.to_datetime(df_raw['Date_YMD'])
    
    # Valid state columns (excluding UN: Unassigned, and non-state fields)
    state_cols = [c for c in df_raw.columns if c in STATE_CODE_TO_NAME and c not in ['UN']]
    
    # Melt dataframe to long format
    id_vars = ['Date_YMD', 'Status']
    df_long = df_raw.melt(id_vars=id_vars, value_vars=state_cols, var_name='State_Code', value_name='Count')
    
    # Convert Count to numeric, handle errors
    df_long['Count'] = pd.to_numeric(df_long['Count'], errors='coerce').fillna(0)
    
    # Pivot Status to separate columns: Confirmed, Recovered, Deceased
    df_pivot = df_long.pivot_table(
        index=['Date_YMD', 'State_Code'],
        columns='Status',
        values='Count',
        aggfunc='first'
    ).reset_index()
    
    # Rename columns and ensure standard names
    df_pivot.rename(columns={
        'Date_YMD': 'Date',
        'Confirmed': 'Confirmed_Daily',
        'Recovered': 'Recovered_Daily',
        'Deceased': 'Deceased_Daily'
    }, inplace=True)
    
    for col in ['Confirmed_Daily', 'Recovered_Daily', 'Deceased_Daily']:
        if col not in df_pivot.columns:
            df_pivot[col] = 0
        # Rectify negative counts (sometimes retrospective adjustments)
        df_pivot[col] = df_pivot[col].clip(lower=0)
        
    df_pivot.sort_values(by=['State_Code', 'Date'], inplace=True)
    
    # Add State Name and Population
    df_pivot['State_Name'] = df_pivot['State_Code'].map(STATE_CODE_TO_NAME)
    df_pivot['Population'] = df_pivot['State_Code'].map(STATE_POPULATION).fillna(1000000).astype(int)
    
    # Calculate 7-day rolling averages and cumulative counts per state
    df_pivot['Confirmed_7D_Avg'] = df_pivot.groupby('State_Code')['Confirmed_Daily'].transform(
        lambda s: s.rolling(window=7, min_periods=1).mean()
    )
    df_pivot['Recovered_7D_Avg'] = df_pivot.groupby('State_Code')['Recovered_Daily'].transform(
        lambda s: s.rolling(window=7, min_periods=1).mean()
    )
    df_pivot['Deceased_7D_Avg'] = df_pivot.groupby('State_Code')['Deceased_Daily'].transform(
        lambda s: s.rolling(window=7, min_periods=1).mean()
    )
    
    df_pivot['Cumulative_Confirmed'] = df_pivot.groupby('State_Code')['Confirmed_Daily'].cumsum()
    df_pivot['Cumulative_Recovered'] = df_pivot.groupby('State_Code')['Recovered_Daily'].cumsum()
    df_pivot['Cumulative_Deceased'] = df_pivot.groupby('State_Code')['Deceased_Daily'].cumsum()
    
    # Incidence metrics per 100,000 population
    df_pivot['Confirmed_Per_100k'] = (df_pivot['Confirmed_Daily'] / df_pivot['Population']) * 100000
    df_pivot['Confirmed_7D_Avg_Per_100k'] = (df_pivot['Confirmed_7D_Avg'] / df_pivot['Population']) * 100000
    df_pivot['Cumulative_Per_100k'] = (df_pivot['Cumulative_Confirmed'] / df_pivot['Population']) * 100000
    
    # Active cases proxy
    df_pivot['Active_Cases_Proxy'] = (
        df_pivot['Cumulative_Confirmed'] - df_pivot['Cumulative_Recovered'] - df_pivot['Cumulative_Deceased']
    ).clip(lower=0)

    return df_pivot

def load_and_clean_vaccine(vaccine_path):
    """
    Load raw Cowin vaccine CSV, harmonize state names,
    and compute cumulative vaccination coverage.
    """
    df_vac = pd.read_csv(vaccine_path)
    
    # Parse dates (formats can be DD/MM/YYYY or YYYY-MM-DD)
    df_vac['Date'] = pd.to_datetime(df_vac['Updated On'], dayfirst=True, errors='coerce')
    df_vac = df_vac.dropna(subset=['Date'])
    
    # Clean state names
    df_vac['State'] = df_vac['State'].str.strip()
    
    # Map state name to state code
    df_vac['State_Code'] = df_vac['State'].map(STATE_NAME_TO_CODE)
    
    # If state is 'India', map to 'TT'
    df_vac.loc[df_vac['State'] == 'India', 'State_Code'] = 'TT'
    
    # Filter valid state codes
    df_vac = df_vac[df_vac['State_Code'].notnull()].copy()
    
    # Select key metrics
    keep_cols = [
        'Date', 'State_Code', 
        'Total Doses Administered', 
        'First Dose Administered', 
        'Second Dose Administered'
    ]
    
    df_vac = df_vac[keep_cols].rename(columns={
        'Total Doses Administered': 'Total_Doses',
        'First Dose Administered': 'Dose1_Administered',
        'Second Dose Administered': 'Dose2_Administered'
    })
    
    for c in ['Total_Doses', 'Dose1_Administered', 'Dose2_Administered']:
        df_vac[c] = pd.to_numeric(df_vac[c], errors='coerce').fillna(0)
        
    # Group by Date and State_Code (take max in case of multiple updates per day)
    df_vac_grouped = df_vac.groupby(['Date', 'State_Code'], as_index=False).max()
    
    return df_vac_grouped

def build_curated_panel():
    """
    Execute full pipeline: download, clean cases, clean vaccine, merge panel,
    and save curated dataset.
    """
    cases_path, vaccine_path = download_data()
    print("Processing COVID-19 epidemic cases data...")
    df_cases = load_and_clean_cases(cases_path)
    
    print("Processing CoWIN vaccination data...")
    df_vac = load_and_clean_vaccine(vaccine_path)
    
    print("Merging cases and vaccination streams into harmonized panel...")
    df_panel = pd.merge(
        df_cases,
        df_vac,
        on=['Date', 'State_Code'],
        how='left'
    )
    
    # Forward-fill vaccination counts (since vaccine reporting began Jan 2021)
    df_panel.sort_values(by=['State_Code', 'Date'], inplace=True)
    df_panel['Dose1_Administered'] = df_panel.groupby('State_Code')['Dose1_Administered'].ffill().fillna(0)
    df_panel['Dose2_Administered'] = df_panel.groupby('State_Code')['Dose2_Administered'].ffill().fillna(0)
    df_panel['Total_Doses'] = df_panel.groupby('State_Code')['Total_Doses'].ffill().fillna(0)
    
    # Calculate vaccination coverage rates (% of population)
    df_panel['Dose1_Coverage_Pct'] = ((df_panel['Dose1_Administered'] / df_panel['Population']) * 100).clip(0, 100)
    df_panel['Dose2_Coverage_Pct'] = ((df_panel['Dose2_Administered'] / df_panel['Population']) * 100).clip(0, 100)
    
    # Save panel to CSV
    output_path = PROCESSED_DATA_DIR / "cleaned_state_panel.csv"
    df_panel.to_csv(output_path, index=False)
    print(f"Curated state-level panel saved successfully: {output_path} (Shape: {df_panel.shape})")
    
    return df_panel

if __name__ == "__main__":
    df = build_curated_panel()
    print("Sample panel preview:")
    print(df[['Date', 'State_Code', 'State_Name', 'Confirmed_Daily', 'Confirmed_7D_Avg', 'Dose1_Coverage_Pct']].head(10))
