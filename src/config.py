"""
COVID-19 Wave Analysis: Growth Modelling and Clustering of Indian States
Configuration and Metadata Module
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FIGURES_DIR = BASE_DIR / "figures"
DASHBOARD_DIR = BASE_DIR / "dashboard"
PAPER_DIR = BASE_DIR / "paper"

# Ensure all directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, DASHBOARD_DIR, PAPER_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Remote URLs (COVID19-India static repository)
CASES_URL = "https://data.covid19india.org/csv/latest/state_wise_daily.csv"
VACCINE_URL = "https://data.covid19india.org/csv/latest/cowin_vaccine_data_statewise.csv"

# State Code to Name Mapping (ISO / Covid19India standard)
STATE_CODE_TO_NAME = {
    "AN": "Andaman and Nicobar Islands",
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CH": "Chandigarh",
    "CT": "Chhattisgarh",
    "DL": "Delhi",
    "DN": "Dadra and Nagar Haveli and Daman and Diu",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HP": "Himachal Pradesh",
    "HR": "Haryana",
    "JH": "Jharkhand",
    "JK": "Jammu and Kashmir",
    "KA": "Karnataka",
    "KL": "Kerala",
    "LA": "Ladakh",
    "LD": "Lakshadweep",
    "MH": "Maharashtra",
    "ML": "Meghalaya",
    "MN": "Manipur",
    "MP": "Madhya Pradesh",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OR": "Odisha",
    "PB": "Punjab",
    "PY": "Puducherry",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TG": "Telangana",
    "TN": "Tamil Nadu",
    "TR": "Tripura",
    "TT": "India Total",
    "UP": "Uttar Pradesh",
    "UT": "Uttarakhand",
    "WB": "West Bengal"
}

STATE_NAME_TO_CODE = {v: k for k, v in STATE_CODE_TO_NAME.items()}

# Projected 2021 Population for Indian States (UIDAI / Census Projections in Millions)
# Source: Report of the Technical Group on Population Projections (MoHFW)
STATE_POPULATION = {
    "AN": 400000,
    "AP": 52787000,
    "AR": 1533000,
    "AS": 35043000,
    "BR": 123083000,
    "CH": 1208000,
    "CT": 29436000,
    "DL": 20571000,
    "DN": 1079000,
    "GA": 1559000,
    "GJ": 69788000,
    "HP": 7394000,
    "HR": 29483000,
    "JH": 38471000,
    "JK": 13384000,
    "KA": 66845000,
    "KL": 35489000,
    "LA": 297000,
    "LD": 68000,
    "MH": 124437000,
    "ML": 3288000,
    "MN": 3165000,
    "MP": 84516000,
    "MZ": 1216000,
    "NL": 2190000,
    "OR": 45696000,
    "PB": 30339000,
    "PY": 1573000,
    "RJ": 79281000,
    "SK": 677000,
    "TG": 37725000,
    "TN": 76402000,
    "TR": 4071000,
    "TT": 1361343000,
    "UP": 230907000,
    "UT": 11392000,
    "WB": 98179000
}

# Epidemic Wave Temporal Segmentation (Indian Context)
WAVE_PERIODS = {
    "Wave 1": {
        "start": "2020-05-01",
        "end": "2020-12-31",
        "description": "First ancestral/wild-type wave peaking around Sep 2020"
    },
    "Wave 2": {
        "start": "2021-03-01",
        "end": "2021-07-31",
        "description": "Delta variant surge characterized by steep exponential growth peaking in May 2021"
    },
    "Full Timeline": {
        "start": "2020-03-14",
        "end": "2021-10-31",
        "description": "Full observation panel"
    }
}
