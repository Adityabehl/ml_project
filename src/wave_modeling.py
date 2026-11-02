"""
COVID-19 Wave Analysis: Growth Modelling via Polynomial & Regularised (Ridge) Regression
Fits dynamic growth curves across epidemic waves, evaluates RMSE & R², and extracts wave kinetics.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from src.config import WAVE_PERIODS, PROCESSED_DATA_DIR

def fit_wave_regressions(df_panel, states=None):
    """
    Fits Polynomial and Ridge regressions for each state across Wave 1 and Wave 2.
    Evaluates RMSE, R2, optimal polynomial degree, and Ridge alpha parameter.
    
    Parameters:
    -----------
    df_panel : pd.DataFrame
        Cleaned state-level panel data.
    states : list, optional
        List of state codes. If None, fits for all states.
        
    Returns:
    --------
    df_metrics : pd.DataFrame
        Detailed regression performance metrics per state and wave.
    fitted_curves : dict
        Fitted curve predictions for plotting and dashboard visualization.
    """
    if states is None:
        states = [s for s in df_panel['State_Code'].unique() if s != 'TT']
        
    metrics_records = []
    fitted_curves = {}
    
    # Analyze both Wave 1 and Wave 2
    target_waves = ["Wave 1", "Wave 2"]
    
    for wave_name in target_waves:
        wave_cfg = WAVE_PERIODS[wave_name]
        start_date = pd.to_datetime(wave_cfg["start"])
        end_date = pd.to_datetime(wave_cfg["end"])
        
        fitted_curves[wave_name] = {}
        
        for state in states:
            df_state = df_panel[
                (df_panel['State_Code'] == state) & 
                (df_panel['Date'] >= start_date) & 
                (df_panel['Date'] <= end_date)
            ].sort_values('Date').copy()
            
            if len(df_state) < 20:
                continue
                
            y = df_state['Confirmed_7D_Avg'].values
            if y.max() <= 5:  # Skip tiny islands or negligible outbreaks for regression fit
                continue
                
            n_obs = len(y)
            # Normalized time index t in [0, 1] for numerical conditioning
            t_norm = (np.arange(n_obs) / float(n_obs - 1)).reshape(-1, 1)
            dates = df_state['Date'].dt.strftime('%Y-%m-%d').tolist()
            
            # 1. Polynomial Regression (Test degrees 2, 3, 4, 5 and select best R2)
            best_poly_r2 = -float('inf')
            best_poly_deg = 3
            best_poly_rmse = float('inf')
            best_poly_pred = None
            
            for deg in [2, 3, 4, 5]:
                poly = PolynomialFeatures(degree=deg, include_bias=True)
                X_poly = poly.fit_transform(t_norm)
                poly_model = LinearRegression()
                poly_model.fit(X_poly, y)
                y_poly_pred = np.clip(poly_model.predict(X_poly), 0, None)
                
                r2 = r2_score(y, y_poly_pred)
                rmse = np.sqrt(mean_squared_error(y, y_poly_pred))
                
                if r2 > best_poly_r2:
                    best_poly_r2 = r2
                    best_poly_deg = deg
                    best_poly_rmse = rmse
                    best_poly_pred = y_poly_pred
            
            # 2. Regularized Ridge Regression on Polynomial Features (degree 4)
            poly_ridge = PolynomialFeatures(degree=4, include_bias=True)
            X_ridge = poly_ridge.fit_transform(t_norm)
            alphas = np.logspace(-3, 3, 20)
            ridge_cv = RidgeCV(alphas=alphas, cv=5)
            ridge_cv.fit(X_ridge, y)
            
            best_alpha = float(ridge_cv.alpha_)
            y_ridge_pred = np.clip(ridge_cv.predict(X_ridge), 0, None)
            ridge_r2 = r2_score(y, y_ridge_pred)
            ridge_rmse = np.sqrt(mean_squared_error(y, y_ridge_pred))
            
            # Peak identification within wave
            peak_idx = int(np.argmax(y))
            peak_val = float(y[peak_idx])
            peak_date = dates[peak_idx]
            pop = df_state['Population'].iloc[0]
            peak_per_100k = (peak_val / pop) * 100000
            
            # Wave kinetics: growth duration (days to peak), decay duration
            growth_duration = peak_idx
            decay_duration = len(y) - peak_idx
            
            metrics_records.append({
                'Wave': wave_name,
                'State_Code': state,
                'State_Name': df_state['State_Name'].iloc[0],
                'Population': pop,
                'Observations': len(y),
                'Peak_Date': peak_date,
                'Peak_Daily_Cases': round(peak_val, 1),
                'Peak_Per_100k': round(peak_per_100k, 2),
                'Days_To_Peak': growth_duration,
                'Days_Post_Peak': decay_duration,
                # Polynomial Regression
                'Poly_Degree': best_poly_deg,
                'Poly_RMSE': round(best_poly_rmse, 2),
                'Poly_R2': round(best_poly_r2, 4),
                # Ridge Regression
                'Ridge_Alpha': round(best_alpha, 4),
                'Ridge_RMSE': round(ridge_rmse, 2),
                'Ridge_R2': round(ridge_r2, 4),
                # Normalized RMSE (NRMSE = RMSE / Range)
                'Poly_NRMSE': round(best_poly_rmse / (y.max() - y.min() + 1e-5), 4),
                'Ridge_NRMSE': round(ridge_rmse / (y.max() - y.min() + 1e-5), 4)
            })
            
            fitted_curves[wave_name][state] = {
                'dates': dates,
                'actual': [round(float(val), 1) for val in y],
                'poly_fit': [round(float(val), 1) for val in best_poly_pred],
                'ridge_fit': [round(float(val), 1) for val in y_ridge_pred],
                'poly_degree': int(best_poly_deg),
                'ridge_alpha': round(float(best_alpha), 4),
                'poly_r2': round(float(best_poly_r2), 4),
                'ridge_r2': round(float(ridge_r2), 4)
            }
            
    df_metrics = pd.DataFrame(metrics_records)
    
    # Save metrics to CSV
    metrics_path = PROCESSED_DATA_DIR / "regression_metrics.csv"
    df_metrics.to_csv(metrics_path, index=False)
    print(f"Regression metrics saved: {metrics_path} (Records: {len(df_metrics)})")
    
    return df_metrics, fitted_curves

def extract_state_wave_features(df_panel, df_metrics):
    """
    Extracts unified epidemiological and vaccination feature profile for each state
    to serve as the input matrix for PCA and Clustering.
    """
    states = [s for s in df_panel['State_Code'].unique() if s not in ['TT', 'UN']]
    state_features = []
    
    # Calculate state vaccination status at end of timeline (2021-10-31)
    df_latest = df_panel[df_panel['Date'] == df_panel['Date'].max()]
    
    for state in states:
        m_w1 = df_metrics[(df_metrics['State_Code'] == state) & (df_metrics['Wave'] == 'Wave 1')]
        m_w2 = df_metrics[(df_metrics['State_Code'] == state) & (df_metrics['Wave'] == 'Wave 2')]
        
        if len(m_w1) == 0 or len(m_w2) == 0:
            continue
            
        w1_peak_per_100k = m_w1['Peak_Per_100k'].values[0]
        w2_peak_per_100k = m_w2['Peak_Per_100k'].values[0]
        w1_days_to_peak = m_w1['Days_To_Peak'].values[0]
        w2_days_to_peak = m_w2['Days_To_Peak'].values[0]
        
        # Surge factor: Wave 2 intensity relative to Wave 1
        delta_surge_ratio = (w2_peak_per_100k + 1.0) / (w1_peak_per_100k + 1.0)
        
        # Cumulative incidence at end of study
        df_st = df_panel[df_panel['State_Code'] == state]
        total_incidence_per_100k = df_st['Cumulative_Per_100k'].max()
        case_fatality_proxy = (df_st['Cumulative_Deceased'].max() / (df_st['Cumulative_Confirmed'].max() + 1)) * 100
        
        # Vaccination coverage as of Oct 2021
        st_vac = df_latest[df_latest['State_Code'] == state]
        dose1_cov = st_vac['Dose1_Coverage_Pct'].values[0] if len(st_vac) > 0 else 0.0
        dose2_cov = st_vac['Dose2_Coverage_Pct'].values[0] if len(st_vac) > 0 else 0.0
        
        # Regression goodness of fit (average R2)
        mean_ridge_r2 = (m_w1['Ridge_R2'].values[0] + m_w2['Ridge_R2'].values[0]) / 2.0
        
        state_features.append({
            'State_Code': state,
            'State_Name': m_w1['State_Name'].values[0],
            'Population': int(m_w1['Population'].values[0]),
            'Wave1_Peak_Per_100k': round(float(w1_peak_per_100k), 2),
            'Wave2_Peak_Per_100k': round(float(w2_peak_per_100k), 2),
            'Wave1_Days_To_Peak': int(w1_days_to_peak),
            'Wave2_Days_To_Peak': int(w2_days_to_peak),
            'Delta_Surge_Ratio': round(float(delta_surge_ratio), 2),
            'Total_Incidence_Per_100k': round(float(total_incidence_per_100k), 2),
            'Case_Fatality_Rate_Pct': round(float(case_fatality_proxy), 2),
            'Dose1_Coverage_Pct': round(float(dose1_cov), 2),
            'Dose2_Coverage_Pct': round(float(dose2_cov), 2),
            'Mean_Ridge_R2': round(float(mean_ridge_r2), 4)
        })
        
    df_feat = pd.DataFrame(state_features)
    output_path = PROCESSED_DATA_DIR / "state_wave_metrics.csv"
    df_feat.to_csv(output_path, index=False)
    print(f"State wave metrics feature panel saved: {output_path} (States: {len(df_feat)})")
    
    return df_feat
