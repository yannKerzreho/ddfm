import pandas as pd
import numpy as np
import os, sys
import jax
import jax.numpy as jnp
from sklearn.model_selection import TimeSeriesSplit
from pandas.tseries.offsets import MonthEnd

# Add parent dir to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.bai_ng_models import BaiNgModels
from models.ltv_ddfm import LTVDDFM
from models.ltv_training import LTVTrainingManager

def run_benchmark():
    print("Starting LTV-DDFM vs Bai & Ng Benchmarks...")
    
    # 1. Load Data (matching the notebook logic)
    parentdir = "/Users/yannkerzreho/Documents/PR/DDFM/"
    data_m_all = pd.read_csv(f"{parentdir}data/mdfred_snapshot_monthly.csv")
    transform_code_m = data_m_all.iloc[0, 1:].copy()
    data_m = data_m_all.iloc[1:, :].set_index("sasdate")
    data_m = data_m.apply(pd.to_numeric, errors='coerce')
    data_m.index = pd.to_datetime(data_m.index) + MonthEnd()

    data_q_all = pd.read_csv(f"{parentdir}data/mdfred_snapshot_quarterly.csv")
    transform_code_q = data_q_all.iloc[1, :].copy() # iloc[1] is the "transform" row
    data_q = data_q_all.iloc[2:, :].set_index("sasdate")
    data_q = data_q.apply(pd.to_numeric, errors='coerce')
    data_q.index = pd.to_datetime(data_q.index) + MonthEnd()

    gdp_q = data_q["GDPC1"]
    data_final = pd.concat([gdp_q, data_m], axis=1).sort_index()
    data_final = data_final.astype(np.float32) # Prevent int64 and match JAX float32
    
    transform_code_final = transform_code_m.copy()
    transform_code_final.loc["GDPC1"] = transform_code_q["GDPC1"]
    # Ensure no nans in transform code
    transform_code_final = transform_code_final.fillna(1).astype(int)

    # 2. Setup OOS Loop
    # re-estimate once a year, last 2 years for trial (increase for final)
    months_oos = 12 * 2 
    test_size = 12
    n_splits = int(months_oos / test_size)
    
    # Models to compare
    K = 3
    TARGET = "GDPC1"
    
    # Instantiate BaiNg helper for stationarity/preprocessing
    bai_ng_helper = BaiNgModels(data_final, transform_code_final,
                    staz=True, standardize=True, target_name=TARGET,
                    start_date=pd.Timestamp("1990-01-01"),
                    n_factors=K, model_name="PC")
    
    X_all = bai_ng_helper.X_all
    y_all = bai_ng_helper.y_all
    full_data = pd.concat([y_all, X_all], axis=1).astype(float) # Force float64

    print(f"Data shape: {full_data.shape}, dtypes: {full_data.dtypes.unique()}")
    if (full_data.dtypes == 'object').any():
        print("Warning: Object columns found!")

    # Identifty original NaNs in target (mixed freq) to re-mask for LTV-DDFM
    # Use data_final which was numeric and timestamp-indexed
    orig_target = data_final[TARGET].reindex(full_data.index)
    nan_mask = orig_target.isna()
    
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    
    results = {"PCA": [], "DFM": [], "LTV-DDFM": []}
    
    # 3. Main Loop
    for i, (train_index, test_index) in enumerate(tscv.split(full_data)):
        print(f"Split {i+1}/{n_splits}...")
        
        # Training window data
        train_df = full_data.iloc[train_index].copy()
        # Re-mask NaNs for LTV-DDFM training (mixed freq)
        train_df.loc[nan_mask.iloc[train_index], TARGET] = np.nan
        
        # Prediction window data
        test_df = full_data.iloc[test_index]
        
        # --- PCA Baseline ---
        pca_model = None
        try:
            print("\n  Fitting PCA...")
            pca_model = BaiNgModels(data_final, transform_code_final, 
                                   n_factors=K, model_name="PC", target_name=TARGET)
            pca_model.fit(X_all.iloc[train_index], y_all.iloc[train_index])
            # Predict over the test window
            y_pred_pca = pca_model.predict(X_all.iloc[list(train_index) + list(test_index)], y_all.iloc[list(train_index) + list(test_index)])
            results["PCA"].append(y_pred_pca.loc[test_df.index])
        except Exception as e:
            print(f"  PCA failed: {e}")

        # --- DFM Baseline (EM-based) ---
        dfm_benchmark_model = None
        try:
            print("\n  Fitting DFM (EM)...")
            dfm_benchmark_model = BaiNgModels(data_final, transform_code_final, 
                                             n_factors=K, model_name="DFM", target_name=TARGET)
            dfm_benchmark_model.fit(X_all.iloc[train_index], y_all.iloc[train_index])
            y_pred_dfm = dfm_benchmark_model.predict(X_all.iloc[list(train_index) + list(test_index)], y_all.iloc[list(train_index) + list(test_index)])
            results["DFM"].append(y_pred_dfm.loc[test_df.index])
        except Exception as e:
            print(f"  DFM failed: {e}")

        # --- LTV-DDFM Variants ---
        strategies = ["Hybrid", "Decoupled", "E2E", "Optimized"]
        for strategy in strategies:
            try:
                name = f"LTV-DDFM ({strategy})"
                if name not in results: results[name] = []
                
                # Use DFM or PCA for initialization
                init_model = dfm_benchmark_model if dfm_benchmark_model is not None else pca_model
                print(f"\n  Training {name}...")
                
                key = jax.random.PRNGKey(i)
                ltv_model = LTVDDFM.init_from_dfm(train_df, n_factors=K, dfm_benchmark=init_model, 
                                                  exposure_type="asymmetric", target_name=TARGET, 
                                                  hidden_size=8, key=key)
                
                training_mgr = LTVTrainingManager(ltv_model, lr=1e-4, val_split=0.1, patience=5)
                
                # Prepare JAX arrays
                times = jnp.arange(len(train_df), dtype=jnp.float32)
                obs = jnp.array(train_df.values, dtype=jnp.float32)
                covariates = obs
                
                target_mask_arr = jnp.zeros(obs.shape[1], dtype=jnp.float32)
                target_idx = list(train_df.columns).index(TARGET)
                target_mask_arr = target_mask_arr.at[target_idx].set(1.0)
                
                # Fit based on strategy
                if strategy == "Hybrid":
                    ltv_model = training_mgr.fit_hybrid(times, obs, covariates, target_mask=target_mask_arr, em_iters=20, m_epochs=100)
                elif strategy == "Decoupled":
                    # Decoupled needs "fixed" factors. We can take them from the init_model via BaiNgModels logic.
                    # Or simpler: just run 1 iter of E-step to get them.
                    delta_L = ltv_model.exposure_model(times, covariates)
                    Lambda_t = ltv_model.Lambda_base + delta_L
                    z_s, _, _, _, _ = ltv_model.ssm_engine.filter_and_smooth(obs, Lambda_t)
                    ltv_model, _ = training_mgr.fit_decoupled(times, obs, covariates, z_s, target_mask=target_mask_arr, max_epochs=100)
                elif strategy == "E2E":
                    ltv_model = training_mgr.fit_e2e(times, obs, covariates, target_mask=target_mask_arr, max_epochs=100)
                elif strategy == "Optimized":
                    # 1. First get states from a basic hybrid run (low iters)
                    delta_L = ltv_model.exposure_model(times, covariates)
                    Lambda_t = ltv_model.Lambda_base + delta_L
                    z_s, _, _, _, _ = ltv_model.ssm_engine.filter_and_smooth(obs, Lambda_t)
                    # 2. Tune weight decay on validation set
                    training_mgr.tune_hyperparameters(times, obs, covariates, z_s, target_mask=target_mask_arr, 
                                                       wd_candidates=[1e-2, 1e-1, 1.0, 10.0])
                    # 3. High-intensity E2E training
                    ltv_model = training_mgr.fit_e2e(times, obs, covariates, target_mask=target_mask_arr, max_epochs=100)
                
                # Nowcast over combined window
                combined_df = full_data.iloc[list(train_index) + list(test_index)].copy()
                combined_df.loc[test_df.index, TARGET] = np.nan
                
                full_obs = jnp.array(combined_df.values, dtype=jnp.float32)
                full_times = jnp.arange(len(full_obs), dtype=jnp.float32)
                
                predictions, _, _ = ltv_model(full_times, full_obs, full_obs)
                y_pred_ltv = pd.Series(np.array(predictions[:, target_idx]), index=combined_df.index)
                test_preds = y_pred_ltv.loc[test_df.index]
                
                if test_preds.isna().any():
                    test_preds = test_preds.fillna(0.0)
                
                results[name].append(test_preds)
                
            except Exception as e:
                print(f"  {name} failed split {i}: {e}")

    # 4. Evaluation
    summary = {}
    print("\nProcessing Results...")
    for name, preds_list in results.items():
        # Ensure preds_list is not empty before concatenating
        if not preds_list:
            summary[name] = np.nan
            continue
        
        all_preds = pd.concat(preds_list)
        # Match with actuals for the predicted dates
        y_true = y_all.loc[all_preds.index]
        
        # Mask NaNs in true data (the mixed-frequency gaps)
        mask = y_true.notna()
        if mask.any():
            rmse = np.sqrt(np.mean((all_preds[mask] - y_true[mask])**2))
            summary[name] = rmse
        else:
            summary[name] = np.nan
        
    print("\nBenchmark Results (RMSFE):")
    for name, val in summary.items():
        print(f"{name}: {val:.4f}" if not np.isnan(val) else f"{name}: nan")
    
    # Relative to PCA
    if "PCA" in summary and "LTV-DDFM" in summary and not np.isnan(summary["PCA"]):
        rel = summary['LTV-DDFM'] / summary['PCA']
        print(f"LTV-DDFM Relative RMSE: {rel:.4f}")

if __name__ == "__main__":
    run_benchmark()
