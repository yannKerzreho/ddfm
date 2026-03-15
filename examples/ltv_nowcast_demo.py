import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
from models.ltv_ddfm import LTVDDFM
from models.ltv_training import LTVTrainingManager

def test_ltv_nowcast():
    print("Starting LTV-DDFM Nowcast Demo...")
    
    # 1. Generate synthetic data (Monthly + quarterly GDP)
    T = 100
    N = 10
    K = 2
    
    # Random factors
    z = np.random.randn(T, K)
    # Random loadings
    L = np.random.randn(N, K)
    
    # Observables
    x = z @ L.T + 0.1 * np.random.randn(T, N)
    
    # Introduce NaNs (Ragged Edge)
    x[-5:, 1:] = np.nan # Last 5 months, only index 0 (GDP) might be missing less
    x[-10:, 0] = np.nan # GDP missing last 10 months (typical quarterly lag)
    
    # Monthly index
    dates = pd.date_range("2000-01-01", periods=T, freq="ME")
    df = pd.DataFrame(x, index=dates, columns=[f"V{i}" for i in range(N)])
    df.columns.values[0] = "GDP"
    
    # 2. Init Model from PCA (Asymmetric: only GDP is LTV)
    key = jax.random.PRNGKey(42)
    model = LTVDDFM.init_from_pca(df, n_factors=K, exposure_type="asymmetric", target_name="GDP", key=key)
    
    # 3. Training
    manager = LTVTrainingManager(model)
    
    times = jnp.arange(T, dtype=jnp.float32)
    obs = jnp.array(df.values)
    covariates = obs # simplified: use data itself as covariates
    
    # Target mask for GDP only
    target_mask = jnp.zeros(N)
    target_mask = target_mask.at[0].set(1.0)
    
    print("Running Hybrid Training...")
    model = manager.fit_hybrid(times, obs, covariates, target_mask=target_mask, em_iters=3, m_epochs=5)
    
    # 4. Filter / Smooth
    predictions, filtered_states, ll = model(times, obs, covariates)
    
    print(f"Final Log-Likelihood: {ll:.4f}")
    print("Predictions for GDP (last 5 months):")
    print(predictions[-5:, 0])
    
    assert not jnp.any(jnp.isnan(predictions))
    print("Success: LTV-DDFM successfully handled NaNs and produced nowcasts.")

if __name__ == "__main__":
    test_ltv_nowcast()
