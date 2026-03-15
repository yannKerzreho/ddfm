import jax
import jax.numpy as jnp
import equinox as eqx
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
from sklearn.decomposition import PCA

from models.ltv_ssm import LTVStateSpace
from models.ltv_exposure import NCDEExposure, AsymmetricNCDEExposure
from models.bai_ng_models import BaiNgModels
from tools.getters_converters_tools import convert_decoder_to_numpy

class LTVDDFM(eqx.Module):
    """
    Unified LTV-DDFM Model.
    Supports symmetric/asymmetric loadings and various initialization/training protocols.
    """
    Lambda_base: jnp.ndarray
    exposure_model: eqx.Module
    ssm_engine: LTVStateSpace
    
    n_series: int = eqx.field(static=True)
    n_factors: int = eqx.field(static=True)
    target_name: str = eqx.field(static=True)

    def __init__(self, Lambda_base: jnp.ndarray, exposure_model: eqx.Module, ssm_engine: LTVStateSpace, target_name: str = "GDP"):
        self.Lambda_base = jnp.asarray(Lambda_base)
        self.exposure_model = exposure_model
        self.ssm_engine = ssm_engine
        self.n_series, self.n_factors = self.Lambda_base.shape
        self.target_name = target_name

    @classmethod
    def init_from_pca(cls, data: pd.DataFrame, n_factors: int, exposure_type: str = "asymmetric", target_name: str = "GDP", hidden_size: int = 8, key: jax.Array = jax.random.PRNGKey(0)):
        """Initialize Lambda_base from standard PCA."""
        # Standardize for PCA
        z = (data - data.mean()) / data.std()
        pca = PCA(n_components=n_factors)
        pca.fit(z.fillna(0)) # crude init
        Lambda_base = pca.components_.T # [N, K]
        
        n_series = data.shape[1]
        n_covariates = n_series # using all series as covariates for now 
        
        keys = jax.random.split(key, 2)
        
        if exposure_type == "asymmetric":
            target_idx = list(data.columns).index(target_name)
            exposure_model = AsymmetricNCDEExposure(n_covariates, n_series, n_factors, keys[0], hidden_size=hidden_size, target_indices=(target_idx,))
        else:
            exposure_model = NCDEExposure(n_covariates, n_series, n_factors, keys[0], hidden_size=hidden_size)
            
        ssm_engine = LTVStateSpace(n_factors, n_series, keys[1])
        return cls(Lambda_base, exposure_model, ssm_engine, target_name)

    @classmethod
    def init_from_dfm(cls, data: pd.DataFrame, n_factors: int, dfm_benchmark: BaiNgModels, exposure_type: str = "asymmetric", target_name: str = "GDP", hidden_size: int = 8, key: jax.Array = jax.random.PRNGKey(0)):
        """Initialize Lambda_base from a pre-fitted DFM (BaiNgModels)."""
        # Extract loadings from the DFM benchmark
        if hasattr(dfm_benchmark, 'decoder'):
            # It's a Neural DDFM model (obsolete but keeping for structure)
            bs, emission = convert_decoder_to_numpy(dfm_benchmark.decoder, dfm_benchmark.use_bias, dfm_benchmark.factor_oder)
            Lambda_base = emission[:, :n_factors]
        elif hasattr(dfm_benchmark, 'res') and hasattr(dfm_benchmark.res, 'model'):
            # It's a Real DFM (statsmodels DynamicFactor)
            # res.model.ssm['design'] has shape [N_subset, k_factors]
            design = dfm_benchmark.res.model.ssm['design']
            Lambda_base_subset = design[:, :n_factors]
            
            # Map back to full data shape [N_full, K]
            Lambda_base = np.zeros((data.shape[1], n_factors))
            # dmf_data used in fit was [y, X_subset]
            # Order in design matrix matches dfm_data columns
            dfm_cols = pd.concat([dfm_benchmark.y_all, dfm_benchmark.X_all if dfm_benchmark.model_name != "DFM" else dfm_benchmark.data.drop(dfm_benchmark.target_name, axis=1)], axis=1).columns
            # Actually, DFM implementation in BaiNgModels uses: dfm_data = pd.concat([y, X_subset], axis=1)
            # Let's just use the column names from that concat.
            # We know it's [target, feat_subset...]
            target_idx_data = list(data.columns).index(target_name)
            Lambda_base[target_idx_data] = Lambda_base_subset[0]
            
            # The rest of the loadings are for the subsetted features used in DFM
            dfm_cols = getattr(dfm_benchmark, 'dfm_cols', None)
            if dfm_cols is not None:
                feat_cols = [c for c in dfm_cols if c != target_name]
                for i, col in enumerate(feat_cols):
                    if col in data.columns:
                        idx = list(data.columns).index(col)
                        # i+1 because index 0 is the target
                        Lambda_base[idx] = Lambda_base_subset[i+1]
            elif hasattr(dfm_benchmark, 'data'):
                # Fallback if dfm_cols not present
                feat_cols = [c for c in dfm_benchmark.data.columns if c != target_name]
                for i, col in enumerate(feat_cols):
                    if i+1 >= Lambda_base_subset.shape[0]: break
                    if col in data.columns:
                        idx = list(data.columns).index(col)
                        Lambda_base[idx] = Lambda_base_subset[i+1]
        elif hasattr(dfm_benchmark, 'eigvect'):
            # PCA logic
            Lambda_feat = dfm_benchmark.eigvect
            target_idx = list(data.columns).index(target_name)
            Lambda_base = np.zeros((data.shape[1], n_factors))
            # If it's standard PCA, eigvect is [N, K]
            if Lambda_feat.shape[0] == data.shape[1]:
                Lambda_base = Lambda_feat
            else:
                # Targeted PCA logic
                Lambda_base[target_idx] = 0 # Target loading will be learned or taken from regression
                # ... existing logic for targeted PCA mapping ...
                feat_indices = [i for i in range(data.shape[1]) if i != target_idx]
                min_len = min(Lambda_feat.shape[0], len(feat_indices))
                idx_array = np.array(feat_indices[:min_len])
                Lambda_base[idx_array] = Lambda_feat[:min_len]
        else:
            # Fallback to PCA if not found
            z = (data - data.mean()) / data.std()
            pca = PCA(n_components=n_factors)
            pca.fit(z.fillna(0))
            Lambda_base = pca.components_.T
            
        n_series = data.shape[1]
        n_covariates = n_series
        
        keys = jax.random.split(key, 2)
        
        if exposure_type == "asymmetric":
            target_idx = list(data.columns).index(target_name)
            exposure_model = AsymmetricNCDEExposure(n_covariates, n_series, n_factors, keys[0], hidden_size=hidden_size, target_indices=(target_idx,))
        else:
            exposure_model = NCDEExposure(n_covariates, n_series, n_factors, keys[0], hidden_size=hidden_size)
            
        ssm_engine = LTVStateSpace(n_factors, n_series, keys[1])
        return cls(Lambda_base, exposure_model, ssm_engine, target_name)

    def get_loadings(self, times: jnp.ndarray, covariates: jnp.ndarray, inference: bool = True, key: Optional[jax.Array] = None) -> jnp.ndarray:
        delta_Lambda = self.exposure_model(times, covariates, inference=inference, key=key)
        return jnp.broadcast_to(self.Lambda_base, (times.shape[0], self.n_series, self.n_factors)) + delta_Lambda

    def __call__(self, times: jnp.ndarray, observations: jnp.ndarray, covariates: jnp.ndarray, inference: bool = True, key: Optional[jax.Array] = None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Full forward pass."""
        Lambda_t = self.get_loadings(times, covariates, inference, key)
        filtered_states, lls = self.ssm_engine.filter(observations, Lambda_t)
        
        # Predictions: Y_hat = Lambda_t @ z_t
        predictions = jax.vmap(lambda L, z: L @ z)(Lambda_t, filtered_states)
        return predictions, filtered_states, jnp.sum(lls)
