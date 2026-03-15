import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
from typing import Tuple, Optional, List

from models.ltv_ddfm import LTVDDFM

def loss_e2e(model: LTVDDFM, times, observations, covariates, target_mask=None, kf_ll_weight=0.05):
    """End-to-End loss: NLL of Kalman Filter + MSE on targets."""
    predictions, filtered_states, total_ll = model(times, observations, covariates)
    
    # MSE part
    v = observations - predictions
    if target_mask is not None:
        v = v * target_mask
    
    # Mask NaNs in MSE calculation
    v = jnp.where(jnp.isnan(observations), 0.0, v)
    mse = jnp.mean(jnp.square(v))
    
    return mse - kf_ll_weight * total_ll

def loss_decoupled(model: LTVDDFM, times, observations, covariates, z_fixed, target_mask=None):
    """Decoupled loss: match delta_Lambda * z_fixed to residuals."""
    Lambda_t = model.get_loadings(times, covariates)
    predictions = jax.vmap(lambda L, z: L @ z)(Lambda_t, z_fixed)
    
    v = observations - predictions
    if target_mask is not None:
        v = v * target_mask
    
    v = jnp.where(jnp.isnan(observations), 0.0, v)
    return jnp.mean(jnp.square(v))

class LTVTrainingManager:
    """Manager for LTV-DDFM training strategies with validation and early stopping."""
    
    optim: optax.GradientTransformation
    opt_state: optax.OptState

    def __init__(self, model: LTVDDFM, lr: float = 1e-3, weight_decay: float = 1e-2, val_split: float = 0.1, patience: int = 5):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.val_split = val_split
        self.patience = patience
        
        # Initialize optimizer and state
        self.optim = optax.adamw(self.lr, weight_decay=self.weight_decay)
        self.opt_state = self.optim.init(eqx.filter(self.model, eqx.is_array))

    def _init_optim(self, weight_decay):
        self.optim = optax.adamw(self.lr, weight_decay=weight_decay)
        self.opt_state = self.optim.init(eqx.filter(self.model, eqx.is_array))

    def _split_data(self, times, observations, covariates):
        """Chronologically split dataset into Train and Validation chunks."""
        T = times.shape[0]
        split_idx = int(T * (1.0 - self.val_split))
        
        train_data = (times[:split_idx], observations[:split_idx], covariates[:split_idx])
        val_data = (times[split_idx:], observations[split_idx:], covariates[split_idx:])
        return train_data, val_data, split_idx

    @eqx.filter_jit
    def step_e2e(self, model, opt_state, optim, times, observations, covariates, target_mask):
        loss_fn = lambda m: loss_e2e(m, times, observations, covariates, target_mask)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, next_opt_state = optim.update(grads, opt_state, model)
        next_model = eqx.apply_updates(model, updates)
        return next_model, next_opt_state, loss

    @eqx.filter_jit
    def step_decoupled(self, model, opt_state, optim, times, observations, covariates, z_fixed, target_mask):
        loss_fn = lambda m: loss_decoupled(m, times, observations, covariates, z_fixed, target_mask)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, next_opt_state = optim.update(grads, opt_state, model)
        next_model = eqx.apply_updates(model, updates)
        return next_model, next_opt_state, loss

    def tune_hyperparameters(self, times, observations, covariates, z_fixed, target_mask=None, 
                              wd_candidates=[1e-2, 1e-1, 1.0, 10.0]):
        """Finds best weight decay using decoupled strategy on validation set."""
        best_wd = self.weight_decay
        best_overall_val_loss = float('inf')
        
        # Save initial model state
        initial_model = self.model
        
        print(f"Tuning weight_decay over candidates: {wd_candidates}")
        for wd in wd_candidates:
            self.model = initial_model
            self._init_optim(wd)
            
            # Simple decoupled fit to test this WD
            # Use lower max_epochs for speed but enough to see difference
            trained_model, val_loss = self.fit_decoupled(times, observations, covariates, z_fixed, target_mask, max_epochs=30)
            print(f"  wd={wd:.4f} -> val_loss={val_loss:.6f}")
            
            if val_loss < best_overall_val_loss:
                best_overall_val_loss = val_loss
                best_wd = wd
        
        print(f"Selected best weight_decay: {best_wd}")
        self.weight_decay = best_wd
        self.model = initial_model
        self._init_optim(best_wd)
        return best_wd

    def fit_e2e(self, times, observations, covariates, target_mask=None, max_epochs: int = 100):
        """End-to-End training with early stopping."""
        train_data, val_data, _ = self._split_data(times, observations, covariates)
        t_t, obs_t, cov_t = train_data
        t_v, obs_v, cov_v = val_data
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        current_model = self.model
        current_opt_state = self.opt_state

        for epoch in range(max_epochs):
            current_model, current_opt_state, train_loss = self.step_e2e(
                current_model, current_opt_state, self.optim, t_t, obs_t, cov_t, target_mask
            )
            val_loss = loss_e2e(current_model, t_v, obs_v, cov_v, target_mask)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.model = current_model
                self.opt_state = current_opt_state
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= self.patience:
                break
        
        return self.model

    def fit_decoupled(self, times, observations, covariates, z_fixed, target_mask=None, max_epochs: int = 100):
        """Decoupled training with early stopping."""
        train_data, val_data, split_idx = self._split_data(times, observations, covariates)
        t_t, obs_t, cov_t = train_data
        t_v, obs_v, cov_v = val_data
        
        z_t = z_fixed[:split_idx]
        z_v = z_fixed[split_idx:]
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        current_model = self.model
        current_opt_state = self.opt_state

        for epoch in range(max_epochs):
            current_model, current_opt_state, train_loss = self.step_decoupled(
                current_model, current_opt_state, self.optim, t_t, obs_t, cov_t, z_t, target_mask
            )
            val_loss = loss_decoupled(current_model, t_v, obs_v, cov_v, z_v, target_mask)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.model = current_model
                self.opt_state = current_opt_state
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= self.patience:
                break
        
        return self.model, best_val_loss

    def fit_hybrid(self, times, observations, covariates, target_mask=None, em_iters=10, m_epochs=20, em_patience=3):
        """Hybrid EM: Analytical SSM update + Grad Descent Exposure update with early stopping."""
        best_val_loss = float('inf')
        iters_no_improve = 0
        best_model = self.model

        for i in range(em_iters):
            # E-step
            delta_L = self.model.exposure_model(times, covariates)
            Lambda_t = self.model.Lambda_base + delta_L
            z_s, P_s, P_c, z_f, lls = self.model.ssm_engine.filter_and_smooth(observations, Lambda_t)
            
            # M-step: Analytical SSM (A, Q, R)
            self.model = self.analytical_m_step(self.model, z_s, P_s, P_c, observations, Lambda_t)
            
            # M-step: Exposure (Decoupled Grad Descent)
            self.model, iter_val_loss = self.fit_decoupled(times, observations, covariates, z_s, target_mask, max_epochs=m_epochs)
            
            if iter_val_loss < best_val_loss:
                best_val_loss = iter_val_loss
                best_model = self.model
                iters_no_improve = 0
            else:
                iters_no_improve += 1
            
            print(f"EM Iter {i} complete. Val Loss: {iter_val_loss:.4f}")
            if iters_no_improve >= em_patience:
                print(f"EM early stopping at iteration {i}")
                break
                
        self.model = best_model
        return self.model

    def analytical_m_step(self, model, z_s, P_s, P_c, observations, Lambda_t):
        """Exact M-Step for A, Q, R."""
        T, K = z_s.shape
        N = observations.shape[1]
        
        # expectations
        z_t_z_t = jnp.einsum('tk,tl->tkl', z_s, z_s) + P_s
        z_t_z_tm1 = jnp.einsum('tk,tl->tkl', z_s[1:], z_s[:-1]) + P_c[1:]
        
        S11 = jnp.sum(z_t_z_t[:-1], axis=0)
        S10 = jnp.sum(z_t_z_tm1, axis=0)
        
        A_new = jax.scipy.linalg.solve(S11 + 1e-8*jnp.eye(K), S10.T, assume_a='pos').T
        A_new = jnp.nan_to_num(A_new)
        
        S00 = jnp.sum(z_t_z_t[1:], axis=0)
        Q_new = (S00 - A_new @ S10.T - S10 @ A_new.T + A_new @ S11 @ A_new.T) / (T - 1)
        Q_new = jnp.nan_to_num(Q_new)
        
        # R update with NaN masking
        y_z_L = jnp.einsum('tn,tk,tnk->tn', observations, z_s, Lambda_t)
        L_z_z_L = jnp.einsum('tnk,tkl,tnl->tn', Lambda_t, z_t_z_t, Lambda_t)
        
        # mean square error only where observations exist
        raw_R = jnp.square(observations) - 2 * y_z_L + L_z_z_L
        R_diag = jnp.nanmean(raw_R, axis=0)
        R_diag = jnp.clip(jnp.nan_to_num(R_diag, nan=0.1), 1e-6, 10.0)
        
        new_model = eqx.tree_at(
            lambda m: (m.ssm_engine.A_raw, m.ssm_engine.log_diag_Q, m.ssm_engine.log_diag_R),
            model,
            (A_new, jnp.log(jnp.diag(Q_new) + 1e-8), jnp.log(R_diag + 1e-8))
        )
        return new_model
