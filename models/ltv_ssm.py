import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Tuple, Optional

class LTVStateSpace(eqx.Module):
    """
    Differentiable Linear Time-Varying (LTV) State Space Model.
    Designed for robustness against missing data (NaNs) and ragged edges.
    """
    A_raw: jnp.ndarray      # Transition matrix [K, K]
    log_diag_Q: jnp.ndarray # Process noise [K]
    log_diag_R: jnp.ndarray # Observation noise [N]
    
    n_factors: int = eqx.field(static=True)
    n_series: int = eqx.field(static=True)
    max_spectral_radius: float = eqx.field(static=True)

    def __init__(self, n_factors: int, n_series: int, key: jax.Array, max_spectral_radius: float = 0.98):
        self.n_factors = n_factors
        self.n_series = n_series
        self.max_spectral_radius = max_spectral_radius
        
        # Initialize A to be stable
        self.A_raw = jnp.eye(n_factors) * 0.5
        self.log_diag_Q = jnp.log(jnp.ones(n_factors) * 0.1)
        self.log_diag_R = jnp.log(jnp.ones(n_series) * 0.5)

    @property
    def A(self) -> jnp.ndarray:
        # Spectral normalization to ensure stability
        sigma_max = jnp.linalg.norm(self.A_raw, ord=2)
        return self.max_spectral_radius * self.A_raw / (sigma_max + 1e-6)

    @property
    def Q(self) -> jnp.ndarray:
        return jnp.diag(jnp.exp(self.log_diag_Q))

    @property
    def R(self) -> jnp.ndarray:
        return jnp.diag(jnp.exp(self.log_diag_R))

    def filter(self, observations: jnp.ndarray, Lambda_t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Kalman Filter forward pass with NaN masking.
        observations: [T, N]
        Lambda_t: [T, N, K]
        """
        T, N = observations.shape
        K = self.n_factors
        
        z0 = jnp.zeros(K)
        P0 = jnp.eye(K) * 1.0
        
        A = self.A
        Q = self.Q
        R = self.R

        def step(carry, x):
            z_prev, P_prev = carry
            y_t, L_t = x
            
            # Predict
            z_pred = A @ z_prev
            P_pred = A @ P_prev @ A.T + Q
            
            # Prediction for observables
            y_pred = L_t @ z_pred
            v = y_t - y_pred # Innovation
            
            # Handle NaNs: mask missing observations
            mask = jnp.isnan(y_t)
            v = jnp.where(mask, 0.0, v)
            
            # Adjust L and R for missing data
            # If all are missing, F = R and K = 0
            F = L_t @ P_pred @ L_t.T + R
            
            # Kalman gain with regularization
            F_reg = F + 1e-4 * jnp.eye(N)
            KT = jax.scipy.linalg.solve(F_reg, L_t @ P_pred, assume_a='pos')
            KalmanGain = KT.T
            
            # Zero out gain for missing entries
            # mask has shape (N,), KalmanGain has shape (K, N)
            KalmanGain = jnp.where(mask[None, :], 0.0, KalmanGain)
            
            # Update
            z_upd = z_pred + KalmanGain @ v
            
            # Joseph form for numerical stability
            # P_upd = (I - K L) P_pred (I - K L).T + K R K.T
            I_KL = jnp.eye(K) - KalmanGain @ L_t
            P_upd = I_KL @ P_pred @ I_KL.T + KalmanGain @ R @ KalmanGain.T
            P_upd = 0.5 * (P_upd + P_upd.T) # Ensure symmetry
            
            # Log likelihood (only for non-missing)
            active_N = jnp.sum(~mask)
            # Use slogdet for stability
            sign, logdet_F = jnp.linalg.slogdet(F_reg)
            mahalanobis = v.T @ jax.scipy.linalg.solve(F_reg, v, assume_a='pos')
            step_ll = -0.5 * (active_N * jnp.log(2 * jnp.pi) + logdet_F + mahalanobis)
            
            return (z_upd, P_upd), (z_upd, step_ll)

        _, (states, lls) = jax.lax.scan(step, (z0, P0), (observations, Lambda_t))
        # Ensure LL is not NaN
        lls = jnp.nan_to_num(lls, nan=-1e6)
        return states, lls

    def filter_and_smooth(self, observations: jnp.ndarray, Lambda_t: jnp.ndarray):
        """
        RTS Smoother with NaN support.
        """
        T, N = observations.shape
        K = self.n_factors
        
        A = self.A
        Q = self.Q
        R = self.R
        
        z0 = jnp.zeros(K)
        P0 = jnp.eye(K)

        def forward(carry, x):
            z_prev, P_prev = carry
            y_t, L_t = x
            z_p = A @ z_prev
            P_p = A @ P_prev @ A.T + Q
            
            mask = jnp.isnan(y_t)
            v = jnp.where(mask, 0.0, y_t - L_t @ z_p)
            F = L_t @ P_p @ L_t.T + R
            F_reg = F + 1e-4 * jnp.eye(N)
            
            KT = jax.scipy.linalg.solve(F_reg, L_t @ P_p, assume_a='pos')
            KalmanGain = KT.T
            KalmanGain = jnp.where(mask[None, :], 0.0, KalmanGain)
            
            z_u = z_p + KalmanGain @ v
            # Joseph form
            I_KL = jnp.eye(K) - KalmanGain @ L_t
            P_u = I_KL @ P_p @ I_KL.T + KalmanGain @ R @ KalmanGain.T
            P_u = 0.5 * (P_u + P_u.T)
            
            active_N = jnp.sum(~mask)
            step_ll = -0.5 * (active_N * jnp.log(2*jnp.pi)) # simplified
            
            return (z_u, P_u), (z_p, P_p, z_u, P_u, step_ll)

        _, (z_pred_h, P_pred_h, z_filt, P_filt, lls) = jax.lax.scan(forward, (z0, P0), (observations, Lambda_t))

        def backward(carry, x):
            z_s_next, P_s_next = carry
            z_f, P_f, z_p_next, P_p_next = x
            
            J = jax.scipy.linalg.solve(P_p_next + 1e-6*jnp.eye(K), A @ P_f, assume_a='pos').T
            z_s = z_f + J @ (z_s_next - z_p_next)
            P_s = P_f + J @ (P_s_next - P_p_next) @ J.T
            P_cross = P_s_next @ J.T
            
            return (z_s, P_s), (z_s, P_s, P_cross)

        init_b = (z_filt[-1], P_filt[-1])
        xs_b = (z_filt[:-1], P_filt[:-1], z_pred_h[1:], P_pred_h[1:])
        _, (z_smooth_vals, P_smooth_vals, P_cross_vals) = jax.lax.scan(backward, init_b, xs_b, reverse=True)
        
        z_smooth = jnp.concatenate([z_smooth_vals, z_filt[-1:]], axis=0)
        P_smooth = jnp.concatenate([P_smooth_vals, P_filt[-1:]], axis=0)
        P_cross = jnp.concatenate([jnp.zeros((1, K, K)), P_cross_vals], axis=0)
        
        return z_smooth, P_smooth, P_cross, z_filt, lls
