import equinox as eqx
import jax
import jax.numpy as jnp
import diffrax
from typing import Optional, Tuple

class AbstractLTVExposure(eqx.Module):
    """Base class for LTV exposure models."""
    def __call__(self, times: jnp.ndarray, covariates: jnp.ndarray, inference: bool = True, key: Optional[jax.Array] = None) -> jnp.ndarray:
        raise NotImplementedError

class NCDEExposure(AbstractLTVExposure):
    """
    Time-varying factor loadings using Neural Controlled Differential Equations.
    """
    initial_hidden: eqx.nn.Linear
    vector_field: eqx.nn.MLP
    readout: eqx.nn.Linear
    
    n_factors: int = eqx.field(static=True)
    n_series: int = eqx.field(static=True)
    n_covariates: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    dropout: float = eqx.field(static=True)

    def __init__(self, n_covariates: int, n_series: int, n_factors: int, key: jax.Array, hidden_size: int = 8, dropout: float = 0.1):
        keys = jax.random.split(key, 3)
        self.n_factors = n_factors
        self.n_series = n_series
        self.n_covariates = n_covariates
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.initial_hidden = eqx.nn.Linear(n_covariates, hidden_size, key=keys[0])
        self.vector_field = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=hidden_size * n_covariates,
            width_size=32,
            depth=2,
            activation=jax.nn.tanh,
            key=keys[1]
        )
        self.readout = eqx.nn.Linear(hidden_size, n_series * n_factors, key=keys[2])
        
        # Initialize readout close to zero so it starts as a standard DFM
        self.readout = eqx.tree_at(lambda l: l.weight, self.readout, self.readout.weight * 0.001)
        self.readout = eqx.tree_at(lambda l: l.bias, self.readout, jnp.zeros_like(self.readout.bias))

    def _vf(self, t, y, args):
        control = args
        dX_dt = control.evaluate(t)
        matrix = self.vector_field(y).reshape(self.hidden_size, self.n_covariates)
        return matrix @ dX_dt

    def __call__(self, times: jnp.ndarray, covariates: jnp.ndarray, inference: bool = True, key: Optional[jax.Array] = None) -> jnp.ndarray:
        # Interpolate covariates for NCDE (fill NaNs to avoid propagation)
        cov_filled = jnp.where(jnp.isnan(covariates), 0.0, covariates)
        coeffs = diffrax.backward_hermite_coefficients(times, cov_filled)
        control = diffrax.CubicInterpolation(times, coeffs)
        
        y0 = self.initial_hidden(cov_filled[0])
        term = diffrax.ODETerm(self._vf)
        solver = diffrax.Dopri5()
        saveat = diffrax.SaveAt(ts=times)
        
        solution = diffrax.diffeqsolve(term, solver, t0=times[0], t1=times[-1], dt0=0.1, y0=y0, args=control, saveat=saveat)
        
        hidden_states = solution.ys
        if not inference and key is not None:
            mask = jax.random.bernoulli(key, 1 - self.dropout, (self.hidden_size,))
            hidden_states = hidden_states * (mask / (1 - self.dropout))
            
        out = jax.vmap(self.readout)(hidden_states)
        return out.reshape(len(times), self.n_series, self.n_factors)

class AsymmetricNCDEExposure(NCDEExposure):
    """
    Limits LTV adjustments to a subset of target indices (e.g. GDP).
    """
    target_indices: Tuple[int, ...] = eqx.field(static=True)
    n_targets: int = eqx.field(static=True)

    def __init__(self, n_covariates: int, n_series: int, n_factors: int, key: jax.Array, hidden_size: int = 8, target_indices: Tuple[int, ...] = (0,), dropout: float = 0.1):
        super().__init__(n_covariates, n_series, n_factors, key, hidden_size, dropout)
        self.target_indices = target_indices
        self.n_targets = len(target_indices)
        
        # Override readout to only output targets
        keys = jax.random.split(key, 4)
        self.readout = eqx.nn.Linear(hidden_size, self.n_targets * n_factors, key=keys[3])
        self.readout = eqx.tree_at(lambda l: l.weight, self.readout, self.readout.weight * 0.001)
        self.readout = eqx.tree_at(lambda l: l.bias, self.readout, jnp.zeros_like(self.readout.bias))

    def __call__(self, times: jnp.ndarray, covariates: jnp.ndarray, inference: bool = True, key: Optional[jax.Array] = None) -> jnp.ndarray:
        # Same logic as NCDE but scatter results
        cov_filled = jnp.where(jnp.isnan(covariates), 0.0, covariates)
        coeffs = diffrax.backward_hermite_coefficients(times, cov_filled)
        control = diffrax.CubicInterpolation(times, coeffs)
        y0 = self.initial_hidden(cov_filled[0])
        term = diffrax.ODETerm(self._vf)
        solver = diffrax.Dopri5()
        saveat = diffrax.SaveAt(ts=times)
        solution = diffrax.diffeqsolve(term, solver, t0=times[0], t1=times[-1], dt0=0.1, y0=y0, args=control, saveat=saveat)
        
        hidden_states = solution.ys
        if not inference and key is not None:
            mask = jax.random.bernoulli(key, 1 - self.dropout, (self.hidden_size,))
            hidden_states = hidden_states * (mask / (1 - self.dropout))
            
        out_targets = jax.vmap(self.readout)(hidden_states)
        out_targets = out_targets.reshape(len(times), self.n_targets, self.n_factors)
        
        # Scatter into full [T, N, K]
        full_delta = jnp.zeros((len(times), self.n_series, self.n_factors))
        full_delta = full_delta.at[:, jnp.array(self.target_indices), :].set(out_targets)
        return full_delta
