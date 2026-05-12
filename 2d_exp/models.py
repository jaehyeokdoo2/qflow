"""
Network definitions and flow integration utilities for 2D toy experiments.

All networks are pure functional JAX MLPs.
"""
import jax
import jax.numpy as jnp
import jax.random as jrandom

from datasets import DOMAIN_MIN, DOMAIN_MAX


# ============================================================================
# MLP primitives
# ============================================================================

def init_mlp(layer_sizes, key, scale=0.02):
    """Xavier-style init with small scale to keep dynamics stable."""
    keys = jrandom.split(key, len(layer_sizes) - 1)
    params = []
    for k, (n_in, n_out) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:])):
        w = jrandom.normal(k, (n_in, n_out)) * scale
        b = jnp.zeros((n_out,))
        params.append({'w': w, 'b': b})
    return params


def apply_mlp(params, x, activation=jax.nn.relu, use_residual=False):
    for i, layer in enumerate(params):
        x = jnp.dot(x, layer['w']) + layer['b']
        if i < len(params) - 1:
            x = activation(x)
    if use_residual:
        x = x + x
    return x


# ============================================================================
# Time embedding
# ============================================================================

def fourier_time_embed(t, embed_dim=64, max_freq=512.0):
    """
    Fourier positional embedding for time.
    Maps scalar t in [0, 1] to a embed_dim-dimensional vector.

    Args:
        t: shape (batch, 1)
        embed_dim: must be even
    Returns:
        shape (batch, embed_dim)
    """
    half_dim = embed_dim // 2
    freqs = jnp.exp(jnp.linspace(0.0, jnp.log(max_freq), half_dim))
    args = t * freqs[None, :]
    return jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)


# ============================================================================
# Forward functions
# ============================================================================

def flow_forward(flow_params, x, t):
    """Flow network: input = [x, t], output = velocity (same dim as x)."""
    xt = jnp.concatenate([x, t], axis=-1)
    return apply_mlp(flow_params, xt)


def flow_forward_with_time_embed(flow_params, x, t, time_embed_dim=64):
    """Flow forward with Fourier time embedding."""
    t_embed = fourier_time_embed(t, embed_dim=time_embed_dim)
    xt = jnp.concatenate([x, t_embed], axis=-1)
    return apply_mlp(flow_params, xt)


def onestep_forward(policy_params, z):
    """One-step policy: maps noise z directly to action."""
    return apply_mlp(policy_params, z)


def critic_forward(critic_params, a, use_residual=False):
    """Outer critic Q(a): maps action to scalar value."""
    return apply_mlp(critic_params, a, use_residual=use_residual).squeeze(-1)


def inner_critic_forward(inner_params, x, t, use_residual=False):
    """Inner critic V(x, t): time-conditioned value function."""
    xt = jnp.concatenate([x, t], axis=-1)
    return apply_mlp(inner_params, xt, use_residual=use_residual).squeeze(-1)


# ============================================================================
# Flow integration
# ============================================================================

def integrate_flow_from(flow_params, x, t_start, steps=10, full_simulate=False):
    """Integrate flow starting at (x, t_start) to t=1."""
    def step_fn(carry, _):
        x_curr, t_curr = carry
        if full_simulate:
            dt = (1.0 - t_curr) / steps
        else:
            dt = jnp.minimum(1.0 - t_curr, 1.0 / steps)
        v = flow_forward(flow_params, x_curr, t_curr)
        x_next = x_curr + v * dt
        t_next = t_curr + dt
        x_next = jnp.clip(x_next, DOMAIN_MIN, DOMAIN_MAX)
        return (x_next, t_next), None

    (x_final, _), _ = jax.lax.scan(step_fn, (x, t_start), None, length=steps)
    return x_final


def integrate_flow(flow_params, z, steps=10):
    """Integrate flow from t=0 to t=1."""
    t0 = jnp.zeros((z.shape[0], 1))
    return integrate_flow_from(flow_params, z, t0, steps=steps)


def integrate_flow_with_time(flow_params, z, steps, t0=None):
    """Alias for integrate_flow (t0 ignored for API compatibility)."""
    return integrate_flow(flow_params, z, steps=steps)
