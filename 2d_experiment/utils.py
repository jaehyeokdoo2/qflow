"""Utility functions and constants for 2D toy experiment."""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from functools import partial

# Domain constants
DOMAIN_MIN = -3.0
DOMAIN_MAX = 3.0
N_DATA = 256

# Synthetic distribution targets
def swiss_roll_2d(n_samples, key):
    """Generate 2D Swiss roll distribution."""
    u = jrandom.uniform(key, (n_samples,)) * 3 * jnp.pi
    h = jrandom.uniform(jrandom.fold_in(key, 1), (n_samples,)) * 21 - 10.5
    x = u * jnp.cos(u)
    y = h
    return jnp.stack([x, y], axis=-1)

def two_spirals_2d(n_samples, key):
    """Generate 2D two spirals distribution."""
    t = jrandom.uniform(key, (n_samples,)) * 4 * jnp.pi
    noise = jrandom.normal(jrandom.fold_in(key, 1), (n_samples,)) * 0.1
    x = (t / (4 * jnp.pi) * 2 - 1) * jnp.cos(t) + noise
    y = (t / (4 * jnp.pi) * 2 - 1) * jnp.sin(t) + noise
    return jnp.stack([x, y], axis=-1)

def generate_target_data(dataset_type='swiss_roll', n_samples=256, key=None):
    """Generate target data based on dataset type."""
    if key is None:
        key = jrandom.PRNGKey(0)

    if dataset_type == 'swiss_roll':
        return swiss_roll_2d(n_samples, key)
    elif dataset_type == 'two_spirals':
        return two_spirals_2d(n_samples, key)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

# Neural network utilities
def init_mlp(layer_sizes, key):
    """Initialize MLP parameters."""
    params = []
    for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        k_w, k_b = jrandom.split(key)
        key = k_w
        w = jrandom.normal(k_w, (in_size, out_size)) * jnp.sqrt(2.0 / in_size)
        b = jnp.zeros(out_size)
        params.append({'w': w, 'b': b})
    return params

def apply_mlp(params, x, use_residual=False):
    """Apply MLP forward pass."""
    for i, layer in enumerate(params[:-1]):
        x = jnp.dot(x, layer['w']) + layer['b']
        x = jax.nn.relu(x)
    # Output layer (no activation)
    x = jnp.dot(x, params[-1]['w']) + params[-1]['b']
    return x

def fourier_time_embed(t, embed_dim=64):
    """Fourier time embedding for time-conditioned networks."""
    if t.ndim == 1:
        t = t[:, None]
    freqs = jnp.arange(embed_dim // 2) * (2.0 * jnp.pi)
    t_embed = jnp.concatenate([
        jnp.sin(t * freqs),
        jnp.cos(t * freqs)
    ], axis=-1)
    return t_embed

# Evaluation utilities
def compute_wd(samples, target_data, grid_size=50):
    """Compute Wasserstein distance between samples and target via grid-based approximation."""
    # Simple grid-based distance: count samples in each grid cell
    grid_x = jnp.linspace(DOMAIN_MIN, DOMAIN_MAX, grid_size)
    grid_y = jnp.linspace(DOMAIN_MIN, DOMAIN_MAX, grid_size)
    dx = (DOMAIN_MAX - DOMAIN_MIN) / grid_size

    # Bin samples and target
    bin_x_samples = ((samples[:, 0] - DOMAIN_MIN) / dx).astype(int)
    bin_y_samples = ((samples[:, 1] - DOMAIN_MIN) / dx).astype(int)
    bin_x_target = ((target_data[:, 0] - DOMAIN_MIN) / dx).astype(int)
    bin_y_target = ((target_data[:, 1] - DOMAIN_MIN) / dx).astype(int)

    # Count in each bin
    hist_samples = jnp.zeros((grid_size, grid_size))
    hist_target = jnp.zeros((grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            count_s = jnp.sum((bin_x_samples == i) & (bin_y_samples == j))
            count_t = jnp.sum((bin_x_target == i) & (bin_y_target == j))
            hist_samples = hist_samples.at[i, j].set(count_s)
            hist_target = hist_target.at[i, j].set(count_t)

    # Normalize histograms
    hist_samples = hist_samples / jnp.sum(hist_samples)
    hist_target = hist_target / jnp.sum(hist_target)

    # Simple L1 distance between histograms
    wd = jnp.sum(jnp.abs(hist_samples - hist_target)) / 2.0
    return wd

def compute_mmd(samples, target_data, bandwidth=0.1):
    """Compute Maximum Mean Discrepancy between samples and target."""
    # Kernel matrix for samples
    diff_s = samples[:, None, :] - samples[None, :, :]
    dist_s = jnp.sum(diff_s ** 2, axis=-1)
    k_ss = jnp.exp(-dist_s / (2 * bandwidth ** 2))

    # Kernel matrix for target
    diff_t = target_data[:, None, :] - target_data[None, :, :]
    dist_t = jnp.sum(diff_t ** 2, axis=-1)
    k_tt = jnp.exp(-dist_t / (2 * bandwidth ** 2))

    # Kernel matrix for cross terms
    diff_cross = samples[:, None, :] - target_data[None, :, :]
    dist_cross = jnp.sum(diff_cross ** 2, axis=-1)
    k_st = jnp.exp(-dist_cross / (2 * bandwidth ** 2))

    # MMD
    mmd = jnp.mean(k_ss) + jnp.mean(k_tt) - 2 * jnp.mean(k_st)
    return jnp.sqrt(jnp.maximum(mmd, 0.0))
