"""
Dataset samplers and oracle reward functions for 2D toy experiments.
"""
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_swiss_roll

DOMAIN_MIN, DOMAIN_MAX = -4.0, 4.0


# ============================================================================
# Samplers (return (data, rewards) tuples)
# ============================================================================

def _sample_moons(n, noise=0.1):
    data, y = make_moons(n_samples=n, noise=noise)
    data = data.astype(np.float32)
    data = data * 2.0 + np.array([-1.0, -0.2])
    rewards = (y > 0.5).astype(np.float32)[:, None]
    return np.clip(data, DOMAIN_MIN, DOMAIN_MAX), rewards


def _sample_two_spirals(n):
    n_half = n // 2
    n_val = np.sqrt(np.random.rand(n_half, 1)) * 3 * np.pi
    d1x = -np.cos(n_val) * n_val + np.random.randn(n_half, 1) * 0.1
    d1y = np.sin(n_val) * n_val + np.random.randn(n_half, 1) * 0.1
    arm1 = np.hstack((d1x, d1y))
    arm2 = np.hstack((-d1x, -d1y))
    x = np.vstack((arm1, arm2)) / 3.0
    n_concat = np.concatenate([n_val, n_val], axis=0)
    rewards = np.clip(1.0 - n_concat / 10.0, 0.0, 1.0)
    return np.clip(x, DOMAIN_MIN, DOMAIN_MAX), rewards


def _sample_swissroll(n, noise=1.0):
    data, _ = make_swiss_roll(n_samples=n, noise=noise)
    data = data.astype(np.float32)[:, [0, 2]]
    data /= 5.0
    rewards = np.sum(data**2, axis=1, keepdims=True) / 9.0
    return np.clip(data, DOMAIN_MIN, DOMAIN_MAX), rewards


def _sample_eight_gaussians(n):
    scale = 4.0
    centers = [
        (0, 1),
        (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (-1, 0),
        (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        (0, -1),
        (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        (1, 0),
        (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
    ]
    centers = np.array([(scale * x, scale * y) for x, y in centers])
    indices = np.random.randint(0, 8, size=n)
    selected_centers = centers[indices]
    noise = np.random.randn(n, 2) * 0.5
    points = (selected_centers + noise) / np.sqrt(2)
    rewards = indices.astype(np.float32)[:, None] / 7.0
    return np.clip(points, DOMAIN_MIN, DOMAIN_MAX), rewards


def sample_from_dataset(n, dataset_type):
    """Sample n points from the specified dataset type."""
    if dataset_type == 'eight_gaussians':
        return _sample_eight_gaussians(n)
    if dataset_type == 'moons':
        return _sample_moons(n)
    if dataset_type == 'two_spirals':
        return _sample_two_spirals(n)
    if dataset_type == 'swissroll':
        return _sample_swissroll(n)
    raise ValueError(f"Unknown dataset_type: {dataset_type}")


# ============================================================================
# Oracle reward functions (JAX)
# ============================================================================

def reward_fn_jnp(a, dataset_type):
    """Per-sample reward function (JAX). Used for landscape plots."""
    a = jnp.asarray(a)
    in_domain = (
        (a[0] >= DOMAIN_MIN) & (a[0] <= DOMAIN_MAX) &
        (a[1] >= DOMAIN_MIN) & (a[1] <= DOMAIN_MAX)
    )
    if dataset_type == 'eight_gaussians':
        a_scaled = a * jnp.sqrt(2)
        scale = 4.0
        centers = jnp.array([
            (0, 1),
            (-1.0 / jnp.sqrt(2), 1.0 / jnp.sqrt(2)),
            (-1, 0),
            (-1.0 / jnp.sqrt(2), -1.0 / jnp.sqrt(2)),
            (0, -1),
            (1.0 / jnp.sqrt(2), -1.0 / jnp.sqrt(2)),
            (1, 0),
            (1.0 / jnp.sqrt(2), 1.0 / jnp.sqrt(2)),
        ]) * scale
        dists = jnp.sum((a_scaled[None, :] - centers) ** 2, axis=1)
        nearest_idx = jnp.argmin(dists)
        val = nearest_idx.astype(jnp.float32) / 7.0
        return jnp.where(in_domain, val, 0.0)

    if dataset_type == 'moons':
        a_orig = (a - jnp.array([-1.0, -0.2])) / 2.0
        dist_c0 = jnp.sum((a_orig - jnp.array([0.0, 0.0])) ** 2)
        dist_c1 = jnp.sum((a_orig - jnp.array([1.0, -0.5])) ** 2)
        val = jnp.where(dist_c1 < dist_c0, 1.0, 0.0)
        return jnp.where(in_domain, val, 0.0)

    if dataset_type == 'two_spirals':
        n_est = 3.0 * jnp.sqrt(jnp.sum(a ** 2))
        val = jnp.clip(1.0 - n_est / 10.0, 0.0, 1.0)
        r_vals = jnp.linspace(0.0, 3.0 * jnp.pi, 100)
        pts = jnp.stack([-jnp.cos(r_vals) * r_vals, jnp.sin(r_vals) * r_vals], axis=1) / 3.0
        d = jnp.min(jnp.sum((pts - a) ** 2, axis=1))
        d2 = jnp.min(jnp.sum((-pts - a) ** 2, axis=1))
        mask = jnp.exp(-0.5 * jnp.minimum(d, d2) / (0.15 ** 2))
        return jnp.where(in_domain, val * mask, 0.0)

    if dataset_type == 'swissroll':
        val = jnp.sum(a ** 2) / 9.0
        return jnp.where(in_domain, val, 0.0)

    return 0.0


def make_reward_fns(dataset_type):
    """Return (reward_fn_scalar, reward_fn_batched) for a given dataset_type."""
    def _reward_jnp(a):
        return reward_fn_jnp(a, dataset_type)

    reward_fn_vec = jax.vmap(_reward_jnp, in_axes=(0,))

    def reward_fn(a):
        return float(_reward_jnp(jnp.asarray(a)))

    return reward_fn, reward_fn_vec


# ============================================================================
# Visualization helpers
# ============================================================================

def plot_reward_landscape(ax, dataset_type, res=50):
    xs = np.linspace(DOMAIN_MIN, DOMAIN_MAX, res)
    ys = np.linspace(DOMAIN_MIN, DOMAIN_MAX, res)
    X, Y = np.meshgrid(xs, ys)
    grid = np.stack([X, Y], axis=-1).reshape(-1, 2)
    _, reward_fn_vec = make_reward_fns(dataset_type)
    R = np.array(reward_fn_vec(jnp.asarray(grid))).reshape(X.shape)
    return ax.contourf(X, Y, R, levels=30, cmap='viridis', alpha=0.8)


def plot_dataset(x_1_data, rewards_data, dataset_type, save_dir):
    """Plot and save dataset samples."""
    import os
    plt.figure(figsize=(4, 4))
    plt.scatter(
        x_1_data[:, 0], x_1_data[:, 1],
        c=rewards_data, cmap='plasma', s=10, alpha=0.6, edgecolors='none'
    )
    plt.title(f"Dataset Samples\nMean reward: {rewards_data.mean():.3f}")
    plt.xlabel('Action dim 1')
    plt.ylabel('Action dim 2')
    plt.xlim(DOMAIN_MIN, DOMAIN_MAX)
    plt.ylim(DOMAIN_MIN, DOMAIN_MAX)
    plt.tight_layout()
    os.makedirs(f'{save_dir}/{dataset_type}', exist_ok=True)
    plt.savefig(f'{save_dir}/{dataset_type}/training_dataset.png', dpi=100, bbox_inches='tight')
    plt.close()
