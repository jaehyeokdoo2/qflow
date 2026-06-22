"""Dataset generation and oracle reward functions for the 2D toy experiment.

Each reference sampler returns a tuple ``(data, rewards)`` where ``data`` has
shape ``(n, 2)`` and ``rewards`` has shape ``(n, 1)``.
"""

import numpy as np
import jax
import jax.numpy as jnp
from sklearn.datasets import make_moons, make_swiss_roll

# Domain bounds for the 2D action space.
DOMAIN_MIN, DOMAIN_MAX = -4.0, 4.0


# ============================================================================
# REFERENCE SAMPLERS (Returns Tuple: Data, Rewards)
# ============================================================================
def _sample_moons(n, noise=0.1):
    # 1. Generate using sklearn
    data, y = make_moons(n_samples=n, noise=noise)
    data = data.astype(np.float32)
    # 2. Transform coordinates: Scale * 2, Shift [-1, -0.2]
    data = data * 2.0 + np.array([-1.0, -0.2])

    # 3. Reference Reward: Class 1 (Bottom Moon) gets 1.0
    rewards = (y > 0.5).astype(np.float32)[:, None]

    return np.clip(data, DOMAIN_MIN, DOMAIN_MAX), rewards


def _sample_two_spirals(n):
    # 1. Latent 'n': 0 to 3pi (1.5 turns)
    n_half = n // 2
    n_val = np.sqrt(np.random.rand(n_half, 1)) * 3 * np.pi  # "n" in reference

    # 2. Reference Geometry: -cos(n)*n (Mirrored), add noise similar to reference (0.1)
    d1x = -np.cos(n_val) * n_val + np.random.randn(n_half, 1) * 0.1
    d1y = np.sin(n_val) * n_val + np.random.randn(n_half, 1) * 0.1

    # Arm 1 & Arm 2 (Mirrored)
    arm1 = np.hstack((d1x, d1y))
    arm2 = np.hstack((-d1x, -d1y))

    x = np.vstack((arm1, arm2)) / 3.0

    # 3. Reference Reward: 1 - n/10 (both arms share the same 'n' distribution)
    n_concat = np.concatenate([n_val, n_val], axis=0)
    rewards = np.clip(1.0 - n_concat / 10.0, 0.0, 1.0)

    return np.clip(x, DOMAIN_MIN, DOMAIN_MAX), rewards


def _sample_swissroll(n, noise=1.0):
    data, _ = make_swiss_roll(n_samples=n, noise=noise)
    data = data.astype(np.float32)[:, [0, 2]]
    data /= 5.0

    # Reference Reward: sum(data^2) / 9
    rewards = np.sum(data ** 2, axis=1, keepdims=True) / 9.0
    return np.clip(data, DOMAIN_MIN, DOMAIN_MAX), rewards


def _sample_eight_gaussians(n):
    scale = 4.0
    # Standard 8 centers around (0,0)
    centers = [
        (0, 1),
        (-1. / np.sqrt(2), 1. / np.sqrt(2)),
        (-1, 0),
        (-1. / np.sqrt(2), -1. / np.sqrt(2)),
        (0, -1),
        (1. / np.sqrt(2), -1. / np.sqrt(2)),
        (1, 0),
        (1. / np.sqrt(2), 1. / np.sqrt(2)),
    ]

    # Apply scale 4.0
    centers = np.array([(scale * x, scale * y) for x, y in centers])  # (8, 2)

    # 1. Choose centers
    indices = np.random.randint(0, 8, size=n)
    selected_centers = centers[indices]

    # 2. Add noise (0.5)
    noise = np.random.randn(n, 2) * 0.5
    points = selected_centers + noise

    # 3. Final scale (/ 1.414)
    points /= np.sqrt(2)

    # 4. Reference Reward: index / 7.0
    rewards = indices.astype(np.float32)[:, None] / 7.0

    return np.clip(points, DOMAIN_MIN, DOMAIN_MAX), rewards


def sample_from_dataset(dataset_type, n):
    """Sample ``n`` points from the named dataset, returning ``(data, rewards)``."""
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
# ORACLE REWARD FUNCTION (For Landscape/Noise Visualization)
# ============================================================================
def make_reward_fns(dataset_type):
    """Build oracle reward helpers for a given dataset.

    Returns a tuple ``(reward_fn_jnp, reward_fn_vec, reward_fn, plot_reward_landscape)``.
    """
    def reward_fn_jnp(a):
        a = jnp.asarray(a)
        mode = dataset_type
        in_domain = (a[0] >= DOMAIN_MIN) & (a[0] <= DOMAIN_MAX) & (a[1] >= DOMAIN_MIN) & (a[1] <= DOMAIN_MAX)
        if mode == 'eight_gaussians':
            # 1. Reconstruct Latent: Reverse the final division (points /= sqrt(2))
            a_scaled = a * jnp.sqrt(2)

            # 2. Define the 8 Centers (Scaled by 4.0)
            scale = 4.0
            centers = jnp.array([
                (0, 1),
                (-1. / jnp.sqrt(2), 1. / jnp.sqrt(2)),
                (-1, 0),
                (-1. / jnp.sqrt(2), -1. / jnp.sqrt(2)),
                (0, -1),
                (1. / jnp.sqrt(2), -1. / jnp.sqrt(2)),
                (1, 0),
                (1. / jnp.sqrt(2), 1. / jnp.sqrt(2)),
            ]) * scale

            # 3. Find Nearest Center
            dists = jnp.sum((a_scaled[None, :] - centers) ** 2, axis=1)  # Shape (8,)
            nearest_idx = jnp.argmin(dists)

            # 4. Reference Reward: index / 7.0
            val = nearest_idx.astype(jnp.float32) / 7.0
            return jnp.where(in_domain, val, 0.0)
        if mode == 'moons':
            a_orig = (a - jnp.array([-1.0, -0.2])) / 2.0
            dist_c0 = jnp.sum((a_orig - jnp.array([0.0, 0.0])) ** 2)
            dist_c1 = jnp.sum((a_orig - jnp.array([1.0, -0.5])) ** 2)
            val = jnp.where(dist_c1 < dist_c0, 1.0, 0.0)
            return jnp.where(in_domain, val, 0.0)

        if mode == 'two_spirals':
            # Reconstruct n ~ 3 * |a|
            n_est = 3.0 * jnp.sqrt(jnp.sum(a ** 2))
            val = jnp.clip(1.0 - n_est / 10.0, 0.0, 1.0)

            # Manifold mask for visualization clarity
            r_vals = jnp.linspace(0.0, 3.0 * jnp.pi, 100)
            pts = jnp.stack([-jnp.cos(r_vals) * r_vals, jnp.sin(r_vals) * r_vals], axis=1) / 3.0
            d = jnp.min(jnp.sum((pts - a) ** 2, axis=1))
            d2 = jnp.min(jnp.sum((-pts - a) ** 2, axis=1))
            mask = jnp.exp(-0.5 * jnp.minimum(d, d2) / (0.15 ** 2))
            return jnp.where(in_domain, val * mask, 0.0)

        if mode == 'swissroll':
            val = jnp.sum(a ** 2) / 9.0
            return jnp.where(in_domain, val, 0.0)

        return 0.0

    reward_fn_vec = jax.vmap(reward_fn_jnp, in_axes=(0,))

    def reward_fn(a):
        return float(reward_fn_jnp(jnp.asarray(a)))

    def plot_reward_landscape(ax, res=50):
        xs = np.linspace(DOMAIN_MIN, DOMAIN_MAX, res)
        ys = np.linspace(DOMAIN_MIN, DOMAIN_MAX, res)
        X, Y = np.meshgrid(xs, ys)
        grid = np.stack([X, Y], axis=-1).reshape(-1, 2)
        R = np.array(reward_fn_vec(jnp.asarray(grid))).reshape(X.shape)
        return ax.contourf(X, Y, R, levels=30, cmap='viridis', alpha=0.8)

    return reward_fn_jnp, reward_fn_vec, reward_fn, plot_reward_landscape
