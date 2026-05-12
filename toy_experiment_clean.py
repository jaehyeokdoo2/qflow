import dotenv
# %load_ext dotenv
# %dotenv

# --- New Cell ---

import math
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
import matplotlib.pyplot as plt
from functools import partial
from sklearn.utils import shuffle as util_shuffle
from tqdm import tqdm, trange

# %matplotlib inline
plt.rcParams['figure.figsize'] = [10, 8]
plt.rcParams['figure.dpi'] = 100

print(f"Using JAX backend: {jax.default_backend()} on {jax.devices()[0]}")


# --- New Cell ---

# ============================================================================
# DATASET DISTRIBUTION: Where the "clean" samples come from
# ============================================================================
# Choose among: 'mixture_gaussians', 'eight_gaussians', 'moons',
#               'swissroll', 'rings', 'checkerboard', 'two_spirals'

from sklearn.datasets import make_moons, make_swiss_roll

DOMAIN_MIN, DOMAIN_MAX = -4.0, 4.0
DATASET_TYPE = 'eight_gaussians'  # change here to try other distributions

# Default mixture (old behavior)
DATASET_CENTERS = [
    np.array([0.6, 0.6]),
    np.array([-0.5, 0.7]),
    np.array([0.3, -0.4]),
]
DATASET_STDS = [0.15, 0.2, 0.25]
DATASET_WEIGHTS = [0.5, 0.3, 0.2]


def _sample_mixture(n):
    samples = []
    for _ in range(n):
        idx = np.random.choice(len(DATASET_CENTERS), p=DATASET_WEIGHTS)
        sample = np.random.randn(2) * DATASET_STDS[idx] + DATASET_CENTERS[idx]
        sample = np.clip(sample, DOMAIN_MIN, DOMAIN_MAX)
        samples.append(sample)
    return np.array(samples)


def _sample_eight_gaussians(n):
    # Matches reference: centers on radius 4, noise 0.5, final scaled by 1/sqrt(2)
    scale = 4.0
    centers = np.array([
        (0, 1), (-1/np.sqrt(2), 1/np.sqrt(2)), (-1, 0), (-1/np.sqrt(2), -1/np.sqrt(2)),
        (0, -1), (1/np.sqrt(2), -1/np.sqrt(2)), (1, 0), (1/np.sqrt(2), 1/np.sqrt(2)),
    ]) * scale
    noise = np.random.randn(n, 2) * 0.5
    idx = np.random.randint(0, 8, size=n)
    samples = centers[idx] + noise
    samples = samples / np.sqrt(2)
    return np.clip(samples, DOMAIN_MIN, DOMAIN_MAX)


def _sample_moons(n, noise=0.1):
    data, y = make_moons(n_samples=n, noise=noise)
    data = data.astype(np.float32)
    data = data * 2.0 + np.array([-1.0, -0.2])  # match reference transform
    return np.clip(data, DOMAIN_MIN, DOMAIN_MAX)


def _sample_swissroll(n, noise=1.0):
    data, _ = make_swiss_roll(n_samples=n, noise=noise)
    data = data[:, [0, 2]] / 5.0  # take x and z, scale down
    return np.clip(data, DOMAIN_MIN, DOMAIN_MAX)


def _sample_rings(n):
    # Four concentric rings with reference scaling and noise
    n4 = n3 = n2 = n // 4
    n1 = n - n4 - n3 - n2
    lin4 = np.linspace(0, 2*np.pi, n4, endpoint=False)
    lin3 = np.linspace(0, 2*np.pi, n3, endpoint=False)
    lin2 = np.linspace(0, 2*np.pi, n2, endpoint=False)
    lin1 = np.linspace(0, 2*np.pi, n1, endpoint=False)
    circ = []
    circ.append(np.stack([np.cos(lin4), np.sin(lin4)], axis=1))           # r=1
    circ.append(np.stack([np.cos(lin3)*0.75, np.sin(lin3)*0.75], axis=1)) # r=0.75
    circ.append(np.stack([np.cos(lin2)*0.5, np.sin(lin2)*0.5], axis=1))   # r=0.5
    circ.append(np.stack([np.cos(lin1)*0.25, np.sin(lin1)*0.25], axis=1)) # r=0.25
    samples = np.vstack(circ) * 3.0
    samples = util_shuffle(samples)
    samples += np.random.normal(scale=0.08, size=samples.shape)
    return np.clip(samples, DOMAIN_MIN, DOMAIN_MAX)


def _sample_checkerboard(n):
    x1 = np.random.rand(n) * 4 - 2
    x2_raw = np.random.rand(n) - np.random.randint(0, 2, n) * 2
    x2 = x2_raw + (np.floor(x1) % 2)
    samples = np.stack([x1, x2], axis=1) * 2.0
    samples += np.random.normal(scale=0.0, size=samples.shape)  # keep deterministic aside from base noise above
    return np.clip(samples, DOMAIN_MIN, DOMAIN_MAX)


def _sample_two_spirals(n):
    n_half = n // 2

    # Angle parameter r (spiral "time")
    r = np.linspace(0.0, 3.0 * np.pi, n_half)  # 1.5 turns, smooth coverage
    r = r.reshape(-1, 1)

    # Spiral 1
    x1 = r * np.cos(r)
    y1 = r * np.sin(r)

    # Spiral 2 = rotated by pi
    x2 = -x1
    y2 = -y1

    pts = np.vstack([
        np.hstack([x1, y1]),
        np.hstack([x2, y2]),
    ])

    # Normalize into dataset domain
    pts /= np.max(np.abs(pts)) / 3.5    # scale to ~[-3.5,3.5]
    pts += np.random.randn(*pts.shape) * 0.15  # noise

    return np.clip(pts, DOMAIN_MIN, DOMAIN_MAX)


def sample_from_dataset(n):
    mode = DATASET_TYPE
    if mode == 'mixture_gaussians':
        return _sample_mixture(n)
    if mode == 'eight_gaussians':
        return _sample_eight_gaussians(n)
    if mode == 'moons':
        return _sample_moons(n)
    if mode == 'swissroll':
        return _sample_swissroll(n)
    if mode == 'rings':
        return _sample_rings(n)
    if mode == 'checkerboard':
        return _sample_checkerboard(n)
    if mode == 'two_spirals':
        return _sample_two_spirals(n)
    raise ValueError(f"Unknown DATASET_TYPE: {mode}")

# ============================================================================
# REWARD FUNCTION: Multimodal reward over clean action space
# ============================================================================
# R(a) = sum of Gaussians at different locations (multimodal)

REWARD_PEAK_CENTERS = jnp.array([
    [0.6, 0.6],
    [-0.5, 0.5],
    [0.0, -0.6],
])
REWARD_PEAK_STDS = jnp.array([0.15, 0.2, 0.18])
REWARD_PEAK_WEIGHTS = jnp.array([1.0, 0.7, 0.5])


def reward_fn_jnp(a):
    """Reward implemented in JAX for vectorization/JIT.
    Zero outside dataset support for each distribution.
    """
    a = jnp.asarray(a)
    mode = DATASET_TYPE
    in_domain = (a[0] >= DOMAIN_MIN) & (a[0] <= DOMAIN_MAX) & (a[1] >= DOMAIN_MIN) & (a[1] <= DOMAIN_MAX)

    if mode == 'mixture_gaussians':
        diff = a - REWARD_PEAK_CENTERS
        dist2 = jnp.sum(diff**2, axis=1)
        gaussian = jnp.exp(-0.5 * dist2 / (REWARD_PEAK_STDS**2))
        weighted = gaussian * REWARD_PEAK_WEIGHTS
        val = jnp.sum(weighted) / jnp.sum(REWARD_PEAK_WEIGHTS)
        return jnp.where(in_domain, val, 0.0)

    if mode == 'eight_gaussians':
        # Match dataset labels: nearest center index / 7
        centers = jnp.array([
            (0, 1), (-1/jnp.sqrt(2), 1/jnp.sqrt(2)), (-1, 0), (-1/jnp.sqrt(2), -1/jnp.sqrt(2)),
            (0, -1), (1/jnp.sqrt(2), -1/jnp.sqrt(2)), (1, 0), (1/jnp.sqrt(2), 1/jnp.sqrt(2)),
        ]) * (4.0 / jnp.sqrt(2.0))
        dists = jnp.sum((centers - a) ** 2, axis=1)
        idx = jnp.argmin(dists)
        val = idx / 7.0
        return jnp.where(in_domain, val, 0.0)

    if mode == 'rings':
        # Match dataset piecewise energies by radius
        r2 = jnp.sum(a**2)
        val = jnp.where(r2 >= 8.5, 0.667,
                        jnp.where(r2 >= 5.0, 0.333,
                                  jnp.where(r2 >= 2.0, 1.0, 0.0)))
        return jnp.where(in_domain, val, 0.0)

    if mode == 'swissroll':
        val = (a[0]**2 + a[1]**2) / 9.0
        return jnp.where(in_domain, val, 0.0)

    if mode == 'checkerboard':
        x1 = a[0]
        judger = ((x1 > 0) & (x1 <= 2)) | (x1 <= -2)
        val = jnp.where(judger, 1.0, 0.0)
        return jnp.where(in_domain, val, 0.0)

    if mode == 'two_spirals':
        # Spiral definition (same as sampler)
        r_vals = jnp.linspace(0.0, 3.0 * jnp.pi, 400)
        x1 = r_vals * jnp.cos(r_vals)
        y1 = r_vals * jnp.sin(r_vals)
        x2 = -x1
        y2 = -y1

        # Stack into arrays
        pts1 = jnp.stack([x1, y1], axis=1)
        pts2 = jnp.stack([x2, y2], axis=1)

        # Distance to nearest spiral arm
        d1 = jnp.min(jnp.linalg.norm(pts1 - a, axis=1))
        d2 = jnp.min(jnp.linalg.norm(pts2 - a, axis=1))
        d = jnp.minimum(d1, d2)

        # Gaussian distance-based reward
        sigma = 0.15
        val = jnp.exp(-0.5 * (d / sigma) ** 2)
        return jnp.where(in_domain, val, 0.0)


    if mode == 'moons':
        val = jnp.where(a[1] > 0, 1.0, 0.0)
        return jnp.where(in_domain, val, 0.0)

    return 0.0

# Vectorized reward for batches
reward_fn_vec = jax.vmap(reward_fn_jnp, in_axes=(0,))


def reward_fn(a):
    """Python-friendly wrapper returning a float."""
    return float(reward_fn_jnp(jnp.asarray(a)))

# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

# Toggle to show/hide reward peak markers
MARK_REWARD_PEAKS = False

def plot_reward_landscape(ax, resolution=50):
    """Plot the reward function R(a) over action space"""
    xs = np.linspace(DOMAIN_MIN, DOMAIN_MAX, resolution)
    ys = np.linspace(DOMAIN_MIN, DOMAIN_MAX, resolution)
    X, Y = np.meshgrid(xs, ys)
    grid = np.stack([X, Y], axis=-1).reshape(-1, 2)
    R = np.array(reward_fn_vec(jnp.asarray(grid))).reshape(X.shape)
    im = ax.contourf(X, Y, R, levels=30, cmap='viridis', alpha=0.8)
    if MARK_REWARD_PEAKS and DATASET_TYPE == 'mixture_gaussians':
        for i, center in enumerate(REWARD_PEAK_CENTERS):
            ax.scatter([center[0]], [center[1]], c='red', s=100*REWARD_PEAK_WEIGHTS[i],
                       marker='*', zorder=10, label=f"Peak {i+1}" if i == 0 else None)
    ax.set_xlim(DOMAIN_MIN, DOMAIN_MAX)
    ax.set_ylim(DOMAIN_MIN, DOMAIN_MAX)
    ax.set_aspect('equal')
    return im

print("Setup complete!")
print(f"Domain: [{DOMAIN_MIN}, {DOMAIN_MAX}]²")
print(f"Reward: Multimodal with {len(REWARD_PEAK_CENTERS)} peaks")
for i, center in enumerate(REWARD_PEAK_CENTERS):
    print(f"  Peak {i+1}: center={np.array(center)}, σ={REWARD_PEAK_STDS[i]}, weight={REWARD_PEAK_WEIGHTS[i]}")
print(f"Dataset type: {DATASET_TYPE}")
if DATASET_TYPE == 'mixture_gaussians':
    print(f"  Mixture centers: {DATASET_CENTERS}")


# --- New Cell ---

# Sample from dataset and visualize
np.random.seed(42)
dataset_samples = sample_from_dataset(2000)
dataset_rewards = np.array([reward_fn(a) for a in dataset_samples])

# Also sample from noise (prior) for comparison
noise_samples = np.random.randn(2000, 2)
noise_rewards = np.array([reward_fn(a) for a in noise_samples])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Reward Landscape R(a)
ax1 = axes[0]
im1 = plot_reward_landscape(ax1)
fig.colorbar(im1, ax=ax1, label='Reward R(a)')
ax1.set_title('Reward Function R(a)\nDefined over clean action space')
ax1.set_xlabel('Action dim 1'); ax1.set_ylabel('Action dim 2')

# 2. Dataset Distribution (clean samples)
ax2 = axes[1]
# im2 = plot_reward_landscape(ax2)
scatter2 = ax2.scatter(dataset_samples[:, 0], dataset_samples[:, 1], 
                       c=dataset_rewards, cmap='plasma', s=10, alpha=0.6, edgecolors='none')
if DATASET_TYPE == 'mixture_gaussians':
    for i, center in enumerate(DATASET_CENTERS):
        ax2.scatter([center[0]], [center[1]], c='white', s=100, marker='x', linewidths=2)
ax2.set_title(f"Dataset: {DATASET_TYPE}\nMean reward: {dataset_rewards.mean():.3f}")
ax2.set_xlabel('Action dim 1'); ax2.set_ylabel('Action dim 2')

# 3. Noise Distribution (prior)
ax3 = axes[2]
# im3 = plot_reward_landscape(ax3)
scatter3 = ax3.scatter(noise_samples[:, 0], noise_samples[:, 1], 
                       c='gray', s=10, alpha=0.3, edgecolors='none')
ax3.set_title(f'Noise Samples N(0,I)\nMean reward: {noise_rewards.mean():.3f}')
ax3.set_xlabel('Action dim 1'); ax3.set_ylabel('Action dim 2')

plt.tight_layout()
plt.show()

# Reward statistics
print("\n" + "="*50)
print("REWARD STATISTICS")
print("="*50)
print(f"Dataset ({DATASET_TYPE}) samples: mean R = {dataset_rewards.mean():.4f}, max R = {dataset_rewards.max():.4f}")
print(f"Noise samples:   mean R = {noise_rewards.mean():.4f}, max R = {noise_rewards.max():.4f}")
print(f"\nFlow model goal: Learn to transform noise → dataset distribution")
print(f"Value model goal: Predict R(a_clean) from intermediate state (x_t, t)")


# --- New Cell ---

# ============================================================================
# Generate Training Data
# ============================================================================
# Dataset: clean samples x_1 from the dataset distribution
# Reward: R(x_1) evaluated on clean samples

np.random.seed(0)
key_data = jrandom.PRNGKey(0)

N_DATA = 10000
FLOW_STEPS = 32

# Clean samples from dataset distribution
x_1_data = sample_from_dataset(N_DATA)

# Rewards evaluated on clean samples (vectorized JAX → NumPy)
rewards_data = np.array(reward_fn_vec(jnp.asarray(x_1_data)))

print(f"Dataset: {N_DATA} clean samples from {DATASET_TYPE}")
print(f"Reward range: [{rewards_data.min():.4f}, {rewards_data.max():.4f}]")
print(f"Mean reward: {rewards_data.mean():.4f}")
print(f"High reward (>0.3): {(rewards_data > 0.3).mean():.2%}")


# --- New Cell ---

# ============================================================================
# MODEL DEFINITIONS (JAX)
# ============================================================================
# All networks are simple MLPs expressed as pure JAX functions.


def init_mlp(layer_sizes, key, scale=0.02):
    """Xavier-style init with small scale to keep dynamics stable."""
    keys = jrandom.split(key, len(layer_sizes) - 1)
    params = []
    for k, (n_in, n_out) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:])):
        w = jrandom.normal(k, (n_in, n_out)) * scale
        b = jnp.zeros((n_out,))
        params.append({'w': w, 'b': b})
    return params


def apply_mlp(params, x, activation=jax.nn.relu):
    for i, layer in enumerate(params):
        x = jnp.dot(x, layer['w']) + layer['b']
        if i < len(params) - 1:
            x = activation(x)
    return x


def fourier_time_embed(t, embed_dim=64, max_freq=1000.0):
    """
    Fourier positional embedding for time.
    Maps scalar t ∈ [0, 1] to a embed_dim-dimensional vector using sinusoidal features.
    
    Args:
        t: Time values of shape (batch, 1)
        embed_dim: Dimension of the embedding (must be even)
        max_freq: Maximum frequency for the Fourier features
        
    Returns:
        Embedding of shape (batch, embed_dim)
    """
    # Frequencies: log-spaced from 1 to max_freq
    half_dim = embed_dim // 2
    freqs = jnp.exp(jnp.linspace(0.0, jnp.log(max_freq), half_dim))
    # Apply frequencies to time: (batch, 1) * (half_dim,) -> (batch, half_dim)
    args = t * freqs[None, :]
    # Concatenate sin and cos: (batch, embed_dim)
    embedding = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
    return embedding


def flow_forward(flow_params, x, t):
    xt = jnp.concatenate([x, t], axis=-1)
    return apply_mlp(flow_params, xt)


def flow_forward_with_time_embed(flow_params, x, t, time_embed_dim=64):
    """Flow forward with Fourier time embedding."""
    t_embed = fourier_time_embed(t, embed_dim=time_embed_dim)
    xt = jnp.concatenate([x, t_embed], axis=-1)
    return apply_mlp(flow_params, xt)


def onestep_forward(policy_params, z):
    return apply_mlp(policy_params, z)


def critic_forward(critic_params, a):
    return apply_mlp(critic_params, a).squeeze(-1)


def inner_critic_forward(inner_params, x, t):
    xt = jnp.concatenate([x, t], axis=-1)
    return apply_mlp(inner_params, xt).squeeze(-1)


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
    t0 = jnp.zeros((z.shape[0], 1))
    return integrate_flow_from(flow_params, z, t0, steps=steps)


print("Models redefined in JAX!")
print("  - FlowPolicy / OneStep / Critics are functional MLPs")
print("  - integrate_flow_from(): integrates from (x, t) → t=1")
print("  - Optimizer: optax.adam")


# --- New Cell ---

# ============================================================================
# FQL ACTOR-CRITIC TRAINING (JAX)
# ============================================================================
# Components: FlowPolicy (teacher), OneStepPolicy, OuterCritic
# Policy improvement: maximize Q(policy(z))

batch_size = 1024

def plot_training_progress(flow_params, policy_params, critic_params, epoch, flow_steps, n_samples=512):
    """Plot the current model predictions during training."""
    from IPython.display import display, clear_output
    
    # Sample actions from flow and one-step policy
    z_eval = jrandom.normal(jrandom.PRNGKey(999), (n_samples, 2))
    flow_actions = np.array(integrate_flow(flow_params, z_eval, steps=flow_steps))
    policy_actions = np.array(onestep_forward(policy_params, z_eval))
    
    # Compute rewards
    flow_rewards = np.array(reward_fn_vec(jnp.asarray(flow_actions)))
    policy_rewards = np.array(reward_fn_vec(jnp.asarray(policy_actions)))
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # 2. One-Step Policy
    ax2 = axes[0]
    sc2 = ax2.scatter(policy_actions[:, 0], policy_actions[:, 1], c=policy_rewards, 
                      cmap='plasma', s=15, alpha=0.7, edgecolors='none')
    ax2.set_title(f'One-Step Policy (Epoch {epoch})\nMean R: {policy_rewards.mean():.4f}')
    ax2.set_xlabel('x₁'); ax2.set_ylabel('x₂')
    ax2.set_xlim(DOMAIN_MIN, DOMAIN_MAX); ax2.set_ylim(DOMAIN_MIN, DOMAIN_MAX)
    ax2.set_aspect('equal')
    
    # 3. Critic Q-values heatmap
    ax3 = axes[1]
    resolution = 50
    x = np.linspace(DOMAIN_MIN, DOMAIN_MAX, resolution)
    y = np.linspace(DOMAIN_MIN, DOMAIN_MAX, resolution)
    X, Y = np.meshgrid(x, y)
    positions = jnp.array(np.stack([X.ravel(), Y.ravel()], axis=-1))
    q_values = np.array(critic_forward(critic_params, positions)).reshape(resolution, resolution)
    im3 = ax3.imshow(q_values, extent=[DOMAIN_MIN, DOMAIN_MAX, DOMAIN_MIN, DOMAIN_MAX], 
                     origin='lower', cmap='viridis', aspect='equal')
    ax3.set_title(f'Critic Q(a) (Epoch {epoch})')
    ax3.set_xlabel('x₁'); ax3.set_ylabel('x₂')
    
    plt.tight_layout()
    plt.show()


def train_fql(x_1_data, rewards_data, epochs=2000, lr=1e-3, alpha=1.0, flow_steps=10, hidden=128, plot_every=None):
    """
    FQL-style actor-critic training (JAX) with jitted update step.
    1. Train flow teacher (BC via flow matching)
    2. Train outer critic Q(a) → R(a)
    3. Train one-step policy: distillation + Q-maximization
    
    Args:
        plot_every: If set, plot model predictions every `plot_every` epochs.
    """
    key = jrandom.PRNGKey(42)
    X1 = jnp.asarray(x_1_data)
    R = jnp.asarray(rewards_data)

    k1, k2, k3 = jrandom.split(key, 3)
    flow_params = init_mlp([3, hidden, hidden, 2], k1)
    policy_params = init_mlp([2, hidden, hidden, 2], k2)
    critic_params = init_mlp([2, hidden, hidden, 1], k3)

    flow_opt = optax.adam(lr)
    policy_opt = optax.adam(lr)
    critic_opt = optax.adam(lr)
    flow_opt_state = flow_opt.init(flow_params)
    policy_opt_state = policy_opt.init(policy_params)
    critic_opt_state = critic_opt.init(critic_params)

    @partial(jax.jit, static_argnames=("flow_steps", "alpha", "batch_size"))
    def step(state, key, flow_steps, alpha, batch_size):
        flow_params, policy_params, critic_params, flow_opt_state, policy_opt_state, critic_opt_state = state
        key, k_idx, k_z, k_t, k_pol = jrandom.split(key, 5)
        idx = jrandom.choice(k_idx, X1.shape[0], shape=(batch_size,), replace=False)
        x_1 = X1[idx]
        r = R[idx]

        z = jrandom.normal(k_z, (batch_size, 2))
        t = jrandom.uniform(k_t, (batch_size, 1))
        x_t = (1 - t) * z + t * x_1
        u_target = x_1 - z

        def flow_loss_fn(params):
            v_pred = flow_forward(params, x_t, t)
            return jnp.mean((v_pred - u_target) ** 2)

        flow_loss, flow_grads = jax.value_and_grad(flow_loss_fn)(flow_params)
        flow_updates, flow_opt_state = flow_opt.update(flow_grads, flow_opt_state, flow_params)
        flow_params = optax.apply_updates(flow_params, flow_updates)

        def critic_loss_fn(params):
            q_pred = critic_forward(params, x_1)
            return jnp.mean((q_pred - r) ** 2)

        critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(critic_params)
        critic_updates, critic_opt_state = critic_opt.update(critic_grads, critic_opt_state, critic_params)
        critic_params = optax.apply_updates(critic_params, critic_updates)

        z_policy = jrandom.normal(k_pol, (batch_size, 2))
        teacher_output = jax.lax.stop_gradient(integrate_flow(flow_params, z_policy, steps=flow_steps))

        def policy_loss_fn(params):
            policy_output = onestep_forward(params, z_policy)
            distill_loss = jnp.mean((policy_output - teacher_output) ** 2)
            q_at_policy = critic_forward(critic_params, policy_output)
            return alpha * distill_loss - jnp.mean(q_at_policy), (distill_loss, jnp.mean(q_at_policy))

        (policy_loss, (distill_loss, q_val)), policy_grads = jax.value_and_grad(policy_loss_fn, has_aux=True)(policy_params)
        policy_updates, policy_opt_state = policy_opt.update(policy_grads, policy_opt_state, policy_params)
        policy_params = optax.apply_updates(policy_params, policy_updates)

        logs = (
            flow_loss,
            critic_loss,
            distill_loss,
            q_val,
            policy_loss,
        )
        new_state = (flow_params, policy_params, critic_params, flow_opt_state, policy_opt_state, critic_opt_state)
        return new_state, (key, logs)

    history = {
        'flow_loss': [], 'critic_loss': [],
        'policy_distill': [], 'policy_q': [], 'policy_total': []
    }
    if plot_every:
        num_plots = epochs // plot_every
        fig, axes = plt.subplots(2, num_plots, figsize=(num_plots*5, 8))
    
    cnt = 0
    n_samples = 512
    # NOTE: Don't shadow the flow_steps parameter!
    state = (flow_params, policy_params, critic_params, flow_opt_state, policy_opt_state, critic_opt_state)
    for epoch in trange(epochs, desc="FQL (JAX)"):
        state, (key, logs) = step(state, key, flow_steps, alpha, batch_size)
        flow_loss, critic_loss, distill_loss, q_val, policy_loss = logs
        history['flow_loss'].append(float(flow_loss))
        history['critic_loss'].append(float(critic_loss))
        history['policy_distill'].append(float(distill_loss))
        history['policy_q'].append(float(q_val))
        history['policy_total'].append(float(policy_loss))
        
        # Plot intermediate results
        if plot_every is not None and (epoch + 1) % plot_every == 0:
            flow_params_curr, policy_params_curr, critic_params_curr, _, _, _ = state
            z_eval = jrandom.normal(jrandom.PRNGKey(999), (n_samples, 2))
            # FIX: Use _curr params instead of initial params
            flow_actions = np.array(integrate_flow(flow_params_curr, z_eval, steps=flow_steps))
            policy_actions = np.array(onestep_forward(policy_params_curr, z_eval))
            
            # Compute rewards
            flow_rewards = np.array(reward_fn_vec(jnp.asarray(flow_actions)))
            policy_rewards = np.array(reward_fn_vec(jnp.asarray(policy_actions)))
            
            ax = axes[0, cnt]
            sc2 = ax.scatter(policy_actions[:, 0], policy_actions[:, 1], c=policy_rewards, 
                            cmap='plasma', s=15, alpha=0.7, edgecolors='none')
            ax.set_title(f'One-Step Policy (Epoch {epoch + 1})\nMean R: {policy_rewards.mean():.4f}')
            ax.set_xlabel('x₁'); ax.set_ylabel('x₂')
            ax.set_xlim(DOMAIN_MIN, DOMAIN_MAX); ax.set_ylim(DOMAIN_MIN, DOMAIN_MAX)
            ax.set_aspect('equal')

            # Critic Q-values heatmap
            ax3 = axes[1, cnt]
            resolution = 50
            x = np.linspace(DOMAIN_MIN, DOMAIN_MAX, resolution)
            y = np.linspace(DOMAIN_MIN, DOMAIN_MAX, resolution)
            X, Y = np.meshgrid(x, y)
            positions = jnp.array(np.stack([X.ravel(), Y.ravel()], axis=-1))
            # FIX: Use critic_params_curr instead of critic_params
            q_values = np.array(critic_forward(critic_params_curr, positions)).reshape(resolution, resolution)
            im3 = ax3.imshow(q_values, extent=[DOMAIN_MIN, DOMAIN_MAX, DOMAIN_MIN, DOMAIN_MAX], 
                            origin='lower', cmap='viridis', aspect='equal')
            ax3.set_title(f'Critic Q(a) (Epoch {epoch + 1})')
            ax3.set_xlabel('x₁'); ax3.set_ylabel('x₂')

            cnt += 1

    flow_params, policy_params, critic_params, _, _, _ = state
    if plot_every is not None:
        plt.tight_layout()
        plt.show()
    return flow_params, policy_params, critic_params, history


# ============================================================================
# FBRAC_INNER ACTOR-CRITIC TRAINING  (JAX)
# ============================================================================
# Matching fbrac_inner.py actor_loss behavior
# Key: t=0 is NOISE, t=1 is CLEAN ACTION
# Policy improvement: V(x_t + v(x_t,t)*dt, t+dt) - lookahead with policy output

def plot_fbrac_training_progress(flow_params, inner_params, epoch, flow_steps, n_samples=512):
    """Plot the current model predictions during fbrac_inner training."""
    from IPython.display import display, clear_output
    
    # Sample actions from flow
    z_eval = jrandom.normal(jrandom.PRNGKey(999), (n_samples, 2))
    flow_actions = np.array(integrate_flow(flow_params, z_eval, steps=flow_steps))
    
    # Compute rewards
    flow_rewards = np.array(reward_fn_vec(jnp.asarray(flow_actions)))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Flow Policy samples
    ax1 = axes[0]
    sc1 = ax1.scatter(flow_actions[:, 0], flow_actions[:, 1], c=flow_rewards, 
                      cmap='plasma', s=15, alpha=0.7, edgecolors='none')
    ax1.set_title(f'Flow Policy (Epoch {epoch})\nMean R: {flow_rewards.mean():.4f}')
    ax1.set_xlabel('x₁'); ax1.set_ylabel('x₂')
    ax1.set_xlim(DOMAIN_MIN, DOMAIN_MAX); ax1.set_ylim(DOMAIN_MIN, DOMAIN_MAX)
    ax1.set_aspect('equal')
    
    # 2. Inner Critic V(x, t=0) - value at noise
    ax2 = axes[1]
    resolution = 50
    x = np.linspace(DOMAIN_MIN, DOMAIN_MAX, resolution)
    y = np.linspace(DOMAIN_MIN, DOMAIN_MAX, resolution)
    X, Y = np.meshgrid(x, y)
    positions = jnp.array(np.stack([X.ravel(), Y.ravel()], axis=-1))
    t_zero = jnp.zeros((resolution * resolution, 1))
    v_values_t0 = np.array(inner_critic_forward(inner_params, positions, t_zero)).reshape(resolution, resolution)
    im2 = ax2.imshow(v_values_t0, extent=[DOMAIN_MIN, DOMAIN_MAX, DOMAIN_MIN, DOMAIN_MAX], 
                     origin='lower', cmap='viridis', aspect='equal')
    ax2.set_title(f'Inner Critic V(x, t=0) (Epoch {epoch})')
    ax2.set_xlabel('x₁'); ax2.set_ylabel('x₂')
    
    # 3. Inner Critic V(x, t=1) - value at clean action
    ax3 = axes[2]
    t_one = jnp.ones((resolution * resolution, 1))
    v_values_t1 = np.array(inner_critic_forward(inner_params, positions, t_one)).reshape(resolution, resolution)
    im3 = ax3.imshow(v_values_t1, extent=[DOMAIN_MIN, DOMAIN_MAX, DOMAIN_MIN, DOMAIN_MAX], 
                     origin='lower', cmap='viridis', aspect='equal')
    ax3.set_title(f'Inner Critic V(x, t=1) (Epoch {epoch})')
    ax3.set_xlabel('x₁'); ax3.set_ylabel('x₂')
    
    plt.tight_layout()
    plt.show()

# Override fbrac_inner to distill from an outer critic (matches fbrac_inner.py behavior)

def train_fbrac_inner(x_1_data, rewards_data, epochs=2000, lr=1e-3, alpha=1.0, flow_steps=10, hidden=128, plot_every=None, fbrac_inner_policy_loss_type="grad_align", use_time_embed=False, time_embed_dim=64, max_time_penalty=0.0,time_weight=True):
    """
    FBRAC-Inner: Flow-based actor-critic with inner (time-conditioned) critic.
    
    Args:
        use_time_embed: If True, use Fourier time embedding for inner critic (not policy)
        time_embed_dim: Dimension of the Fourier time embedding (default 64)
    """
    key = jrandom.PRNGKey(123)
    X1 = jnp.asarray(x_1_data)
    R = jnp.asarray(rewards_data)

    k1, k2, k3, k4 = jrandom.split(key, 4)
    # Flow network input: x (2D) + time (1D) - always uses raw time
    flow_input_dim = 3
    # Inner critic input: x (2D) + time (1D or time_embed_dim)  
    inner_input_dim = 2 + (time_embed_dim if use_time_embed else 1)
    
    flow_params = init_mlp([flow_input_dim, hidden, hidden, 2], k1)
    inner_params = init_mlp([inner_input_dim, hidden, hidden, 1], k2)
    outer_params = init_mlp([2, hidden, hidden, 1], k3)
    
    # Define local forward functions
    # Policy (flow) always uses raw time
    def _flow_forward(params, x, t):
        return flow_forward(params, x, t)
    
    # Inner critic optionally uses time embedding
    def _inner_critic_forward(params, x, t):
        if use_time_embed:
            t_embed = fourier_time_embed(t, embed_dim=time_embed_dim)
            xt = jnp.concatenate([x, t_embed], axis=-1)
        else:
            xt = jnp.concatenate([x, t], axis=-1)
        return apply_mlp(params, xt).squeeze(-1)

    flow_opt = optax.adam(lr)
    inner_opt = optax.adam(lr)
    outer_opt = optax.adam(lr)
    flow_opt_state = flow_opt.init(flow_params)
    inner_opt_state = inner_opt.init(inner_params)
    outer_opt_state = outer_opt.init(outer_params)
    
    # Pre-create figure for grid plotting
    n_samples = 512
    if plot_every:
        num_plots = epochs // plot_every
        fig, axes = plt.subplots(2, num_plots, figsize=(num_plots*5, 8))
        cnt = 0

    dt_const = 1.0 / flow_steps

    distillation_batch_scale = 1

    # Local integrate functions that use the time-embed-aware forward
    def _integrate_flow_from(flow_params, x, t_start, steps=10):
        """Integrate flow starting at (x, t_start) to t=1 using local _flow_forward."""
        def step_fn(carry, _):
            x_curr, t_curr = carry
            dt = jnp.minimum(1.0 - t_curr, 1.0 / steps)
            v = _flow_forward(flow_params, x_curr, t_curr)
            x_next = x_curr + v * dt
            t_next = t_curr + dt
            x_next = jnp.clip(x_next, DOMAIN_MIN, DOMAIN_MAX)
            return (x_next, t_next), None
        (x_final, _), _ = jax.lax.scan(step_fn, (x, t_start), None, length=steps)
        return x_final
    
    def _integrate_flow(flow_params, z, steps=10):
        """Integrate flow from t=0 to t=1 using local _flow_forward."""
        t0 = jnp.zeros((z.shape[0], 1))
        return _integrate_flow_from(flow_params, z, t0, steps=steps)

    @partial(jax.jit, static_argnames=("flow_steps", "alpha", "batch_size"))
    def step(state, key, flow_steps, alpha, batch_size):
        flow_params, inner_params, outer_params, flow_opt_state, inner_opt_state, outer_opt_state = state
        key, k_idx, k_noise, k_t, k_pol1, k_pol2, _, _ = jrandom.split(key, 8)
        idx = jrandom.choice(k_idx, X1.shape[0], shape=(batch_size,), replace=False)
        x_1 = X1[idx]
        r = R[idx]

        # outer critic on clean actions
        def outer_loss_fn(params):
            q_pred = critic_forward(params, x_1)
            
            return jnp.mean((q_pred - r) ** 2)

        outer_loss, outer_grads = jax.value_and_grad(outer_loss_fn)(outer_params)
        outer_updates, outer_opt_state = outer_opt.update(outer_grads, outer_opt_state, outer_params)
        outer_params = optax.apply_updates(outer_params, outer_updates)

        # critic distillation (inner) from outer critic along flow + boundary
        x_0 = jrandom.normal(k_noise, (batch_size*distillation_batch_scale, 2))
        t = jrandom.uniform(k_t, (batch_size*distillation_batch_scale, 1), minval=0.0, maxval=1.0)
        x_t = (1 - t) * x_0 + t * x_1.repeat(distillation_batch_scale, axis=0)
        x_1_pred = _integrate_flow_from(flow_params, x_t, t, steps=flow_steps)

        time_penalty = max_time_penalty * (1.0 - t)
        target_r = jax.lax.stop_gradient(critic_forward(outer_params, x_1_pred))  - time_penalty # ODE simulation
        # one step forward
        # dt = jnp.minimum(1.0 - t, 1.0/flow_steps)
        # x_t_next = x_t + _flow_forward(flow_params, x_t, t) * dt
        # target_r = jax.lax.stop_gradient(_inner_critic_forward(inner_params, x_t_next, t + dt))

        # target_r = jax.lax.stop_gradient(critic_forward(outer_params, x_1)) # Optimal Transport

        def critic_loss_fn(params):
            v_pred = _inner_critic_forward(params, x_t, t)
            distill_loss = jnp.mean((v_pred - target_r) ** 2)
            # v_clean_pred = _inner_critic_forward(params, x_1, jnp.ones_like(t))
            # boundary_loss = jnp.mean((v_clean_pred - r) ** 2)
            return distill_loss, (distill_loss, 0.0)

        (critic_loss, (distill_loss, _)), critic_grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(inner_params)
        inner_updates, inner_opt_state = inner_opt.update(critic_grads, inner_opt_state, inner_params)
        inner_params = optax.apply_updates(inner_params, inner_updates)

        # flow update
        x_0_policy = jrandom.normal(k_pol1, (batch_size, 2))
        t_policy = jrandom.uniform(k_pol2, (batch_size, 1))
        x_t_policy = (1 - t_policy) * x_0_policy + t_policy * x_1
        u_target_policy = x_1 - x_0_policy
        if fbrac_inner_policy_loss_type == "grad_align":

            def flow_loss_fn(params):
                pred_vel = _flow_forward(params, x_t_policy, t_policy)
                # 1. BC Loss (Base Requirement)
                bc_loss = jnp.mean((pred_vel - u_target_policy) ** 2)

                # 2. Q-Maximization (Explicit Gradient Alignment)
                # We want the velocity vector to align with the spatial gradient of the Value function.
                # Objective: Maximize (v • ∇x V)
                
                def value_sum_fn(x_in):
                    # Sum for batch gradient computation
                    return jnp.sum(_inner_critic_forward(inner_params, x_in, t_policy))
                
                # Calculate ∇x V(x_t, t)
                # We use stop_gradient so the actor doesn't try to 'trick' the gradient calculation
                # It just treats the direction as a fixed target.
                grad_v = jax.grad(value_sum_fn)(x_t_policy)
                grad_v = jax.lax.stop_gradient(grad_v)

                # Alignment Loss: Minimize negative dot product
                # This pushes 'pred_vel' to rotate towards the direction of increasing value
                alignment = jnp.sum(pred_vel * grad_v, axis=-1)
                
                # Optional: Gradient Normalization
                # If the value function has steep cliffs, gradients can be huge. 
                # Normalizing helps balance with BC loss.
                # grad_norm = jnp.linalg.norm(grad_v, axis=-1, keepdims=True) + 1e-8
                # alignment = alignment / grad_norm
                
                q_loss = -jnp.mean(alignment)

                total_loss = alpha * bc_loss + q_loss
                return total_loss, (bc_loss, q_loss)
        elif fbrac_inner_policy_loss_type == "next_step":
            def flow_loss_fn(params):
                pred_vel = _flow_forward(params, x_t_policy, t_policy)
                bc_loss = jnp.mean((pred_vel - u_target_policy) ** 2)
                dist_to_end = 1.0 - t_policy
                step_dt = jnp.clip(dist_to_end, 0.0, 1.0 / flow_steps)
                x_next_pred = x_t_policy + pred_vel * step_dt
                q_next = critic_forward(outer_params, x_next_pred)
                q_loss = -jnp.mean(q_next/step_dt)
                
                total_loss = alpha * bc_loss + q_loss
                return total_loss, (bc_loss, q_loss)
        elif fbrac_inner_policy_loss_type == "full":
            def flow_loss_fn(params):
                # 1. BC Loss (flow matching)
                v_pred = _flow_forward(params, x_t_policy, t_policy)
                bc_loss = jnp.mean((v_pred - u_target_policy) ** 2)

                # 2. Q-Maximization: integrate flow and maximize Q at the end
                # Sample fresh noise for policy evaluation
                z_policy = jrandom.normal(k4, (batch_size, 2))
                
                # Differentiable flow integration from t=0 to t=1
                def integrate_for_grad(flow_p, x_start, n_steps):
                    """Differentiable flow integration."""
                    dt = 1.0 / n_steps
                    x = x_start
                    t_curr = jnp.zeros((x.shape[0], 1))
                    for _ in range(n_steps):
                        v = _flow_forward(flow_p, x, t_curr)
                        x = x + v * dt
                        t_curr = t_curr + dt
                    return jnp.clip(x, DOMAIN_MIN, DOMAIN_MAX)
                
                # Generate actions using current flow parameters (differentiable)
                clean_actions = integrate_for_grad(params, z_policy, flow_steps)
                
                # Maximize Q(clean_actions) - use stop_gradient on critic
                q_at_actions = _inner_critic_forward(jax.lax.stop_gradient(inner_params), clean_actions, jnp.ones((batch_size, 1)))
                q_loss = -jnp.mean(q_at_actions)

                return alpha * bc_loss + q_loss, (bc_loss, jnp.mean(q_at_actions))
        elif fbrac_inner_policy_loss_type == "vgg_flow_matching":
            def flow_loss_fn(params):
                pred_vel = _flow_forward(params, x_t_policy, t_policy)
                
                # 1. Base Flow (Behavior Cloning Target / Data Flow)
                # In VGG-Flow, this is the pre-trained or optimal transport flow.
                # Here calculate it as the straight line target u_target_policy = (x1 - x0).
                v_base = u_target_policy 

                # 2. Value Gradient (Steering Term)
                # "Forward-Looking" parametrization: Estimate Grad(V) by looking ahead 
                def value_sum_fn(x_in):
                    return jnp.sum(_inner_critic_forward(inner_params, x_in, t_policy))
                
                # Standard backprop through value function at x_t
                grad_v = jax.grad(value_sum_fn)(x_t_policy)
                grad_v = jax.lax.stop_gradient(grad_v)

                # 3. Construct Target Vector Field (Matching Problem)
                # v_target = v_{base} + beta * grad_v
                # Use alpha as the beta (reward scale/inverse temperature) as requested.
                beta = 1/alpha
                if time_weight:
                    v_target = v_base + (1-t)*beta * grad_v
                else:
                    v_target = v_base + beta * grad_v
                
                # 4. Matching Loss (Regression)
                # Force the policy to Match the target vector field
                # Regress policy towards the vector sum
                matching_loss = jnp.mean((pred_vel - jax.lax.stop_gradient(v_target)) ** 2)

                # No separate BC loss - BC acts as the anchor in v_target
                return matching_loss, (matching_loss, 0.0)

        (flow_loss, (bc_loss, q_val)), flow_grads = jax.value_and_grad(flow_loss_fn, has_aux=True)(flow_params)
        flow_updates, flow_opt_state = flow_opt.update(flow_grads, flow_opt_state, flow_params)
        flow_params = optax.apply_updates(flow_params, flow_updates)

        logs = (bc_loss, distill_loss, q_val, flow_loss, outer_loss)
        new_state = (flow_params, inner_params, outer_params, flow_opt_state, inner_opt_state, outer_opt_state)
        return new_state, (key, logs)

    history = {
        'flow_bc_loss': [], 'critic_distill': [],
        'flow_q_loss': [], 'flow_total': [], 'outer_loss': []
    }

    state = (flow_params, inner_params, outer_params, flow_opt_state, inner_opt_state, outer_opt_state)
    for epoch in trange(epochs, desc="fbrac_inner (JAX)"):
        state, (key, logs) = step(state, key, flow_steps, alpha, batch_size)
        bc_loss, distill_loss, q_val, flow_loss, outer_loss = logs
        history['flow_bc_loss'].append(float(bc_loss))
        history['critic_distill'].append(float(distill_loss))
        history['flow_q_loss'].append(float(q_val))
        history['flow_total'].append(float(flow_loss))
        history['outer_loss'].append(float(outer_loss))
        # Plot intermediate results
        if plot_every is not None and (epoch + 1) % plot_every == 0:
            flow_params_curr, inner_params_curr, _, _, _, _ = state
            z_eval = jrandom.normal(jrandom.PRNGKey(999), (n_samples, 2))
            flow_actions = np.array(_integrate_flow(flow_params_curr, z_eval, steps=flow_steps))
            flow_rewards = np.array(reward_fn_vec(jnp.asarray(flow_actions)))
            
            # Flow Policy samples
            ax = axes[0, cnt]
            sc = ax.scatter(flow_actions[:, 0], flow_actions[:, 1], c=flow_rewards, 
                           cmap='plasma', s=15, alpha=0.7, edgecolors='none')
            ax.set_title(f'Flow Policy (Epoch {epoch + 1})\nMean R: {flow_rewards.mean():.4f}')
            ax.set_xlabel('x₁'); ax.set_ylabel('x₂')
            ax.set_xlim(DOMAIN_MIN, DOMAIN_MAX); ax.set_ylim(DOMAIN_MIN, DOMAIN_MAX)
            ax.set_aspect('equal')

            # Inner Critic V(x, t=1) heatmap
            ax2 = axes[1, cnt]
            resolution = 50
            x = np.linspace(DOMAIN_MIN, DOMAIN_MAX, resolution)
            y = np.linspace(DOMAIN_MIN, DOMAIN_MAX, resolution)
            X, Y = np.meshgrid(x, y)
            positions = jnp.array(np.stack([X.ravel(), Y.ravel()], axis=-1))
            t_one = jnp.ones((resolution * resolution, 1))
            v_values = np.array(_inner_critic_forward(inner_params_curr, positions, t_one)).reshape(resolution, resolution)
            im = ax2.imshow(v_values, extent=[DOMAIN_MIN, DOMAIN_MAX, DOMAIN_MIN, DOMAIN_MAX], 
                           origin='lower', cmap='viridis', aspect='equal')
            ax2.set_title(f'Inner Critic V(x, t=1) (Epoch {epoch + 1})')
            ax2.set_xlabel('x₁'); ax2.set_ylabel('x₂')

            cnt += 1

    flow_params, inner_params, _, _, _, _ = state
    if plot_every is not None:
        plt.tight_layout()
        plt.show()
    return flow_params, inner_params, history


# ============================================================================
# FBRAC (Flow-Based Actor-Critic) - NO Inner Critic
# ============================================================================
# Like FQL but without one-step policy distillation.
# Directly maximizes Q(flow(z)) through the full flow integration.
# Components: FlowPolicy, OuterCritic Q(a)
# Policy improvement: maximize Q(integrate_flow(z))

def train_fbrac(x_1_data, rewards_data, epochs=2000, lr=1e-3, alpha=1.0, flow_steps=10, hidden=128, plot_every=None):
    """
    FBRAC: Flow-based actor-critic without inner critic.
    
    1. Train outer critic Q(a) → R(a) on clean actions
    2. Train flow policy with BC loss + Q-maximization through full flow integration
    
    Unlike FQL: No one-step policy, uses full flow model directly.
    Unlike fbrac_inner: No inner critic V(x,t), uses outer critic Q(a) directly.
    
    Args:
        plot_every: If set, plot model predictions every `plot_every` epochs.
    """
    key = jrandom.PRNGKey(42)
    X1 = jnp.asarray(x_1_data)
    R = jnp.asarray(rewards_data)

    k1, k2 = jrandom.split(key, 2)
    flow_params = init_mlp([3, hidden, hidden, 2], k1)
    critic_params = init_mlp([2, hidden, hidden, 1], k2)

    flow_opt = optax.adam(lr)
    critic_opt = optax.adam(lr)
    flow_opt_state = flow_opt.init(flow_params)
    critic_opt_state = critic_opt.init(critic_params)

    # Pre-create figure for grid plotting
    n_samples = 512
    if plot_every:
        num_plots = epochs // plot_every
        fig, axes = plt.subplots(2, num_plots, figsize=(num_plots*5, 8))
        cnt = 0

    @partial(jax.jit, static_argnames=("flow_steps", "alpha", "batch_size"))
    def step(state, key, flow_steps, alpha, batch_size):
        flow_params, critic_params, flow_opt_state, critic_opt_state = state
        key, k_idx, k_z, k_t, k_pol = jrandom.split(key, 5)
        idx = jrandom.choice(k_idx, X1.shape[0], shape=(batch_size,), replace=False)
        x_1 = X1[idx]
        r = R[idx]

        # --- Critic Update: Q(a) → R(a) on clean actions ---
        def critic_loss_fn(params):
            q_pred = critic_forward(params, x_1)
            return jnp.mean((q_pred - r) ** 2)

        critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(critic_params)
        critic_updates, critic_opt_state = critic_opt.update(critic_grads, critic_opt_state, critic_params)
        critic_params = optax.apply_updates(critic_params, critic_updates)

        # --- Flow Update: BC + Q-maximization ---
        z = jrandom.normal(k_z, (batch_size, 2))
        t = jrandom.uniform(k_t, (batch_size, 1))
        x_t = (1 - t) * z + t * x_1
        u_target = x_1 - z  # Flow matching target

        def flow_loss_fn(params):
            # 1. BC Loss (flow matching)
            v_pred = flow_forward(params, x_t, t)
            bc_loss = jnp.mean((v_pred - u_target) ** 2)

            # 2. Q-Maximization: integrate flow and maximize Q at the end
            # Sample fresh noise for policy evaluation
            z_policy = jrandom.normal(k_pol, (batch_size, 2))
            
            # Differentiable flow integration from t=0 to t=1
            def integrate_for_grad(flow_p, x_start, n_steps):
                """Differentiable flow integration."""
                dt = 1.0 / n_steps
                x = x_start
                t_curr = jnp.zeros((x.shape[0], 1))
                for _ in range(n_steps):
                    v = flow_forward(flow_p, x, t_curr)
                    x = x + v * dt
                    t_curr = t_curr + dt
                return jnp.clip(x, DOMAIN_MIN, DOMAIN_MAX)
            
            # Generate actions using current flow parameters (differentiable)
            clean_actions = integrate_for_grad(params, z_policy, flow_steps)
            
            # Maximize Q(clean_actions) - use stop_gradient on critic
            q_at_actions = critic_forward(jax.lax.stop_gradient(critic_params), clean_actions)
            q_loss = -jnp.mean(q_at_actions)

            return alpha * bc_loss + q_loss, (bc_loss, jnp.mean(q_at_actions))

        (flow_loss, (bc_loss, q_val)), flow_grads = jax.value_and_grad(flow_loss_fn, has_aux=True)(flow_params)
        flow_updates, flow_opt_state = flow_opt.update(flow_grads, flow_opt_state, flow_params)
        flow_params = optax.apply_updates(flow_params, flow_updates)

        logs = (bc_loss, critic_loss, q_val, flow_loss)
        new_state = (flow_params, critic_params, flow_opt_state, critic_opt_state)
        return new_state, (key, logs)

    history = {
        'flow_bc_loss': [], 'critic_loss': [],
        'flow_q_val': [], 'flow_total': []
    }

    state = (flow_params, critic_params, flow_opt_state, critic_opt_state)
    for epoch in trange(epochs, desc="FBRAC (JAX)"):
        state, (key, logs) = step(state, key, flow_steps, alpha, batch_size)
        bc_loss, critic_loss, q_val, flow_loss = logs
        history['flow_bc_loss'].append(float(bc_loss))
        history['critic_loss'].append(float(critic_loss))
        history['flow_q_val'].append(float(q_val))
        history['flow_total'].append(float(flow_loss))
        
        # Plot intermediate results
        if plot_every is not None and (epoch + 1) % plot_every == 0:
            flow_params_curr, critic_params_curr, _, _ = state
            z_eval = jrandom.normal(jrandom.PRNGKey(999), (n_samples, 2))
            flow_actions = np.array(integrate_flow(flow_params_curr, z_eval, steps=flow_steps))
            flow_rewards = np.array(reward_fn_vec(jnp.asarray(flow_actions)))
            
            # Flow Policy samples
            ax = axes[0, cnt]
            sc = ax.scatter(flow_actions[:, 0], flow_actions[:, 1], c=flow_rewards, 
                           cmap='plasma', s=15, alpha=0.7, edgecolors='none')
            ax.set_title(f'Flow Policy (Epoch {epoch + 1})\nMean R: {flow_rewards.mean():.4f}')
            ax.set_xlabel('x₁'); ax.set_ylabel('x₂')
            ax.set_xlim(DOMAIN_MIN, DOMAIN_MAX); ax.set_ylim(DOMAIN_MIN, DOMAIN_MAX)
            ax.set_aspect('equal')

            # Outer Critic Q(a) heatmap
            ax2 = axes[1, cnt]
            resolution = 50
            x = np.linspace(DOMAIN_MIN, DOMAIN_MAX, resolution)
            y = np.linspace(DOMAIN_MIN, DOMAIN_MAX, resolution)
            X, Y = np.meshgrid(x, y)
            positions = jnp.array(np.stack([X.ravel(), Y.ravel()], axis=-1))
            q_values = np.array(critic_forward(critic_params_curr, positions)).reshape(resolution, resolution)
            im = ax2.imshow(q_values, extent=[DOMAIN_MIN, DOMAIN_MAX, DOMAIN_MIN, DOMAIN_MAX], 
                           origin='lower', cmap='viridis', aspect='equal')
            ax2.set_title(f'Outer Critic Q(a) (Epoch {epoch + 1})')
            ax2.set_xlabel('x₁'); ax2.set_ylabel('x₂')

            cnt += 1

    flow_params, critic_params, _, _ = state
    if plot_every is not None:
        plt.tight_layout()
        plt.show()
    return flow_params, critic_params, history

# --- New Cell ---

# Plot the prediction over the training
plot_step = 1000
print("="*60)
print("Training FQL (Policy Distillation) [JAX]")
print("="*60)
fql_flow_params, fql_policy_params, fql_critic_params, fql_history = train_fql(
    x_1_data, rewards_data, 
    epochs=3000,
    lr=3e-4, 
    alpha=0.1,  # Weight for Q-maximization
    flow_steps=FLOW_STEPS,
    plot_every=plot_step  # Plot every 1000 epochs
)
print(f"Final Q at policy: {fql_history['policy_q'][-1]:.4f}")

# --- New Cell ---

# Train FBRAC (Flow-based Actor-Critic with outer Q only, no inner critic)
plot_step = 1000
print("="*60)
print("Training FBRAC (Flow + Outer Q, no distillation) [JAX]")
print("="*60)
fbrac_outer_flow_params, fbrac_outer_critic_params, fbrac_outer_history = train_fbrac(
    x_1_data, rewards_data, 
    epochs=3000, 
    lr=3e-4, 
    alpha=0.3,  # BC regularization weight
    flow_steps=FLOW_STEPS,
    plot_every=plot_step  # Plot every 1000 epochs
)
print(f"Final Q at flow output: {fbrac_outer_history['flow_q_val'][-1]:.4f}")

# --- New Cell ---

plot_step = 1000
print("="*60)
print("Training fbrac_inner (Value Distillation) [JAX]")
print("="*60)
alpha=0.1

fbrac_flow_params, fbrac_critic_params, fbrac_history = train_fbrac_inner(
    x_1_data, rewards_data, 
    epochs=3000,
    lr=3e-4,
    alpha=alpha,  # Weight for V-maximization
    flow_steps=FLOW_STEPS,
    plot_every=plot_step,  # Plot every 1000 epochs
    use_time_embed=True,
    time_embed_dim=16,
    max_time_penalty=0.0,
    time_weight=False,
    fbrac_inner_policy_loss_type="vgg_flow_matching"
)
print(f"Final V at noise: {fbrac_history['flow_q_loss'][-1]:.4f}")

# --- New Cell ---

# Evaluate Policy Quality (JAX)
n_eval = 512
key_eval = jrandom.PRNGKey(np.random.randint(0, 1000000))

z_eval = jrandom.normal(key_eval, (n_eval, 2))

# FQL: One-step policy
fql_actions = np.array(onestep_forward(fql_policy_params, z_eval))
fql_rewards = np.array(reward_fn_vec(jnp.asarray(fql_actions)))

# FQL: Flow teacher (for comparison)
fql_flow_actions = np.array(integrate_flow(fql_flow_params, z_eval, steps=FLOW_STEPS))
fql_flow_rewards = np.array(reward_fn_vec(jnp.asarray(fql_flow_actions)))

# fbrac_inner: Flow policy
fbrac_inner_actions = np.array(integrate_flow(fbrac_flow_params, z_eval, steps=FLOW_STEPS))
fbrac_inner_rewards = np.array(reward_fn_vec(jnp.asarray(fbrac_inner_actions)))

# fbrac_outer: Flow policy
fbrac_outer_actions = np.array(integrate_flow(fbrac_outer_flow_params, z_eval, steps=FLOW_STEPS))
fbrac_outer_rewards = np.array(reward_fn_vec(jnp.asarray(fbrac_outer_actions)))

print("="*60)
print("POLICY QUALITY (Mean Reward of Generated Actions)")
print("="*60)
print(f"Dataset (ground truth):    {rewards_data.mean():.4f}")
print(f"FQL One-Step Policy:       {fql_rewards.mean():.4f}")
print(f"FQL Flow Teacher:          {fql_flow_rewards.mean():.4f}")
print(f"fbrac_inner Flow Policy:   {fbrac_inner_rewards.mean():.4f}")
print(f"fbrac_outer Flow Policy:   {fbrac_outer_rewards.mean():.4f}")

# --- New Cell ---

# Visualize Relevant Results
fig, axes = plt.subplots(1, 4, figsize=(18, 10))
axis_limits = [DOMAIN_MIN, DOMAIN_MAX]

# 0. Dataset Distribution
ax0 = axes[0]
ax0.scatter(x_1_data[:, 0], x_1_data[:, 1], c=rewards_data, cmap='plasma', 
            s=10, alpha=0.6, edgecolors='none')
ax0.set_title(f'Dataset Distribution\nMean R: {rewards_data.mean():.4f}')
ax0.set_xlabel('Action dim 1'); ax0.set_ylabel('Action dim 2')
ax0.set_xlim(axis_limits); ax0.set_ylim(axis_limits); ax0.set_aspect('equal')

# 1. FQL One-Step Policy
ax1 = axes[1]
ax1.scatter(fql_actions[:, 0], fql_actions[:, 1], c=fql_rewards, cmap='plasma', 
            s=10, alpha=0.6, edgecolors='none')
ax1.set_title(f'FQL One-Step Policy\nMean R: {fql_rewards.mean():.4f}')
ax1.set_xlabel('Action dim 1'); ax1.set_ylabel('Action dim 2')
ax1.set_xlim(axis_limits); ax1.set_ylim(axis_limits); ax1.set_aspect('equal')

# 3. fbrac_outer Flow Policy
ax3 = axes[2]
ax3.scatter(fbrac_outer_actions[:, 0], fbrac_outer_actions[:, 1], c=fbrac_outer_rewards, cmap='plasma', 
            s=10, alpha=0.6, edgecolors='none')
ax3.set_title(f'FBRAC outer Flow Policy\nMean R: {fbrac_outer_rewards.mean():.4f}')
ax3.set_xlabel('Action dim 1'); ax3.set_ylabel('Action dim 2')
ax3.set_xlim(axis_limits); ax3.set_ylim(axis_limits); ax3.set_aspect('equal')

# 2. fbrac_inner Flow Policy
ax2 = axes[3]
ax2.scatter(fbrac_inner_actions[:, 0], fbrac_inner_actions[:, 1], c=fbrac_inner_rewards, cmap='plasma', 
            s=10, alpha=0.6, edgecolors='none')
ax2.set_title(f'FBRAC inner Flow Policy\nMean R: {fbrac_inner_rewards.mean():.4f}')
ax2.set_xlabel('Action dim 1'); ax2.set_ylabel('Action dim 2')
ax2.set_xlim(axis_limits); ax2.set_ylim(axis_limits); ax2.set_aspect('equal')

plt.tight_layout()
plt.show()

# --- New Cell ---

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# 1. Helper for fbrac_inner V-value with time embedding
def _inner_critic_forward_plotting(params, x, t):
    # Consistently use time_embed_dim=16 as in the training call
    t_embed = fourier_time_embed(t, embed_dim=16)
    xt = jnp.concatenate([x, t_embed], axis=-1)
    return apply_mlp(params, xt).squeeze(-1)

# 2. Evaluation Setup
n_eval_traj = 128
key_traj = jax.random.PRNGKey(42)
z_init = jax.random.normal(key_traj, (n_eval_traj, 2))

# 3. Collect fbrac_inner Trajectory Values
v_inner_history = []
x_inner = z_init
times = jnp.linspace(0.0, 1.0, FLOW_STEPS + 1)

for t_val in times:
    t_batch = jnp.full((n_eval_traj, 1), t_val)
    v_val = _inner_critic_forward_plotting(fbrac_critic_params, x_inner, t_batch)
    v_inner_history.append(jnp.mean(v_val))
    
    if t_val < 1.0:
        dt = 1.0 / FLOW_STEPS
        v_flow = flow_forward(fbrac_flow_params, x_inner, t_batch)
        x_inner = x_inner + v_flow * dt
        x_inner = jnp.clip(x_inner, DOMAIN_MIN, DOMAIN_MAX)

# 4. Collect fbrac_outer Trajectory Values (integrated Q)
v_outer_history = []
x_outer = z_init

for t_val in times:
    t_batch = jnp.full((n_eval_traj, 1), t_val)
    
    # For fbrac_outer, we integrate the flow from x_t to t=1 to get x_1
    # and then evaluate Q(x_1)
    x_1_pred = integrate_flow_from(fbrac_outer_flow_params, x_outer, t_batch, steps=FLOW_STEPS)
    q_val = critic_forward(fbrac_outer_critic_params, x_1_pred)
    v_outer_history.append(jnp.mean(q_val))
    
    if t_val < 1.0:
        dt = 1.0 / FLOW_STEPS
        v_flow = flow_forward(fbrac_outer_flow_params, x_outer, t_batch)
        x_outer = x_outer + v_flow * dt
        x_outer = jnp.clip(x_outer, DOMAIN_MIN, DOMAIN_MAX)

# 5. Plotting
plt.figure(figsize=(8, 5))
plt.plot(times, v_inner_history, label="fbrac_inner V(x,t) (state-value)", marker='o')
plt.plot(times, v_outer_history, label="fbrac_outer Q(x_1|x_t) (integrated Q)", marker='s')
plt.xlabel("Time (t)")
plt.ylabel("Estimated Value")
plt.title("Value along Policy Trajectory (Average)")
plt.legend()
plt.grid(True)
plt.show()

print("Trajectory Value Comparison Plot complete.")
