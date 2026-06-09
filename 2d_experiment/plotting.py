"""Plotting utilities for 2D toy experiment visualization."""

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
from utils import DOMAIN_MIN, DOMAIN_MAX

def plot_agent_samples(flow_params, agent_name, integrate_fn, n_samples=512,
                       flow_steps=10, key=None, title_suffix=""):
    """Plot samples from a flow-based agent."""
    if key is None:
        key = jrandom.PRNGKey(0)

    z = jrandom.normal(key, (n_samples, 2))
    samples = integrate_fn(flow_params, z, steps=flow_steps)
    samples = np.array(samples)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(samples[:, 0], samples[:, 1], s=20, alpha=0.6, edgecolors='none')
    ax.set_xlim(DOMAIN_MIN, DOMAIN_MAX)
    ax.set_ylim(DOMAIN_MIN, DOMAIN_MAX)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title(f'{agent_name} Samples{title_suffix}')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    return fig, ax, samples

def plot_target_distribution(target_data, title="Target Distribution"):
    """Plot target distribution."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(target_data[:, 0], target_data[:, 1], s=20, alpha=0.6,
               color='red', edgecolors='none')
    ax.set_xlim(DOMAIN_MIN, DOMAIN_MAX)
    ax.set_ylim(DOMAIN_MIN, DOMAIN_MAX)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    return fig, ax

def plot_comparison(target_data, samples_fql, samples_fbrac, samples_qflow,
                   figsize=(15, 5)):
    """Compare samples from three agents."""
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # Target
    axes[0].scatter(target_data[:, 0], target_data[:, 1], s=20, alpha=0.6,
                   color='red', edgecolors='none')
    axes[0].set_title('Target Distribution')

    # FQL
    axes[1].scatter(samples_fql[:, 0], samples_fql[:, 1], s=20, alpha=0.6,
                   color='blue', edgecolors='none')
    axes[1].set_title('FQL Samples')

    # FBRAC
    axes[2].scatter(samples_fbrac[:, 0], samples_fbrac[:, 1], s=20, alpha=0.6,
                   color='green', edgecolors='none')
    axes[2].set_title('FBRAC Samples')

    # QFlow
    axes[3].scatter(samples_qflow[:, 0], samples_qflow[:, 1], s=20, alpha=0.6,
                   color='purple', edgecolors='none')
    axes[3].set_title('QFlow Samples')

    for ax in axes:
        ax.set_xlim(DOMAIN_MIN, DOMAIN_MAX)
        ax.set_ylim(DOMAIN_MIN, DOMAIN_MAX)
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def plot_training_curves(histories, figsize=(12, 4)):
    """Plot training curves for all agents."""
    agents = ['FQL', 'FBRAC', 'QFlow']

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Flow loss
    for i, (agent, hist) in enumerate(zip(agents, histories)):
        if 'flow_loss' in hist:
            axes[0].plot(hist['flow_loss'], label=agent, alpha=0.7)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Flow Loss')
    axes[0].set_title('Flow Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Critic loss
    for i, (agent, hist) in enumerate(zip(agents, histories)):
        if 'critic_loss' in hist:
            axes[1].plot(hist['critic_loss'], label=agent, alpha=0.7)
        elif 'outer_loss' in hist:
            axes[1].plot(hist['outer_loss'], label=agent, alpha=0.7)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Critic Loss')
    axes[1].set_title('Critic Training Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def plot_value_heatmap(critic_params, critic_forward_fn, title="Critic V(x)",
                      figsize=(6, 6)):
    """Plot value function as a heatmap."""
    resolution = 50
    x = np.linspace(DOMAIN_MIN, DOMAIN_MAX, resolution)
    y = np.linspace(DOMAIN_MIN, DOMAIN_MAX, resolution)
    X, Y = np.meshgrid(x, y)

    positions = jnp.array(np.stack([X.ravel(), Y.ravel()], axis=-1))
    v_values = np.array(critic_forward_fn(critic_params, positions))
    v_values = v_values.reshape(resolution, resolution)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(v_values, extent=[DOMAIN_MIN, DOMAIN_MAX, DOMAIN_MIN, DOMAIN_MAX],
                   origin='lower', cmap='viridis', aspect='equal')
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Value')

    return fig, ax
