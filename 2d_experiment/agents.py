"""Training functions for the 2D toy experiment: FQL, FBRAC, and QFlow.

All three share the same offline RL setup: a flow behavior-cloning objective
plus a value/critic-based policy-improvement term, with a BC-only warmup.
"""

import os
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
import matplotlib.pyplot as plt
from tqdm import trange

from networks import (
    DOMAIN_MIN, DOMAIN_MAX,
    init_mlp, apply_mlp, fourier_time_embed,
    flow_forward, onestep_forward, critic_forward,
    integrate_flow,
)


def plot_training_progress(flow_params, policy_params, critic_params, epoch, flow_steps,
                           reward_fn_vec, n_samples=512, save_path=None):
    """Plot the current model predictions during training."""
    # Sample actions from flow and one-step policy
    z_eval = jrandom.normal(jrandom.PRNGKey(999), (n_samples, 2))
    flow_actions = np.array(integrate_flow(flow_params, z_eval, steps=flow_steps))
    policy_actions = np.array(onestep_forward(policy_params, z_eval))

    # Compute rewards
    flow_rewards = np.array(reward_fn_vec(jnp.asarray(flow_actions)))
    policy_rewards = np.array(reward_fn_vec(jnp.asarray(policy_actions)))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # One-Step Policy
    ax2 = axes[0]
    ax2.scatter(policy_actions[:, 0], policy_actions[:, 1], c=policy_rewards,
                cmap='plasma', s=15, alpha=0.7, edgecolors='none')
    ax2.set_title(f'One-Step Policy (Epoch {epoch})\nMean R: {policy_rewards.mean():.4f}')
    ax2.set_xlabel('x₁'); ax2.set_ylabel('x₂')
    ax2.set_xlim(DOMAIN_MIN, DOMAIN_MAX); ax2.set_ylim(DOMAIN_MIN, DOMAIN_MAX)
    ax2.set_aspect('equal')

    # Critic Q-values heatmap
    ax3 = axes[1]
    resolution = 50
    x = np.linspace(DOMAIN_MIN, DOMAIN_MAX, resolution)
    y = np.linspace(DOMAIN_MIN, DOMAIN_MAX, resolution)
    X, Y = np.meshgrid(x, y)
    positions = jnp.array(np.stack([X.ravel(), Y.ravel()], axis=-1))
    q_values = np.array(critic_forward(critic_params, positions)).reshape(resolution, resolution)
    ax3.imshow(q_values, extent=[DOMAIN_MIN, DOMAIN_MAX, DOMAIN_MIN, DOMAIN_MAX],
               origin='lower', cmap='viridis', aspect='equal')
    ax3.set_title(f'Critic Q(a) (Epoch {epoch})')
    ax3.set_xlabel('x₁'); ax3.set_ylabel('x₂')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


# ============================================================================
# FQL ACTOR-CRITIC TRAINING (JAX)
# ============================================================================
# Components: FlowPolicy (teacher), OneStepPolicy, OuterCritic
# Policy improvement: maximize Q(policy(z))
def train_fql(x_1_data, rewards_data, epochs=2000, lr=1e-3, alpha=1.0, flow_steps=10,
              hidden=128, num_layers=4, plot_every=None, use_target_network=False, tau=0.005,
              batch_size=4096, bc_epochs=2000, num_evals=512,
              save_dir='toy_experiment', dataset_type='swissroll'):
    """
    FQL-style actor-critic training (JAX) with jitted update step.
    1. Train flow teacher (BC via flow matching)
    2. Train outer critic Q(a) → R(a)
    3. Train one-step policy: distillation + Q-maximization

    Args:
        plot_every: If set, plot model predictions every `plot_every` epochs.
        num_layers: Number of hidden layers in each network.
        use_target_network: Whether to use target networks for critic.
        tau: EMA rate for target network updates.
    """
    key = jrandom.PRNGKey(42)
    X1 = jnp.asarray(x_1_data)
    R = jnp.asarray(rewards_data)

    k1, k2, k3 = jrandom.split(key, 3)
    hidden_half = max(1, hidden // 2)
    flow_params = init_mlp([3] + [hidden] * num_layers + [hidden_half] + [2], k1)
    policy_params = init_mlp([2] + [hidden] * num_layers + [hidden_half] + [2], k2)
    critic_params = init_mlp([2] + [hidden] * num_layers + [1], k3)

    # Initialize target network (copy of critic)
    target_critic_params = jax.tree_util.tree_map(lambda x: x.copy(), critic_params) if use_target_network else critic_params

    flow_opt = optax.adam(lr)
    policy_opt = optax.adam(lr)
    critic_opt = optax.adam(lr)
    flow_opt_state = flow_opt.init(flow_params)
    policy_opt_state = policy_opt.init(policy_params)
    critic_opt_state = critic_opt.init(critic_params)

    @partial(jax.jit, static_argnames=("flow_steps", "alpha", "batch_size", "use_target_network", "tau"))
    def step(state, key, flow_steps, alpha, batch_size, bc_only, use_target_network, tau):
        flow_params, policy_params, critic_params, target_critic_params, flow_opt_state, policy_opt_state, critic_opt_state = state
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
            # Use target network if enabled, otherwise use main critic
            critic_params_to_use = target_critic_params if use_target_network else critic_params
            q_at_policy = critic_forward(critic_params_to_use, policy_output)
            # BC warmup: first phase ignores critic in the loss
            total_loss = jax.lax.select(bc_only, alpha * distill_loss, alpha * distill_loss - jnp.mean(q_at_policy))
            return total_loss, (distill_loss, jnp.mean(q_at_policy))

        (policy_loss, (distill_loss, q_val)), policy_grads = jax.value_and_grad(policy_loss_fn, has_aux=True)(policy_params)
        policy_updates, policy_opt_state = policy_opt.update(policy_grads, policy_opt_state, policy_params)
        policy_params = optax.apply_updates(policy_params, policy_updates)

        # Update target network with EMA if enabled
        if use_target_network:
            target_critic_params = jax.tree_util.tree_map(
                lambda p, tp: p * tau + tp * (1 - tau),
                critic_params,
                target_critic_params
            )
        else:
            target_critic_params = critic_params  # Keep in sync when not using target

        logs = (flow_loss, critic_loss, distill_loss, q_val, policy_loss)
        new_state = (flow_params, policy_params, critic_params, target_critic_params, flow_opt_state, policy_opt_state, critic_opt_state)
        return new_state, (key, logs)

    history = {
        'flow_loss': [], 'critic_loss': [],
        'policy_distill': [], 'policy_q': [], 'policy_total': []
    }
    if plot_every:
        num_plots = epochs // plot_every
        fig, axes = plt.subplots(2, num_plots, figsize=(num_plots * 5, 8))
    else:
        fig, axes = None, None

    cnt = 0
    n_samples = num_evals
    state = (flow_params, policy_params, critic_params, target_critic_params, flow_opt_state, policy_opt_state, critic_opt_state)
    # Track policy parameters at the end of BC-only warmup
    bc_policy_params = None

    # Calculate steps per epoch to iterate through entire dataset
    steps_per_epoch = max(1, len(x_1_data) // batch_size)

    for epoch in trange(epochs, desc="FQL (JAX)"):
        bc_only = epoch < bc_epochs

        # Accumulate losses across all steps in this epoch
        epoch_flow_loss = 0.0
        epoch_critic_loss = 0.0
        epoch_distill_loss = 0.0
        epoch_q_val = 0.0
        epoch_policy_loss = 0.0

        # Run multiple steps per epoch to cover the dataset
        for step_idx in range(steps_per_epoch):
            state, (key, logs) = step(state, key, flow_steps, alpha, batch_size, bc_only, use_target_network, tau)
            flow_loss, critic_loss, distill_loss, q_val, policy_loss = logs
            epoch_flow_loss += float(flow_loss)
            epoch_critic_loss += float(critic_loss)
            epoch_distill_loss += float(distill_loss)
            epoch_q_val += float(q_val)
            epoch_policy_loss += float(policy_loss)

        # Average losses across steps
        history['flow_loss'].append(epoch_flow_loss / steps_per_epoch)
        history['critic_loss'].append(epoch_critic_loss / steps_per_epoch)
        history['policy_distill'].append(epoch_distill_loss / steps_per_epoch)
        history['policy_q'].append(epoch_q_val / steps_per_epoch)
        history['policy_total'].append(epoch_policy_loss / steps_per_epoch)

        # Capture BC-only snapshot at the end of warmup
        if (not bc_only) and (bc_policy_params is None):
            _, policy_params_curr, _, _, _, _, _ = state
            bc_policy_params = policy_params_curr

        # Plot intermediate results
        if plot_every is not None and (epoch + 1) % plot_every == 0:
            flow_params_curr, policy_params_curr, critic_params_curr, _, _, _, _ = state
            z_eval = jrandom.normal(jrandom.PRNGKey(999), (n_samples, 2))
            policy_actions = np.array(onestep_forward(policy_params_curr, z_eval))

            ax = axes[0, cnt]
            ax.scatter(policy_actions[:, 0], policy_actions[:, 1], s=15, alpha=0.7, edgecolors='none')
            ax.set_title(f'One-Step Policy (Epoch {epoch + 1})')
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
            q_values = np.array(critic_forward(critic_params_curr, positions)).reshape(resolution, resolution)
            ax3.imshow(q_values, extent=[DOMAIN_MIN, DOMAIN_MAX, DOMAIN_MIN, DOMAIN_MAX],
                       origin='lower', cmap='viridis', aspect='equal')
            ax3.set_title(f'Critic Q(a) (Epoch {epoch + 1})')
            ax3.set_xlabel('x₁'); ax3.set_ylabel('x₂')

            cnt += 1

    flow_params, policy_params, critic_params, _, _, _, _ = state
    # Fallback in case epochs < 2
    if bc_policy_params is None:
        bc_policy_params = policy_params

    # Add final plot at last epoch if not already plotted
    if plot_every is not None and (epochs % plot_every != 0):
        flow_params_curr, policy_params_curr, critic_params_curr, _, _, _, _ = state
        z_eval = jrandom.normal(jrandom.PRNGKey(999), (n_samples, 2))
        policy_actions = np.array(onestep_forward(policy_params_curr, z_eval))

        ax = axes[0, cnt]
        ax.scatter(policy_actions[:, 0], policy_actions[:, 1], s=15, alpha=0.7, edgecolors='none')
        ax.set_title(f'One-Step Policy (Epoch {epochs})')
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
        q_values = np.array(critic_forward(critic_params_curr, positions)).reshape(resolution, resolution)
        ax3.imshow(q_values, extent=[DOMAIN_MIN, DOMAIN_MAX, DOMAIN_MIN, DOMAIN_MAX],
                   origin='lower', cmap='viridis', aspect='equal')
        ax3.set_title(f'Critic Q(a) (Epoch {epochs})')
        ax3.set_xlabel('x₁'); ax3.set_ylabel('x₂')

    if plot_every is not None:
        plt.tight_layout()
        out_dir = f'{save_dir}/{dataset_type}/fql'
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(f'{out_dir}/alpha{alpha}_flow{flow_steps}_layers{num_layers}_hidden{hidden}_data{len(x_1_data)}_epochs{epochs}_training_progress.png', dpi=100, bbox_inches='tight')
        plt.close()
    return flow_params, policy_params, critic_params, bc_policy_params, history


# ============================================================================
# QFlow (Q-guided Flow) - flow matching with inner time-conditioned critic
# ============================================================================
# Components: FlowPolicy, InnerCritic V(x, t), OuterCritic Q(a)
# Policy improvement: steer the BC flow by the gradient of the inner critic.
def train_qflow(x_1_data, rewards_data, epochs=2000, lr=1e-3, alpha=1.0, flow_steps=10,
                hidden=128, num_layers=4, plot_every=None,
                use_time_embed=False, time_embed_dim=64,
                use_target_network=False, tau=0.005,
                batch_size=4096, bc_epochs=2000, num_evals=512,
                save_dir='toy_experiment', dataset_type='swissroll'):
    """
    QFlow: flow-based actor-critic with an inner (time-conditioned) critic.

    1. Train outer critic Q(a) → R(a) on clean actions.
    2. Distill the inner critic V(x, t) from the outer critic along the flow.
    3. Train the flow policy to match a value-steered velocity field.

    Args:
        use_time_embed: If True, use Fourier time embedding for the inner critic.
        time_embed_dim: Dimension of the Fourier time embedding (default 64).
        num_layers: Number of hidden layers in each network.
        use_target_network: Whether to use target networks for critics.
        tau: EMA rate for target network updates.
    """
    key = jrandom.PRNGKey(123)
    X1 = jnp.asarray(x_1_data)
    R = jnp.asarray(rewards_data)

    k1, k2, k3, k4 = jrandom.split(key, 4)
    # Flow network input: x (2D) + scalar time (1D)
    flow_input_dim = 2 + 1
    # Inner critic input: x (2D) + time (1D or time_embed_dim)
    inner_input_dim = 2 + (time_embed_dim if use_time_embed else 1)

    hidden_half = max(1, hidden // 2)
    flow_params = init_mlp([flow_input_dim] + [hidden] * num_layers + [hidden_half] + [2], k1)
    inner_params = init_mlp([inner_input_dim] + [hidden_half] * num_layers + [1], k2)
    outer_params = init_mlp([2] + [hidden_half] * num_layers + [1], k3)

    # Initialize target networks (copies of critics)
    target_inner_params = jax.tree_util.tree_map(lambda x: x.copy(), inner_params) if use_target_network else inner_params
    target_outer_params = jax.tree_util.tree_map(lambda x: x.copy(), outer_params) if use_target_network else outer_params

    # Policy (flow) uses scalar time.
    def _flow_forward(params, x, t):
        xt = jnp.concatenate([x, t], axis=-1)
        return apply_mlp(params, xt)

    # Inner critic optionally uses Fourier time embedding.
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

    n_samples = num_evals
    if plot_every:
        num_plots = epochs // plot_every
        fig, axes = plt.subplots(2, num_plots, figsize=(num_plots * 5, 8))
        cnt = 0

    distillation_batch_scale = 1

    def _integrate_flow_from(flow_params, x, t_start, steps=10):
        """Integrate flow starting at (x, t_start) to t=1 using the Euler method."""
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
        t0 = jnp.zeros((z.shape[0], 1))
        return _integrate_flow_from(flow_params, z, t0, steps=steps)

    @partial(jax.jit, static_argnames=("flow_steps", "alpha", "batch_size", "use_target_network", "tau"))
    def step(state, key, flow_steps, alpha, batch_size, bc_only, use_target_network, tau):
        flow_params, inner_params, outer_params, target_inner_params, target_outer_params, flow_opt_state, inner_opt_state, outer_opt_state = state
        key, k_idx, k_noise, k_t, k_pol1, k_pol2, _, _ = jrandom.split(key, 8)
        idx = jrandom.choice(k_idx, X1.shape[0], shape=(batch_size,), replace=False)
        x_1 = X1[idx]
        r = R[idx]

        # Outer critic on clean actions.
        def outer_loss_fn(params):
            q_pred = critic_forward(params, x_1)
            return jnp.mean((q_pred - r) ** 2)

        outer_loss, outer_grads = jax.value_and_grad(outer_loss_fn)(outer_params)
        outer_updates, outer_opt_state = outer_opt.update(outer_grads, outer_opt_state, outer_params)
        outer_params = optax.apply_updates(outer_params, outer_updates)

        # Inner critic distillation from the outer critic along the flow.
        x_0 = jrandom.normal(k_noise, (batch_size * distillation_batch_scale, 2))
        t = jrandom.uniform(k_t, (batch_size * distillation_batch_scale, 1), minval=0.0, maxval=1.0)
        x_t = (1 - t) * x_0 + t * x_1.repeat(distillation_batch_scale, axis=0)
        x_1_pred = _integrate_flow_from(flow_params, x_t, t, steps=flow_steps)

        # Use target network if enabled, otherwise use the main outer critic.
        outer_params_to_use = target_outer_params if use_target_network else outer_params
        target_r = jax.lax.stop_gradient(critic_forward(outer_params_to_use, x_1_pred))

        def critic_loss_fn(params):
            v_pred = _inner_critic_forward(params, x_t, t)
            distill_loss = jnp.mean((v_pred - target_r) ** 2)
            return distill_loss, (distill_loss, 0.0)

        (critic_loss, (distill_loss, _)), critic_grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(inner_params)
        inner_updates, inner_opt_state = inner_opt.update(critic_grads, inner_opt_state, inner_params)
        inner_params = optax.apply_updates(inner_params, inner_updates)

        # Flow update.
        x_0_policy = jrandom.normal(k_pol1, (batch_size, 2))
        t_policy = jrandom.uniform(k_pol2, (batch_size, 1), minval=0.0, maxval=1.0)
        x_t_policy = (1 - t_policy) * x_0_policy + t_policy * x_1
        u_target_policy = x_1 - x_0_policy

        def flow_loss_fn(params):
            pred_vel = _flow_forward(params, x_t_policy, t_policy)

            # Base Flow (Behavior Cloning target / straight-line data flow).
            v_base = u_target_policy

            # Value Gradient (Steering term). Backprop through the inner critic at x_t.
            inner_params_to_use = target_inner_params if use_target_network else inner_params
            def value_sum_fn(x_in):
                return jnp.sum(_inner_critic_forward(inner_params_to_use, x_in, t_policy))

            grad_v = jax.grad(value_sum_fn)(x_t_policy)
            grad_v = jax.lax.stop_gradient(grad_v)

            # Target Vector Field: v_target = v_base + beta * grad_v.
            beta = 1.0 / alpha
            v_target = v_base + beta * grad_v

            # Matching loss. During warmup, match only the base flow (no value gradient).
            bc_matching_loss = jnp.mean((pred_vel - jax.lax.stop_gradient(v_base)) ** 2)
            full_matching_loss = jnp.mean((pred_vel - jax.lax.stop_gradient(v_target)) ** 2)
            total_loss = jax.lax.select(bc_only, bc_matching_loss, full_matching_loss)

            return total_loss, (total_loss, 0.0)

        (flow_loss, (bc_loss, q_val)), flow_grads = jax.value_and_grad(flow_loss_fn, has_aux=True)(flow_params)
        flow_updates, flow_opt_state = flow_opt.update(flow_grads, flow_opt_state, flow_params)
        flow_params = optax.apply_updates(flow_params, flow_updates)

        # Update target networks with EMA if enabled.
        if use_target_network:
            target_inner_params = jax.tree_util.tree_map(
                lambda p, tp: p * tau + tp * (1 - tau),
                inner_params,
                target_inner_params
            )
            target_outer_params = jax.tree_util.tree_map(
                lambda p, tp: p * tau + tp * (1 - tau),
                outer_params,
                target_outer_params
            )
        else:
            target_inner_params = inner_params
            target_outer_params = outer_params

        logs = (bc_loss, distill_loss, q_val, flow_loss, outer_loss)
        new_state = (flow_params, inner_params, outer_params, target_inner_params, target_outer_params, flow_opt_state, inner_opt_state, outer_opt_state)
        return new_state, (key, logs)

    history = {
        'flow_bc_loss': [], 'critic_distill': [],
        'flow_q_loss': [], 'flow_total': [], 'outer_loss': []
    }
    bc_policy_params = None

    state = (flow_params, inner_params, outer_params, target_inner_params, target_outer_params, flow_opt_state, inner_opt_state, outer_opt_state)

    # Calculate steps per epoch to iterate through entire dataset
    steps_per_epoch = max(1, len(x_1_data) // batch_size)

    for epoch in trange(epochs, desc="qflow (JAX)"):
        bc_only = epoch < bc_epochs

        # Accumulate losses across all steps in this epoch
        epoch_bc_loss = 0.0
        epoch_distill_loss = 0.0
        epoch_q_val = 0.0
        epoch_flow_loss = 0.0
        epoch_outer_loss = 0.0

        # Run multiple steps per epoch to cover the dataset
        for step_idx in range(steps_per_epoch):
            state, (key, logs) = step(state, key, flow_steps, alpha, batch_size, bc_only, use_target_network, tau)
            bc_loss, distill_loss, q_val, flow_loss, outer_loss = logs
            epoch_bc_loss += float(bc_loss)
            epoch_distill_loss += float(distill_loss)
            epoch_q_val += float(q_val)
            epoch_flow_loss += float(flow_loss)
            epoch_outer_loss += float(outer_loss)

        # Average losses across steps
        history['flow_bc_loss'].append(epoch_bc_loss / steps_per_epoch)
        history['critic_distill'].append(epoch_distill_loss / steps_per_epoch)
        history['flow_q_loss'].append(epoch_q_val / steps_per_epoch)
        history['flow_total'].append(epoch_flow_loss / steps_per_epoch)
        history['outer_loss'].append(epoch_outer_loss / steps_per_epoch)

        # Capture BC-only snapshot at the end of warmup
        if (not bc_only) and (bc_policy_params is None):
            flow_params, _, _, _, _, _, _, _ = state
            bc_policy_params = flow_params

        # Plot intermediate results
        if plot_every is not None and (epoch + 1) % plot_every == 0:
            flow_params_curr, inner_params_curr, _, _, _, _, _, _ = state
            z_eval = jrandom.normal(jrandom.PRNGKey(999), (n_samples, 2))
            flow_actions = np.array(_integrate_flow(flow_params_curr, z_eval, steps=flow_steps))

            # Flow Policy samples
            ax = axes[0, cnt]
            ax.scatter(flow_actions[:, 0], flow_actions[:, 1], s=15, alpha=0.7, edgecolors='none')
            ax.set_title(f'Flow Policy (Epoch {epoch + 1})')
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
            ax2.imshow(v_values, extent=[DOMAIN_MIN, DOMAIN_MAX, DOMAIN_MIN, DOMAIN_MAX],
                       origin='lower', cmap='viridis', aspect='equal')
            ax2.set_title(f'Inner Critic V(x, t=1) (Epoch {epoch + 1})')
            ax2.set_xlabel('x₁'); ax2.set_ylabel('x₂')

            cnt += 1

    flow_params, inner_params, _, _, _, _, _, _ = state

    # Add final plot at last epoch if not already plotted
    if plot_every is not None and (epochs % plot_every != 0):
        flow_params_curr, inner_params_curr, _, _, _, _, _, _ = state
        z_eval = jrandom.normal(jrandom.PRNGKey(999), (n_samples, 2))
        flow_actions = np.array(_integrate_flow(flow_params_curr, z_eval, steps=flow_steps))

        # Flow Policy samples
        ax = axes[0, cnt]
        ax.scatter(flow_actions[:, 0], flow_actions[:, 1], s=15, alpha=0.7, edgecolors='none')
        ax.set_title(f'Flow Policy (Epoch {epochs})')
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
        ax2.imshow(v_values, extent=[DOMAIN_MIN, DOMAIN_MAX, DOMAIN_MIN, DOMAIN_MAX],
                   origin='lower', cmap='viridis', aspect='equal')
        ax2.set_title(f'Inner Critic V(x, t=1) (Epoch {epochs})')
        ax2.set_xlabel('x₁'); ax2.set_ylabel('x₂')

    if plot_every is not None:
        plt.tight_layout()
        out_dir = f'{save_dir}/{dataset_type}/qflow'
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(f'{out_dir}/alpha{alpha}_flow{flow_steps}_layers{num_layers}_hidden{hidden}_data{len(x_1_data)}_epochs{epochs}_training_progress.png', dpi=100, bbox_inches='tight')
        plt.close()
    return flow_params, inner_params, bc_policy_params, history


# ============================================================================
# FBRAC (Flow-Based Actor-Critic) - NO Inner Critic
# ============================================================================
# Like FQL but without one-step policy distillation.
# Directly maximizes Q(flow(z)) through the full flow integration.
def train_fbrac(x_1_data, rewards_data, epochs=2000, lr=1e-3, alpha=1.0, flow_steps=10,
                hidden=128, num_layers=4, plot_every=None, use_target_network=False, tau=0.005,
                batch_size=4096, bc_epochs=2000, num_evals=512,
                save_dir='toy_experiment', dataset_type='swissroll'):
    """
    FBRAC: Flow-based actor-critic without inner critic.

    1. Train outer critic Q(a) → R(a) on clean actions.
    2. Train flow policy with BC loss + Q-maximization through full flow integration.

    Unlike FQL: No one-step policy, uses the full flow model directly.
    Unlike QFlow: No inner critic V(x, t), uses the outer critic Q(a) directly.

    Args:
        plot_every: If set, plot model predictions every `plot_every` epochs.
        num_layers: Number of hidden layers in each network.
        use_target_network: Whether to use target networks for critic.
        tau: EMA rate for target network updates.
    """
    key = jrandom.PRNGKey(42)
    X1 = jnp.asarray(x_1_data)
    R = jnp.asarray(rewards_data)

    k1, k2 = jrandom.split(key, 2)
    hidden_half = max(1, hidden // 2)
    flow_params = init_mlp([3] + [hidden] * num_layers + [hidden_half] + [2], k1)
    critic_params = init_mlp([2] + [hidden] * num_layers + [1], k2)

    # Initialize target network (copy of critic)
    target_critic_params = jax.tree_util.tree_map(lambda x: x.copy(), critic_params) if use_target_network else critic_params

    flow_opt = optax.adam(lr)
    critic_opt = optax.adam(lr)
    flow_opt_state = flow_opt.init(flow_params)
    critic_opt_state = critic_opt.init(critic_params)

    # Pre-create figure for grid plotting (only plot during offline epochs).
    offline_epochs = epochs - bc_epochs
    n_samples = num_evals
    if plot_every:
        num_plots = offline_epochs // plot_every if offline_epochs > 0 else 0
        if num_plots > 0:
            fig, axes = plt.subplots(2, num_plots, figsize=(num_plots * 5, 8))
        else:
            fig, axes = None, None
        cnt = 0
    else:
        fig, axes = None, None
        cnt = 0

    @partial(jax.jit, static_argnames=("flow_steps", "alpha", "batch_size", "use_target_network", "tau"))
    def step(state, key, flow_steps, alpha, batch_size, bc_only, use_target_network, tau):
        flow_params, critic_params, target_critic_params, flow_opt_state, critic_opt_state = state
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

            # 2. Q-Maximization: integrate flow and maximize Q at the end.
            z_policy = jrandom.normal(k_pol, (batch_size, 2))

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

            # Maximize Q(clean_actions) - use target network if enabled, otherwise main critic
            critic_params_to_use = target_critic_params if use_target_network else critic_params
            q_at_actions = critic_forward(jax.lax.stop_gradient(critic_params_to_use), clean_actions)
            q_loss = -jnp.mean(q_at_actions)

            # BC warmup: ignore q_loss during the warmup phase.
            total_loss = jax.lax.select(bc_only, alpha * bc_loss, alpha * bc_loss + q_loss)
            return total_loss, (bc_loss, jnp.mean(q_at_actions))

        (flow_loss, (bc_loss, q_val)), flow_grads = jax.value_and_grad(flow_loss_fn, has_aux=True)(flow_params)
        flow_updates, flow_opt_state = flow_opt.update(flow_grads, flow_opt_state, flow_params)
        flow_params = optax.apply_updates(flow_params, flow_updates)

        # Update target network with EMA if enabled
        if use_target_network:
            target_critic_params = jax.tree_util.tree_map(
                lambda p, tp: p * tau + tp * (1 - tau),
                critic_params,
                target_critic_params
            )
        else:
            target_critic_params = critic_params  # Keep in sync when not using target

        logs = (bc_loss, critic_loss, q_val, flow_loss)
        new_state = (flow_params, critic_params, target_critic_params, flow_opt_state, critic_opt_state)
        return new_state, (key, logs)

    history = {
        'flow_bc_loss': [], 'critic_loss': [],
        'flow_q_val': [], 'flow_total': []
    }

    state = (flow_params, critic_params, target_critic_params, flow_opt_state, critic_opt_state)
    bc_policy_params = None

    # Calculate steps per epoch to iterate through entire dataset
    steps_per_epoch = max(1, len(x_1_data) // batch_size)

    for epoch in trange(epochs, desc="FBRAC (JAX)"):
        bc_only = epoch < bc_epochs

        # Accumulate losses across all steps in this epoch
        epoch_bc_loss = 0.0
        epoch_critic_loss = 0.0
        epoch_q_val = 0.0
        epoch_flow_loss = 0.0

        # Run multiple steps per epoch to cover the dataset
        for step_idx in range(steps_per_epoch):
            state, (key, logs) = step(state, key, flow_steps, alpha, batch_size, bc_only, use_target_network, tau)
            bc_loss, critic_loss, q_val, flow_loss = logs
            epoch_bc_loss += float(bc_loss)
            epoch_critic_loss += float(critic_loss)
            epoch_q_val += float(q_val)
            epoch_flow_loss += float(flow_loss)

        # Average losses across steps
        history['flow_bc_loss'].append(epoch_bc_loss / steps_per_epoch)
        history['critic_loss'].append(epoch_critic_loss / steps_per_epoch)
        history['flow_q_val'].append(epoch_q_val / steps_per_epoch)
        history['flow_total'].append(epoch_flow_loss / steps_per_epoch)

        # Capture BC-only snapshot at the end of warmup
        if (not bc_only) and (bc_policy_params is None):
            flow_params_curr, _, _, _, _ = state
            bc_policy_params = flow_params_curr

    flow_params, critic_params, _, _, _ = state
    return flow_params, critic_params, bc_policy_params, history
