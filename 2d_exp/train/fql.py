"""
FQL (Flow Q-Learning) training for 2D toy experiments.

Components: FlowPolicy (teacher), OneStepPolicy, OuterCritic
Policy improvement: maximize Q(policy(z))
"""
import os
import pickle
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax
import matplotlib.pyplot as plt
from tqdm import trange

from datasets import DOMAIN_MIN, DOMAIN_MAX
from models import (
    init_mlp,
    flow_forward,
    onestep_forward,
    critic_forward,
    integrate_flow,
)


def _plot_progress(flow_params, policy_params, critic_params, axes, cnt, epoch,
                   flow_steps, n_samples, domain_min, domain_max):
    z_eval = jrandom.normal(jrandom.PRNGKey(999), (n_samples, 2))
    policy_actions = np.array(onestep_forward(policy_params, z_eval))

    ax = axes[0, cnt]
    ax.scatter(policy_actions[:, 0], policy_actions[:, 1], s=15, alpha=0.7, edgecolors='none')
    ax.set_title(f'One-Step Policy (Epoch {epoch})')
    ax.set_xlabel('x₁'); ax.set_ylabel('x₂')
    ax.set_xlim(domain_min, domain_max); ax.set_ylim(domain_min, domain_max)
    ax.set_aspect('equal')

    ax3 = axes[1, cnt]
    resolution = 50
    xs = np.linspace(domain_min, domain_max, resolution)
    ys = np.linspace(domain_min, domain_max, resolution)
    X, Y = np.meshgrid(xs, ys)
    positions = jnp.array(np.stack([X.ravel(), Y.ravel()], axis=-1))
    q_values = np.array(critic_forward(critic_params, positions)).reshape(resolution, resolution)
    ax3.imshow(q_values, extent=[domain_min, domain_max, domain_min, domain_max],
               origin='lower', cmap='viridis', aspect='equal')
    ax3.set_title(f'Critic Q(a) (Epoch {epoch})')
    ax3.set_xlabel('x₁'); ax3.set_ylabel('x₂')


def train_fql(
    x_1_data,
    rewards_data,
    epochs=2100,
    lr=3e-4,
    alpha=1.0,
    flow_steps=10,
    hidden=128,
    num_layers=4,
    plot_every=None,
    use_target_network=False,
    tau=0.005,
    batch_size=4096,
    bc_epochs=2000,
    num_evals=512,
    save_dir='toy_experiment',
    dataset_type='two_spirals',
    n_data=10000,
):
    """
    FQL-style actor-critic training (JAX).

    1. Train flow teacher via BC (flow matching)
    2. Train outer critic Q(a) -> R(a)
    3. Train one-step policy: distillation + Q-maximization

    Returns:
        flow_params, policy_params, critic_params, bc_policy_params, history
    """
    key = jrandom.PRNGKey(42)
    X1 = jnp.asarray(x_1_data)
    R = jnp.asarray(rewards_data)

    k1, k2, k3 = jrandom.split(key, 3)
    hidden_half = max(1, hidden // 2)
    flow_params = init_mlp([3] + [hidden] * num_layers + [hidden_half] + [2], k1)
    policy_params = init_mlp([2] + [hidden] * num_layers + [hidden_half] + [2], k2)
    critic_params = init_mlp([2] + [hidden] * num_layers + [1], k3)

    target_critic_params = (
        jax.tree_util.tree_map(lambda x: x.copy(), critic_params)
        if use_target_network else critic_params
    )

    flow_opt = optax.adam(lr)
    policy_opt = optax.adam(lr)
    critic_opt = optax.adam(lr)
    flow_opt_state = flow_opt.init(flow_params)
    policy_opt_state = policy_opt.init(policy_params)
    critic_opt_state = critic_opt.init(critic_params)

    def _l2_norm(grads):
        leaves = jax.tree_util.tree_leaves(grads)
        return jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves))

    @partial(jax.jit, static_argnames=("flow_steps", "alpha", "batch_size", "use_target_network", "tau"))
    def step(state, key, flow_steps, alpha, batch_size, bc_only, use_target_network, tau):
        flow_params, policy_params, critic_params, target_critic_params, \
            flow_opt_state, policy_opt_state, critic_opt_state = state
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
        flow_grad_norm = _l2_norm(flow_grads)
        flow_updates, flow_opt_state = flow_opt.update(flow_grads, flow_opt_state, flow_params)
        flow_params = optax.apply_updates(flow_params, flow_updates)

        def critic_loss_fn(params):
            q_pred = critic_forward(params, x_1)
            return jnp.mean((q_pred - r) ** 2)

        critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(critic_params)
        critic_grad_norm = _l2_norm(critic_grads)
        critic_updates, critic_opt_state = critic_opt.update(critic_grads, critic_opt_state, critic_params)
        critic_params = optax.apply_updates(critic_params, critic_updates)

        z_policy = jrandom.normal(k_pol, (batch_size, 2))
        teacher_output = jax.lax.stop_gradient(integrate_flow(flow_params, z_policy, steps=flow_steps))

        def policy_loss_fn(params):
            policy_output = onestep_forward(params, z_policy)
            distill_loss = jnp.mean((policy_output - teacher_output) ** 2)
            critic_params_to_use = target_critic_params if use_target_network else critic_params
            q_at_policy = critic_forward(critic_params_to_use, policy_output)
            total_loss = jax.lax.select(
                bc_only,
                alpha * distill_loss,
                alpha * distill_loss - jnp.mean(q_at_policy),
            )
            return total_loss, (distill_loss, jnp.mean(q_at_policy))

        (policy_loss, (distill_loss, q_val)), policy_grads = jax.value_and_grad(
            policy_loss_fn, has_aux=True
        )(policy_params)
        policy_grad_norm = _l2_norm(policy_grads)
        policy_updates, policy_opt_state = policy_opt.update(policy_grads, policy_opt_state, policy_params)
        policy_params = optax.apply_updates(policy_params, policy_updates)

        if use_target_network:
            target_critic_params = jax.tree_util.tree_map(
                lambda p, tp: p * tau + tp * (1 - tau),
                critic_params, target_critic_params,
            )
        else:
            target_critic_params = critic_params

        logs = (flow_loss, critic_loss, distill_loss, q_val, policy_loss,
                flow_grad_norm, critic_grad_norm, policy_grad_norm)
        new_state = (
            flow_params, policy_params, critic_params, target_critic_params,
            flow_opt_state, policy_opt_state, critic_opt_state,
        )
        return new_state, (key, logs)

    history = {
        'flow_loss': [], 'critic_loss': [],
        'policy_distill': [], 'policy_q': [], 'policy_total': [],
        'flow_grad_norm': [], 'critic_grad_norm': [], 'policy_grad_norm': [],
    }

    if plot_every:
        num_plots = epochs // plot_every
        fig, axes = plt.subplots(2, num_plots, figsize=(num_plots * 5, 8))
    else:
        fig, axes = None, None

    cnt = 0
    state = (
        flow_params, policy_params, critic_params, target_critic_params,
        flow_opt_state, policy_opt_state, critic_opt_state,
    )
    bc_policy_params = None
    steps_per_epoch = max(1, len(x_1_data) // batch_size)

    for epoch in trange(epochs, desc="FQL"):
        bc_only = epoch < bc_epochs

        epoch_flow_loss = epoch_critic_loss = epoch_distill_loss = 0.0
        epoch_q_val = epoch_policy_loss = 0.0
        epoch_flow_gn = epoch_critic_gn = epoch_policy_gn = 0.0

        for _ in range(steps_per_epoch):
            state, (key, logs) = step(
                state, key, flow_steps, alpha, batch_size, bc_only, use_target_network, tau
            )
            fl, cl, dl, qv, pl, fgn, cgn, pgn = logs
            epoch_flow_loss += float(fl)
            epoch_critic_loss += float(cl)
            epoch_distill_loss += float(dl)
            epoch_q_val += float(qv)
            epoch_policy_loss += float(pl)
            epoch_flow_gn += float(fgn)
            epoch_critic_gn += float(cgn)
            epoch_policy_gn += float(pgn)

        history['flow_loss'].append(epoch_flow_loss / steps_per_epoch)
        history['critic_loss'].append(epoch_critic_loss / steps_per_epoch)
        history['policy_distill'].append(epoch_distill_loss / steps_per_epoch)
        history['policy_q'].append(epoch_q_val / steps_per_epoch)
        history['policy_total'].append(epoch_policy_loss / steps_per_epoch)
        history['flow_grad_norm'].append(epoch_flow_gn / steps_per_epoch)
        history['critic_grad_norm'].append(epoch_critic_gn / steps_per_epoch)
        history['policy_grad_norm'].append(epoch_policy_gn / steps_per_epoch)

        if epoch == bc_epochs - 1:
            _, policy_params_curr, _, _, _, _, _ = state
            bc_policy_params = policy_params_curr

        if plot_every is not None and (epoch + 1) % plot_every == 0:
            flow_params_curr, policy_params_curr, critic_params_curr, _, _, _, _ = state
            _plot_progress(
                flow_params_curr, policy_params_curr, critic_params_curr,
                axes, cnt, epoch + 1, flow_steps, num_evals,
                DOMAIN_MIN, DOMAIN_MAX,
            )
            cnt += 1

    flow_params, policy_params, critic_params, _, _, _, _ = state
    if bc_policy_params is None:
        bc_policy_params = policy_params

    if plot_every is not None and (epochs % plot_every != 0):
        flow_params_curr, policy_params_curr, critic_params_curr, _, _, _, _ = state
        _plot_progress(
            flow_params_curr, policy_params_curr, critic_params_curr,
            axes, cnt, epochs, flow_steps, num_evals, DOMAIN_MIN, DOMAIN_MAX,
        )

    if plot_every is not None:
        plt.tight_layout()
        out_dir = f'{save_dir}/{dataset_type}/fql'
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(
            f'{out_dir}/alpha{alpha}_flow{flow_steps}_layers{num_layers}'
            f'_hidden{hidden}_data{n_data}_epochs{epochs}_training_progress.png',
            dpi=100, bbox_inches='tight',
        )
        plt.close()

    out_dir = f'{save_dir}/{dataset_type}/fql'
    os.makedirs(out_dir, exist_ok=True)
    pkl_path = (
        f'{out_dir}/weights_alpha{alpha}_flow{flow_steps}_layers{num_layers}'
        f'_hidden{hidden}_data{n_data}_epochs{epochs}.pkl'
    )
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'flow_params': jax.tree_util.tree_map(np.array, flow_params),
            'policy_params': jax.tree_util.tree_map(np.array, policy_params),
            'critic_params': jax.tree_util.tree_map(np.array, critic_params),
            'bc_policy_params': jax.tree_util.tree_map(np.array, bc_policy_params),
            'history': history,
        }, f)

    return flow_params, policy_params, critic_params, bc_policy_params, history
