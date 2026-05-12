"""
FBRAC (Flow-Based Actor-Critic) training for 2D toy experiments.

Directly maximizes Q(flow(z)) through full flow integration.
No inner critic — uses outer critic Q(a) = R(a) directly.
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
from models import init_mlp, flow_forward, critic_forward


def train_fbrac(
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
    offline_epochs=100,
    num_evals=512,
    save_dir='toy_experiment',
    dataset_type='two_spirals',
    n_data=10000,
):
    """
    FBRAC: flow-based actor-critic without inner critic.

    1. Train outer critic Q(a) -> R(a) on clean actions
    2. Train flow policy with BC loss + Q-maximization via full flow integration

    Returns:
        flow_params, critic_params, bc_policy_params, history
    """
    key = jrandom.PRNGKey(42)
    X1 = jnp.asarray(x_1_data)
    R = jnp.asarray(rewards_data)

    k1, k2 = jrandom.split(key, 2)
    hidden_half = max(1, hidden // 2)
    flow_params = init_mlp([3] + [hidden] * num_layers + [hidden_half] + [2], k1)
    critic_params = init_mlp([2] + [hidden] * num_layers + [1], k2)

    target_critic_params = (
        jax.tree_util.tree_map(lambda x: x.copy(), critic_params)
        if use_target_network else critic_params
    )

    flow_opt = optax.adam(lr)
    critic_opt = optax.adam(lr)
    flow_opt_state = flow_opt.init(flow_params)
    critic_opt_state = critic_opt.init(critic_params)

    if plot_every:
        num_plots = offline_epochs // plot_every if offline_epochs > 0 else 0
        fig, axes = (plt.subplots(2, num_plots, figsize=(num_plots * 5, 8))
                     if num_plots > 0 else (None, None))
        cnt = 0
    else:
        fig, axes = None, None
        cnt = 0

    def _l2_norm(grads):
        leaves = jax.tree_util.tree_leaves(grads)
        return jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves))

    @partial(jax.jit, static_argnames=("flow_steps", "alpha", "batch_size", "use_target_network", "tau"))
    def step(state, key, flow_steps, alpha, batch_size, bc_only, use_target_network, tau):
        flow_params, critic_params, target_critic_params, flow_opt_state, critic_opt_state = state
        key, k_idx, k_z, k_t, k_pol = jrandom.split(key, 5)
        idx = jrandom.choice(k_idx, X1.shape[0], shape=(batch_size,), replace=False)
        x_1 = X1[idx]
        r = R[idx]

        # Critic update: Q(x_1) -> R(x_1)
        def critic_loss_fn(params):
            return jnp.mean((critic_forward(params, x_1) - r) ** 2)

        critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(critic_params)
        critic_grad_norm = _l2_norm(critic_grads)
        critic_updates, critic_opt_state = critic_opt.update(critic_grads, critic_opt_state, critic_params)
        critic_params = optax.apply_updates(critic_params, critic_updates)

        # Flow policy update: BC + Q-maximization through full integration
        z = jrandom.normal(k_z, (batch_size, 2))
        t = jrandom.uniform(k_t, (batch_size, 1))
        x_t = (1 - t) * z + t * x_1
        u_target = x_1 - z

        def flow_loss_fn(params):
            v_pred = flow_forward(params, x_t, t)
            bc_loss = jnp.mean((v_pred - u_target) ** 2)

            z_policy = jrandom.normal(k_pol, (batch_size, 2))

            def integrate_for_grad(flow_p, x_start, n_steps):
                dt = 1.0 / n_steps
                x = x_start
                t_curr = jnp.zeros((x.shape[0], 1))
                for _ in range(n_steps):
                    v = flow_forward(flow_p, x, t_curr)
                    x = x + v * dt
                    t_curr = t_curr + dt
                return jnp.clip(x, DOMAIN_MIN, DOMAIN_MAX)

            clean_actions = integrate_for_grad(params, z_policy, flow_steps)
            q_use = target_critic_params if use_target_network else critic_params
            q_loss = -jnp.mean(critic_forward(jax.lax.stop_gradient(q_use), clean_actions))
            total_loss = jax.lax.select(bc_only, alpha * bc_loss, alpha * bc_loss + q_loss)
            return total_loss, (bc_loss, -q_loss)

        (flow_loss, (bc_loss, q_val)), flow_grads = jax.value_and_grad(
            flow_loss_fn, has_aux=True
        )(flow_params)
        flow_grad_norm = _l2_norm(flow_grads)
        flow_updates, flow_opt_state = flow_opt.update(flow_grads, flow_opt_state, flow_params)
        flow_params = optax.apply_updates(flow_params, flow_updates)

        if use_target_network:
            target_critic_params = jax.tree_util.tree_map(
                lambda p, tp: p * tau + tp * (1 - tau), critic_params, target_critic_params,
            )
        else:
            target_critic_params = critic_params

        logs = (bc_loss, critic_loss, q_val, flow_loss, flow_grad_norm, critic_grad_norm)
        new_state = (flow_params, critic_params, target_critic_params, flow_opt_state, critic_opt_state)
        return new_state, (key, logs)

    history = {
        'flow_bc_loss': [], 'critic_loss': [],
        'flow_q_val': [], 'flow_total': [],
        'flow_grad_norm': [], 'critic_grad_norm': [],
    }
    state = (flow_params, critic_params, target_critic_params, flow_opt_state, critic_opt_state)
    bc_policy_params = None
    steps_per_epoch = max(1, len(x_1_data) // batch_size)

    for epoch in trange(epochs, desc="FBRAC"):
        bc_only = epoch < bc_epochs
        epoch_bc = epoch_cl = epoch_q = epoch_fl = 0.0
        epoch_flow_gn = epoch_critic_gn = 0.0

        for _ in range(steps_per_epoch):
            state, (key, logs) = step(
                state, key, flow_steps, alpha, batch_size, bc_only, use_target_network, tau
            )
            bc, cl, q, fl, fgn, cgn = logs
            epoch_bc += float(bc); epoch_cl += float(cl)
            epoch_q += float(q);   epoch_fl += float(fl)
            epoch_flow_gn += float(fgn); epoch_critic_gn += float(cgn)

        history['flow_bc_loss'].append(epoch_bc / steps_per_epoch)
        history['critic_loss'].append(epoch_cl / steps_per_epoch)
        history['flow_q_val'].append(epoch_q / steps_per_epoch)
        history['flow_total'].append(epoch_fl / steps_per_epoch)
        history['flow_grad_norm'].append(epoch_flow_gn / steps_per_epoch)
        history['critic_grad_norm'].append(epoch_critic_gn / steps_per_epoch)

        if epoch == bc_epochs - 1:
            flow_params_curr, _, _, _, _ = state
            bc_policy_params = flow_params_curr

    flow_params, critic_params, _, _, _ = state
    if bc_policy_params is None:
        bc_policy_params = flow_params

    out_dir = f'{save_dir}/{dataset_type}/fbrac'
    os.makedirs(out_dir, exist_ok=True)
    pkl_path = (
        f'{out_dir}/weights_alpha{alpha}_flow{flow_steps}_layers{num_layers}'
        f'_hidden{hidden}_data{n_data}_epochs{epochs}.pkl'
    )
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'flow_params': jax.tree_util.tree_map(np.array, flow_params),
            'critic_params': jax.tree_util.tree_map(np.array, critic_params),
            'bc_policy_params': jax.tree_util.tree_map(np.array, bc_policy_params),
            'history': history,
        }, f)

    return flow_params, critic_params, bc_policy_params, history
