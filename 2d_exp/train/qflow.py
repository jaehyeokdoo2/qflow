"""
QFlow training for 2D toy experiments.

Flow-based actor-critic with an inner (time-conditioned) critic V(x, t).
The flow policy is trained to maximize V via value gradient alignment or
VGG-flow matching, backed by an outer critic Q(a) = R(a).
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
from models import init_mlp, apply_mlp, fourier_time_embed, critic_forward


def train_qflow(
    x_1_data,
    rewards_data,
    epochs=2100,
    lr=3e-4,
    alpha=1.0,
    flow_steps=10,
    hidden=128,
    num_layers=4,
    plot_every=None,
    use_time_embed=False,
    time_embed_dim=64,
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
    QFlow: flow-based actor-critic with inner (time-conditioned) critic.

    Architecture:
        - flow_params:  vector field u(x, t) -> velocity (2D)
        - inner_params: time-conditioned value V(x, t) -> scalar
        - outer_params: terminal value Q(x_1) = R(x_1) -> scalar

    Training:
        1. Outer critic: supervised regression Q(x_1) -> R(x_1)
        2. Inner critic: distilled from outer critic via ODE simulation
        3. Flow policy:  BC + value gradient steering (VGG-flow matching or grad_align)

    Returns:
        flow_params, inner_params, bc_policy_params, history
    """
    key = jrandom.PRNGKey(123)
    X1 = jnp.asarray(x_1_data)
    R = jnp.asarray(rewards_data)

    k1, k2, k3, _ = jrandom.split(key, 4)
    flow_input_dim = 2 + 1
    inner_input_dim = 2 + (time_embed_dim if use_time_embed else 1)

    hidden_half = max(1, hidden // 2)
    flow_params = init_mlp([flow_input_dim] + [hidden] * num_layers + [hidden_half] + [2], k1)
    inner_params = init_mlp([inner_input_dim] + [hidden_half] * num_layers + [1], k2)
    outer_params = init_mlp([2] + [hidden_half] * num_layers + [1], k3)

    target_inner_params = (
        jax.tree_util.tree_map(lambda x: x.copy(), inner_params)
        if use_target_network else inner_params
    )
    target_outer_params = (
        jax.tree_util.tree_map(lambda x: x.copy(), outer_params)
        if use_target_network else outer_params
    )

    # ---- Local forward functions ----
    def _flow_forward(params, x, t):
        xt = jnp.concatenate([x, t], axis=-1)
        return apply_mlp(params, xt)

    def _inner_critic_forward(params, x, t):
        if use_time_embed:
            t_embed = fourier_time_embed(t, embed_dim=time_embed_dim)
            xt = jnp.concatenate([x, t_embed], axis=-1)
        else:
            xt = jnp.concatenate([x, t], axis=-1)
        return apply_mlp(params, xt).squeeze(-1)

    def _integrate_from(fp, x, t_start, steps):
        def step_fn(carry, _):
            x_c, t_c = carry
            dt = jnp.minimum(1.0 - t_c, 1.0 / steps)
            x_n = jnp.clip(x_c + _flow_forward(fp, x_c, t_c) * dt, DOMAIN_MIN, DOMAIN_MAX)
            return (x_n, t_c + dt), None
        (x_final, _), _ = jax.lax.scan(step_fn, (x, t_start), None, length=steps)
        return x_final

    def _integrate(fp, z, steps):
        t0 = jnp.zeros((z.shape[0], 1))
        return _integrate_from(fp, z, t0, steps)

    def _l2_norm(grads):
        leaves = jax.tree_util.tree_leaves(grads)
        return jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves))

    # ---- Optimizers ----
    flow_opt = optax.adam(lr)
    inner_opt = optax.adam(lr)
    outer_opt = optax.adam(lr)
    flow_opt_state = flow_opt.init(flow_params)
    inner_opt_state = inner_opt.init(inner_params)
    outer_opt_state = outer_opt.init(outer_params)

    # ---- Plot setup ----
    if plot_every:
        num_plots = epochs // plot_every
        fig, axes = plt.subplots(2, num_plots, figsize=(num_plots * 5, 8))
        cnt = 0

    # ---- JIT-compiled training step ----
    @partial(jax.jit, static_argnames=(
        "flow_steps", "alpha", "batch_size", "use_target_network", "tau"
    ))
    def step(state, key, flow_steps, alpha, batch_size, bc_only, use_target_network, tau):
        flow_params, inner_params, outer_params, \
            target_inner_params, target_outer_params, \
            flow_opt_state, inner_opt_state, outer_opt_state = state

        key, k_idx, k_noise, k_t, k_pol1, k_pol2, _, _ = jrandom.split(key, 8)
        idx = jrandom.choice(k_idx, X1.shape[0], shape=(batch_size,), replace=False)
        x_1 = X1[idx]
        r = R[idx]

        # 1. Outer critic: Q(x_1) -> R(x_1)
        def outer_loss_fn(params):
            return jnp.mean((critic_forward(params, x_1) - r) ** 2)

        outer_loss, outer_grads = jax.value_and_grad(outer_loss_fn)(outer_params)
        outer_grad_norm = _l2_norm(outer_grads)
        outer_updates, outer_opt_state = outer_opt.update(outer_grads, outer_opt_state, outer_params)
        outer_params = optax.apply_updates(outer_params, outer_updates)

        # 2. Inner critic: V(x_t, t) <- Q(integrate(x_t, t))
        x_0 = jrandom.normal(k_noise, (batch_size, 2))
        t = jrandom.uniform(k_t, (batch_size, 1), minval=0.0, maxval=1.0)
        x_t = (1 - t) * x_0 + t * x_1
        x_1_pred = _integrate_from(flow_params, x_t, t, steps=flow_steps)

        outer_use = target_outer_params if use_target_network else outer_params
        target_r = jax.lax.stop_gradient(critic_forward(outer_use, x_1_pred))

        def critic_loss_fn(params):
            v_pred = _inner_critic_forward(params, x_t, t)
            distill_loss = jnp.mean((v_pred - target_r) ** 2)
            return distill_loss, distill_loss

        (critic_loss, distill_loss), critic_grads = jax.value_and_grad(
            critic_loss_fn, has_aux=True
        )(inner_params)
        inner_grad_norm = _l2_norm(critic_grads)
        inner_updates, inner_opt_state = inner_opt.update(critic_grads, inner_opt_state, inner_params)
        inner_params = optax.apply_updates(inner_params, inner_updates)

        # 3. Flow policy: BC + value gradient steering
        x_0_p = jrandom.normal(k_pol1, (batch_size, 2))
        t_p = jrandom.uniform(k_pol2, (batch_size, 1), minval=0.0, maxval=1.0)
        x_t_p = (1 - t_p) * x_0_p + t_p * x_1
        u_target = x_1 - x_0_p
        inner_use = target_inner_params if use_target_network else inner_params

        def flow_loss_fn(params):
            pred_vel = _flow_forward(params, x_t_p, t_p)
            v_base = u_target

            def value_sum_fn(x_in):
                return jnp.sum(_inner_critic_forward(inner_use, x_in, t_p))

            grad_v = jax.lax.stop_gradient(jax.grad(value_sum_fn)(x_t_p))
            beta = 1.0 / (alpha + 1e-8)
            v_target = v_base + beta * grad_v

            bc_loss = jnp.mean((pred_vel - jax.lax.stop_gradient(v_base)) ** 2)
            full_loss = jnp.mean((pred_vel - jax.lax.stop_gradient(v_target)) ** 2)
            total = jax.lax.select(bc_only, bc_loss, full_loss)
            return total, (bc_loss, 0.0)

        (flow_loss, (bc_loss, q_val)), flow_grads = jax.value_and_grad(
            flow_loss_fn, has_aux=True
        )(flow_params)
        flow_grad_norm = _l2_norm(flow_grads)
        flow_updates, flow_opt_state = flow_opt.update(flow_grads, flow_opt_state, flow_params)
        flow_params = optax.apply_updates(flow_params, flow_updates)

        # EMA target network updates
        if use_target_network:
            target_inner_params = jax.tree_util.tree_map(
                lambda p, tp: p * tau + tp * (1 - tau), inner_params, target_inner_params,
            )
            target_outer_params = jax.tree_util.tree_map(
                lambda p, tp: p * tau + tp * (1 - tau), outer_params, target_outer_params,
            )
        else:
            target_inner_params = inner_params
            target_outer_params = outer_params

        logs = (bc_loss, distill_loss, q_val, flow_loss, outer_loss,
                flow_grad_norm, inner_grad_norm, outer_grad_norm)
        new_state = (
            flow_params, inner_params, outer_params,
            target_inner_params, target_outer_params,
            flow_opt_state, inner_opt_state, outer_opt_state,
        )
        return new_state, (key, logs)

    # ---- Training loop ----
    history = {
        'flow_bc_loss': [], 'critic_distill': [],
        'flow_q_loss': [], 'flow_total': [], 'outer_loss': [],
        'flow_grad_norm': [], 'inner_grad_norm': [], 'outer_grad_norm': [],
    }
    bc_policy_params = None
    state = (
        flow_params, inner_params, outer_params,
        target_inner_params, target_outer_params,
        flow_opt_state, inner_opt_state, outer_opt_state,
    )
    steps_per_epoch = max(1, len(x_1_data) // batch_size)

    for epoch in trange(epochs, desc="QFlow"):
        bc_only = epoch < bc_epochs
        epoch_bc = epoch_dist = epoch_q = epoch_fl = epoch_ol = 0.0
        epoch_flow_gn = epoch_inner_gn = epoch_outer_gn = 0.0

        for _ in range(steps_per_epoch):
            state, (key, logs) = step(
                state, key, flow_steps, alpha, batch_size, bc_only,
                use_target_network, tau,
            )
            bc, dist, q, fl, ol, fgn, ign, ogn = logs
            epoch_bc += float(bc); epoch_dist += float(dist)
            epoch_q += float(q);   epoch_fl += float(fl); epoch_ol += float(ol)
            epoch_flow_gn += float(fgn); epoch_inner_gn += float(ign); epoch_outer_gn += float(ogn)

        history['flow_bc_loss'].append(epoch_bc / steps_per_epoch)
        history['critic_distill'].append(epoch_dist / steps_per_epoch)
        history['flow_q_loss'].append(epoch_q / steps_per_epoch)
        history['flow_total'].append(epoch_fl / steps_per_epoch)
        history['outer_loss'].append(epoch_ol / steps_per_epoch)
        history['flow_grad_norm'].append(epoch_flow_gn / steps_per_epoch)
        history['inner_grad_norm'].append(epoch_inner_gn / steps_per_epoch)
        history['outer_grad_norm'].append(epoch_outer_gn / steps_per_epoch)

        if epoch == bc_epochs - 1:
            bc_policy_params = state[0]  # flow_params at end of last BC epoch

        if plot_every is not None and (epoch + 1) % plot_every == 0:
            flow_p, inner_p, *_ = state
            z_eval = jrandom.normal(jrandom.PRNGKey(999), (num_evals, 2))
            flow_actions = np.array(_integrate(flow_p, z_eval, steps=flow_steps))

            ax = axes[0, cnt]
            ax.scatter(flow_actions[:, 0], flow_actions[:, 1], s=15, alpha=0.7, edgecolors='none')
            ax.set_title(f'QFlow Policy (Epoch {epoch + 1})')
            ax.set_xlabel('x₁'); ax.set_ylabel('x₂')
            ax.set_xlim(DOMAIN_MIN, DOMAIN_MAX); ax.set_ylim(DOMAIN_MIN, DOMAIN_MAX)
            ax.set_aspect('equal')

            ax2 = axes[1, cnt]
            res = 50
            xs = np.linspace(DOMAIN_MIN, DOMAIN_MAX, res)
            ys = np.linspace(DOMAIN_MIN, DOMAIN_MAX, res)
            Xg, Yg = np.meshgrid(xs, ys)
            positions = jnp.array(np.stack([Xg.ravel(), Yg.ravel()], axis=-1))
            t_one = jnp.ones((res * res, 1))
            v_vals = np.array(
                _inner_critic_forward(inner_p, positions, t_one)
            ).reshape(res, res)
            ax2.imshow(v_vals, extent=[DOMAIN_MIN, DOMAIN_MAX, DOMAIN_MIN, DOMAIN_MAX],
                       origin='lower', cmap='viridis', aspect='equal')
            ax2.set_title(f'V(x, t=1) (Epoch {epoch + 1})')
            ax2.set_xlabel('x₁'); ax2.set_ylabel('x₂')
            cnt += 1

    flow_params, inner_params, *_ = state
    if bc_policy_params is None:
        bc_policy_params = flow_params

    if plot_every is not None and (epochs % plot_every != 0):
        flow_p, inner_p, *_ = state
        z_eval = jrandom.normal(jrandom.PRNGKey(999), (num_evals, 2))
        flow_actions = np.array(_integrate(flow_p, z_eval, steps=flow_steps))
        ax = axes[0, cnt]
        ax.scatter(flow_actions[:, 0], flow_actions[:, 1], s=15, alpha=0.7, edgecolors='none')
        ax.set_title(f'QFlow Policy (Epoch {epochs})')
        ax.set_xlim(DOMAIN_MIN, DOMAIN_MAX); ax.set_ylim(DOMAIN_MIN, DOMAIN_MAX)
        ax.set_aspect('equal')

    if plot_every is not None:
        plt.tight_layout()
        out_dir = f'{save_dir}/{dataset_type}/qflow'
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(
            f'{out_dir}/alpha{alpha}_flow{flow_steps}_layers{num_layers}'
            f'_hidden{hidden}_data{n_data}_epochs{epochs}_training_progress.png',
            dpi=100, bbox_inches='tight',
        )
        plt.close()

    outer_params_final = state[2]

    out_dir = f'{save_dir}/{dataset_type}/qflow'
    os.makedirs(out_dir, exist_ok=True)
    pkl_path = (
        f'{out_dir}/weights_alpha{alpha}_flow{flow_steps}_layers{num_layers}'
        f'_hidden{hidden}_data{n_data}_epochs{epochs}.pkl'
    )
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'flow_params': jax.tree_util.tree_map(np.array, flow_params),
            'inner_params': jax.tree_util.tree_map(np.array, inner_params),
            'outer_params': jax.tree_util.tree_map(np.array, outer_params_final),
            'bc_policy_params': jax.tree_util.tree_map(np.array, bc_policy_params),
            'history': history,
        }, f)

    return flow_params, inner_params, bc_policy_params, history
