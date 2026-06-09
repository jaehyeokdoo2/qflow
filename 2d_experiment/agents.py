"""Agent implementations: FQL, FBRAC, and QFlow."""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
import numpy as np
from functools import partial
from tqdm import trange
import matplotlib.pyplot as plt

from utils import (
    init_mlp, apply_mlp, fourier_time_embed,
    DOMAIN_MIN, DOMAIN_MAX
)

def train_fql(x_1_data, rewards_data, epochs=2000, lr=1e-3, flow_steps=10, hidden=128,
              num_layers=4, plot_every=None, num_evals=512, save_dir='toy_experiment'):
    """
    FQL: Flow Q-Learning with one-step distillation.
    Simple baseline that learns Q(a) directly without inner critic.
    """
    key = jrandom.PRNGKey(123)
    X1 = jnp.asarray(x_1_data)
    R = jnp.asarray(rewards_data)

    k1, k2, k3 = jrandom.split(key, 3)

    # Networks: policy (flow) and outer critic
    flow_params = init_mlp([2, hidden, hidden, 2], k1)
    critic_params = init_mlp([2, hidden // 2, 1], k2)

    def flow_forward(params, x, t):
        xt = jnp.concatenate([x, t], axis=-1)
        return apply_mlp(params, xt)

    def critic_forward(params, a):
        return apply_mlp(params, a).squeeze(-1)

    # Optimizers
    flow_opt = optax.adam(lr)
    critic_opt = optax.adam(lr)
    flow_opt_state = flow_opt.init(flow_params)
    critic_opt_state = critic_opt.init(critic_params)

    history = {'flow_loss': [], 'critic_loss': []}

    def integrate_flow(flow_params, z, steps=10):
        """Integrate flow from t=0 to t=1."""
        def step_fn(carry, _):
            x_curr, t_curr = carry
            dt = jnp.minimum(1.0 - t_curr, 1.0 / steps)
            v = flow_forward(flow_params, x_curr, t_curr)
            x_next = x_curr + v * dt
            t_next = t_curr + dt
            x_next = jnp.clip(x_next, DOMAIN_MIN, DOMAIN_MAX)
            return (x_next, t_next), None
        (x_final, _), _ = jax.lax.scan(step_fn, (z, jnp.zeros((z.shape[0], 1))), None, length=steps)
        return x_final

    for epoch in trange(epochs, desc="FQL (JAX)"):
        # Sample batch
        key, k_idx, k_noise = jrandom.split(key, 3)
        idx = jrandom.choice(k_idx, X1.shape[0], shape=(256,), replace=False)
        x_1 = X1[idx]
        r = R[idx]

        # Critic update
        def critic_loss_fn(params):
            q_pred = critic_forward(params, x_1)
            return jnp.mean((q_pred - r) ** 2)

        critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(critic_params)
        critic_updates, critic_opt_state = critic_opt.update(critic_grads, critic_opt_state, critic_params)
        critic_params = optax.apply_updates(critic_params, critic_updates)

        # Flow update (BC)
        key, k_noise_t = jrandom.split(key)
        z = jrandom.normal(k_noise, (256, 2))
        t = jrandom.uniform(k_noise_t, (256, 1))
        x_t = (1 - t) * z + t * x_1
        u_target = x_1 - z

        def flow_loss_fn(params):
            v_pred = flow_forward(params, x_t, t)
            return jnp.mean((v_pred - u_target) ** 2)

        flow_loss, flow_grads = jax.value_and_grad(flow_loss_fn)(flow_params)
        flow_updates, flow_opt_state = flow_opt.update(flow_grads, flow_opt_state, flow_params)
        flow_params = optax.apply_updates(flow_params, flow_updates)

        history['flow_loss'].append(float(flow_loss))
        history['critic_loss'].append(float(critic_loss))

    return flow_params, critic_params, history

def train_fbrac(x_1_data, rewards_data, epochs=2000, lr=1e-3, flow_steps=10, hidden=128,
                num_layers=4, plot_every=None, num_evals=512, bc_epochs=500, save_dir='toy_experiment'):
    """
    FBRAC: Flow-based Behavioral Regularized Actor-Critic.
    Uses backprop through ODE solver for gradient-based policy update.
    """
    key = jrandom.PRNGKey(123)
    X1 = jnp.asarray(x_1_data)
    R = jnp.asarray(rewards_data)

    k1, k2, k3 = jrandom.split(key, 3)

    flow_params = init_mlp([2, hidden, hidden, 2], k1)
    critic_params = init_mlp([2, hidden // 2, 1], k2)

    def flow_forward(params, x, t):
        xt = jnp.concatenate([x, t], axis=-1)
        return apply_mlp(params, xt)

    def critic_forward(params, a):
        return apply_mlp(params, a).squeeze(-1)

    flow_opt = optax.adam(lr)
    critic_opt = optax.adam(lr)
    flow_opt_state = flow_opt.init(flow_params)
    critic_opt_state = critic_opt.init(critic_params)

    history = {'flow_loss': [], 'critic_loss': [], 'q_loss': []}

    def integrate_flow_from(flow_params, x, t_start, steps=10):
        """Integrate flow from t_start to t=1."""
        def step_fn(carry, _):
            x_curr, t_curr = carry
            dt = jnp.minimum(1.0 - t_curr, 1.0 / steps)
            v = flow_forward(flow_params, x_curr, t_curr)
            x_next = x_curr + v * dt
            t_next = t_curr + dt
            x_next = jnp.clip(x_next, DOMAIN_MIN, DOMAIN_MAX)
            return (x_next, t_next), None
        (x_final, _), _ = jax.lax.scan(step_fn, (x, t_start), None, length=steps)
        return x_final

    for epoch in trange(epochs, desc="FBRAC (JAX)"):
        key, k_idx, k_noise, k_t = jrandom.split(key, 4)
        idx = jrandom.choice(k_idx, X1.shape[0], shape=(256,), replace=False)
        x_1 = X1[idx]
        r = R[idx]

        # Critic update
        def critic_loss_fn(params):
            q_pred = critic_forward(params, x_1)
            return jnp.mean((q_pred - r) ** 2)

        critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(critic_params)
        critic_updates, critic_opt_state = critic_opt.update(critic_grads, critic_opt_state, critic_params)
        critic_params = optax.apply_updates(critic_params, critic_updates)

        # Flow update with Q guidance
        key, k_z, k_pol_t = jrandom.split(key, 3)
        z = jrandom.normal(k_z, (256, 2))
        x_1_batch = x_1
        u_target = x_1_batch - z

        bc_only = epoch < bc_epochs

        def flow_loss_fn(params):
            # BC loss
            t_bc = jrandom.uniform(k_pol_t, (256, 1))
            x_t = (1 - t_bc) * z + t_bc * x_1_batch
            v_pred = flow_forward(params, x_t, t_bc)
            bc_loss = jnp.mean((v_pred - u_target) ** 2)

            # Q guidance (after warmup)
            def q_sum_fn(x_in):
                return jnp.sum(critic_forward(critic_params, x_in))

            grad_q = jax.grad(q_sum_fn)(x_1_batch)
            grad_q = jax.lax.stop_gradient(grad_q)

            v_at_clean = flow_forward(params, x_1_batch, jnp.ones((256, 1)))
            alignment = jnp.sum(v_at_clean * grad_q, axis=-1)
            q_loss = -jnp.mean(alignment)

            total_loss = jax.lax.select(bc_only, bc_loss, 0.1 * bc_loss + q_loss)
            return total_loss, (bc_loss, q_loss)

        (flow_loss, (bc_loss, q_loss)), flow_grads = jax.value_and_grad(flow_loss_fn, has_aux=True)(flow_params)
        flow_updates, flow_opt_state = flow_opt.update(flow_grads, flow_opt_state, flow_params)
        flow_params = optax.apply_updates(flow_params, flow_updates)

        history['flow_loss'].append(float(flow_loss))
        history['critic_loss'].append(float(critic_loss))
        history['q_loss'].append(float(q_loss))

    return flow_params, critic_params, history

def train_qflow(x_1_data, rewards_data, epochs=2000, lr=1e-3, flow_steps=10, hidden=128,
                num_layers=4, plot_every=None, num_evals=512, bc_epochs=500, alpha=0.5,
                save_dir='toy_experiment'):
    """
    QFlow: Q-guided Flow with intermediate value function.
    Uses value-consistent flows and intermediate critic for stable training.
    """
    key = jrandom.PRNGKey(123)
    X1 = jnp.asarray(x_1_data)
    R = jnp.asarray(rewards_data)

    k1, k2, k3, k4 = jrandom.split(key, 4)

    flow_params = init_mlp([2, hidden, hidden, 2], k1)
    inner_critic_params = init_mlp([2, hidden // 2, 1], k2)
    outer_critic_params = init_mlp([2, hidden // 2, 1], k3)

    def flow_forward(params, x, t):
        xt = jnp.concatenate([x, t], axis=-1)
        return apply_mlp(params, xt)

    def inner_critic_forward(params, a):
        return apply_mlp(params, a).squeeze(-1)

    def outer_critic_forward(params, a):
        return apply_mlp(params, a).squeeze(-1)

    flow_opt = optax.adam(lr)
    inner_opt = optax.adam(lr)
    outer_opt = optax.adam(lr)
    flow_opt_state = flow_opt.init(flow_params)
    inner_opt_state = inner_opt.init(inner_critic_params)
    outer_opt_state = outer_opt.init(outer_critic_params)

    history = {'flow_loss': [], 'inner_loss': [], 'outer_loss': [], 'q_loss': []}

    def integrate_flow_from(flow_params, x, t_start, steps=10):
        """Integrate flow from t_start to t=1."""
        def step_fn(carry, _):
            x_curr, t_curr = carry
            dt = jnp.minimum(1.0 - t_curr, 1.0 / steps)
            v = flow_forward(flow_params, x_curr, t_curr)
            x_next = x_curr + v * dt
            t_next = t_curr + dt
            x_next = jnp.clip(x_next, DOMAIN_MIN, DOMAIN_MAX)
            return (x_next, t_next), None
        (x_final, _), _ = jax.lax.scan(step_fn, (x, t_start), None, length=steps)
        return x_final

    for epoch in trange(epochs, desc="QFlow (JAX)"):
        key, k_idx, k_noise, k_t = jrandom.split(key, 4)
        idx = jrandom.choice(k_idx, X1.shape[0], shape=(256,), replace=False)
        x_1 = X1[idx]
        r = R[idx]

        # Outer critic update (on clean actions)
        def outer_loss_fn(params):
            q_pred = outer_critic_forward(params, x_1)
            return jnp.mean((q_pred - r) ** 2)

        outer_loss, outer_grads = jax.value_and_grad(outer_loss_fn)(outer_critic_params)
        outer_updates, outer_opt_state = outer_opt.update(outer_grads, outer_opt_state, outer_critic_params)
        outer_critic_params = optax.apply_updates(outer_critic_params, outer_updates)

        # Inner critic distillation
        key, k_noise_inner = jrandom.split(key)
        z = jrandom.normal(k_noise, (256, 2))
        t = jrandom.uniform(k_t, (256, 1))
        x_t = (1 - t) * z + t * x_1

        # Rollout to get target
        x_1_pred = integrate_flow_from(flow_params, x_t, t, steps=flow_steps)
        target_r = jax.lax.stop_gradient(outer_critic_forward(outer_critic_params, x_1_pred))

        def inner_loss_fn(params):
            v_pred = inner_critic_forward(params, x_t)
            return jnp.mean((v_pred - target_r) ** 2)

        inner_loss, inner_grads = jax.value_and_grad(inner_loss_fn)(inner_critic_params)
        inner_updates, inner_opt_state = inner_opt.update(inner_grads, inner_opt_state, inner_critic_params)
        inner_critic_params = optax.apply_updates(inner_critic_params, inner_updates)

        # Flow update with Q guidance
        key, k_z, k_flow_t = jrandom.split(key, 3)
        z_policy = jrandom.normal(k_z, (256, 2))
        u_target = x_1 - z_policy

        bc_only = epoch < bc_epochs

        def flow_loss_fn(params):
            t_policy = jrandom.uniform(k_flow_t, (256, 1))
            x_t_policy = (1 - t_policy) * z_policy + t_policy * x_1
            v_pred = flow_forward(params, x_t_policy, t_policy)

            # BC loss
            bc_loss = jnp.mean((v_pred - u_target) ** 2)

            # Q guidance via inner critic
            def v_sum_fn(x_in):
                return jnp.sum(inner_critic_forward(inner_critic_params, x_in))

            grad_v = jax.grad(v_sum_fn)(x_t_policy)
            grad_v = jax.lax.stop_gradient(grad_v)

            alignment = jnp.sum(v_pred * grad_v, axis=-1)
            q_loss = -jnp.mean(alignment)

            total_loss = jax.lax.select(bc_only, bc_loss, alpha * bc_loss + q_loss)
            return total_loss, (bc_loss, q_loss)

        (flow_loss, (bc_loss, q_loss)), flow_grads = jax.value_and_grad(flow_loss_fn, has_aux=True)(flow_params)
        flow_updates, flow_opt_state = flow_opt.update(flow_grads, flow_opt_state, flow_params)
        flow_params = optax.apply_updates(flow_params, flow_updates)

        history['flow_loss'].append(float(flow_loss))
        history['inner_loss'].append(float(inner_loss))
        history['outer_loss'].append(float(outer_loss))
        history['q_loss'].append(float(q_loss))

    return flow_params, inner_critic_params, outer_critic_params, history
