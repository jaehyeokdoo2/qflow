"""Network definitions and flow integration for the 2D toy experiment.

All networks are simple MLPs expressed as pure JAX functions.
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom

# Domain bounds for the 2D action space.
DOMAIN_MIN, DOMAIN_MAX = -4.0, 4.0


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


def fourier_time_embed(t, embed_dim=64, max_freq=512.0):
    """
    Fourier positional embedding for time.
    Maps scalar t in [0, 1] to an embed_dim-dimensional vector using sinusoidal features.

    Args:
        t: Time values of shape (batch, 1)
        embed_dim: Dimension of the embedding (must be even)
        max_freq: Maximum frequency for the Fourier features

    Returns:
        Embedding of shape (batch, embed_dim)
    """
    half_dim = embed_dim // 2
    freqs = jnp.exp(jnp.linspace(0.0, jnp.log(max_freq), half_dim))
    args = t * freqs[None, :]
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
    """Integrate flow starting at (x, t_start) to t=1 using the Euler method."""
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
    """Integrate flow from t=0 to t=1 starting from noise z."""
    t0 = jnp.zeros((z.shape[0], 1))
    return integrate_flow_from(flow_params, z, t0, steps=steps)


def integrate_flow_with_time(flow_params, z, steps, t0=None):
    # t0 is ignored, kept for a consistent signature.
    return integrate_flow(flow_params, z, steps=steps)
