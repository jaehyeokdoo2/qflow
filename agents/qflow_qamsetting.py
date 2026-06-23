import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value, TimeConditionedCritic


def fourier_time_embed(t, embed_dim=64, max_freq=256.0):
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


class QFlowAgent(flax.struct.PyTreeNode):
    """QFlow agent with intermediate value learning and intermediate guidance."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _flatten_actions(self, actions):
        if self.config["action_chunking"]:
            if actions.shape[-2:] == (self.config["horizon_length"], self.config["action_dim"]):
                return actions.reshape(actions.shape[:-2] + (-1,))
            return actions.reshape(actions.shape[:-1] + (-1,))
        if actions.ndim > 2:
            return actions[..., 0, :]
        return actions

    def _get_time_features(self, t):
        if self.config['use_time_embed']:
            return fourier_time_embed(t, embed_dim=self.config['time_embed_dim'])
        return t

    def critic_loss(self, batch, grad_params, rng):
        """
        Critic loss combining:
        1. Outer critic TD loss (s,a) -> Q
        2. Inner critic TD loss (s, x_t, t) -> Q_i (time-conditioned)
        """
        batch_actions = self._flatten_actions(batch["actions"])

        batch_size, action_dim = batch_actions.shape
        rng, next_rng, x_rng, t_rng = jax.random.split(rng, 4)

        # --- Outer critic TD update ---
        next_actions = self._sample_policy_actions(batch['next_observations'][..., -1, :], rng=next_rng)
        next_qs = self.network.select('target_critic')(batch['next_observations'][..., -1, :], actions=next_actions)
        next_q = next_qs.mean(axis=0) - self.config["rho"] * next_qs.std(axis=0)
        target_q_outer = batch['rewards'][..., -1] + \
            (self.config['discount'] ** self.config["horizon_length"]) * batch['masks'][..., -1] * next_q

        q_outer = self.network.select('critic')(batch['observations'], actions=batch_actions, params=grad_params)
        outer_loss = (jnp.square(q_outer - target_q_outer) * batch['valid'][..., -1]).mean()

        # --- Inner critic (time-conditioned) distillation loss ---
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions

        # Uniform continuous time sampling
        t_continuous = jax.random.uniform(t_rng, (batch_size, 1), minval=0.0, maxval=1.0)
        x_t = (1.0 - t_continuous) * x_0 + t_continuous * x_1

        t_features = self._get_time_features(t_continuous)

        inner_qs = self.network.select('tc_critic')(
            batch['observations'], x_t, t_features, params=grad_params
        )
        if inner_qs.shape[0] > 1:
            inner_q_t = inner_qs.mean(axis=0)
        else:
            inner_q_t = inner_qs.squeeze(0)

        # Flow integration to get x_1_pred
        T_steps = self.config['flow_steps']
        max_dt = 1.0 / T_steps
        x_curr = x_t
        t_curr = t_continuous

        for _ in range(self.config['flow_steps']):
            dist_to_end = 1.0 - t_curr
            step_dt = jnp.clip(dist_to_end, 0.0, max_dt)

            # Query policy for velocity
            vels = self.network.select('actor_bc_flow')(
                batch['observations'], x_curr, t_curr, is_encoded=True
            )

            # Euler update
            x_curr = x_curr + vels * step_dt
            t_curr = t_curr + step_dt

        x_1_pred = jnp.clip(x_curr, -1, 1)
        target_q_outers = self.network.select('target_critic')(
            batch['observations'], x_1_pred
        )
        target_value = target_q_outers.mean(axis=0) - self.config["rho"] * target_q_outers.std(axis=0)
        target_value = jax.lax.stop_gradient(target_value)

        distillation_loss = (jnp.square(inner_q_t - target_value) * batch['valid'][..., -1]).mean()
        total_loss = outer_loss + distillation_loss

        info = {
            'outer_loss': outer_loss,
            'distillation_loss': distillation_loss,
            'q_outer_mean': q_outer.mean(),
            'q_inner_mean': inner_q_t.mean(),
        }

        return total_loss, info

    def actor_loss(self, batch, grad_params, rng):
        """Actor loss using 'grad' type: matching target vector field."""
        batch_actions = self._flatten_actions(batch["actions"])

        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions

        t = jax.random.uniform(t_rng, (batch_size, 1), minval=0.0, maxval=1.0)

        x_t = (1.0 - t) * x_0 + t * x_1
        u_t = x_1 - x_0

        observations = batch['observations']

        # Actor prediction (velocity)
        pred_vel = self.network.select('actor_bc_flow')(
            observations, x_t, t, is_encoded=True, params=grad_params
        )

        # Base flow (behavior-cloning target)
        v_base = u_t

        # Value gradient (steering term)
        t_features = self._get_time_features(t)
        q_grad = jax.grad(lambda actions: self.network.select('tc_critic')(
            observations, actions, t_features
        ).mean(axis=0).sum())

        grad_v = q_grad(x_t)
        grad_v = jax.lax.stop_gradient(grad_v)

        beta = 1.0 / (self.config['guidance_lambda'] + 1e-8)
        v_target = v_base + beta * grad_v
        loss_per_sample = jnp.mean((pred_vel - jax.lax.stop_gradient(v_target)) ** 2, axis=-1)

        # Apply valid mask if available
        if 'valid' in batch:
            total_loss = jnp.mean(loss_per_sample * batch['valid'][..., -1])
        else:
            total_loss = jnp.mean(loss_per_sample)

        # No separate BC loss - BC acts as the anchor in v_target
        info = {'total_loss': total_loss}

        return total_loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @staticmethod
    def _update(agent, batch):
        """Static update method for use with jax.lax.scan."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'critic')
        agent.target_update(new_network, 'tc_critic')
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        return self._update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        """Batch update using jax.lax.scan."""
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @jax.jit
    def _sample_policy_actions(
        self,
        observations,
        rng=None,
    ):
        """Sample actions from the flow policy using best-of-n."""
        action_dim = self.config['action_dim'] * (
            self.config['horizon_length'] if self.config["action_chunking"] else 1
        )
        noises = jax.random.normal(
            rng,
            (
                *observations.shape[: -len(self.config['ob_dims'])],  # batch_size
                self.config["best_of_n"], action_dim
            ),
        )
        observations = jnp.repeat(observations[..., None, :], self.config["best_of_n"], axis=-2)

        # Flow integration
        actions = noises
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)

        # Best-of-n selection using critic
        q = self.network.select("critic")(observations, actions).mean(axis=0)
        indices = jnp.argmax(q, axis=-1)

        bshape = indices.shape
        indices = indices.reshape(-1)
        bsize = len(indices)
        actions = jnp.reshape(actions, (-1, self.config["best_of_n"], action_dim))[jnp.arange(bsize), indices, :].reshape(
            bshape + (action_dim,))

        return actions

    @jax.jit
    def sample_actions(
        self,
        observations,
        rng=None,
    ):
        policy_actions = self._sample_policy_actions(observations, rng=rng)
        return jnp.clip(policy_actions, -1, 1)

    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        actions = noises
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]

        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]

        # Handle action chunking
        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions
        policy_actions = full_actions
        policy_action_dim = policy_actions.shape[-1]

        # Define networks.
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=config['num_qs'],
        )

        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=policy_action_dim,
            layer_norm=config['actor_layer_norm'],
        )

        tc_critic_def = TimeConditionedCritic(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=config['num_qs'],
        )

        # Prepare ex_times for tc_critic if embedding is used.
        # Note: ex_times keeps its original shape for actor_bc_flow and must match
        # ex_observations/ex_actions dimensionality. tc_ex_times is adjusted only for
        # the time-conditioned critic.
        if config.get('use_time_embed', False):
            if ex_times.ndim == 0:
                tc_ex_times = fourier_time_embed(
                    ex_times[None, None],
                    embed_dim=config.get('time_embed_dim', 64),
                ).squeeze(0)
            elif ex_times.ndim == 1:
                tc_ex_times = fourier_time_embed(
                    ex_times[None, :],
                    embed_dim=config.get('time_embed_dim', 64),
                ).squeeze(0)
            else:
                tc_ex_times = fourier_time_embed(
                    ex_times,
                    embed_dim=config.get('time_embed_dim', 64),
                )
        else:
            tc_ex_times = ex_times

        network_info = dict(
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, policy_actions, ex_times)),
            critic=(critic_def, (ex_observations, policy_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, policy_actions)),
            tc_critic=(tc_critic_def, (ex_observations, policy_actions, tc_ex_times)),
            target_tc_critic=(copy.deepcopy(tc_critic_def), (ex_observations, policy_actions, tc_ex_times)),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)

        network_tx = optax.adam(learning_rate=config['lr'])

        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']
        params['modules_target_tc_critic'] = params['modules_tc_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        config['policy_action_dim'] = policy_action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='qflow',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),   # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int), # Action dimension (will be set automatically).

            ## Common hyperparameters
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            actor_layer_norm=False,
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            value_layer_norm=True,

            ## Q-chunking hyperparameters
            horizon_length=ml_collections.config_dict.placeholder(int), # Will be set
            action_chunking=False,                                      # Use Q-chunking or just n-step return

            ## RL hyperparameters
            num_qs=10,       # Critic ensemble size
            rho=0.5,        # Pessimistic backup

            discount=0.99,  # Discount factor.
            tau=0.005,      # Target network update rate.
            flow_steps=10,  # Number of flow steps.

            best_of_n=1,    # Best-of-n for computing Q-targets and sampling actions.

            ## Main hyperparameter(s)
            guidance_lambda=1.0,    # Guidance coefficient (inverse temperature for value gradient).

            ## Time-conditioned critic hyperparameters
            use_time_embed=False, # Whether to use Fourier time embedding for TimeConditionedCritic
            time_embed_dim=64, # Dimension of Fourier time embedding
        )
    )
    return config
