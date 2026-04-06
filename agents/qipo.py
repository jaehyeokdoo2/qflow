import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value


class QIPOAgent(flax.struct.PyTreeNode):
    """QIPO: in-support softmax Q-learning with softmax Q-weighted BC flow actor.

    Critic: soft Bellman backup — next-state value is estimated as a
    softmax-weighted average of Q over M policy-sampled support actions:

        V(x') = sum_j softmax(beta * Q(x', a'_j)) * Q(x', a'_j)

    Actor: BC flow loss weighted by the dataset action's softmax Q weight
    relative to M policy-sampled support actions at the same state:

        w_i = exp(beta * Q(x_i, a_i)) / sum_j exp(beta * Q(x_i, a_{ij}))

    No separate V network is needed.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _softmax_value(self, observations, rng):
        """Compute implicit V(x) via softmax-weighted Q over M policy-sampled actions.

        Args:
            observations: (batch_size, obs_dim)
            rng: random key

        Returns:
            v: (batch_size,) implicit value
            support_qs: (M, batch_size) Q-values for all support actions
        """
        batch_size = observations.shape[0]
        M = self.config['num_support_actions']

        # Sample M sets of noises and run the flow for each.
        # vmap over M rng keys; observations and self are closed over.
        rngs = jax.random.split(rng, M)

        def sample_and_q(rng_i):
            noises = jax.random.normal(rng_i, (batch_size, self.config['action_dim']))
            actions = noises
            for i in range(self.config['flow_steps']):
                t = jnp.full((batch_size, 1), i / self.config['flow_steps'])
                vels = self.network.select('actor_bc_flow')(observations, actions, t)
                actions = actions + vels / self.config['flow_steps']
            actions = jnp.clip(actions, -1, 1)
            qs = self.network.select('target_critic')(observations, actions=actions)
            if self.config['q_agg'] == 'min':
                return qs.min(axis=0)  # (batch_size,)
            else:
                return qs.mean(axis=0)

        support_qs = jax.vmap(sample_and_q)(rngs)  # (M, batch_size)

        # Softmax-weighted average: V(x) = sum_j w_j * Q_j
        log_w = self.config['beta'] * support_qs
        log_w = log_w - log_w.max(axis=0, keepdims=True)  # numerical stability
        w = jnp.exp(log_w)
        w = w / w.sum(axis=0, keepdims=True)
        v = (w * support_qs).sum(axis=0)  # (batch_size,)

        return v

    def critic_loss(self, batch, grad_params, rng):
        """Soft Bellman backup: target value from softmax-weighted Q over M support actions."""
        rng, v_rng = jax.random.split(rng)
        softmax_q_target = self._softmax_value(batch['next_observations'], v_rng)

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * softmax_q_target

        q = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        critic_loss = jnp.square(q - target_q).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
            'softmax_q_target_mean': softmax_q_target.mean(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Softmax Q-weighted BC flow loss.

        The dataset action's BC flow loss is weighted by its softmax Q weight
        relative to M policy-sampled support actions at the same state:

            w_i = exp(beta * Q(x_i, a_i)) / sum_j exp(beta * Q(x_i, a_{ij}))

        where a_{i0} = a_i (dataset) and a_{i1..M} are sampled from the policy.
        No V function needed.
        """
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng, support_rng = jax.random.split(rng, 4)
        M = self.config['num_support_actions']

        # BC flow loss for dataset actions.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)
        bc_flow_loss = jnp.mean(jnp.square(pred - vel), axis=-1)  # (batch_size,)

        # Q(x, a) for dataset actions.
        qs_dataset = self.network.select('critic')(batch['observations'], actions=batch['actions'])
        if self.config['q_agg'] == 'min':
            q_dataset = qs_dataset.min(axis=0)
        else:
            q_dataset = qs_dataset.mean(axis=0)  # (batch_size,)

        # Sample M support actions and compute their Q values.
        support_rngs = jax.random.split(support_rng, M)

        def sample_and_q(rng_i):
            noises = jax.random.normal(rng_i, (batch_size, action_dim))
            actions = noises
            for i in range(self.config['flow_steps']):
                t_step = jnp.full((batch_size, 1), i / self.config['flow_steps'])
                vels = self.network.select('actor_bc_flow')(batch['observations'], actions, t_step)
                actions = actions + vels / self.config['flow_steps']
            actions = jnp.clip(actions, -1, 1)
            qs = self.network.select('critic')(batch['observations'], actions=actions)
            if self.config['q_agg'] == 'min':
                return qs.min(axis=0)
            else:
                return qs.mean(axis=0)

        support_qs = jax.vmap(sample_and_q)(support_rngs)  # (M, batch_size)

        # Softmax weight for the dataset action over the full support set (dataset + M samples).
        all_qs = jnp.concatenate([q_dataset[None], support_qs], axis=0)  # (M+1, batch_size)
        log_w = self.config['beta'] * all_qs
        log_w = log_w - log_w.max(axis=0, keepdims=True)  # numerical stability
        w = jnp.exp(log_w)
        w = w / w.sum(axis=0, keepdims=True)
        w_dataset = jax.lax.stop_gradient(w[0])  # (batch_size,)

        actor_loss = (bc_flow_loss * w_dataset).mean()

        return actor_loss, {
            'actor_loss': actor_loss,
            'w_dataset_mean': w_dataset.mean(),
        }

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

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the flow policy."""
        noises = jax.random.normal(
            seed,
            (*observations.shape[: -len(self.config['ob_dims'])], self.config['action_dim']),
        )
        actions = noises
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, actions, t)
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
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()

        # Define networks.
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, ex_actions, ex_times)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='qipo',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg='mean',  # Aggregation method for Q ensemble.
            flow_steps=10,  # Number of flow steps for action sampling.
            num_support_actions=4,  # Number of policy-sampled support actions M.
            beta=1.0,  # Inverse temperature for softmax Q weighting (both critic target and actor loss).
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
