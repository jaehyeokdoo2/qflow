import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, Value


class IQLAgent(flax.struct.PyTreeNode):
    """Implicit Q-learning (IQL) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        """Compute the IQL value loss."""
        q1, q2 = self.network.select('target_critic')(batch['observations'], actions=batch['actions'])
        q = jnp.minimum(q1, q2)
        v = self.network.select('value')(batch['observations'], params=grad_params)
        value_loss = self.expectile_loss(q - v, q - v, self.config['expectile']).mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def critic_loss(self, batch, grad_params):
        """Compute the IQL critic loss."""
        next_v = self.network.select('value')(batch['next_observations'])
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v

        q1, q2 = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()

        # Value Jacobian regularization
        jacobian_reg_loss = 0.0
        jacobian_reg_coeff = self.config.get('value_jacobian_reg', 0.0)
        if jacobian_reg_coeff > 0:
            def q_fn(state):
                q1, q2 = self.network.select('critic')(state[None, :], actions=batch['actions'][:1], params=grad_params)
                return jnp.mean(jnp.minimum(q1, q2))  # Use min like IQL does
            
            jacobians = jax.vmap(jax.grad(q_fn))(batch['observations'])
            jacobian_reg_loss = jnp.sum(jacobians**2, axis=-1)
            
            if self.config['weight_value_jacobian_reg']:
                # --- Value-based Weighting Logic ---
                # 1. Get the value for each state. Stop gradient to use it only for weighting.
                v = self.network.select('value')(batch['observations'])
                v = jax.lax.stop_gradient(v)

                # 2. Calculate weights using softmax over negative values.
                # Lower value -> higher weight. Temperature controls sharpness.
                weighting_temp = self.config.get('jacobian_weighting_temp', 1.0)
                weights = jax.nn.softmax(-v / weighting_temp) * batch['observations'].shape[0] # Multiply by N for mean-like scale

                # 3. Apply weights to the per-state penalty scores
                jacobian_reg_loss = (weights * jacobian_reg_loss).mean()
            else:
                jacobian_reg_loss = jnp.mean(jacobian_reg_loss)

            jacobian_reg_loss = jacobian_reg_coeff * jacobian_reg_loss
            critic_loss += jacobian_reg_loss

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
            'value_jacb_reg_loss': jacobian_reg_loss,
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the actor loss (AWR or DDPG+BC)."""
        # Add weight decay for actor parameters
        # Use current network params if grad_params is None (e.g., during validation)
        params_to_use = grad_params if grad_params is not None else self.network.params
        actor_params = params_to_use['modules_actor']
        weight_decay = self.config.get('actor_weight_decay', 0.0)

        # --- Hessian Regularization (Computationally Expensive) ---
        hessian_reg_coeff = self.config.get('actor_hessian_reg', 0.0)
        hessian_reg_loss = 0.0
        if hessian_reg_coeff > 0:
            def scalar_policy_output(state):
                dist = self.network.select('actor')(state[None, :], params=grad_params)
                return jnp.sum(dist.mode()[0]) # Hessian requires a scalar output

            hessians = jax.vmap(jax.hessian(scalar_policy_output))(batch['observations'])
            # hessians shape: (batch_size, state_dim, state_dim)
            hessian_reg_loss = hessian_reg_coeff * jnp.mean(jnp.sum(hessians**2, axis=(-2, -1)))

         # --- Jacobian Regularization ---
        jacobian_reg_coeff = self.config.get('actor_jacobian_reg', 0.0)
        jacobian_reg_loss = 0.0
        if jacobian_reg_coeff > 0:
            def policy_fn(state):
                # Helper function for a single state -> deterministic action mapping
                dist = self.network.select('actor')(state[None, :], params=grad_params)
                return dist.mode()[0]  # Get deterministic action, remove batch dim

            # Compute Jacobians for the entire batch using vmap
            jacobians = jax.vmap(jax.jacobian(policy_fn))(batch['observations'])
            # jacobians shape: (batch_size, action_dim, state_dim)

            # Calculate the squared Frobenius norm for each Jacobian and average over the batch
            jacobian_reg_loss = jnp.sum(jacobians**2, axis=(-2, -1))
            
            if self.config['weight_jacobian_reg']:
                # --- New Q-Value Weighting Logic ---
                # 1. Get the value (as a proxy for Q) for each state. Stop gradient to use it only for weighting.
                v = self.network.select('value')(batch['observations'])
                v = jax.lax.stop_gradient(v)

                # 2. Calculate weights using softmax over negative values.
                # Lower value -> higher weight. Temperature controls sharpness.
                weighting_temp = self.config.get('jacobian_weighting_temp', 1.0)
                weights = jax.nn.softmax(-v / weighting_temp) * batch['observations'].shape[0] # Multiply by N for mean-like scale

                # 3. Apply weights to the per-state penalty scores
                jacobian_reg_loss = (weights * jacobian_reg_loss).mean()
            else:
                jacobian_reg_loss = jnp.mean(jacobian_reg_loss)

            jacobian_reg_loss = jacobian_reg_coeff * jacobian_reg_loss
        
        # --- L2 regularization ---
        l2_reg = 0.0
        if weight_decay > 0:
            l2_reg = weight_decay * sum(jnp.sum(p**2) for p in jax.tree_util.tree_leaves(actor_params))
        
        if self.config['actor_loss'] == 'awr':
            # AWR loss.
            v = self.network.select('value')(batch['observations'])
            q1, q2 = self.network.select('critic')(batch['observations'], actions=batch['actions'])
            q = jnp.minimum(q1, q2)
            adv = q - v

            exp_a = jnp.exp(adv * self.config['alpha'])
            exp_a = jnp.minimum(exp_a, 100.0)

            dist = self.network.select('actor')(batch['observations'], params=grad_params)
            log_prob = dist.log_prob(batch['actions'])

            actor_loss = -(exp_a * log_prob).mean()
            
            # Add weight decay to loss
            actor_loss = actor_loss + l2_reg

            actor_info = {
                'actor_loss': actor_loss,
                'adv': adv.mean(),
                'bc_log_prob': log_prob.mean(),
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
                'l2_reg': l2_reg,
                'jacb_reg_loss': jacobian_reg_loss,
            }

            return actor_loss, actor_info
        elif self.config['actor_loss'] == 'ddpgbc':
            # DDPG+BC loss.
            dist = self.network.select('actor')(batch['observations'], params=grad_params)
            if self.config['const_std']:
                q_actions = jnp.clip(dist.mode(), -1, 1)
            else:
                q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)
            q1, q2 = self.network.select('critic')(batch['observations'], actions=q_actions)
            q = jnp.minimum(q1, q2)

            # Normalize Q loss by the absolute mean to make the loss scale invariant.
            if self.config['normalize_q_loss']:
                q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean())
            else:
                q_loss = -q.mean()

            if self.config['use_mse']:
                mse = jnp.square(dist.mode() - batch['actions']).sum(axis=-1)
                bc_loss = (self.config['alpha'] * mse).mean()
            else:
                log_prob = dist.log_prob(batch['actions'])

                bc_loss = -(self.config['alpha'] * log_prob).mean()

            actor_loss = q_loss + bc_loss
            
            # Add weight decay to loss
            actor_loss = actor_loss + l2_reg + jacobian_reg_loss + hessian_reg_loss

            return actor_loss, {
                'actor_loss': actor_loss,
                'q_loss': q_loss,
                'bc_loss': bc_loss,
                'q_mean': q.mean(),
                'q_abs_mean': jnp.abs(q).mean(),
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
                'l2_reg': l2_reg,
                'jacb_reg_loss': jacobian_reg_loss,
                'hessian_reg_loss': hessian_reg_loss,
            }
        else:
            raise ValueError(f'Unsupported actor loss: {self.config["actor_loss"]}')

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = value_loss + critic_loss + actor_loss
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
        """Sample actions from the actor."""
        dist = self.network.select('actor')(observations, temperature=temperature)
        actions = dist.sample(seed=seed)
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

        action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['value'] = encoder_module()
            encoders['critic'] = encoder_module()
            encoders['actor'] = encoder_module()

        # Define networks.
        value_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=1,
            encoder=encoders.get('value'),
        )
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        actor_def = Actor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            state_dependent_std=False,
            const_std=config['const_std'],
            encoder=encoders.get('actor'),
        )

        network_info = dict(
            value=(value_def, (ex_observations,)),
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor=(actor_def, (ex_observations,)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_critic'] = params['modules_critic']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='iql',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.9,  # IQL expectile.
            actor_loss='awr',  # Actor loss type ('awr' or 'ddpgbc').
            alpha=10.0,  # Temperature in AWR or BC coefficient in DDPG+BC.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            use_mse=False,  # Whether to use MSE loss for the BC term.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            actor_weight_decay=1e-4,  # Weight decay coefficient for actor network.
            actor_jacobian_reg=0.0,  # Jacobian regularization coefficient.
            actor_hessian_reg=0.0,  # Hessian regularization coefficient.
            weight_jacobian_reg=False,  # Weight Jacobian regularization.
            jacobian_weighting_temp=1.0,  # Temperature for Jacobian weighting.
            value_jacobian_reg=0.0, # Value Jacobian regularization coefficient.
            weight_value_jacobian_reg=False, # Weight value Jacobian regularization.
        )
    )
    return config
