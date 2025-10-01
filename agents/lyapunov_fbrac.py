import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value, LatentLyapunovFunction


class LyapunovFBRACAgent(flax.struct.PyTreeNode):
    """Flow Q-learning agent with Lyapunov function learning (simultaneous learning)."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()


    def critic_loss(self, batch, grad_params, rng):
        """Compute the FQL critic loss."""
        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(batch['next_observations'], seed=sample_rng)
        next_actions = jnp.clip(next_actions, -1, 1)

        next_qs = self.network.select('target_critic')(batch['next_observations'], actions=next_actions)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = self.config['reward_scale'] * batch['rewards'] + self.config['discount'] * batch['masks'] * next_q

        q = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        critic_loss = jnp.square(q - target_q).mean()

        # Value Jacobian regularization
        jacobian_reg_loss = 0.0
        jacobian_reg_coeff = self.config.get('value_jacobian_reg', 0.0)
        if jacobian_reg_coeff > 0:
            def q_fn(action):
                qs = self.network.select('critic')(batch['observations'][:1], actions=action[None, :], params=grad_params)
                if self.config['q_agg'] == 'min':
                    return jnp.min(qs)
                else:
                    return jnp.mean(qs)
            
            jacobians = jax.vmap(jax.grad(q_fn))(batch['actions'])
            jacobian_reg_loss = jnp.sum(jacobians**2, axis=-1)
            
            if self.config.get('weight_value_jacobian_reg', False):
                # Q-value-based Weighting Logic
                q_values = self.network.select('critic')(batch['observations'], actions=batch['actions'])
                if self.config['q_agg'] == 'min':
                    q_for_weighting = jnp.min(q_values, axis=0)
                else:
                    q_for_weighting = jnp.mean(q_values, axis=0)
                q_for_weighting = jax.lax.stop_gradient(q_for_weighting)

                weighting_temp = self.config.get('jacobian_weighting_temp', 1.0)
                weights = jax.nn.softmax(-q_for_weighting / weighting_temp) * batch['observations'].shape[0]

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

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss (simplified version without EWR)."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # Add weight decay for actor parameters
        params_to_use = grad_params if grad_params is not None else self.network.params
        actor_params = params_to_use['modules_actor_bc_flow']
        weight_decay = self.config.get('actor_weight_decay', 0.0)

        # L2 regularization on actor parameters
        l2_reg = 0.0
        if weight_decay > 0:
            l2_reg = weight_decay * sum(jnp.sum(p**2) for p in jax.tree_util.tree_leaves(actor_params))

        if self.config['encoder'] is not None:
            observations = self.network.select('actor_bc_flow_encoder')(batch['observations'])
        else:
            observations = batch['observations']

        # BC flow loss
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc_flow')(observations, x_t, t, is_encoded=True, params=grad_params)
        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        # Q loss
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        actor_actions = noises
        
        # action sampling over flow steps
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, actor_actions, t, is_encoded=True, params=grad_params)
            actor_actions = actor_actions + vels / self.config['flow_steps']
        actor_actions = jnp.clip(actor_actions, -1, 1)
        qs = self.network.select('critic')(batch['observations'], actions=actor_actions)
        q = jnp.mean(qs, axis=0)

        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        # Lyapunov safety term for actor - use pre-trained Lyapunov function for safety guidance
        # Note: Expert actions get HIGHER V(s,a) values, so we want actor to maximize V(s,a) for safety
        # We minimize -V(s,a) to maximize V(s,a)
        # import pdb; pdb.set_trace()
        # Use the Lyapunov network with the loaded parameters
        lyapunov_values = self.network.select('lyapunov')(batch['observations'], actor_actions, params=grad_params)
        lyapunov_safety_loss = -jnp.mean(lyapunov_values)  # Minimize -V(s,a) to maximize V(s,a)
        
        # Apply Lyapunov regularization coefficient
        lyapunov_reg_coeff = self.config.get('lyapunov_reg', 0.0)
        lyapunov_actor_loss = lyapunov_reg_coeff * lyapunov_safety_loss

        # Total loss
        actor_loss = self.config['alpha'] * bc_flow_loss + q_loss + l2_reg + lyapunov_actor_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'q_loss': q_loss,
            'q': q.mean(),
            'l2_reg': l2_reg,
            'lyapunov_actor_loss': lyapunov_actor_loss,
            'lyapunov_safety_loss': lyapunov_safety_loss,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss including Lyapunov loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        # Standard FBRAC losses
        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        total_loss = critic_loss + actor_loss
        
        info['total_loss'] = total_loss
        
        return total_loss, info

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

        # Compute gradients but exclude Lyapunov network
        grads, info = jax.grad(loss_fn, has_aux=True)(self.network.params)
        
        # Zero out gradients for Lyapunov network to keep it frozen
        if 'modules_lyapunov' in grads:
            grads['modules_lyapunov'] = jax.tree_util.tree_map(jnp.zeros_like, grads['modules_lyapunov'])
        
        # Apply gradients manually
        new_network = self.network.apply_gradients(grads=grads)
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the one-step policy."""
        action_seed, noise_seed = jax.random.split(seed)
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        )
        actions = noises
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, actions, t)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)

        return actions

    def get_lyapunov_value(self, observations, actions):
        """Get the actual Lyapunov value: -V(s,a) (negated for proper interpretation)."""
        return -self.network.select('lyapunov')(observations, actions)

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
        
        # Define Lyapunov network
        lyapunov_def = LatentLyapunovFunction(
            state_dim=ob_dims[0] if len(ob_dims) == 1 else ob_dims[-1],
            action_dim=action_dim,
            latent_dim=config.get('lyapunov_latent_dim', 16),
            hidden_dim=config.get('lyapunov_hidden_dims', (64, 64)),
            layer_norm=config.get('lyapunov_layer_norm', False)
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, ex_actions, ex_times)),
            lyapunov=(lyapunov_def, (ex_observations, ex_actions)),
        )
        if encoders.get('actor_bc_flow') is not None:
            # Add actor_bc_flow_encoder to ModuleDict to make it separately callable.
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))
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
            agent_name='lyapunov_fbrac',  # Agent name.
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
            q_agg='mean',  # Aggregation method for target Q values.
            alpha=10.0,  # BC coefficient (need to be tuned for each environment).
            flow_steps=10,  # Number of flow steps.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            reward_scale=1.0,  # Reward scale.
            actor_weight_decay=1e-4,  # Weight decay coefficient for actor network.
            value_jacobian_reg=0.0,  # Value Jacobian regularization coefficient.
            weight_value_jacobian_reg=False,  # Weight value Jacobian regularization.
            jacobian_weighting_temp=1.0,  # Temperature for Jacobian weighting.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            
            # Lyapunov-specific hyperparameters
            lyapunov_reg=0.1,  # Lyapunov regularization coefficient (reduced for stability).
            lyapunov_latent_dim=16,  # Latent dimension for Lyapunov network.
            lyapunov_hidden_dims=(512, 512, 512),  # Hidden layer dimensions for Lyapunov network.
            lyapunov_layer_norm=False,  # Whether to use layer normalization in Lyapunov network.
            
            actor_loss='pgbc',  # Actor loss type.
        )
    )
    return config
