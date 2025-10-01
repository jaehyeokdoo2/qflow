import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP


class IDMAgent(flax.struct.PyTreeNode):
    """Inverse Dynamics Model (IDM) agent.

    This agent learns to predict actions from state transitions.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def idm_loss(self, batch, grad_params, rng):
        """Compute the IDM loss (action prediction loss)."""
        # Concatenate current and next observations
        current_obs = batch['observations']
        next_obs = batch['next_observations']
        
        # Input to IDM: concatenated current and next states
        state_transition = jnp.concatenate([current_obs, next_obs], axis=-1)
        
        # Predict action from state transition
        predicted_actions = self.network.select('idm')(state_transition, params=grad_params)
        
        # True actions from the dataset
        true_actions = batch['actions']
        
        # Compute MSE loss
        action_loss = jnp.mean((predicted_actions - true_actions) ** 2)
        
        # Compute additional metrics
        action_mae = jnp.mean(jnp.abs(predicted_actions - true_actions))
        action_cosine_similarity = self.compute_cosine_similarity(predicted_actions, true_actions)
        
        return action_loss, {
            'idm_loss': action_loss,
            'action_mae': action_mae,
            'action_cosine_similarity': action_cosine_similarity,
            'predicted_action_mean': predicted_actions.mean(),
            'predicted_action_std': predicted_actions.std(),
            'true_action_mean': true_actions.mean(),
            'true_action_std': true_actions.std(),
        }

    def compute_cosine_similarity(self, pred_actions, true_actions):
        """Compute cosine similarity between predicted and true actions."""
        # Normalize vectors
        pred_norm = pred_actions / (jnp.linalg.norm(pred_actions, axis=-1, keepdims=True) + 1e-8)
        true_norm = true_actions / (jnp.linalg.norm(true_actions, axis=-1, keepdims=True) + 1e-8)
        
        # Compute cosine similarity
        cosine_sim = jnp.sum(pred_norm * true_norm, axis=-1)
        return jnp.mean(cosine_sim)

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        rng = rng if rng is not None else self.rng
        
        idm_loss, idm_info = self.idm_loss(batch, grad_params, rng)
        
        info = {}
        for k, v in idm_info.items():
            info[k] = v
        
        return idm_loss, info

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def predict_action(self, current_obs, next_obs):
        """Predict action from current and next observations."""
        state_transition = jnp.concatenate([current_obs, next_obs], axis=-1)
        predicted_action = self.network.select('idm')(state_transition)
        return jnp.clip(predicted_action, -1, 1)

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions using the IDM (requires next observations).
        
        Note: This method is not typically used for IDM since we need next states.
        For evaluation, use predict_action() instead.
        """
        # For IDM, we typically don't sample actions without next states
        # This is kept for compatibility but should not be used
        raise NotImplementedError("IDM requires next observations to predict actions. Use predict_action() instead.")

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new IDM agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        # Extract dimensions from example data (like FQL does)
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]
        
        # Input dimension: current_obs + next_obs
        input_dim = ex_observations.shape[-1] * 2

        # Get activation function from config
        activation_name = config.get('activation', 'gelu')
        if activation_name == 'gelu':
            activation = jax.nn.gelu
        elif activation_name == 'relu':
            activation = jax.nn.relu
        elif activation_name == 'tanh':
            activation = jax.nn.tanh
        elif activation_name == 'swish':
            activation = jax.nn.swish
        else:
            activation = jax.nn.gelu  # default

        # Define IDM network with proper output dimension
        idm_def = MLP(
            hidden_dims=(*config['idm_hidden_dims'], action_dim),  # Add action_dim as final layer
            activations=activation,
            activate_final=False,  # No activation on final layer for action prediction
            layer_norm=config['layer_norm'],
        )

        network_info = dict(
            idm=(idm_def, (jnp.zeros((1, input_dim)),)),  # Input: state transition
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        # Update config with extracted dimensions (like FQL does)
        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='idm',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            idm_hidden_dims=(512, 512, 512, 512),  # IDM network hidden dimensions.
            activation='gelu',  # Activation function name (string).
            layer_norm=True,  # Whether to use layer normalization.
        )
    )
    return config
