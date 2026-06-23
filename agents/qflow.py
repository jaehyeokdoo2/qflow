import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value, InnerValue, TimeConditionedCritic

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

    def _get_time_features(self, t):
        if self.config['use_time_embed']:
            return fourier_time_embed(t, embed_dim=self.config['time_embed_dim'])
        return t

    def critic_loss(self, batch, grad_params, rng):
        """
        Critic loss combining:
        1. Outer critic TD loss (s,a) -> Q
        2. Inner critic TD loss (s, x_t, u_t) -> Q_i
        """
        batch_size, action_dim = batch['actions'].shape
        rng, next_rng, x_rng, t_rng = jax.random.split(rng, 4)

        # --- Outer critic TD update ---
        next_actions = self.sample_actions(batch['next_observations'], seed=next_rng)
        next_qs = self.network.select('target_critic')(batch['next_observations'], next_actions)
        next_q = next_qs.mean(axis=0) if self.config['q_agg'] == 'mean' else next_qs.min(axis=0)
        target_q_outer = self.config['reward_scale'] * batch['rewards'] + self.config['discount'] * batch['masks'] * next_q

        q_outer = self.network.select('critic')(batch['observations'], batch['actions'], params=grad_params)
        outer_loss = jnp.mean((q_outer - target_q_outer) ** 2)
        
        batch_size = batch['actions'].shape[0]
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']

        # Uniform continuous time sampling
        t_continuous = jax.random.uniform(t_rng, (batch_size, 1), minval=0.0, maxval=1.0)
        x_t = (1.0 - t_continuous) * x_0 + t_continuous * x_1

        t_features = self._get_time_features(t_continuous)
        
        inner_qs = self.network.select('tc_critic')(
            batch['observations'], x_t, t_features, params=grad_params
        )
        if inner_qs.shape[0] > 1:
            inner_q_t = (
                inner_qs.mean(axis=0)
                if self.config['q_agg'] == 'mean'
                else inner_qs.min(axis=0)
            )
        else:
            inner_q_t = inner_qs.squeeze(0)
        

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
        target_value = (
            target_q_outers.mean(axis=0)
            if self.config['q_agg'] == 'mean'
            else target_q_outers.min(axis=0)
        )
        target_value = jax.lax.stop_gradient(target_value)
        distillation_loss = jnp.mean((inner_q_t - target_value) ** 2)
        total_loss = outer_loss + distillation_loss

        info = {
            'outer_loss': outer_loss,
            'distillation_loss': distillation_loss,
            'q_outer_mean': q_outer.mean(),
            'q_inner_mean': inner_q_t.mean(),
        }

        return total_loss, info
    
    def actor_loss(self, batch, grad_params, rng):
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1), minval=0.0, maxval=1.0)

        x_t = (1.0 - t) * x_0 + t * x_1
        u_t = x_1 - x_0 

        observations = batch['observations']

        # Actor prediction (velocity)
        pred_vel = self.network.select('actor_bc_flow')(
            observations, x_t, t, is_encoded=True, params=grad_params
        )
        loss_type=self.config['actor_loss_type']

        if loss_type == "grad":
            # Base flow (behavior-cloning target)
            v_base = u_t

            # Value gradient (steering term)
            t_features = self._get_time_features(t)
            q_grad = jax.grad(lambda actions: self.network.select('tc_critic')(
                observations, actions, t_features).mean(axis=0).sum())
            grad_v = q_grad(x_t)
            grad_v = jax.lax.stop_gradient(grad_v)

            # Target vector matching
            beta = 1.0 / (self.config['alpha'] + 1e-8)
            v_target = v_base + beta * grad_v

            total_loss = jnp.mean((pred_vel - jax.lax.stop_gradient(v_target)) ** 2)
            
            info = {'total_loss': total_loss}

            return total_loss, info
        elif loss_type == "bptt":
            bc_loss = jnp.mean((pred_vel - u_t) ** 2)

            # Q loss.
            rng, noise_rng = jax.random.split(rng)
            noises = jax.random.normal(noise_rng, (batch_size, action_dim))

            # Push forward upto sampled t
            for i in range(self.config['flow_steps']):
                cur_step = i / self.config['flow_steps']
                step_t = jnp.full((batch_size, 1), cur_step)
                remaining_t = t - step_t
                dt = jnp.minimum(1.0 / self.config['flow_steps'], jnp.maximum(remaining_t, 0.0))

                # Apply upto t
                mask = (step_t < t)
                vels = self.network.select('actor_bc_flow')(observations, actor_actions, step_t, is_encoded=True, params=grad_params)
                actor_actions = jnp.where(
                    mask, 
                    actor_actions + vels * dt, 
                    actor_actions
                )

            clipped_sampling = True
        
            if clipped_sampling:
                actor_actions = jnp.clip(actor_actions, -1, 1)
            
            t_features = self._get_time_features(t)
            qs = self.network.select('tc_critic')(observations, actor_actions, t_features)
            q = jnp.mean(qs, axis=0)

            q_loss = -q.mean()
            if self.config['normalize_q_loss']:
                lam = jax.lax.stop_gradient(1 / (jnp.abs(q).mean() + 1e-8))
                q_loss = lam * q_loss

            total_loss = self.config['alpha'] * bc_loss + q_loss
            info = {'total_loss': total_loss, 'bc_loss': bc_loss, 'q_loss': q_loss}
            return total_loss, info
        else:
            raise ValueError(f"Invalid actor loss type: {loss_type}")

            
            

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
        self.target_update(new_network, 'tc_critic')

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

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        # --- Encoders ---
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['inner'] = encoder_module()  # shared encoder for inner networks

        # --- Networks ---
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_ensembles'],
            encoder=encoders.get('critic'),
            use_residual=False,
        )
        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
            use_residual=False,
        )

        tc_critic_def = TimeConditionedCritic(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_ensembles'],
            encoder=encoders.get('tc_critic'),
            use_residual=False,
        )

        # Prepare ex_times for tc_critic if embedding is used
        tc_ex_times = ex_times
        if config['use_time_embed']:
            tc_ex_times = fourier_time_embed(ex_times, embed_dim=config['time_embed_dim'])

        # --- Network info ---
        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, ex_actions, ex_times)),
            tc_critic=(tc_critic_def, (ex_observations, ex_actions, tc_ex_times)),
            target_tc_critic=(copy.deepcopy(tc_critic_def), (ex_observations, ex_actions, tc_ex_times)),
        )

        if encoders.get('actor_bc_flow') is not None:
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        # Set target params for outer critic and optionally inner value
        network_params['modules_target_critic'] = network_params['modules_critic']
        network_params['modules_target_tc_critic'] = network_params['modules_tc_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))



def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='qflow',  # Agent name.
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
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            num_samples=32, # Number of action samples for rejection sampling
            use_time_embed=False, # Whether to use Fourier time embedding for TimeConditionedCritic
            time_embed_dim=64, # Dimension of Fourier time embedding
            num_ensembles=2,
            actor_loss_type="grad",
        )

    )
    return config
