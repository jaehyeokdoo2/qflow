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

# Global cache for compiled gradient functions
_COMPILED_GRAD_FNS = {}


class IFQLAgent(flax.struct.PyTreeNode):
    """Implicit flow Q-learning (IFQL) agent.

    IFQL is the flow variant of implicit diffusion Q-learning (IDQL).
    """

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

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the behavioral flow-matching actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_flow')(batch['observations'], x_t, t, params=grad_params)
        actor_loss = jnp.mean((pred - vel) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
        }

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
        return_full_candidates=False,
    ):
        """Sample actions from the actor."""
        orig_observations = observations
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_flow_encoder')(observations)
        action_seed, noise_seed = jax.random.split(seed)

        # Sample `num_samples` noises and propagate them through the flow.
        actions = jax.random.normal(
            action_seed,
            (
                self.config['num_samples'],
                *observations.shape[:-1],
                self.config['action_dim'],
            ),
        )
        n_observations = jnp.repeat(jnp.expand_dims(observations, 0), self.config['num_samples'], axis=0)
        n_orig_observations = jnp.repeat(jnp.expand_dims(orig_observations, 0), self.config['num_samples'], axis=0)

        for i in range(self.config['flow_steps']):
            t = jnp.full((self.config['num_samples'], *observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_flow')(n_observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        full_candidates = actions

        # Pick the action with the highest Q-value.
        q = self.network.select('critic')(n_orig_observations, actions=actions).min(axis=0)
        actions = actions[jnp.argmax(q)]

        if return_full_candidates:
            return actions, full_candidates
        else:
            return actions

    def _get_energy_grad_fn(self, energy_fn):
        """
        Get or create a pre-compiled gradient function for the energy function.
        This avoids recompiling the gradient computation in every flow step.
        
        Args:
            energy_fn: Energy function that takes (observations, actions) and returns energy values
            
        Returns:
            energy_grad_fn: Pre-compiled gradient function
        """
        # Create a unique key for this energy function (using its id)
        energy_fn_id = id(energy_fn)
        
        # Use global module-level cache
        global _COMPILED_GRAD_FNS
        
        if energy_fn_id not in _COMPILED_GRAD_FNS:
            # Pre-compile the gradient function once with JIT compilation
            def energy_grad_fn(observations, actions):
                def energy_sum_fn(actions):
                    energy_values = energy_fn(observations, actions)
                    return jnp.sum(energy_values)
                return jax.grad(energy_sum_fn)(actions)
            
            # JIT compile the gradient function for maximum performance
            _COMPILED_GRAD_FNS[energy_fn_id] = jax.jit(energy_grad_fn)
        
        return _COMPILED_GRAD_FNS[energy_fn_id]

    def sample_action_with_guidance(
        self,
        observations,
        energy_fn,
        seed=None,
        guidance_coeff=1.0,
        temperature=1.0,
        partial_guidance=-1,
    ):
        """
        Sample actions from the policy with energy-based guidance (FBRAC-style),
        adapted to IFQL's multi-sample generation and action selection.
        """
        # Scheduling and options (matching FBRAC)
        rescale_strategy = "normalize"
        pred_clean = True
        guidance_scheduling = "fixed"
        last_step_guidance = True

        if guidance_coeff == 0.0:
            partial_guidance = 0

        def get_guidance_strength(i, exp_k=5):
            """Treat the guidance_coeff as max strength"""
            # If guidance_coeff is 0, always return 0 regardless of scheduling
            if guidance_coeff == 0.0:
                return 0.0
                
            min_guidance = 1e-4

            if guidance_scheduling == "fixed":
                return guidance_coeff
            elif guidance_scheduling == "linear":
                return min_guidance + (guidance_coeff - min_guidance) * (i+1) / flow_steps
            elif guidance_scheduling == "exp":
                factor = (jnp.exp((i+1)*exp_k/flow_steps) - 1)/(jnp.exp(exp_k) - 1)
                return min_guidance + (guidance_coeff - min_guidance) * factor
            else:
                raise ValueError(f"Invalid guidance_scheduling: {guidance_scheduling}")
        
        energy_vals = []
        gradient_vals = []
        cosine_sim_vals = []
        
        # Split seed: one for action sampling, one for guidance
        action_seed, guidance_rng = jax.random.split(seed)
        
        # Handle encoder if needed
        orig_observations = observations
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_flow_encoder')(observations)

        num_samples = self.config['num_samples']

        if guidance_coeff != 0.0 and partial_guidance != 0:
            num_samples = 1
        
        # Sample `num_samples` noises and propagate through flow with guidance
        actions = jax.random.normal(
            action_seed,
            (
                *observations.shape[:-1],
                num_samples,
                self.config['action_dim'],
            ),
        )
        
        # Determine when to apply guidance (matching FBRAC logic)
        flow_steps = self.config['flow_steps']
        if partial_guidance == -1:
            guidance_start_step = 0
            guidance_end_step = flow_steps
        elif 0 < partial_guidance < flow_steps:
            if last_step_guidance:
                guidance_start_step = flow_steps - partial_guidance
                guidance_end_step = flow_steps
            else:
                guidance_start_step = 0
                guidance_end_step = partial_guidance
        else:
            guidance_start_step = flow_steps
            guidance_end_step = flow_steps
            print(f"Warning: Invalid partial_guidance value {partial_guidance}. Using no guidance.")
        
        # Pre-compile gradient function
        energy_grad_fn = None
        if guidance_start_step < flow_steps:
            try:
                energy_grad_fn = self._get_energy_grad_fn(energy_fn)
            except Exception as e:
                print(f"Error pre-compiling gradient function: {e}")
                energy_grad_fn = None
        
        # Tile observations for multi-candidate approach
        n_observations = jnp.repeat(jnp.expand_dims(observations, 0), num_samples, axis=0)
        n_orig_observations = jnp.repeat(jnp.expand_dims(orig_observations, 0), num_samples, axis=0)
        vs = self.network.select('value')(n_orig_observations).min(axis=0)
        
        # Removed EMA guidance smoothing
        
        # Apply flow steps with conditional guidance
        for i in range(flow_steps):
            t = jnp.full((*observations.shape[:-1], num_samples, 1), i / flow_steps)
            
            # Get flow velocity from actor
            vels = self.network.select('actor_flow')(n_observations, actions, t, is_encoded=True)
            
            # Only compute guidance weight if we might need it
            guidance_weight = jnp.where(
                guidance_coeff != 0.0,
                get_guidance_strength(i),
                0.0
            )

            # a_0_t = actions - (1 - t) * vels
            # a_0_t = jnp.clip(a_0_t, -1, 1)
            # qs = self.network.select('critic')(n_orig_observations, a_0_t).min(axis=0)
            # adv = qs - vs
    
            # guidance_weight = guidance_weight * jnp.tanh(adv)
            # guidance_weight = guidance_weight[..., None] # For broadcasting

            def apply_guidance():
                if energy_grad_fn is not None:
                    # Batched gradient computation without vmap
                    # Define Q function that takes entire batch and returns scalar
                    def q_fn_batched(actions_batch):
                        # actions_batch: (num_samples, action_dim)
                        if pred_clean:
                            a_1 = actions_batch + (1 - t) * vels
                        else:
                            a_1 = actions_batch
                        qs = self.network.select('critic')(n_observations, a_1)  # (E, num_samples)
                        # Sum of min Q-values across samples (scalar)
                        return jnp.sum(jnp.min(qs, axis=0))
                    
                    # Compute gradient w.r.t. entire batch (returns same shape as actions)
                    q_grad = jax.grad(q_fn_batched)(actions)
                    guidance_grad_raw = q_grad
                    
                    if rescale_strategy == "normalize":
                        # Normalize to match velocity magnitude
                        guidance_grad_raw = guidance_grad_raw / (1e-8 + jnp.linalg.norm(guidance_grad_raw, axis=-1, keepdims=True))
                    
                    return guidance_grad_raw, 0.0
                else:
                    return jnp.zeros_like(vels), 0.0
            
            def no_guidance():
                return jnp.zeros_like(vels), 0.0
            
            # Check if we should apply guidance
            should_apply = (guidance_coeff != 0.0 and 
                          i >= guidance_start_step and 
                          i < guidance_end_step and
                          energy_grad_fn is not None)
            
            # Use JAX conditional
            guidance_grad_raw, _ = jax.lax.cond(
                should_apply,
                apply_guidance,
                no_guidance
            )
            
            # Use raw guidance gradient without EMA smoothing
            guidance_grad = guidance_grad_raw
            
            # Check for NaN/Inf
            guidance_grad = jnp.where(
                jnp.isnan(guidance_grad) | jnp.isinf(guidance_grad),
                0.0,
                guidance_grad
            )
            guidance_grad = jnp.clip(guidance_grad, -1, 1)

            gradient_vals.append(0.0)
            cosine_sim_vals.append(0.0)
            
            # Only apply guidance weight if we should apply guidance
            effective_guidance_weight = jnp.where(should_apply, guidance_weight, 0.0)
            guided_vels = vels + effective_guidance_weight * guidance_grad
            
            # Update actions
            actions = actions + guided_vels / flow_steps
            energy_vals.append(0.0)
        
        actions = jnp.clip(actions, -1, 1)
        
        # Pick action with highest Q-value
        q = self.network.select('critic')(n_orig_observations, actions=actions).min(axis=0)

        if guidance_coeff == 0.0 or partial_guidance == 0:
            # falls back to rejection sampling
            actions = actions[jnp.argmax(q)]
        else:
            action_expectile = 1.0
            q_sorted_indices = jnp.argsort(q)
            if action_expectile == 1.0:
                expectile_idx = q.shape[0] - 1
            else:
                expectile_idx = int(action_expectile * (q.shape[0] - 1))
            chosen_idx = q_sorted_indices[expectile_idx]
            actions = actions[chosen_idx]
        return actions, energy_vals, gradient_vals, cosine_sim_vals

    # def sample_action_with_guidance(
    #     self,
    #     observations,
    #     energy_fn,
    #     seed=None,
    #     guidance_coeff=1.0,
    #     temperature=1.0,
    #     partial_guidance=-1,
    # ):
    #     """
    #     Sample actions from the policy with energy-based guidance (FBRAC-style),
    #     adapted to IFQL's multi-sample generation and action selection.
    #     """
    #     # Scheduling and options (matching FBRAC)
    #     rescale_strategy = "normalize"
    #     pred_clean = True
    #     guidance_scheduling = "fixed"
    #     last_step_guidance = True

    #     if guidance_coeff == 0.0:
    #         partial_guidance = 0

    #     def get_guidance_strength(i, exp_k=5):
    #         """Treat the guidance_coeff as max strength"""
    #         # If guidance_coeff is 0, always return 0 regardless of scheduling
    #         if guidance_coeff == 0.0:
    #             return 0.0

    #         min_guidance = 1e-4

    #         if guidance_scheduling == "fixed":
    #             return guidance_coeff
    #         elif guidance_scheduling == "linear":
    #             return min_guidance + (guidance_coeff - min_guidance) * (i + 1) / flow_steps
    #         elif guidance_scheduling == "exp":
    #             factor = (jnp.exp((i + 1) * exp_k / flow_steps) - 1) / (jnp.exp(exp_k) - 1)
    #             return min_guidance + (guidance_coeff - min_guidance) * factor
    #         else:
    #             raise ValueError(f"Invalid guidance_scheduling: {guidance_scheduling}")

    #     energy_vals = []
    #     gradient_vals = []
    #     cosine_sim_vals = []

    #     # Split seed: one for action sampling, one for guidance
    #     action_seed, guidance_rng = jax.random.split(seed)

    #     # Handle encoder if needed
    #     orig_observations = observations
    #     if self.config["encoder"] is not None:
    #         observations = self.network.select("actor_flow_encoder")(observations)

    #     # Sample `num_samples` noises and propagate through flow with guidance
    #     actions = jax.random.normal(
    #         action_seed,
    #         (
    #             *observations.shape[:-1],
    #             self.config["num_samples"],
    #             self.config["action_dim"],
    #         ),
    #     )

    #     # Determine when to apply guidance (matching FBRAC logic)
    #     flow_steps = self.config["flow_steps"]
    #     if partial_guidance == -1:
    #         guidance_start_step = 0
    #         guidance_end_step = flow_steps
    #     elif 0 < partial_guidance < flow_steps:
    #         if last_step_guidance:
    #             guidance_start_step = flow_steps - partial_guidance
    #             guidance_end_step = flow_steps
    #         else:
    #             guidance_start_step = 0
    #             guidance_end_step = partial_guidance
    #     else:
    #         guidance_start_step = flow_steps
    #         guidance_end_step = flow_steps
    #         print(f"Warning: Invalid partial_guidance value {partial_guidance}. Using no guidance.")

    #     # Pre-compile gradient function
    #     energy_grad_fn = None
    #     if guidance_start_step < flow_steps:
    #         try:
    #             energy_grad_fn = self._get_energy_grad_fn(energy_fn)
    #         except Exception as e:
    #             print(f"Error pre-compiling gradient function: {e}")
    #             energy_grad_fn = None

    #     # Tile observations for multi-candidate approach
    #     n_observations = jnp.repeat(jnp.expand_dims(observations, 0), self.config["num_samples"], axis=0)
    #     n_orig_observations = jnp.repeat(
    #         jnp.expand_dims(orig_observations, 0), self.config["num_samples"], axis=0
    #     )
    #     vs = self.network.select("value")(n_orig_observations).min(axis=0)

    #     # expectile used for final selection; keep consistent with your original
    #     action_expectile = 0.9

    #     # Apply flow steps with conditional guidance
    #     for i in range(flow_steps):
    #         t = jnp.full((*observations.shape[:-1], self.config["num_samples"], 1), i / flow_steps)

    #         # Get flow velocity from actor
    #         vels = self.network.select("actor_flow")(n_observations, actions, t, is_encoded=True)

    #         # Only compute guidance weight if we might need it
    #         guidance_weight = jnp.where(guidance_coeff != 0.0, get_guidance_strength(i), 0.0)

    #         def apply_guidance():
    #             # Use quantile-steering guidance when gradient machinery is available
    #             if energy_grad_fn is not None:
    #                 def q_fn_batched(actions_batch):
    #                     # actions_batch: (..., num_samples, action_dim)
    #                     if pred_clean:
    #                         a_0 = actions_batch - (1 - t) * vels
    #                         a_0 = jnp.clip(a_0, -1, 1)
    #                     else:
    #                         a_0 = actions_batch

    #                     qs = self.network.select("critic")(n_observations, a_0)  # (E, num_samples, ...)
    #                     q_min = jnp.min(qs, axis=0)  # (num_samples, ...)

    #                     # Reduce per-sample to scalar Q by averaging any remaining dims (keeps simple)
    #                     if q_min.ndim > 1:
    #                         q_per_sample = jnp.mean(q_min.reshape(q_min.shape[0], -1), axis=1)
    #                     else:
    #                         q_per_sample = q_min  # shape (num_samples,)

    #                     # empirical quantile (τ)
    #                     Q_tau = jnp.quantile(q_per_sample, action_expectile)

    #                     # softness parameter tied to temperature (non-zero)
    #                     beta_soft = jnp.maximum(1e-6, 10.0 / (temperature + 1e-8))

    #                     # smooth utility focusing on the upper tail
    #                     u = jax.nn.sigmoid(beta_soft * (q_per_sample - Q_tau))

    #                     # scalar objective: sum utilities
    #                     return jnp.sum(u)

    #                 # gradient of scalar objective wrt batched actions
    #                 q_grad = jax.grad(q_fn_batched)(actions)  # same shape as actions
    #                 guidance_grad = q_grad

    #                 if rescale_strategy == "normalize":
    #                     # Normalize guidance magnitude to match velocity magnitude
    #                     guidance_grad = guidance_grad * jnp.linalg.norm(vels, axis=-1, keepdims=True) / (
    #                         1e-8 + jnp.linalg.norm(guidance_grad, axis=-1, keepdims=True)
    #                     )

    #                 return guidance_grad, 0.0
    #             else:
    #                 return jnp.zeros_like(vels), 0.0

    #         def no_guidance():
    #             return jnp.zeros_like(vels), 0.0

    #         # Check if we should apply guidance
    #         should_apply = (
    #             guidance_coeff != 0.0
    #             and i >= guidance_start_step
    #             and i < guidance_end_step
    #             and energy_grad_fn is not None
    #         )

    #         # Use JAX conditional
    #         guidance_grad, _ = jax.lax.cond(should_apply, apply_guidance, no_guidance)

    #         # Check for NaN/Inf and clip
    #         guidance_grad = jnp.where(jnp.isnan(guidance_grad) | jnp.isinf(guidance_grad), 0.0, guidance_grad)
    #         guidance_grad = jnp.clip(guidance_grad, -1, 1)

    #         # record placeholders for diagnostics (kept simple)
    #         gradient_vals.append(0.0)
    #         cosine_sim_vals.append(0.0)

    #         # Only apply guidance weight if we should apply guidance
    #         effective_guidance_weight = jnp.where(should_apply, guidance_weight, 0.0)
    #         guided_vels = vels + effective_guidance_weight * guidance_grad

    #         # Update actions
    #         actions = actions + guided_vels / flow_steps
    #         energy_vals.append(0.0)

    #     actions = jnp.clip(actions, -1, 1)

    #     # Pick action with highest Q-value or the chosen expectile
    #     q = self.network.select("critic")(n_orig_observations, actions=actions).min(axis=0)
    #     # actions = actions[jnp.argmax(q)]

    #     if guidance_coeff == 0.0 or partial_guidance == 0:
    #         # falls back to rejection sampling
    #         actions = actions[jnp.argmax(q)]
    #     else:
    #         q_sorted_indices = jnp.argsort(q)
    #         if action_expectile == 1.0:
    #             expectile_idx = q.shape[0] - 1
    #         else:
    #             expectile_idx = int(action_expectile * (q.shape[0] - 1))
    #         chosen_idx = q_sorted_indices[expectile_idx]
    #         actions = actions[chosen_idx]
    #     return actions, energy_vals, gradient_vals, cosine_sim_vals

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
        action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['value'] = encoder_module()
            encoders['critic'] = encoder_module()
            encoders['actor_flow'] = encoder_module()

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
        actor_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_flow'),
        )

        network_info = dict(
            value=(value_def, (ex_observations,)),
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_flow=(actor_flow_def, (ex_observations, ex_actions, ex_times)),
        )
        if encoders.get('actor_flow') is not None:
            # Add actor_flow_encoder to ModuleDict to make it separately callable.
            network_info['actor_flow_encoder'] = (encoders.get('actor_flow'), (ex_observations,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_critic'] = params['modules_critic']

        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='ifql',  # Agent name.
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.9,  # IQL expectile.
            num_samples=32,  # Number of action samples for rejection sampling.
            flow_steps=10,  # Number of flow steps.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
