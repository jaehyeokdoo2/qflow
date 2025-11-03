import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value, TimeDependentValue

# Global cache for compiled gradient functions
_COMPILED_GRAD_FNS = {}


class FBRAC_TDQAgent(flax.struct.PyTreeNode):
    """Flow Q-learning agent with BPTT and time-dependent value function."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the critic loss with time-dependent value function."""
        rng, sample_rng, x_rng, t_rng = jax.random.split(rng, 4)
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

        # === 2. Anchor Loss (Q_t at t=1) ===
        # Q_t(s, a, 1.0) should match the stable Q(s, a)
        # Your convention has clean data at t=1
        q_t_1 = self.network.select('time_dependent_critic')(
            batch['observations'], 
            actions=batch['actions'], 
            times=jnp.full((*batch['observations'].shape[:-1], 1), 1.0), # Use 1.0 for clean
            params=grad_params
        )
        # Use stop_gradient on 'q' so we only train Q_t, not Q
        q_t_anchor_loss = jnp.square(q_t_1 - jax.lax.stop_gradient(q)).mean() 

        # === 3. Q_t Consistency Loss (for t < 1) ===
        batch_size, action_dim = batch['actions'].shape
        
        # A. Get noisy action x_t
        x_0_noise = jax.random.normal(x_rng, (batch_size, action_dim)) # Noise at t=0
        x_1_data = batch['actions']                                   # Data at t=1
        
        # Sample t in [0, 1-dt] to avoid t_denoised > 1
        dt = 1.0 / self.config.get('diffusion_steps', 100) # Define your step size
        t = jax.random.uniform(t_rng, (batch_size, 1), minval=0.0, maxval=1.0 - dt) 
        x_t = (1 - t) * x_0_noise + t * x_1_data                      # Interpolate to get x_t

        # B. Denoise x_t by one step (dt) using the target actor
        # This is a FORWARD Euler step from t to t+dt
        vel = self.network.select('actor_bc_flow')(
            batch['observations'], x_t, t, is_encoded=True
        )
        x_t_denoised = x_t + vel * dt # Denoised state at t+dt
        t_denoised = t + dt                # Denoised time

        # C. Get the consistency target: Q_t_target(s, x_{t+dt}, t+dt)
        target_q_t = self.network.select('target_time_dependent_critic')(
            batch['observations'], 
            actions=jax.lax.stop_gradient(x_t_denoised), 
            times=jax.lax.stop_gradient(t_denoised)
        )
        if self.config['q_agg'] == 'min':
            target_q_t = target_q_t.min(axis=0)
        else:
            target_q_t = target_q_t.mean(axis=0)
        target_q_t = jax.lax.stop_gradient(target_q_t)

        # D. Get the current Q_t prediction: Q_t(s, x_t, t)
        current_q_t = self.network.select('time_dependent_critic')(
            batch['observations'], 
            actions=x_t, 
            times=t, 
            params=grad_params
        )

        # E. Calculate the consistency loss
        q_t_consistency_loss = jnp.square(current_q_t - target_q_t).mean()

        # === 4. Total Combined Loss ===
        # This loss will update parameters for both 'critic' and 'time_dependent_critic'
        total_critic_loss = critic_loss + q_t_anchor_loss + q_t_consistency_loss

        return total_critic_loss, {
            'critic_loss': critic_loss,
            'q_t_anchor_loss': q_t_anchor_loss,
            'q_t_consistency_loss': q_t_consistency_loss,
            'total_critic_loss': total_critic_loss,
            'q_mean': q.mean(),
            'q_t_1_mean': q_t_1.mean(),
            'q_t_current_mean': current_q_t.mean(),
            'q_t_target_mean': target_q_t.mean(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng, support_noise_rng = jax.random.split(rng, 4)


        if self.config['encoder'] is not None:
            observations = self.network.select('actor_bc_flow_encoder')(batch['observations'])
        else:
            observations = batch['observations']

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc_flow')(observations, x_t, t, is_encoded=True, params=grad_params)
        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        # Q loss.
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        actor_actions = noises

        # Policy extraction with Q_t
        # Re-sample inter_i to be in the correct range [0, T-2]
        inter_i = jax.random.randint(t_rng, (batch_size, 1), 0, self.config['flow_steps'] - 1)
        
        inter_actions = jnp.zeros_like(actor_actions)

        # action sampling over flow steps
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, actor_actions, t, is_encoded=True, params=grad_params)
            actor_actions = actor_actions + vels / self.config['flow_steps']
            
            # Extract the actions at the intermediate timestep for each sample
            mask = (inter_i == i).astype(actor_actions.dtype)
            
            # Store actor_actions (which is a_{i+1}) when mask is true
            inter_actions = mask * actor_actions + (1 - mask) * inter_actions

        actor_actions = jnp.clip(actor_actions, -1, 1)
        qs = self.network.select('critic')(batch['observations'], actions=actor_actions)
        q = jnp.mean(qs, axis=0)

        q_loss = -q.mean()

        # Q_t loss
        q_ts = self.network.select(
            'time_dependent_critic')(batch['observations'], 
            actions=inter_actions, 
            # The action is at step i+1, so the time is (i+1)/T
            times=jnp.full((*observations.shape[:-1], 1), (inter_i + 1) / self.config['flow_steps']))
        
        q_t = jnp.mean(q_ts, axis=0)
        q_t_loss = -q_t.mean()


        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        # Total loss.
        actor_loss = self.config['alpha'] * bc_flow_loss + q_loss + self.config['q_t_weight'] * q_t_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'q_loss': q_loss,
            'q': q.mean(),
            'q_t': q_t.mean(),
            'q_t_loss': q_t_loss,
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
        self.target_update(new_network, 'time_dependent_critic')

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
        Sample actions from the policy with energy-based guidance.
        
        Args:
            observations: Current observations
            energy_fn: Energy function that takes (observations, actions) and returns energy values
            seed: Random seed for sampling
            guidance_coeff: Guidance strength coefficient
            temperature: Sampling temperature
            partial_guidance: If -1, apply guidance for entire sampling process.
                            If > 0 and < flow_steps, apply guidance only to last {partial_guidance} steps.
            
        Returns:
            actions: Sampled actions with guidance applied
            energy_vals: List of energy values computed during sampling
            gradient_vals: List of gradient values computed during sampling
            cosine_sim_vals: List of cosine similarity values between Q-gradient and safety gradient
        """
        
        # In place condition for experiment purpose
        rescale_strategy="normalize" # to be move to guidance_evaluation main function
        pred_clean=True
        guidance_scheduling="fixed"
        last_step_guidance=True

        def get_guidance_strength(i, exp_k=5):
            """
            Treat the guidance_coeff as max strength
            """
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
        
        # Original fixed guidance coefficient method
        energy_vals = []
        gradient_vals = []
        cosine_sim_vals = []
        action_seed, noise_seed = jax.random.split(seed)
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        )
        actions = noises
        
        # Determine when to apply guidance based on partial_guidance and last_step_guidance parameters
        flow_steps = self.config['flow_steps']
        if partial_guidance == -1:
            # Apply guidance for entire sampling process
            guidance_start_step = 0
            guidance_end_step = flow_steps
        elif 0 < partial_guidance < flow_steps:
            # Apply guidance to partial steps based on last_step_guidance flag
            if last_step_guidance:
                # Apply guidance only to last {partial_guidance} steps
                guidance_start_step = flow_steps - partial_guidance
                guidance_end_step = flow_steps
            else:
                # Apply guidance only to first {partial_guidance} steps
                guidance_start_step = 0
                guidance_end_step = partial_guidance
        else:
            # Invalid partial_guidance value, default to no guidance
            guidance_start_step = flow_steps
            guidance_end_step = flow_steps
            print(f"Warning: Invalid partial_guidance value {partial_guidance}. Using no guidance.")
        
        # Pre-compile the gradient function once for better performance (only if we'll use guidance)
        energy_grad_fn = None
        if guidance_start_step < flow_steps:
            try:
                energy_grad_fn = self._get_energy_grad_fn(energy_fn)
            except Exception as e:
                print(f"Error pre-compiling gradient function: {e}")
                energy_grad_fn = None
        
        # Apply flow steps with conditional guidance
        for i in range(flow_steps):
            t = jnp.full((*observations.shape[:-1], 1), i / flow_steps)
            
            # Get the flow velocity from the actor
            vels = self.network.select('actor_bc_flow')(observations, actions, t)

            guidance_weight = get_guidance_strength(i)

            # Only compute energy and guidance if guidance_coeff is not zero and energy > -5
            # Use JAX conditional operations for JIT compatibility
            def apply_guidance():
                if energy_grad_fn is not None:
                    # safety_grad = energy_grad_fn(observations, target_actions)

                    # Compute Q value at target_actions, but gradient w.r.t. current actions
                    def q_fn_at_target(a_t):
                        if pred_clean:
                            # Compute target_actions from a_t
                            a_0 = a_t - (1 - t) * vels
                            a_0 = jnp.clip(a_0, -1, 1)
                        else:
                            a_0 = a_t
                        qs = self.network.select('critic')(observations, a_0)
                        return jnp.mean(qs)
                    
                    q_grad = jax.grad(q_fn_at_target)(actions)  # Gradient w.r.t. a_t

                    # guidance_grad = q_grad/jnp.linalg.norm(q_grad) - safety_grad/jnp.linalg.norm(safety_grad) # to match a scale
                    guidance_grad = q_grad

                    if rescale_strategy == "normalize":
                        # normalize
                        guidance_grad = guidance_grad * jnp.linalg.norm(vels) / (1e-8 + jnp.linalg.norm(guidance_grad))
                    
                    return guidance_grad, 0.0
                else:
                    return jnp.zeros_like(vels), 0.0
            
            def no_guidance():
                return jnp.zeros_like(vels), 0.0
            
            # Check if we should apply guidance
            should_apply = (guidance_coeff != 0.0 and 
                          i >= guidance_start_step and 
                          i < guidance_end_step and
                          energy_grad_fn is not None)
            
            # Use JAX conditional to choose guidance or no guidance
            guidance_grad, _ = jax.lax.cond(
                should_apply,
                apply_guidance,
                no_guidance
            )
            
            # Check for NaN/Inf in guidance gradient
            guidance_grad = jnp.where(
                jnp.isnan(guidance_grad) | jnp.isinf(guidance_grad),
                0.0,
                guidance_grad
            )
            guidance_grad = jnp.clip(guidance_grad, -1, 1)

            gradient_vals.append(guidance_grad)
            cosine_sim_vals.append(0.0)
            guided_vels = vels + guidance_weight * guidance_grad
            
            # Update actions
            actions = actions + guided_vels / flow_steps
            # energy_val = energy_fn(observations, actions)
            energy_vals.append(0.0)
        
        actions = jnp.clip(actions, -1, 1)
        return actions, energy_vals, gradient_vals, cosine_sim_vals

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
        time_dependent_critic_def = TimeDependentValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('time_dependent_critic'),
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
            time_dependent_critic=(time_dependent_critic_def, (ex_observations, ex_actions, ex_times)),
            target_time_dependent_critic=(copy.deepcopy(time_dependent_critic_def), (ex_observations, ex_actions, ex_times)),
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, ex_actions, ex_times))
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
        params['modules_target_time_dependent_critic'] = params['modules_time_dependent_critic']
        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='fbrac_tdq',  # Agent name.
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
            q_t_weight=1.0,  # Weight for the Q_t loss.
        )
    )
    return config
