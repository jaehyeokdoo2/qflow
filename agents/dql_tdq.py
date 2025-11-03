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


class DQL_TDQAgent(flax.struct.PyTreeNode):
    """DQL agent with time-dependent critic and diffusion policy."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    betas: Any
    alphas: Any
    sqrt_alphas: Any
    alphas_cumprod: Any
    alphas_cumprod_prev: Any
    sqrt_alphas_cumprod: Any
    sqrt_one_minus_alphas_cumprod: Any

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

        # === 2. Anchor Loss (Q_t at t=0) ===
        # Q_t(s, a, 0.0) should match the stable Q(s, a)
        # In diffusion convention: t=0 is clean data, t=1 is pure noise
        q_t_0 = self.network.select('time_dependent_critic')(
            batch['observations'], 
            actions=batch['actions'], 
            times=jnp.full((*batch['observations'].shape[:-1], 1), 0.0), # Use 0.0 for clean in diffusion
            params=grad_params
        )
        # Use stop_gradient on 'q' so we only train Q_t, not Q
        q_t_anchor_loss = jnp.square(q_t_0 - jax.lax.stop_gradient(q)).mean() 

        # === 3. Q_t Consistency Loss (for t > dt) ===
        batch_size, action_dim = batch['actions'].shape
        
        # A. Get noisy action x_t using diffusion noise schedule
        # In diffusion: x_0 is clean data, x_1 is pure noise
        x_0_data = batch['actions']                                   # Clean data at t=0
        noise = jax.random.normal(x_rng, (batch_size, action_dim))    # Pure noise
        
        # Sample t in [dt, 1] to avoid t_denoised < 0 (we denoise backward)
        dt = 1.0 / self.config['diffusion_steps'] # Define step size
        t = jax.random.uniform(t_rng, (batch_size, 1), minval=dt, maxval=1.0) 
        
        # Use diffusion noise schedule for interpolation: x_t = sqrt_alpha_bar * x_0 + sqrt(1-alpha_bar) * noise
        t_int = (t * (self.config['diffusion_steps'] - 1)).astype(jnp.int32)
        sqrt_alphas_cumprod = self.extract(self.sqrt_alphas_cumprod, t_int, x_0_data.shape)
        sqrt_one_minus_alphas_cumprod = self.extract(self.sqrt_one_minus_alphas_cumprod, t_int, x_0_data.shape)
        x_t = sqrt_alphas_cumprod * x_0_data + sqrt_one_minus_alphas_cumprod * noise

        # B. Denoise x_t by one step (dt) backward using the actor (DDPM reverse step from t to t-dt)
        pred_noise = self.network.select('actor_diffusion')(
            batch['observations'], x_t, t, is_encoded=True
        )
        
        # Compute x_{t-dt} using DDPM reverse process
        t_denoised = t - dt
        t_denoised = jnp.maximum(t_denoised, 0.0)  # Clamp to avoid negative time
        t_denoised_int = (t_denoised * (self.config['diffusion_steps'] - 1)).astype(jnp.int32)
        
        # Get noise schedule parameters for t and t-dt
        betas_t = self.extract(self.betas, t_int, x_t.shape)
        sqrt_alphas_t = self.extract(self.sqrt_alphas, t_int, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t_int, x_t.shape)
        
        # DDPM reverse step: x_{t-dt} = (x_t - beta_t / sqrt(1-alpha_bar_t) * pred_noise) / sqrt(alpha_t)
        x_t_denoised = (x_t / sqrt_alphas_t) - (betas_t / (sqrt_alphas_t * sqrt_one_minus_alphas_cumprod_t)) * pred_noise

        # C. Get the consistency target: Q_t_target(s, x_{t-dt}, t-dt)
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
            'q_t_0_mean': q_t_0.mean(),
            'q_t_current_mean': current_q_t.mean(),
            'q_t_target_mean': target_q_t.mean(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the DQL actor loss with time-dependent Q."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng, noise_rng = jax.random.split(rng, 4)

        if self.config['encoder'] is not None:
            observations = self.network.select('actor_diffusion_encoder')(batch['observations'])
        else:
            observations = batch['observations']

        # Diffusion loss.
        # Action noising (Here, x1 is the pure noise, x0 is the clean)
        x_1 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_0 = batch['actions']
        
        t = jax.random.randint(t_rng, (batch_size, 1), 0, self.config['diffusion_steps']) # For noise schedule extraction
        sqrt_alphas_cumprod = self.extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        t_normalized = t.astype(jnp.float32) / self.config['diffusion_steps']
        x_t = sqrt_alphas_cumprod * x_0 + sqrt_one_minus_alphas_cumprod * x_1

        pred = self.network.select('actor_diffusion')(observations, x_t, t_normalized, is_encoded=True, params=grad_params)
        diffusion_loss = jnp.mean((pred - x_1) ** 2)

        # Q loss.
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        actor_actions = noises
        
        # DDPM action sampling over diffusion steps
        for i in range(self.config['diffusion_steps'])[::-1]:
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['diffusion_steps'])
            t_int = jnp.full((*observations.shape[:-1], 1), i)

            betas = self.extract(self.betas, t_int, actor_actions.shape)
            sqrt_alphas = self.extract(self.sqrt_alphas, t_int, actor_actions.shape)
            sqrt_one_minus_alphas_cumprod = self.extract(self.sqrt_one_minus_alphas_cumprod, t_int, actor_actions.shape)
            pred = self.network.select('actor_diffusion')(observations, actor_actions, t, is_encoded=True, params=grad_params)
            
            # Check for NaN/Inf in predictions and replace with zeros
            pred = jnp.where(
                jnp.isnan(pred) | jnp.isinf(pred),
                0.0,
                pred
            )
            
            # Split seed BEFORE generating noise
            noise_rng, step_noise_rng = jax.random.split(noise_rng)
            step_noise = jax.random.normal(
                step_noise_rng,
                actor_actions.shape
            )

            # Don't add noise in the final step (i=0)
            noise_scale = jnp.where(i > 0, jnp.sqrt(betas), 0.0)
            actor_actions = actor_actions / sqrt_alphas - betas / (sqrt_alphas * sqrt_one_minus_alphas_cumprod) * pred + noise_scale * step_noise
        actor_actions = jnp.clip(actor_actions, -1, 1)

        qs = self.network.select('critic')(batch['observations'], actions=actor_actions)
        q = jnp.mean(qs, axis=0)

        q_loss = -q.mean()

        # Q_t loss - sample intermediate actions during diffusion
        # Sample a random timestep index in range [1, diffusion_steps-1] (we want intermediate steps)
        # We'll capture the action BEFORE denoising at step i
        inter_i = jax.random.randint(t_rng, (batch_size, 1), 1, self.config['diffusion_steps'])
        
        inter_actions = jnp.zeros_like(actor_actions)
        temp_actions = noises

        # action sampling over diffusion steps to get intermediate actions
        for i in range(self.config['diffusion_steps'])[::-1]:
            # Capture action BEFORE denoising (action is at time t=i/T)
            mask = (inter_i == i).astype(temp_actions.dtype)
            inter_actions = mask * temp_actions + (1 - mask) * inter_actions
            
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['diffusion_steps'])
            t_int = jnp.full((*observations.shape[:-1], 1), i)

            betas = self.extract(self.betas, t_int, temp_actions.shape)
            sqrt_alphas = self.extract(self.sqrt_alphas, t_int, temp_actions.shape)
            sqrt_one_minus_alphas_cumprod = self.extract(self.sqrt_one_minus_alphas_cumprod, t_int, temp_actions.shape)
            pred = self.network.select('actor_diffusion')(observations, temp_actions, t, is_encoded=True, params=grad_params)
            
            # Check for NaN/Inf in predictions and replace with zeros
            pred = jnp.where(
                jnp.isnan(pred) | jnp.isinf(pred),
                0.0,
                pred
            )
            
            # Split seed BEFORE generating noise
            noise_rng, step_noise_rng = jax.random.split(noise_rng)
            step_noise = jax.random.normal(
                step_noise_rng,
                temp_actions.shape
            )

            # Don't add noise in the final step (i=0)
            noise_scale = jnp.where(i > 0, jnp.sqrt(betas), 0.0)
            temp_actions = temp_actions / sqrt_alphas - betas / (sqrt_alphas * sqrt_one_minus_alphas_cumprod) * pred + noise_scale * step_noise

        # Q_t loss - evaluate Q_t at the captured intermediate action
        # inter_actions was captured at timestep i, so time should be i/T
        q_ts = self.network.select(
            'time_dependent_critic')(batch['observations'], 
            actions=inter_actions, 
            # The action was captured at step i, so the time is i/T
            times=jnp.full((*observations.shape[:-1], 1), inter_i / self.config['diffusion_steps']))
        
        q_t = jnp.mean(q_ts, axis=0)
        q_t_loss = -q_t.mean()

        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        # Total loss.
        actor_loss = self.config['alpha'] * diffusion_loss + q_loss + self.config['q_t_weight'] * q_t_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'diffusion_loss': diffusion_loss,
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
        """Sample actions from the policy using DDPM sampling."""
        action_seed, noise_seed = jax.random.split(seed)
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        )
        actions = noises

        # DDPM Sampling (following DDPM as described in DQL)
        for i in range(self.config['diffusion_steps'])[::-1]:
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['diffusion_steps'])
            t_int = jnp.full((*observations.shape[:-1], 1), i)

            betas = self.extract(self.betas, t_int, actions.shape)
            sqrt_alphas = self.extract(self.sqrt_alphas, t_int, actions.shape)
            sqrt_one_minus_alphas_cumprod = self.extract(self.sqrt_one_minus_alphas_cumprod, t_int, actions.shape)
            pred = self.network.select('actor_diffusion')(observations, actions, t)
            
            # Check for NaN/Inf in predictions and replace with zeros
            pred = jnp.where(
                jnp.isnan(pred) | jnp.isinf(pred),
                0.0,
                pred
            )
            
            # Split seed BEFORE generating noise
            noise_seed, step_noise_seed = jax.random.split(noise_seed)
            step_noise = jax.random.normal(
                step_noise_seed,
                (
                    *observations.shape[: -len(self.config['ob_dims'])],
                    self.config['action_dim'],
                ),
            )

            # Don't add noise in the final step (i=0)
            noise_scale = jnp.where(i > 0, jnp.sqrt(betas), 0.0)
            actions = actions / sqrt_alphas - betas / (sqrt_alphas * sqrt_one_minus_alphas_cumprod) * pred + noise_scale * step_noise
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
                            If > 0 and < diffusion_steps, apply guidance only to last {partial_guidance} steps.
            
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

        def get_guidance_strength(i, exp_k=5):
            """
            Treat the guidance_coeff as max strength
            """
            min_guidance = 0.0

            if guidance_scheduling == "fixed":
                return guidance_coeff
            elif guidance_scheduling == "neg_linear": # start from guidance_coeff and decrease linearly
                return guidance_coeff - (guidance_coeff - min_guidance) * (i+1) / diffusion_steps
            elif guidance_scheduling == "linear":
                return min_guidance + (guidance_coeff - min_guidance) * (i+1) / diffusion_steps
            elif guidance_scheduling == "exp":
                factor = (jnp.exp((i+1)*exp_k/diffusion_steps) - 1)/(jnp.exp(exp_k) - 1)
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

        # Determine when to apply guidance based on partial_guidance parameter
        diffusion_steps = self.config['diffusion_steps']
        if partial_guidance == -1:
            # Apply guidance for entire sampling process
            guidance_start_step = 0
        elif 0 < partial_guidance < diffusion_steps:
            # Apply guidance only to last {partial_guidance} steps
            guidance_start_step = diffusion_steps - partial_guidance
        else:
            # Invalid partial_guidance value, default to no guidance
            guidance_start_step = diffusion_steps
            print(f"Warning: Invalid partial_guidance value {partial_guidance}. Using no guidance.")
        
        # Pre-compile the gradient function once for better performance (only if we'll use guidance)
        energy_grad_fn = None
        if guidance_start_step < diffusion_steps:
            try:
                energy_grad_fn = self._get_energy_grad_fn(energy_fn)
            except Exception as e:
                print(f"Error pre-compiling gradient function: {e}")
                energy_grad_fn = None
        
        # Apply DDPM steps with conditional guidance
        for i in range(diffusion_steps)[::-1]:
            t = jnp.full((*observations.shape[:-1], 1), i / diffusion_steps)
            t_int = jnp.full((*observations.shape[:-1], 1), i)

            betas = self.extract(self.betas, t_int, actions.shape)
            sqrt_alphas = self.extract(self.sqrt_alphas, t_int, actions.shape)
            sqrt_alphas_cumprod = self.extract(self.sqrt_alphas_cumprod, t_int, actions.shape)
            sqrt_one_minus_alphas_cumprod = self.extract(self.sqrt_one_minus_alphas_cumprod, t_int, actions.shape)
            
            preds = self.network.select('actor_diffusion')(observations, actions, t)

            # Split seed BEFORE generating noise
            noise_seed, step_noise_seed = jax.random.split(noise_seed)
            step_noise = jax.random.normal(
                    step_noise_seed,
                    (
                        *observations.shape[: -len(self.config['ob_dims'])],
                        self.config['action_dim'],
                    ),
                )

            # Don't add noise in the final step (i=0)
            noise_scale = jnp.where(i > 0, jnp.sqrt(betas), 0.0)

            guidance_weight = get_guidance_strength(i)

            # Only compute guidance if guidance_coeff is not zero
            # Use JAX conditional operations for JIT compatibility
            def apply_guidance():
                if energy_grad_fn is not None:
                    # Compute Q value at target_actions (a_0), but gradient w.r.t. a_t
                    def q_fn_at_target(a_t):
                        if pred_clean:
                            # Compute a_0 from a_t
                            a_0 = (a_t - sqrt_one_minus_alphas_cumprod * preds) / sqrt_alphas_cumprod
                            a_0 = jnp.clip(a_0, -1, 1)
                        else:
                            a_0 = a_t
                        qs = self.network.select('critic')(observations, a_0)
                        return jnp.mean(qs)
                    
                    q_grad = jax.grad(q_fn_at_target)(actions)  # Gradient w.r.t. a_t

                    guidance_grad = q_grad

                    if rescale_strategy == "normalize":
                        # normalize
                        guidance_grad = guidance_grad * jnp.linalg.norm(preds) / jnp.linalg.norm(guidance_grad)
                    
                    return guidance_grad, 0.0
                else:
                    return jnp.zeros_like(preds), 0.0
            
            def no_guidance():
                return jnp.zeros_like(preds), 0.0
            
            # Check if we should apply guidance
            should_apply = (guidance_coeff != 0.0 and 
                          i >= guidance_start_step and 
                          energy_grad_fn is not None)
            
            # Use JAX conditional to choose guidance or no guidance
            guidance_grad, _ = jax.lax.cond(
                should_apply,
                apply_guidance,
                no_guidance
            )

            guidance_grad = jnp.where(
                jnp.isnan(guidance_grad) | jnp.isinf(guidance_grad),
                0.0,
                guidance_grad
            )
            guidance_grad = jnp.clip(guidance_grad, -1, 1)

            gradient_vals.append(0)
            cosine_sim_vals.append(0)
            
            # Combine predicted noise with guidance (convert guidance to noise space)
            # Standard DDPM update: a_t -> a_{t-1}
            actions_pred = actions / sqrt_alphas - betas / (sqrt_alphas * sqrt_one_minus_alphas_cumprod) * preds
            
            # Apply guidance in action space
            actions_guided = actions_pred + guidance_weight * guidance_grad
            
            # Add noise for non-final steps
            actions = actions_guided + noise_scale * step_noise
            energy_vals.append(0)
        
        actions = jnp.clip(actions, -1, 1)
        return actions, energy_vals, gradient_vals, cosine_sim_vals

    def extract(self, a, t, x_shape):
        """
        Extracts values from array a at indices t.
        Equivalent to PyTorch's a.gather(-1, t).

        Args:
            a: Array to extract from (1D array of shape [num_steps]).
            t: Indices (shape [batch, 1] or [batch]).
            x_shape: Original shape to broadcast to.

        Returns:
            Extracted and properly reshaped array.
        """
        b, *_ = t.shape
        t = t.astype(jnp.int32)
        out = a[t]
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

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
            encoders['actor_diffusion'] = encoder_module()

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
        actor_diffusion_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_diffusion'),
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            time_dependent_critic=(time_dependent_critic_def, (ex_observations, ex_actions, ex_times)),
            target_time_dependent_critic=(copy.deepcopy(time_dependent_critic_def), (ex_observations, ex_actions, ex_times)),
            actor_diffusion=(actor_diffusion_def, (ex_observations, ex_actions, ex_times)),
        )
        if encoders.get('actor_diffusion') is not None:
            # Add actor_diffusion_encoder to ModuleDict to make it separately callable.
            network_info['actor_diffusion_encoder'] = (encoders.get('actor_diffusion'), (ex_observations,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        # Add gradient clipping to prevent explosion
        network_tx = optax.chain(
            optax.clip_by_global_norm(1.0),  # Clip gradients by global norm
            optax.adam(learning_rate=config['lr'])
        )
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']
        params['modules_target_time_dependent_critic'] = params['modules_time_dependent_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        # Define beta schedule
        N = config['diffusion_steps']
        beta_min = config['beta_min']
        beta_max = config['beta_max']

        if config['beta_schedule'] == "vp":
            t = jnp.arange(1, N+1)
            alpha = jnp.exp(-beta_min / N - 0.5 * (beta_max - beta_min) * (2 * t - 1) / N ** 2)
            betas = 1 - alpha
        elif config['beta_schedule'] == "linear":
            betas = jnp.linspace(beta_min, beta_max, N)
        else:
            raise ValueError(f"Invalid noise schedule type: {config['beta_schedule']}")

        alphas = 1. - betas
        sqrt_alphas = jnp.sqrt(alphas)
        alphas_cumprod = jnp.cumprod(alphas, axis=0)
        alphas_cumprod_prev = jnp.concatenate([jnp.ones(1), alphas_cumprod[:-1]])
        sqrt_alphas_cumprod = jnp.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = jnp.sqrt(1 - alphas_cumprod)
        return cls(rng, network=network, config=flax.core.FrozenDict(**config), betas=betas, alphas=alphas, sqrt_alphas=sqrt_alphas, alphas_cumprod=alphas_cumprod, alphas_cumprod_prev=alphas_cumprod_prev, sqrt_alphas_cumprod=sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod)


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='dql_tdq',  # Agent name.
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
            diffusion_steps=10,  # Number of diffusion steps.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            reward_scale=1.0,  # Reward scale.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            q_t_weight=1.0,  # Weight for the Q_t loss.
            beta_schedule="vp",
            beta_min=0.1,
            beta_max=10.0,
        )
    )
    return config
