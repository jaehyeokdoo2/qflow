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


class FBRACAgent(flax.struct.PyTreeNode):
    """Flow Q-learning agent with BPTT."""

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
                # --- Q-value-based Weighting Logic ---
                # 1. Get the Q-values for each state. Stop gradient to use it only for weighting.
                q_values = self.network.select('critic')(batch['observations'], actions=batch['actions'])
                if self.config['q_agg'] == 'min':
                    q_for_weighting = jnp.min(q_values, axis=0)
                else:
                    q_for_weighting = jnp.mean(q_values, axis=0)
                q_for_weighting = jax.lax.stop_gradient(q_for_weighting)

                # 2. Calculate weights using softmax over negative values.
                # Lower Q-value -> higher weight. Temperature controls sharpness.
                weighting_temp = self.config.get('jacobian_weighting_temp', 1.0)
                weights = jax.nn.softmax(-q_for_weighting / weighting_temp) * batch['observations'].shape[0]

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

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng, support_noise_rng = jax.random.split(rng, 4)

        # Add weight decay for actor parameters
        # Use current network params if grad_params is None (e.g., during validation)
        params_to_use = grad_params if grad_params is not None else self.network.params
        actor_params = params_to_use['modules_actor_bc_flow']
        weight_decay = self.config.get('actor_weight_decay', 0.0)

        # Jacobian regularization on actor parameters
        jacobian_reg_coeff = self.config.get('actor_jacobian_reg', 0.0)
        jacobian_reg_loss = 0.0
        
        # L2 regularization on actor parameters
        l2_reg = 0.0
        if weight_decay > 0:
            l2_reg = weight_decay * sum(jnp.sum(p**2) for p in jax.tree_util.tree_leaves(actor_params))

        if self.config['encoder'] is not None:
            observations = self.network.select('actor_bc_flow_encoder')(batch['observations'])
        else:
            observations = batch['observations']

        if self.config['actor_loss'] == 'pgbc':
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

            # Total loss.
            actor_loss = self.config['alpha'] * bc_flow_loss + q_loss + l2_reg

            return actor_loss, {
                'actor_loss': actor_loss,
                'bc_flow_loss': bc_flow_loss,
                'q_loss': q_loss,
                'q': q.mean(),
                'l2_reg': l2_reg,
            }
        elif self.config['actor_loss'] == 'ewr':
            # --- Energy-Weighted Regression (Q-weighted Flow Matching) ---
            # Implementation of Algorithm 4 from the paper

            # 1. Sample a support action set a_ij using the current policy (vector field v^θ).
            M = 16
            support_noises = jax.random.normal(support_noise_rng, (batch_size, M, action_dim))
            
            # Use a scan for iterative flow sampling to save memory
            def flow_step(actions, i):
                t = jnp.full((batch_size, M, 1), i / self.config['flow_steps'])
                # Observations need to be broadcastable to the shape of actions
                # Repeat observations along the M dimension to match actions shape
                obs_for_flow = jnp.repeat(jnp.expand_dims(observations, 1), M, axis=1)
                vels = self.network.select('actor_bc_flow')(
                    obs_for_flow, actions, t, is_encoded=True, params=grad_params
                )
                actions = actions + vels / self.config['flow_steps']
                return actions, None

            # Sampled support actions
            sampled_support_actions, _ = jax.lax.scan(
                flow_step, support_noises, jnp.arange(self.config['flow_steps'])
            )
            sampled_support_actions = jnp.clip(sampled_support_actions, -1, 1)

            # 2. Construct the full support action set by including the original batch actions (a_i0).
            original_actions = batch['actions'][:, None, :]  # Shape: (B, 1, A)
            full_support_actions = jnp.concatenate(
                [original_actions, sampled_support_actions], axis=1
            ) # Shape: (B, M+1, A)

            # 3. Calculate guidance weights (g_ij) using Q-values.
            # Prepare observations for the critic: repeat for each support action.
            obs_for_critic = jnp.repeat(jnp.expand_dims(batch['observations'], 1), M + 1, axis=1) # Shape: (B, M+1, O)
            
            support_qs_ensembles = self.network.select('critic')(obs_for_critic, actions=full_support_actions) # Shape: (num_ensembles, B, M+1)
            support_qs = jnp.mean(support_qs_ensembles, axis=0) # Shape: (B, M+1)
            support_qs = jax.lax.stop_gradient(support_qs)

            # Softmax over the Q-values for each state to get weights (g_ij).
            guidance_weights = jax.nn.softmax(self.config['alpha'] * support_qs, axis=-1) # Shape: (B, M+1)
            
            # 4. Compute the weighted flow matching loss.
            # Sample noise and time for every action in the full support set.
            x_0 = jax.random.normal(x_rng, (batch_size, M + 1, action_dim))
            x_1 = full_support_actions
            t = jax.random.uniform(t_rng, (batch_size, M + 1, 1))

            # Create noisy actions and the target velocity field.
            x_t = (1 - t) * x_0 + t * x_1
            target_vel = x_1 - x_0

            # Get predictions from the flow model. We need to flatten inputs to a single batch dim.
            flat_obs = jnp.repeat(observations, M + 1, axis=0)
            flat_x_t = x_t.reshape(-1, action_dim)
            flat_t = t.reshape(-1, 1)

            pred_vel = self.network.select('actor_bc_flow')(
                flat_obs, flat_x_t, flat_t, is_encoded=True, params=grad_params
            )
            
            # Calculate the per-sample flow matching (BC) loss.
            flat_target_vel = target_vel.reshape(-1, action_dim)
            per_sample_loss = jnp.mean((pred_vel - flat_target_vel) ** 2, axis=-1)

            # Reshape weights to match the flattened loss.
            flat_guidance_weights = guidance_weights.reshape(-1)

            # The final actor loss is the mean of the weighted per-sample losses.
            actor_loss = jnp.mean(flat_guidance_weights * per_sample_loss) + l2_reg

            return actor_loss, {
                'actor_loss': actor_loss,
                'bc_flow_loss': jnp.mean(per_sample_loss), # Unweighted loss for logging
                'avg_support_q': jnp.mean(support_qs),
                'avg_guidance_weight': jnp.mean(guidance_weights),
                'l2_reg': l2_reg,
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
        """
        # Use covariance-based guidance if enabled
        if self.config.get('use_cov', False):
            return self.sample_action_with_guidance_cov(
                observations=observations,
                energy_fn=energy_fn,
                seed=seed,
                n_samples=16,
                eta=1.0,
                eps=1e-6,
                lambda_min=0.0,
                lambda_max=3.0,
                temperature=temperature,
                partial_guidance=partial_guidance,
            )[:3]  # Return only first 3 values to match original signature
        
        # In place condition for experiment purpose
        rescale_strategy="normalize" # to be move to guidance_evaluation main function
        
        # Original fixed guidance coefficient method
        energy_vals = []
        gradient_vals = []
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
        flow_steps = self.config['flow_steps']
        if partial_guidance == -1:
            # Apply guidance for entire sampling process
            guidance_start_step = 0
        elif 0 < partial_guidance < flow_steps:
            # Apply guidance only to last {partial_guidance} steps
            guidance_start_step = flow_steps - partial_guidance
        else:
            # Invalid partial_guidance value, default to no guidance
            guidance_start_step = flow_steps
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
            energy_val = energy_fn(observations, actions)
            

            # Only compute energy and guidance if guidance_coeff is not zero and energy > -5
            # Use JAX conditional operations for JIT compatibility
            def apply_guidance():
                if energy_grad_fn is not None:
                    safety_grad = energy_grad_fn(observations, actions)

                    # Compute q value gradient w.r.t. actions
                    def q_fn(actions):
                        qs = self.network.select('critic')(observations, actions)
                        return jnp.mean(qs)
                    
                    q_grad = jax.grad(q_fn)(actions)

                    # guidance_grad = q_grad/jnp.linalg.norm(q_grad) - safety_grad/jnp.linalg.norm(safety_grad) # to match a scale
                    guidance_grad = q_grad

                    if rescale_strategy == "normalize":
                        # normalize
                        # guidance_grad = guidance_grad * jnp.linalg.norm(vels) / jnp.linalg.norm(guidance_grad)

                        # compute cosine similarity between vels and guidance_grad
                        cos_sim = jnp.dot(q_grad, -safety_grad) / (jnp.linalg.norm(q_grad) * jnp.linalg.norm(-safety_grad))
                        cos_sim = jnp.clip(cos_sim, -1, 1)
                        guidance_grad = guidance_grad * (cos_sim + 1) / 2
                        pass
                        
                    guidance_grad = guidance_coeff * guidance_grad
                    
                    return guidance_grad
                else:
                    return jnp.zeros_like(vels)
            
            def no_guidance():
                return jnp.zeros_like(vels)
            
            # Check if we should apply guidance
            should_apply = (guidance_coeff != 0.0 and 
                          i >= guidance_start_step and 
                          energy_grad_fn is not None)
            
            # Use JAX conditional to choose guidance or no guidance
            guidance_grad = jax.lax.cond(
                should_apply,
                apply_guidance,
                no_guidance
            )

            if rescale_strategy == "sigmoid":
                guidance_grad = guidance_grad / (1 + jnp.exp(guidance_coeff * energy_val))

            gradient_vals.append(guidance_grad)
            
            # Combine flow velocity with guidance
            if should_apply:
                guided_vels = vels + guidance_grad
            else:
                guided_vels = vels + guidance_grad
            
            # Update actions
            actions = actions + guided_vels / flow_steps
            energy_val = energy_fn(observations, actions)
            energy_vals.append(energy_val)
        
        actions = jnp.clip(actions, -1, 1)
        return actions, energy_vals, gradient_vals

    def sample_action_with_guidance_cov(
        self,
        observations,
        energy_fn,
        seed=None,
        n_samples=16,
        eta=1.0,
        eps=1e-6,
        lambda_min=0.0,
        lambda_max=3.0,
        temperature=1.0,
        partial_guidance=-1,
    ):
        """
        Sample actions with covariance-based guidance coefficient estimation.
        
        Args:
            observations: Current observations
            energy_fn: Energy function that takes (observations, actions) and returns energy values
            seed: Random seed for sampling
            n_samples: Number of samples to use for covariance estimation
            eta: Trust penalty parameter
            eps: Small value to avoid division by zero
            lambda_min: Minimum guidance coefficient (0 for safety-positive only)
            lambda_max: Maximum guidance coefficient
            temperature: Sampling temperature
            partial_guidance: If -1, apply guidance for entire sampling process.
                            If > 0 and < flow_steps, apply guidance only to last {partial_guidance} steps.
            
        Returns:
            actions: Sampled actions with guidance applied
            energy_vals: List of energy values computed during sampling
            gradient_vals: List of gradient values computed during sampling
            lambda_vals: List of computed guidance coefficients
        """
        energy_vals = []
        gradient_vals = []
        lambda_vals = []
        
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
        flow_steps = self.config['flow_steps']
        if partial_guidance == -1:
            # Apply guidance for entire sampling process
            guidance_start_step = 0
        elif 0 < partial_guidance < flow_steps:
            # Apply guidance only to last {partial_guidance} steps
            guidance_start_step = flow_steps - partial_guidance
        else:
            # Invalid partial_guidance value, default to no guidance
            guidance_start_step = flow_steps
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
            energy_val = energy_fn(observations, actions)
            energy_vals.append(energy_val)

            # Only compute guidance if we're in the guidance phase
            if i >= guidance_start_step and energy_grad_fn is not None:
                try:
                    # Sample n_samples unguided actions for covariance estimation
                    sample_seeds = jax.random.split(noise_seed, n_samples)
                    unguided_actions = []
                    unguided_qs = []
                    unguided_energies = []
                    
                    for j in range(n_samples):
                        # Sample unguided action using current flow
                        sample_noise = jax.random.normal(sample_seeds[j], noises.shape)
                        sample_action = sample_noise
                        
                        # Apply flow steps up to current step
                        for k in range(i):
                            t_k = jnp.full((*observations.shape[:-1], 1), k / flow_steps)
                            vels_k = self.network.select('actor_bc_flow')(observations, sample_action, t_k)
                            sample_action = sample_action + vels_k / flow_steps
                        
                        sample_action = jnp.clip(sample_action, -1, 1)
                        unguided_actions.append(sample_action)
                        
                        # Compute Q-values for the unguided action
                        q_values = self.network.select('critic')(observations, sample_action)
                        q_mean = jnp.mean(q_values, axis=0)
                        unguided_qs.append(q_mean)
                        
                        # Compute energy (safety) for the unguided action
                        energy_val_sample = energy_fn(observations, sample_action)
                        unguided_energies.append(energy_val_sample)
                    
                    # Stack the samples
                    unguided_actions = jnp.stack(unguided_actions, axis=0)  # (n_samples, action_dim)
                    unguided_qs = jnp.stack(unguided_qs, axis=0)  # (n_samples,)
                    unguided_energies = jnp.stack(unguided_energies, axis=0)  # (n_samples,)
                    
                    # Compute sample means
                    q_bar = jnp.mean(unguided_qs)
                    s_bar = jnp.mean(unguided_energies)
                    
                    # Compute covariance and variance (unbiased estimators)
                    q_diff = unguided_qs - q_bar
                    s_diff = unguided_energies - s_bar
                    
                    cov_qs = jnp.mean(q_diff * s_diff)  # Cov(Q, S)
                    var_s = jnp.mean(s_diff * s_diff)   # Var(S)
                    
                    # Compute optimal lambda
                    lambda_star = cov_qs / (eta * (var_s + eps))
                    
                    # Clamp lambda (ignoring clipping for now as requested)
                    lambda_coeff = jnp.clip(lambda_star, lambda_min, lambda_max)
                    lambda_vals.append(lambda_coeff)
                    
                    # Apply guidance with computed coefficient
                    guidance_grad = energy_grad_fn(observations, actions)
                    guidance_grad = lambda_coeff * guidance_grad
                    gradient_vals.append(guidance_grad)
                    
                except Exception as e:
                    print(f"Error computing covariance-based guidance at step {i}: {e}")
                    # Fall back to no guidance if there's an error
                    guidance_grad = jnp.zeros_like(vels)
                    gradient_vals.append(guidance_grad)
                    lambda_vals.append(0.0)
            else:
                # No guidance for this step
                guidance_grad = jnp.zeros_like(vels)
                gradient_vals.append(guidance_grad)
                lambda_vals.append(0.0)
            
            # Combine flow velocity with guidance
            guided_vels = vels - guidance_grad
            
            # Update actions
            actions = actions + guided_vels / flow_steps
        
        actions = jnp.clip(actions, -1, 1)
        return actions, energy_vals, gradient_vals, lambda_vals

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
            agent_name='fbrac',  # Agent name.
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
            lyapunov_reg=0.0,  # Lyapunov regularization coefficient.
            actor_loss='pgbc',  # Actor loss type.
            use_cov=False,  # Whether to use covariance-based guidance coefficient estimation.
        )
    )
    return config
