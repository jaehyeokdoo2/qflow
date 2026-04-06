from typing import Any, Optional, Sequence

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0, 'intermediates': 0},
        split_rngs={'params': True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class Identity(nn.Module):
    """Identity layer."""

    def __call__(self, x):
        return x


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
        use_residual: Whether to use a residual connection from input to final output.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False
    use_residual: bool = False

    @nn.compact
    def __call__(self, x):
        # Architecture: input_projection → hidden_layers → output_projection
        # Residual: from output of input_projection to input of output_projection
        
        # Input projection (first layer)
        if len(self.hidden_dims) > 0:
            x = nn.Dense(self.hidden_dims[0], kernel_init=self.kernel_init, name='layer_0')(x)
            if len(self.hidden_dims) > 1 or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm(name='layer_norm_0')(x)
            
            # Store output of input projection for residual connection
            input_proj_output = x if self.use_residual else None
            
            # Hidden layers (middle layers, if any)
            for i in range(1, len(self.hidden_dims) - 1):
                size = self.hidden_dims[i]
                x = nn.Dense(size, kernel_init=self.kernel_init, name=f'layer_{i}')(x)
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm(name=f'layer_norm_{i}')(x)
                if i == len(self.hidden_dims) - 2:
                    self.sow('intermediates', 'feature', x)
            
            # Before output projection, add residual from input projection output
            if self.use_residual and input_proj_output is not None and len(self.hidden_dims) > 1:
                # At this point, x is the input to the output projection
                # Add residual from input projection output (dimensions should match)
                x = x + input_proj_output
            
            # Output projection (final layer)
            if len(self.hidden_dims) > 1:
                final_size = self.hidden_dims[-1]
                x = nn.Dense(final_size, kernel_init=self.kernel_init, name=f'layer_{len(self.hidden_dims)-1}')(x)
                if self.activate_final:
                    x = self.activations(x)
                    if self.layer_norm:
                        x = nn.LayerNorm(name=f'layer_norm_{len(self.hidden_dims)-1}')(x)
        else:
            # Edge case: no hidden dims (shouldn't happen in practice)
            pass
        
        return x


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_value)


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class Actor(nn.Module):
    """Gaussian actor network.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        temperature=1.0,
    ):
        """Return action distributions.

        Args:
            observations: Observations.
            temperature: Scaling factor for the standard deviation.
        """
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution


class Value(nn.Module):
    """Value/critic network.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        encoder: Optional encoder module to encode the inputs.
        use_residual: Whether to use residual connections in the MLP.
    """

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None
    use_residual: bool = False

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        value_net = mlp_class(
            (*self.hidden_dims, 1), 
            activate_final=False, 
            layer_norm=self.layer_norm,
            use_residual=self.use_residual
        )

        self.value_net = value_net

    def __call__(self, observations, actions=None, scores=None):
        """Return values or critic values.

        Args:
            observations: Observations.
            actions: Actions (optional).
        """
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        if actions is not None and scores is not None:
            inputs.append(actions)
            inputs.append(scores)
        elif actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs).squeeze(-1)

        return v


class InnerValue(nn.Module):
    """Value/critic network.

    This module can be used for both value V(s, a, t) and critic Q(s, a, u, t) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        value_net = mlp_class((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)

        self.value_net = value_net

    def __call__(self, observations, actions=None, times=None, scores=None):
        """Return values or critic values.

        Args:
            observations: Observations.
            actions: Actions (optional).
            times: Times (optional).
            scores: Scores (optional).
        """
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        if actions is not None and scores is not None and times is not None:
            inputs.append(actions)
            inputs.append(scores)
            inputs.append(times)
        elif actions is not None and times is not None:
            inputs.append(actions)
            inputs.append(times)
        
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs).squeeze(-1)

        return v


class TimeConditionedCritic(nn.Module):
    """
    Time-dependent value/critic network for denoising distillation.
    Maps (s, x_t, t) -> Value, representing E[Q(s, a_clean) | x_t].
    
    Strictly takes (observations, actions, times) as input, unlike InnerValue
    which accepts velocity/scores.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        encoder: Optional encoder module to encode the inputs.
        use_residual: Whether to use residual connections in the MLP.
    """

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None
    use_residual: bool = False

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        
        # Output is scalar value
        self.value_net = mlp_class(
            (*self.hidden_dims, 1), 
            activate_final=False, 
            layer_norm=self.layer_norm,
            use_residual=self.use_residual
        )

    def __call__(self, observations, actions, times):
        """Return values.

        Args:
            observations: Observations (s).
            actions: Noisy Actions (x_t).
            times: Time steps (t).
        """
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        
        # Concatenate s, x_t, t directly
        inputs.append(actions)
        inputs.append(times)
        
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs).squeeze(-1)

        return v


class ActorVectorField(nn.Module):
    """Actor vector field network for flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
        use_residual: Whether to use residual connections in the MLP.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None
    use_residual: bool = False

    def setup(self) -> None:
        self.mlp = MLP(
            (*self.hidden_dims, self.action_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
            use_residual=self.use_residual
        )

    @nn.compact
    def __call__(self, observations, actions, times=None, is_encoded=False):
        """Return the vectors at the given states, actions, and times (optional).

        Args:
            observations: Observations.
            actions: Actions.
            times: Times (optional).
            is_encoded: Whether the observations are already encoded.
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        if times is None:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        else:
            inputs = jnp.concatenate([observations, actions, times], axis=-1)

        v = self.mlp(inputs)

        return v