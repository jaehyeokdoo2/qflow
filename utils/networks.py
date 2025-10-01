from typing import Any, Optional, Sequence

import distrax
import flax.linen as nn
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
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
            if i == len(self.hidden_dims) - 2:
                self.sow('intermediates', 'feature', x)
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

    def __call__(self, observations, actions=None):
        """Return values or critic values.

        Args:
            observations: Observations.
            actions: Actions (optional).
        """
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        if actions is not None:
            inputs.append(actions)
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
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None

    def setup(self) -> None:
        self.mlp = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)

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

class LatentLyapunovFunction(nn.Module):
    """
    Lyapunov function with direct latent projection
    
    Architecture:
    1. Direct projection: (s,a) -> latent dimension (no layer norm here)
    2. Hidden layers with optional layer normalization  
    3. Output layer -> V(z) (scalar Lyapunov value)
    
    Attributes:
        state_dim: State dimension.
        action_dim: Action dimension.
        latent_dim: Latent dimension for projection.
        hidden_dim: Hidden layer dimension.
        layer_norm: Whether to apply layer normalization.
    """
    
    hidden_dim: Sequence[int]
    state_dim: int = 2
    action_dim: int = 2
    latent_dim: int = 16
    layer_norm: bool = False

    def setup(self):
        # Direct latent projection layer with proper initialization and activation
        self.latent_projection = nn.Dense(
            self.latent_dim, 
            kernel_init=nn.initializers.variance_scaling(0.1, 'fan_avg', 'uniform')  # Smaller initialization
        )
        
        # Lyapunov network using MLP for hidden layers
        self.lyapunov_mlp = MLP(
            hidden_dims=(*self.hidden_dim, 1),
            activate_final=False,  # No activation on final layer to allow negative values
            layer_norm=self.layer_norm,
            kernel_init=nn.initializers.variance_scaling(0.1, 'fan_avg', 'uniform')  # Smaller initialization
        )
        
    def __call__(self, state, action):
        """
        Forward pass: (s,a) -> latent projection -> hidden layers -> V(z)
        This returns the RAW network output (higher values for expert actions)
        
        Args:
            state: State tensor.
            action: Action tensor.
        """
        # Handle single dimension inputs by expanding if needed
        if len(state.shape) == 1:
            state = jnp.expand_dims(state, axis=0)
        if len(action.shape) == 1:
            action = jnp.expand_dims(action, axis=0)
            
        # Concatenate state and action
        sa_input = jnp.concatenate([state, action], axis=-1)
        
        # Direct latent projection with activation to prevent unbounded growth
        latent = self.latent_projection(sa_input)
        latent = nn.tanh(latent)  # Bound the latent representation
        
        # Process through MLP to get Lyapunov value
        v_value = self.lyapunov_mlp(latent)
        
        return v_value.squeeze(-1)
    
    def lyapunov_value(self, state, action):
        """
        Get the actual Lyapunov value: -V(s,a) (negated for proper interpretation)
        Use this for evaluation and plotting where lower values should indicate safer actions
        
        Args:
            state: State tensor.
            action: Action tensor.
        """
        return -self.__call__(state, action)
    
class EncoderDecoderLyapunovFunction(nn.Module):
    """
    Lyapunov function with separate encoder-decoder components
    
    Architecture:
    1. Encoder: (s,a) -> latent dimension
    2. Decoder: latent -> (s,a) reconstruction  
    3. Lyapunov MLP: latent -> V(z) (scalar Lyapunov value)
    
    Attributes:
        state_dim: State dimension.
        action_dim: Action dimension.
        latent_dim: Latent dimension for projection.
        hidden_dim: Hidden layer dimension.
        layer_norm: Whether to apply layer normalization.
    """

    encoder_hidden_dim: Sequence[int]
    decoder_hidden_dim: Sequence[int]
    hidden_dim: Sequence[int]
    state_dim: int = 2
    action_dim: int = 2
    latent_dim: int = 16
    layer_norm: bool = False
    module_layer_norm: bool = False

    def setup(self):
        # Encoder
        self.encoder = MLP(
            hidden_dims=(*self.encoder_hidden_dim, self.latent_dim),
            activate_final=False,
            layer_norm=self.module_layer_norm
        )

        # Decoder
        self.decoder = MLP(
            hidden_dims=(*self.decoder_hidden_dim, self.state_dim + self.action_dim),
            activate_final=False,
            layer_norm=self.module_layer_norm
        )

        # Lyapunov MLP
        self.lyapunov_mlp = MLP(
            hidden_dims=(*self.hidden_dim, 1),
            activate_final=False,
            layer_norm=self.layer_norm
        )
    
    def encode(self, state, action):
        """Encode state-action pair to latent representation."""
        sa_input = jnp.concatenate([state, action], axis=-1)
        return self.encoder(sa_input)
    
    def decode(self, latent):
        """Decode latent representation back to state-action pair."""
        decoded = self.decoder(latent)
        decoded_state = decoded[:, :self.state_dim]
        decoded_action = decoded[:, self.state_dim:]
        return decoded_state, decoded_action
    
    def lyapunov_value(self, latent):
        """Compute Lyapunov value from latent representation."""
        return self.lyapunov_mlp(latent).squeeze(-1)
    
    def __call__(self, state, action):
        """
        Forward pass: (s,a) -> latent -> V(z)
        
        Returns only the Lyapunov value.
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            lyapunov_value: Scalar Lyapunov value
        """
        latent = self.encode(state, action)
        return self.lyapunov_value(latent)
    
    def process_pair(self, state, action):
        """
        Encode and decode the state and action (used for reconstruction)
        """

        latent = self.encoder(jnp.concatenate([state, action], axis=-1))
        decoded = self.decoder(latent)
        decoded_state = decoded[:, :self.state_dim]
        decoded_action = decoded[:, self.state_dim:]
        return decoded_state, decoded_action
    
    def reconstruction_loss(self, state, action):
        """
        Compute reconstruction loss: MSE between original and reconstructed (s,a) pairs
        
        Args:
            state: State tensor.
            action: Action tensor.
            
        Returns:
            reconstruction_loss: MSE loss between original and reconstructed inputs
        """
        decoded_state, decoded_action = self.process_pair(state, action)
        
        # Compute MSE loss for both state and action reconstruction
        state_loss = jnp.mean((state - decoded_state) ** 2)
        action_loss = jnp.mean((action - decoded_action) ** 2)
        
        return state_loss + action_loss
    
    def encode_only(self, sa_input):
        """Apply encoder only."""
        return self.encoder(sa_input)
    
    def decode_only(self, latent):
        """Apply decoder only."""
        return self.decoder(latent)
    
    def reconstruct(self, state, action):
        """Reconstruct state and action through encoder-decoder."""
        sa_input = jnp.concatenate([state, action], axis=-1)
        latent = self.encoder(sa_input)
        decoded = self.decoder(latent)
        decoded_state = decoded[:, :self.state_dim]
        decoded_action = decoded[:, self.state_dim:]
        return decoded_state, decoded_action
    
    def lyapunov_value(self, state, action):
        """
        Get the actual Lyapunov value: -V(s,a) (negated for proper interpretation)
        Use this for evaluation and plotting where lower values should indicate safer actions
        
        Args:
            state: State tensor.
            action: Action tensor.
        """
        return -self.__call__(state, action)