"""
Flow-QL with Energy-Based Guidance using Lyapunov Functions.
This implementation uses the Lyapunov function as an energy function to guide the flow matching sampling process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .ql_flow import QL_Flow


class QL_Flow_Guidance(QL_Flow):
    """
    Flow-QL with energy-based guidance using Lyapunov functions.
    
    The guidance is applied during sampling, not during training.
    The Lyapunov function acts as an energy function: E(a) = V(s, a)
    The guided sampling process becomes:
    a_{t+1} = a_t + dt * (v_θ(a_t, s, t) + γ ∇_a V(s, a_t))
    """
    
    def __init__(self, state_dim, action_dim, max_action, device, discount=0.99, tau=0.005, 
                 eta=2.5, n_timesteps=50, model_type='MLP', 
                 hidden_dim=128, lr=3e-4, r_fun=None, mode='whole_grad', 
                 lyapunov_fn=None, guidance_coeff=1.0, **kwargs):
        """
        Initialize Flow-QL with guidance.
        
        Args:
            lyapunov_fn: Lyapunov function to use as energy function for guidance
            guidance_coeff: Guidance strength coefficient (γ in the formulation)
        """
        # Initialize parent class without lyapunov_fn (we don't use it for training)
        super().__init__(
            state_dim=state_dim, action_dim=action_dim, max_action=max_action, 
            device=device, discount=discount, tau=tau, eta=eta, 
            n_timesteps=n_timesteps, model_type=model_type, hidden_dim=hidden_dim, 
            lr=lr, r_fun=r_fun, mode=mode, lyapunov_fn=None, **kwargs
        )
        
        self.lyapunov_fn = lyapunov_fn
        self.guidance_coeff = guidance_coeff
        
        print(f"Initialized Flow-QL with guidance:")
        print(f"  Guidance coefficient: {guidance_coeff}")
        print(f"  Lyapunov function provided: {lyapunov_fn is not None}")
    
    def compute_guidance_gradient(self, state, action):
        """
        Compute the gradient of the Lyapunov function w.r.t. actions.
        
        Args:
            state: Current state tensor (batch_size, state_dim)
            action: Current action tensor (batch_size, action_dim)
            
        Returns:
            guidance_grad: Gradient of Lyapunov function w.r.t. actions (batch_size, action_dim)
        """
        if self.lyapunov_fn is None:
            return torch.zeros_like(action)
        
        # Ensure gradients are enabled for the action tensor
        action.requires_grad_(True)
        
        # Compute Lyapunov value
        lyapunov_value = self.lyapunov_fn.lyapunov_value(state, action)
        
        # Compute gradient w.r.t. actions
        guidance_grad = torch.autograd.grad(
            outputs=lyapunov_value.sum(),  # Sum to get scalar for gradient
            inputs=action,
            create_graph=False,  # We don't need second-order gradients
            retain_graph=False
        )[0]
        
        return guidance_grad
    
    def sample_action_with_guidance(self, state, deterministic=False, num_samples=1):
        """
        Sample action using guided flow matching process.
        
        Args:
            state: State tensor (batch_size, state_dim) or (state_dim,) or numpy array
            deterministic: Whether to use deterministic sampling
            num_samples: Number of samples to generate
            
        Returns:
            action: Sampled action (batch_size, action_dim) or (action_dim,)
        """
        if self.lyapunov_fn is None:
            # Fall back to regular sampling if no guidance function
            return super().sample_action(state, deterministic, num_samples)
        
        # Convert numpy array to tensor if needed
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        
        # Handle single state input
        if state.dim() == 1:
            state = state.unsqueeze(0)
            single_state = True
        else:
            single_state = False
        
        batch_size = state.shape[0]
        
        # Repeat state for multiple samples if needed
        if num_samples > 1:
            state = state.repeat(num_samples, 1)
            batch_size = state.shape[0]
        
        # Define guidance function with proper scaling
        def guidance_function(s, a):
            # Ensure action tensor requires gradients
            a = a.requires_grad_(True)
            # lyapunov_value returns -forward(), so we need the raw forward() for energy
            raw_lyapunov = self.lyapunov_fn.forward(s, a)
            energy = -raw_lyapunov  # Energy = -V(s,a), lower for safer actions
            
            # Scale the energy by guidance coefficient
            return self.guidance_coeff * energy
        
        # Use the built-in guided sampling (no_grad is handled inside guided_sample)
        action = self.actor.guided_sample(state, guidance_function, start=0.2, verbose=False)
        
        # Return single action if input was single state
        if single_state and num_samples == 1:
            return action.squeeze(0).detach().cpu().numpy()
        
        return action.detach().cpu().numpy()
    
    def sample_action(self, state, deterministic=False, num_samples=1):
        """
        Override sample_action to use guidance when available.
        """
        return self.sample_action_with_guidance(state, deterministic, num_samples)


def create_guided_flow_agent(args, data_sampler, device, lyapunov_fn=None, guidance_coeff=1.0):
    """
    Create a guided flow agent for the organized script.
    
    Args:
        args: Command line arguments
        data_sampler: Data sampler for training
        device: Device to place model on
        lyapunov_fn: Lyapunov function for guidance
        guidance_coeff: Guidance coefficient
        
    Returns:
        agent: Trained guided flow agent
    """
    print("Training Guided Flow-QL...")
    
    # Initialize guided agent
    agent = QL_Flow_Guidance(
        state_dim=2,
        action_dim=2,
        max_action=1.0,
        device=device,
        discount=0.99,
        tau=0.005,
        eta=args.eta,
        n_timesteps=50,
        model_type='MLP',
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        r_fun=None,
        mode=args.mode,
        lyapunov_fn=lyapunov_fn,
        guidance_coeff=guidance_coeff
    )
    
    # Training loop (same as regular flow)
    iterations = int(args.num_data / args.batch_size)
    
    for i in range(1, args.num_epochs + 1):
        b_loss, q_loss = agent.train(data_sampler, iterations=iterations, batch_size=args.batch_size)
        
        if i % 10 == 0:
            print(f'Guided Flow-QL Epoch: {i} B_loss {b_loss:.6f} Q_loss {q_loss:.6f}')
    
    print("Guided Flow-QL training completed!")
    return agent
