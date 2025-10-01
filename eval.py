import os
import platform

import json
import random
import time

import jax
import numpy as np
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset, ReplayBuffer
from utils.evaluation import evaluate, evaluate_with_immediate_video_saving, flatten, supply_rng
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-double-play-singletask-v0', 'Environment (dataset) name.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')
flags.DEFINE_string('save_tag', "eval_results", 'Evaluation results save tag.')

flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')
flags.DEFINE_integer('balanced_sampling', 0, 'Whether to use balanced sampling for online fine-tuning.')
flags.DEFINE_integer('q_percentile', 20, 'Percentile threshold for Q-value filtering in plots (0-100). Only shows states with Q-values below this percentile.')
flags.DEFINE_boolean('plot_directions', True, 'Whether to plot movement direction arrows on the state coverage plot.')

config_flags.DEFINE_config_file('agent', 'agents/fql.py', lock_config=False)


def plot_traj(traj, env, env_name, save_dir):
    """
    Visualize agent trajectory based on xy coordinates from trajectory data.
    
    Args:
        traj: List of evaluation results, where each result is a dictionary containing 'info'
        env: Environment object to retrieve maze dimensions
    """
    # Extract maze dimensions from environment
    # For OGBench environments, we can try to get dimensions from observation space
    try:
        # Try to get maze dimensions from environment observation space
        if hasattr(env, 'observation_space'):
            obs_space = env.observation_space
            if hasattr(obs_space, 'spaces') and 'state' in obs_space.spaces:
                # For dict observation spaces, check if we can infer dimensions
                state_space = obs_space.spaces['state']
                if hasattr(state_space, 'shape'):
                    # This might give us some clue about the state dimensions
                    print(f"State space shape: {state_space.shape}")
        
        # For now, we'll use a reasonable default maze size
        # You may need to adjust this based on your specific environment
        maze_width, maze_height = 100, 100  # Default maze dimensions
        
        # Try to get more specific dimensions if available
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'maze_size'):
            maze_width, maze_height = env.unwrapped.maze_size
        elif hasattr(env, 'maze_size'):
            maze_width, maze_height = env.maze_size
        elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'width') and hasattr(env.unwrapped, 'height'):
            maze_width, maze_height = env.unwrapped.width, env.unwrapped.height
            
    except Exception as e:
        print(f"Warning: Could not determine maze dimensions from environment: {e}")
        print("Using default maze dimensions (100x100)")
        maze_width, maze_height = 100, 100
    
    print(f"Using maze dimensions: {maze_width} x {maze_height}")
    
    # Try to get goal state from environment
    goal_pos = None
    try:
        # For OGBench goal-conditioned environments, goal is typically in observation
        # Try to get goal from a recent observation first
        if hasattr(env, 'observation_space') and hasattr(env.observation_space, 'spaces'):
            if 'desired_goal' in env.observation_space.spaces:
                # This is a goal-conditioned environment, try to get goal from reset
                try:
                    obs, info = env.reset()
                    if isinstance(obs, dict) and 'desired_goal' in obs:
                        goal_pos = obs['desired_goal']
                        print(f"Retrieved goal from observation: {goal_pos}")
                except:
                    pass
        
        # Fallback: try different ways to get goal position from environment attributes
        if goal_pos is None:
            if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'goal_pos'):
                goal_pos = env.unwrapped.goal_pos
            elif hasattr(env, 'goal_pos'):
                goal_pos = env.goal_pos
            elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'goal'):
                goal_pos = env.unwrapped.goal
            elif hasattr(env, 'goal'):
                goal_pos = env.goal
            elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'target_pos'):
                goal_pos = env.unwrapped.target_pos
            elif hasattr(env, 'target_pos'):
                goal_pos = env.target_pos
                
    except Exception as e:
        print(f"Warning: Could not retrieve goal position from environment: {e}")
    
    # Plot each trajectory separately
    for i, trajectory in enumerate(traj):
        # Extract xy coordinates from trajectory info
        xy_coords = []
        
        # Navigate through the trajectory structure
        if 'info' in trajectory:
            for step_info in trajectory['info']:
                if 'xy' in step_info:
                    xy_coords.append(step_info['xy'])
        
        if xy_coords:
            xy_coords = np.array(xy_coords)
            
            # Create separate figure for each trajectory
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot trajectory
            ax.plot(xy_coords[:, 0], xy_coords[:, 1], 
                   marker='o', markersize=3, linewidth=1, alpha=0.7,
                   label=f'Trajectory {i+1}')
            
            # Mark start and end points
            ax.plot(xy_coords[0, 0], xy_coords[0, 1], 'go', markersize=8, label='Start')
            ax.plot(xy_coords[-1, 0], xy_coords[-1, 1], 'ro', markersize=8, label='End')
            
            # Plot goal state if available
            if goal_pos is not None:
                if isinstance(goal_pos, (list, tuple, np.ndarray)):
                    if len(goal_pos) >= 2:
                        ax.plot(goal_pos[0], goal_pos[1], 'k*', markersize=15, label='Goal', markeredgecolor='white', markeredgewidth=2)
                        print(f"Goal position plotted at: ({goal_pos[0]}, {goal_pos[1]})")
            
            # Set plot properties
            ax.set_xlim(0, maze_width)
            ax.set_ylim(0, maze_height)
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_title(f'Agent Trajectory {i+1} in Maze Environment')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_aspect('equal')

            save_file_name = os.path.join(save_dir, f'agent_trajectory_{i+1}.png')
            
            # Save the plot
            plt.tight_layout()
            plt.savefig(save_file_name, dpi=300, bbox_inches='tight')
            plt.close()  # Close the figure to free memory
            
            print(f"Trajectory {i+1} visualization saved as '{save_file_name}'")
    
    print(f"All trajectory visualizations saved as separate files.")


def plot_q_function_drift_along_trajectory(agent, traj, env, env_name, save_dir):
    """
    Plot Q-function drift along the actual agent trajectory.
    Shows how Q-values change as the agent moves through the environment.
    """
    import matplotlib.pyplot as plt

    save_file_name = os.path.join(save_dir, 'q_function_drift_along_trajectory.png')
    
    # Create actor function similar to evaluation
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(0))
    
    for traj_idx, trajectory in enumerate(traj):
        if 'info' not in trajectory:
            continue
            
        # Extract xy coordinates from trajectory
        xy_coords = []
        observations = []
        for info in trajectory['info']:
            if 'xy' in info and len(info['xy']) >= 2:
                xy_coords.append(info['xy'])
        
        # Get observations from trajectory
        if 'observation' in trajectory:
            observations = trajectory['observation']
        else:
            # If no observations in trajectory, we can't get Q-values
            print(f"Trajectory {traj_idx+1}: No observations available, skipping Q-function drift plot")
            continue
        
        if len(xy_coords) < 2 or len(observations) < 2:
            continue
            
        xy_coords = np.array(xy_coords)
        
        # Get Q-values for each observation along the trajectory
        q_values = []
        
        for i, observation in enumerate(tqdm(observations)):
            try:
                # Get Q-values by sampling actions and computing Q for each action
                # This is a simplified approach - we'll sample multiple actions and get their Q-values
                num_action_samples = 10
                sampled_actions = []
                
                # Sample actions using the agent
                for _ in range(num_action_samples):
                    action = actor_fn(observations=observation, temperature=0.1)  # Low temperature for consistent sampling
                    sampled_actions.append(action)
                
                sampled_actions = np.array(sampled_actions)
                
                # Get Q-values for sampled actions
                # We'll use the critic network directly if available
                if hasattr(agent, 'network') and hasattr(agent.network, 'select'):
                    try:
                        # Try to get Q-values from critic network
                        # Need to handle batch dimensions properly
                        # Repeat observation to match action batch size
                        batch_observation = np.repeat(observation[None, :], len(sampled_actions), axis=0)
                        q_vals = agent.network.select('critic')(batch_observation, actions=sampled_actions)
                        if isinstance(q_vals, (list, tuple)):
                            q_vals = q_vals[0]  # Take first critic if there are multiple
                        q_values.append(np.array(q_vals))
                    except Exception as e:
                        print(f"Error getting Q-values from critic: {e}")
                        # Fallback: use action values as proxy
                        q_values.append(np.zeros(len(sampled_actions)))
                else:
                    # Fallback: use action values as proxy
                    q_values.append(np.zeros(len(sampled_actions)))
                
            except Exception as e:
                print(f"Error getting Q-values for step {i}: {e}")
                continue
        
        if len(q_values) == 0:
            continue
            
        q_values = np.array(q_values)
        
        # Debug information
        print(f"Trajectory {traj_idx+1}: Q-values shape: {q_values.shape}, xy_coords shape: {xy_coords.shape}")
        
        # Create the plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Q-values over time for each action sample
        num_action_samples = q_values.shape[1] if len(q_values.shape) > 1 else 1
        if len(q_values.shape) > 1:
            if len(q_values.shape) == 3:
                # For 3D array (steps, critics, actions), take mean over critics
                q_values_mean = np.mean(q_values, axis=1)  # Shape: (steps, actions)
                for action_idx in range(min(q_values_mean.shape[1], 5)):  # Plot first 5 actions to avoid clutter
                    axes[0, 0].plot(q_values_mean[:, action_idx], label=f'Action {action_idx}', alpha=0.7)
            else:
                # For 2D array (steps, actions)
                for action_idx in range(min(num_action_samples, 5)):  # Plot first 5 actions to avoid clutter
                    axes[0, 0].plot(q_values[:, action_idx], label=f'Action {action_idx}', alpha=0.7)
            axes[0, 0].set_title('Q-values over Trajectory Steps')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Q-value')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Maximum Q-value over time
        max_q = np.max(q_values, axis=1) if len(q_values.shape) > 1 else q_values
        # If q_values is 3D (steps, critics, actions), take max over both critics and actions
        if len(q_values.shape) == 3:
            max_q = np.max(q_values, axis=(1, 2))  # Max over critics and actions
        elif len(q_values.shape) == 2:
            max_q = np.max(q_values, axis=1)  # Max over actions
        axes[0, 1].plot(max_q, 'b-', linewidth=2, label='Max Q-value')
        axes[0, 1].set_title('Maximum Q-value over Trajectory')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Max Q-value')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Q-value drift (change in Q-values)
        if len(q_values) > 1:
            if len(q_values.shape) == 3:
                # For 3D array, take mean over critics first
                q_values_mean = np.mean(q_values, axis=1)  # Shape: (steps, actions)
                q_drift = np.diff(q_values_mean, axis=0)  # Shape: (steps-1, actions)
            else:
                q_drift = np.diff(q_values, axis=0)
            
            if len(q_drift.shape) > 1:
                for action_idx in range(min(q_drift.shape[1], 5)):  # Plot first 5 actions
                    axes[1, 0].plot(q_drift[:, action_idx], label=f'Action {action_idx}', alpha=0.7)
            else:
                axes[1, 0].plot(q_drift, 'r-', linewidth=2)
            axes[1, 0].set_title('Q-value Drift (Change in Q-values)')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Q-value Change')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Plot 4: Trajectory with Q-value heatmap
        if len(xy_coords) > 0:
            # Ensure max_q has the same length as xy_coords
            if len(max_q) != len(xy_coords):
                print(f"Warning: Q-values length ({len(max_q)}) doesn't match trajectory length ({len(xy_coords)}). Truncating.")
                min_len = min(len(max_q), len(xy_coords))
                xy_coords = xy_coords[:min_len]
                max_q = max_q[:min_len]
            
            # Ensure max_q is 1D
            if len(max_q.shape) > 1:
                max_q = max_q.flatten()
            
            scatter = axes[1, 1].scatter(xy_coords[:, 0], xy_coords[:, 1], 
                                       c=max_q, cmap='viridis', s=50, alpha=0.8)
            axes[1, 1].plot(xy_coords[:, 0], xy_coords[:, 1], 'k-', alpha=0.3, linewidth=1)
            axes[1, 1].set_title('Trajectory with Q-value Heatmap')
            axes[1, 1].set_xlabel('X coordinate')
            axes[1, 1].set_ylabel('Y coordinate')
            plt.colorbar(scatter, ax=axes[1, 1], label='Max Q-value')
        
        plt.tight_layout()
        plt.savefig(save_file_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Q-function drift plot saved: {save_file_name}")
    
    print(f"All Q-function drift plots saved in {save_dir}")


def plot_training_dataset_coverage(train_dataset, env_name, agent=None, q_percentile=0, plot_directions=True, save_dir="./"):
    """
    Plot the state coverage of the training dataset to visualize how well it covers the state space.

    Args:
        train_dataset: Training dataset containing observations and actions
        env_name: Environment name for file naming
        agent: Optional agent to compute Q-values for states
        q_percentile: Only plot states with Q-values below this percentile (0-100).
                     If 0, plot all states. If 20, only plot bottom 20% of states by Q-value.
        plot_directions: Whether to plot movement direction arrows on the state coverage plot
    """
    import matplotlib.pyplot as plt

    save_file_name = os.path.join(save_dir, 'training_dataset_coverage.png')
    # Configure JAX to use GPU if available
    if agent is not None:
        try:
            # Set JAX to use GPU
            jax.config.update('jax_platform_name', 'gpu')
            print("JAX configured to use GPU for Q-value computation")
        except:
            print("Could not configure JAX for GPU, using default device")
    
    print("Analyzing training dataset state coverage...")
    
    # Extract states from training dataset
    states = []
    next_states = []
    actions = []
    observations = []

    # Sample a subset of the dataset for visualization (to avoid memory issues)
    num_samples = min(10000, train_dataset.size)  # Limit to 10k samples
    print(f"Sampling {num_samples} states from training dataset...")

    for i in tqdm(range(num_samples)):
        try:
            batch = train_dataset.sample(1)
            if 'observations' in batch:
                obs = batch['observations'][0]
                observations.append(obs)

                # Extract current state
                current_state = None
                if isinstance(obs, dict):
                    # For dict observations, try to extract state
                    if 'state' in obs:
                        current_state = obs['state']
                    elif 'xy' in obs:
                        current_state = obs['xy']
                    else:
                        # Use the first available observation key
                        first_key = list(obs.keys())[0]
                        current_state = obs[first_key]
                else:
                    # For array observations
                    current_state = obs

                states.append(current_state)

                # Try to get next state from the same batch or next sample
                next_state = None
                if 'next_observations' in batch:
                    next_obs = batch['next_observations'][0]
                    if isinstance(next_obs, dict):
                        if 'state' in next_obs:
                            next_state = next_obs['state']
                        elif 'xy' in next_obs:
                            next_state = next_obs['xy']
                        else:
                            first_key = list(next_obs.keys())[0]
                            next_state = next_obs[first_key]
                    else:
                        next_state = next_obs

                next_states.append(next_state)

                if 'actions' in batch:
                    actions.append(batch['actions'][0])
        except Exception as e:
            print(f"Error sampling batch {i}: {e}")
            continue
    
    if not states:
        print("No states found in training dataset")
        return
    
    states = np.array(states)
    actions = np.array(actions) if actions else None

    print(f"Extracted {len(states)} states with shape: {states.shape}")
    print(f"Next states available: {sum(1 for ns in next_states if ns is not None)}/{len(next_states)}")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: State distribution with Q-values (first 2 dimensions if available)
    if states.shape[1] >= 2:
        # Use first two dimensions for 2D plot
        next_states_filtered = next_states  # Initialize with original next_states
        if agent is not None and observations:
            # Compute Q-values for states with their corresponding actions using batching
            q_values = []
            batch_size = 256
            
            # Convert to numpy arrays for batching
            obs_array = np.array(observations)
            action_array = np.array(actions)
            
            print(f"Computing Q-values for {len(observations)} samples in batches of {batch_size}...")
            
            for i in tqdm(range(0, len(observations), batch_size)):
                batch_end = min(i + batch_size, len(observations))
                batch_obs = obs_array[i:batch_end]
                batch_actions = action_array[i:batch_end]
                
                try:
                    if hasattr(agent, 'network') and hasattr(agent.network, 'select'):
                        # Get Q-values for the batch
                        q_vals = agent.network.select('critic')(batch_obs, actions=batch_actions)
                        if len(q_vals) > 1:
                            # Take minimum of multiple critics (conservative approach)
                            q_vals = np.minimum(q_vals[0], q_vals[1])
                        else:
                            q_vals = q_vals[0]
                        q_values.extend(np.array(q_vals).flatten())
                    else:
                        q_values.extend([0.0] * (batch_end - i))
                except Exception as e:
                    print(f"Error in batch {i//batch_size}: {e}")
                    q_values.extend([0.0] * (batch_end - i))
            
            q_values = np.array(q_values)
            
            # Ensure q_values has the same length as states
            if len(q_values) != len(states):
                print(f"Warning: Q-values length ({len(q_values)}) doesn't match states length ({len(states)}). Truncating.")
                min_len = min(len(q_values), len(states))
                q_values = q_values[:min_len]
                states_plot = states[:min_len]  # Use local copy for plotting
                observations = observations[:min_len]
                actions = actions[:min_len]
            else:
                states_plot = states

            # Apply percentile filtering if specified
            if q_percentile > 0 and len(q_values) > 0:
                percentile_threshold = np.percentile(q_values, q_percentile)
                mask = q_values <= percentile_threshold
                q_values = q_values[mask]
                states_plot = states_plot[mask] if len(states_plot) == len(mask) else states_plot
                next_states_filtered = [next_states[i] for i in range(len(next_states)) if mask[i]] if len(next_states) == len(mask) else next_states
                next_states_filtered = np.array(next_states_filtered) if isinstance(next_states_filtered, list) else next_states_filtered
                observations = [obs for i, obs in enumerate(observations) if mask[i]] if len(observations) == len(mask) else observations
                actions = [act for i, act in enumerate(actions) if mask[i]] if len(actions) == len(mask) else actions
                actions = np.array(actions) if isinstance(actions, list) else actions

                print(f"Applied {q_percentile}th percentile filter: {np.sum(mask)}/{len(mask)} states kept (threshold: {percentile_threshold:.3f})")
            
            # Normalize Q-values to [0, 1] range for color mapping
            if len(q_values) > 0:
                q_min, q_max = q_values.min(), q_values.max()
                if q_max > q_min:
                    q_values_normalized = (q_values - q_min) / (q_max - q_min)
                else:
                    q_values_normalized = np.zeros_like(q_values)
                
                                # Plot movement directions with Q-value coloring
                if plot_directions and next_states_filtered is not None:
                    try:
                        # Calculate direction vectors and collect Q-values
                        directions = []
                        valid_indices = []
                        valid_q_values = []

                        for i, (current, next_state) in enumerate(zip(states_plot, next_states_filtered)):
                            if next_state is not None and len(current) >= 2 and len(next_state) >= 2:
                                # Calculate direction vector
                                dx = next_state[0] - current[0]
                                dy = next_state[1] - current[1]
                                directions.append([dx, dy])
                                valid_indices.append(i)
                                valid_q_values.append(q_values_normalized[i])

                        if directions:
                            directions = np.array(directions)
                            valid_states = states_plot[valid_indices]
                            valid_q_colors = np.array(valid_q_values)

                            # Scale arrows for better visibility
                            scale_factor = 2.0
                            directions_scaled = directions * scale_factor

                            # Plot arrows using quiver with Q-value coloring
                            quiver_plot = axes[0, 0].quiver(valid_states[:, 0], valid_states[:, 1],
                                                           directions_scaled[:, 0], directions_scaled[:, 1],
                                                           valid_q_colors, cmap='viridis',
                                                           angles='xy', scale_units='xy', scale=1,
                                                           alpha=0.8, width=0.005,
                                                           headwidth=6, headlength=6, headaxislength=4.5)

                            # Add colorbar for Q-values
                            plt.colorbar(quiver_plot, ax=axes[0, 0], label='Q-value (normalized)')
                            print(f"Plotted {len(valid_states)} Q-value colored direction arrows")
                        else:
                            # Fallback: plot regular scatter if no directions available
                            scatter = axes[0, 0].scatter(states_plot[:, 0], states_plot[:, 1], c=q_values_normalized,
                                                        cmap='viridis', alpha=0.7, s=30)
                            plt.colorbar(scatter, ax=axes[0, 0], label='Q-value (normalized)')
                            print("No direction data available, plotted scatter points instead")

                    except Exception as e:
                        print(f"Warning: Could not plot movement directions: {e}")
                        # Fallback: plot regular scatter
                        scatter = axes[0, 0].scatter(states_plot[:, 0], states_plot[:, 1], c=q_values_normalized,
                                                    cmap='viridis', alpha=0.7, s=30)
                        plt.colorbar(scatter, ax=axes[0, 0], label='Q-value (normalized)')

                title_suffix = f"\nQ-range: [{q_min:.3f}, {q_max:.3f}]"
                if q_percentile > 0:
                    title_suffix += f"\nBottom {q_percentile}% states shown"
                if plot_directions and next_states_filtered is not None:
                    plot_type = "Movement Directions"
                else:
                    plot_type = "State Coverage"
                axes[0, 0].set_title(f'Training Dataset {plot_type} with Q-values (2D){title_suffix}')
            else:
                axes[0, 0].scatter(states_plot[:, 0], states_plot[:, 1], alpha=0.6, s=10)
                axes[0, 0].set_title('Training Dataset State Coverage (2D)')
        else:
            axes[0, 0].scatter(states[:, 0], states[:, 1], alpha=0.6, s=10)
            axes[0, 0].set_title('Training Dataset State Coverage (2D)')
        
        axes[0, 0].set_xlabel('State Dimension 1')
        axes[0, 0].set_ylabel('State Dimension 2')
        axes[0, 0].grid(True, alpha=0.3)
    else:
        # 1D state space
        next_states_filtered = next_states  # Initialize with original next_states
        if agent is not None and observations:
            # Compute Q-values for 1D states using batching
            q_values = []
            batch_size = 256
            
            # Convert to numpy arrays for batching
            obs_array = np.array(observations)
            action_array = np.array(actions)
            
            print(f"Computing Q-values for {len(observations)} samples in batches of {batch_size}...")
            
            for i in tqdm(range(0, len(observations), batch_size)):
                batch_end = min(i + batch_size, len(observations))
                batch_obs = obs_array[i:batch_end]
                batch_actions = action_array[i:batch_end]
                
                try:
                    if hasattr(agent, 'network') and hasattr(agent.network, 'select'):
                        # Get Q-values for the batch
                        q_vals = agent.network.select('critic')(batch_obs, actions=batch_actions)
                        if len(q_vals) > 1:
                            # Take minimum of multiple critics (conservative approach)
                            q_vals = np.minimum(q_vals[0], q_vals[1])
                        else:
                            q_vals = q_vals[0]
                        q_values.extend(np.array(q_vals).flatten())
                    else:
                        q_values.extend([0.0] * (batch_end - i))
                except Exception as e:
                    print(f"Error in batch {i//batch_size}: {e}")
                    q_values.extend([0.0] * (batch_end - i))
            
            q_values = np.array(q_values)
            
            # Ensure q_values has the same length as states
            if len(q_values) != len(states):
                print(f"Warning: Q-values length ({len(q_values)}) doesn't match states length ({len(states)}). Truncating.")
                min_len = min(len(q_values), len(states))
                q_values = q_values[:min_len]
                states_plot = states[:min_len]  # Use local copy for plotting
                observations = observations[:min_len]
                actions = actions[:min_len]
            else:
                states_plot = states

            # Apply percentile filtering if specified
            if q_percentile > 0 and len(q_values) > 0:
                percentile_threshold = np.percentile(q_values, q_percentile)
                mask = q_values <= percentile_threshold
                q_values = q_values[mask]
                states_plot = states_plot[mask] if len(states_plot) == len(mask) else states_plot
                next_states_filtered = [next_states[i] for i in range(len(next_states)) if mask[i]] if len(next_states) == len(mask) else next_states
                next_states_filtered = np.array(next_states_filtered) if isinstance(next_states_filtered, list) else next_states_filtered
                observations = [obs for i, obs in enumerate(observations) if mask[i]] if len(observations) == len(mask) else observations
                actions = [act for i, act in enumerate(actions) if mask[i]] if len(actions) == len(mask) else actions
                actions = np.array(actions) if isinstance(actions, list) else actions

                print(f"Applied {q_percentile}th percentile filter: {np.sum(mask)}/{len(mask)} states kept (threshold: {percentile_threshold:.3f})")
            
            # Normalize Q-values to [0, 1] range for color mapping
            if len(q_values) > 0:
                q_min, q_max = q_values.min(), q_values.max()
                if q_max > q_min:
                    q_values_normalized = (q_values - q_min) / (q_max - q_min)
                else:
                    q_values_normalized = np.zeros_like(q_values)
                
                scatter = axes[0, 0].scatter(states_plot.flatten(), q_values, alpha=0.7, s=10, c=q_values_normalized, cmap='viridis')
                plt.colorbar(scatter, ax=axes[0, 0], label='Q-value (normalized)')
                axes[0, 0].set_ylabel('Q-value')
                title_suffix = f"\nQ-range: [{q_min:.3f}, {q_max:.3f}]"
                if q_percentile > 0:
                    title_suffix += f"\nBottom {q_percentile}% states shown"
                axes[0, 0].set_title(f'Training Dataset State-Q-value Mapping (1D){title_suffix}')
            else:
                axes[0, 0].scatter(states_plot.flatten(), q_values, alpha=0.7, s=10)
                axes[0, 0].set_ylabel('Q-value')
                axes[0, 0].set_title('Training Dataset State-Q-value Mapping (1D)')
        else:
            axes[0, 0].hist(states.flatten(), bins=50, alpha=0.7, edgecolor='black')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Training Dataset State Distribution (1D)')
        
        axes[0, 0].set_xlabel('State Value')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: State coverage heatmap (if we have more than 2 dimensions)
    if states.shape[1] > 2:
        # Create a correlation matrix of state dimensions
        corr_matrix = np.corrcoef(states.T)
        im = axes[0, 1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[0, 1].set_title('State Dimension Correlations')
        axes[0, 1].set_xlabel('State Dimension')
        axes[0, 1].set_ylabel('State Dimension')
        plt.colorbar(im, ax=axes[0, 1])
        
        # Add dimension labels
        tick_labels = [f'Dim {i}' for i in range(states.shape[1])]
        axes[0, 1].set_xticks(range(states.shape[1]))
        axes[0, 1].set_yticks(range(states.shape[1]))
        axes[0, 1].set_xticklabels(tick_labels)
        axes[0, 1].set_yticklabels(tick_labels)
    else:
        # If only 2D, show action distribution instead
        if actions is not None:
            axes[0, 1].hist(actions.flatten(), bins=50, alpha=0.7, edgecolor='black')
            axes[0, 1].set_xlabel('Action Value')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Training Dataset Action Distribution')
            axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: State space coverage statistics
    if states.shape[1] >= 2:
        # Calculate coverage metrics
        x_range = states[:, 0].max() - states[:, 0].min()
        y_range = states[:, 1].max() - states[:, 1].min()
        total_area = x_range * y_range
        
        # Create a grid to estimate coverage
        x_min, x_max = states[:, 0].min(), states[:, 0].max()
        y_min, y_max = states[:, 1].min(), states[:, 1].max()
        
        # Create a fine grid
        grid_size = 100
        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)
        
        # Count points in each grid cell
        coverage_grid = np.zeros((grid_size-1, grid_size-1))
        for i in range(len(states)):
            x_idx = int((states[i, 0] - x_min) / (x_max - x_min) * (grid_size - 1))
            y_idx = int((states[i, 1] - y_min) / (y_max - y_min) * (grid_size - 1))
            x_idx = np.clip(x_idx, 0, grid_size-2)
            y_idx = np.clip(y_idx, 0, grid_size-2)
            coverage_grid[y_idx, x_idx] += 1
        
        # Plot coverage heatmap
        im = axes[1, 0].imshow(coverage_grid, cmap='viridis', origin='lower')
        axes[1, 0].set_title('State Space Coverage Heatmap')
        axes[1, 0].set_xlabel('X Grid')
        axes[1, 0].set_ylabel('Y Grid')
        plt.colorbar(im, ax=axes[1, 0], label='Number of States')
        
        # Calculate coverage percentage
        covered_cells = np.sum(coverage_grid > 0)
        total_cells = coverage_grid.size
        coverage_percentage = (covered_cells / total_cells) * 100
        
        axes[1, 0].text(0.02, 0.98, f'Coverage: {coverage_percentage:.1f}%', 
                        transform=axes[1, 0].transAxes, 
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        # For 1D states, show distribution across dimensions
        if states.shape[1] > 1:
            # Show distribution of each dimension
            for dim in range(min(states.shape[1], 5)):  # Limit to first 5 dimensions
                axes[1, 0].hist(states[:, dim], bins=30, alpha=0.5, label=f'Dim {dim}')
            axes[1, 0].set_xlabel('State Value')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('State Distribution Across Dimensions')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: IQL State Values or Dataset statistics
    if agent is not None and hasattr(agent, 'config') and 'iql' in agent.config.get('agent_name', '').lower():
        # For IQL agents, plot state values using the V-function
        if states.shape[1] >= 2 and observations:
            print("Computing state values using IQL V-function...")
            
            # Compute V-values for states using batching
            v_values = []
            batch_size = 256
            
            # Convert to numpy arrays for batching
            obs_array = np.array(observations)
            
            for i in tqdm(range(0, len(observations), batch_size)):
                batch_end = min(i + batch_size, len(observations))
                batch_obs = obs_array[i:batch_end]
                
                try:
                    if hasattr(agent, 'network') and hasattr(agent.network, 'select'):
                        # Get V-values for the batch using the value network
                        v_vals = agent.network.select('value')(batch_obs)
                        v_values.extend(np.array(v_vals).flatten())
                    else:
                        v_values.extend([0.0] * (batch_end - i))
                except Exception as e:
                    print(f"Error computing V-values in batch {i//batch_size}: {e}")
                    v_values.extend([0.0] * (batch_end - i))
            
            v_values = np.array(v_values)
            
            # Ensure v_values has the same length as states
            if len(v_values) != len(states):
                print(f"Warning: V-values length ({len(v_values)}) doesn't match states length ({len(states)}). Truncating.")
                min_len = min(len(v_values), len(states))
                v_values = v_values[:min_len]
                states_v_plot = states[:min_len]
            else:
                states_v_plot = states
            
            # Apply same percentile filtering if it was applied to Q-values
            if q_percentile > 0 and 'mask' in locals():
                v_values = v_values[mask] if len(v_values) == len(mask) else v_values
                states_v_plot = states_v_plot[mask] if len(states_v_plot) == len(mask) else states_v_plot
            
            # Normalize V-values for color mapping
            if len(v_values) > 0:
                v_min, v_max = v_values.min(), v_values.max()
                if v_max > v_min:
                    v_values_normalized = (v_values - v_min) / (v_max - v_min)
                else:
                    v_values_normalized = np.zeros_like(v_values)
                
                # Create scatter plot with V-value coloring
                scatter = axes[1, 1].scatter(states_v_plot[:, 0], states_v_plot[:, 1], 
                                           c=v_values_normalized, cmap='plasma', 
                                           alpha=0.7, s=30)
                plt.colorbar(scatter, ax=axes[1, 1], label='V-value (normalized)')
                
                title_suffix = f"\nV-range: [{v_min:.3f}, {v_max:.3f}]"
                if q_percentile > 0:
                    title_suffix += f"\nBottom {q_percentile}% states shown"
                axes[1, 1].set_title(f'State Values (IQL V-function){title_suffix}')
                axes[1, 1].set_xlabel('State Dimension 1')
                axes[1, 1].set_ylabel('State Dimension 2')
                axes[1, 1].grid(True, alpha=0.3)
                
                print(f"Plotted {len(states_v_plot)} states with V-values")
            else:
                # Fallback to regular scatter if no V-values
                axes[1, 1].scatter(states_v_plot[:, 0], states_v_plot[:, 1], alpha=0.6, s=30)
                axes[1, 1].set_title('State Coverage (V-values unavailable)')
                axes[1, 1].set_xlabel('State Dimension 1')
                axes[1, 1].set_ylabel('State Dimension 2')
                axes[1, 1].grid(True, alpha=0.3)
        else:
            # For 1D states or no observations, show text summary
            axes[1, 1].axis('off')
            summary_text = f"""IQL State Value Analysis
            
Dataset Size: {train_dataset.size} samples
Analyzed Samples: {len(states)}
State Dimensions: {states.shape[1]}

Note: V-function plot requires 2D+ states and observations
"""
            axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes, 
                            verticalalignment='top', fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    else:
        # For non-IQL agents, show dataset statistics and summary
        axes[1, 1].axis('off')  # Turn off axis for text display
        
        # Create summary text
        summary_text = f"""Training Dataset State Coverage Analysis
        
Dataset Size: {train_dataset.size} samples
Analyzed Samples: {len(states)}
State Dimensions: {states.shape[1]}

State Space Statistics:
"""
        
        if states.shape[1] >= 2:
            summary_text += f"""X Range: [{states[:, 0].min():.3f}, {states[:, 0].max():.3f}]
Y Range: [{states[:, 1].min():.3f}, {states[:, 1].max():.3f}]
X Std: {states[:, 0].std():.3f}
Y Std: {states[:, 1].std():.3f}
"""
            if states.shape[1] > 2:
                summary_text += f"Additional Dimensions: {states.shape[1] - 2}\n"
        
        if actions is not None:
            summary_text += f"""
Action Space:
Action Dimensions: {actions.shape[1]}
Action Range: [{actions.min():.3f}, {actions.max():.3f}]
Action Std: {actions.std():.3f}
"""
        
        # Add coverage metrics if available
        if states.shape[1] >= 2 and 'coverage_percentage' in locals():
            summary_text += f"\nCoverage Analysis:\nCovered Grid Cells: {covered_cells}/{total_cells}\nCoverage: {coverage_percentage:.1f}%"
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes, 
                        verticalalignment='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_file_name, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training dataset coverage plot saved: {save_file_name}")
    
    # Print summary statistics
    print("\nTraining Dataset Coverage Summary:")
    print(f"Dataset size: {train_dataset.size} samples")
    print(f"State dimensions: {states.shape[1]}")
    if states.shape[1] >= 2:
        print(f"X range: [{states[:, 0].min():.3f}, {states[:, 0].max():.3f}]")
        print(f"Y range: [{states[:, 1].min():.3f}, {states[:, 1].max():.3f}]")
        if 'coverage_percentage' in locals():
            print(f"State space coverage: {coverage_percentage:.1f}%")
    if actions is not None:
        print(f"Action dimensions: {actions.shape[1]}")
        print(f"Action range: [{actions.min():.3f}, {actions.max():.3f}]")


def main(_):
    # Set up logger.
    agent_name = FLAGS.agent.agent_name
    env_name = FLAGS.env_name

    save_dir = os.path.join('plots', env_name, agent_name, FLAGS.save_tag)
    os.makedirs(save_dir, exist_ok=True)

    # Make environment and datasets.
    config = FLAGS.agent
    print(config)
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, frame_stack=FLAGS.frame_stack)
    if FLAGS.video_episodes > 0:
        assert 'singletask' in FLAGS.env_name, 'Rendering is currently only supported for OGBench environments.'
    
    train_dataset = Dataset.create(**train_dataset)
    
    # Use the training dataset as the replay buffer.
    train_dataset = ReplayBuffer.create_from_initial_dataset(
        dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
    )
    replay_buffer = train_dataset

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Create agent.
    example_batch = train_dataset.sample(1)
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Restore agent.
    if FLAGS.restore_path is not None:
        # For lyapunov_fbrac models, exclude the lyapunov module during evaluation
        exclude_modules = None
        if config['agent_name'] == 'lyapunov_fbrac':
            exclude_modules = ['modules_lyapunov']
            print("Excluding Lyapunov module from loading for evaluation")
        
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch, exclude_modules=exclude_modules)

    # Plot training dataset coverage
    plot_training_dataset_coverage(replay_buffer, env_name, agent,
                                  q_percentile=FLAGS.q_percentile,
                                  plot_directions=FLAGS.plot_directions,
                                  save_dir=save_dir)

    # Evaluation with immediate video saving
    def get_success_from_traj(traj):
        """Extract success information from trajectory."""
        if 'info' in traj and len(traj['info']) > 0:
            return traj['info'][-1].get('success', False)
        return False

    
    eval_info, trajs, renders = evaluate(
        agent=agent,
        env=eval_env,
        config=config,
        num_eval_episodes=FLAGS.eval_episodes,
        num_video_episodes=FLAGS.video_episodes,
        video_frame_skip=FLAGS.video_frame_skip,
        render_always=False,
        # save_dir=save_dir,
        # success_info_getter=get_success_from_traj,
    )
    
    print(f"All videos have been saved immediately during evaluation in {save_dir}")

    # plot_traj(trajs, eval_env, env_name)

    # Plot Q-function drift along trajectories
    plot_q_function_drift_along_trajectory(agent, trajs, eval_env, env_name, save_dir)


if __name__ == '__main__':
    app.run(main)
