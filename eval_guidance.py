import os

import json
import random
import pickle

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset, ReplayBuffer
from utils.evaluation import flatten, supply_rng
from utils.flax_utils import restore_agent
from utils.networks import LatentLyapunovFunction

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-double-play-singletask-v0', 'Environment (dataset) name.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')
flags.DEFINE_string('save_tag', "eval_guidance_results", 'Evaluation results save tag.')

flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')
flags.DEFINE_integer('balanced_sampling', 0, 'Whether to use balanced sampling for online fine-tuning.')
flags.DEFINE_integer('q_percentile', 20, 'Percentile threshold for Q-value filtering in plots (0-100). Only shows states with Q-values below this percentile.')
flags.DEFINE_boolean('plot_directions', True, 'Whether to plot movement direction arrows on the state coverage plot.')

# Lyapunov model loading flags
flags.DEFINE_string('lyapunov_model_path', None, 'Path to saved Lyapunov model directory.')
flags.DEFINE_integer('lyapunov_model_step', None, 'Lyapunov model step to load (e.g., 1000000).')
flags.DEFINE_integer('latent_dim', 16, 'Latent dimension for Lyapunov function.')
flags.DEFINE_list('lyapunov_hidden_dims', [64, 64], 'Hidden layer dimensions for Lyapunov function.')
flags.DEFINE_boolean('lyapunov_layer_norm', False, 'Whether to use layer normalization in Lyapunov function.')

# Guidance parameters
flags.DEFINE_float('guidance_coeff', 1.0, 'Guidance strength coefficient.')
flags.DEFINE_integer('partial_guidance', -1, 'If -1, apply guidance for entire sampling process. If > 0 and < flow_steps, apply guidance only to last {partial_guidance} steps.')

config_flags.DEFINE_config_file('agent', 'agents/fbrac.py', lock_config=False)


def load_lyapunov_network(model_path, model_step, lyapunov_fn, state_dim, action_dim):
    """Load the saved Lyapunov network parameters."""
    # Load parameters
    params_path = os.path.join(model_path, f'lyapunov_params_step_{model_step}.npz')
    tree_path = os.path.join(model_path, f'lyapunov_tree_step_{model_step}.pkl')
    
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Parameters file not found: {params_path}")
    if not os.path.exists(tree_path):
        raise FileNotFoundError(f"Tree structure file not found: {tree_path}")
    
    # Load tree definition
    with open(tree_path, 'rb') as f:
        tree_def = pickle.load(f)
    
    # Load flattened parameters
    params_data = np.load(params_path)
    flat_params = [params_data[f'arr_{i}'] for i in range(len(params_data.files))]
    
    # Reconstruct parameter tree
    params = jax.tree_util.tree_unflatten(tree_def, flat_params)
    
    print(f"Loaded Lyapunov network from {model_path} at step {model_step}")
    return params


def create_energy_function(lyapunov_fn, lyapunov_params):
    """
    Create an energy function from the Lyapunov function.
    The energy function should return lower values for safer actions.
    """
    def energy_fn(observations, actions):
        try:
            # Use the Lyapunov function to compute energy
            # Lower Lyapunov values (more negative) should correspond to lower energy (safer actions)
            lyapunov_values = lyapunov_fn.apply(lyapunov_params, observations, actions)
            # Return negative Lyapunov values as energy (lower energy = safer)
            return -lyapunov_values
        except Exception as e:
            print(f"Error in energy function: {e}")
            # Return zero energy if there's an error
            return jnp.zeros(observations.shape[0])
    
    # JIT compile the energy function for better performance
    return jax.jit(energy_fn)


def evaluate_with_guidance(
    agent,
    env,
    energy_fn,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    render_always=False,
    guidance_coeff=1.0,
    partial_guidance=-1,
):
    """Evaluate the agent in the environment using guided sampling.

    Args:
        agent: Agent with sample_action_with_guidance method.
        env: Environment.
        energy_fn: Energy function for guidance.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        render_always: Whether to render all episodes.
        guidance_coeff: Guidance strength coefficient.
        partial_guidance: If -1, apply guidance for entire sampling process.
                         If > 0 and < flow_steps, apply guidance only to last {partial_guidance} steps.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    from collections import defaultdict
    
    # Create guided actor function
    def guided_actor_fn(observations, seed=None, temperature=eval_temperature):
        try:
            return agent.sample_action_with_guidance(
                observations=observations,
                energy_fn=energy_fn,
                seed=seed,
                guidance_coeff=guidance_coeff,
                temperature=temperature,
                partial_guidance=partial_guidance
            )
        except Exception as e:
            print(f"Error in guided actor function: {e}")
            # Fall back to regular sampling if guidance fails
            return agent.sample_actions(observations, seed=seed, temperature=temperature)
    
    # JIT-compile the guided actor function for better performance
    print("JIT-compiling guided actor function...")
    guided_actor_fn = jax.jit(guided_actor_fn)
    actor_fn = supply_rng(guided_actor_fn, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        if not render_always:
            should_render = i >= num_eval_episodes
        else:
            should_render = True

        observation, info = env.reset()
        done = False
        step = 0
        render = []
        while not done:
            action = actor_fn(observations=observation, temperature=eval_temperature)
            action = np.array(action)
            action = np.clip(action, -1, 1)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            from utils.evaluation import add_to
            add_to(traj, transition)
            observation = next_observation
        if i < num_eval_episodes and render_always:
            add_to(stats, flatten(info))
            trajs.append(traj)
            renders.append(np.array(render))
        elif i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders


def evaluate_with_guidance_and_immediate_video_saving(
    agent,
    env,
    energy_fn,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    render_always=False,
    save_dir=None,
    success_info_getter=None,
    guidance_coeff=1.0,
    partial_guidance=-1,
):
    """Evaluate the agent with guidance and save videos immediately after each episode.

    Args:
        agent: Agent with sample_action_with_guidance method.
        env: Environment.
        energy_fn: Energy function for guidance.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        render_always: Whether to render all episodes (including evaluation episodes).
        save_dir: Directory to save videos to. If None, videos are not saved.
        success_info_getter: Function to extract success information from trajectory info.
        guidance_coeff: Guidance strength coefficient.
        partial_guidance: If -1, apply guidance for entire sampling process.
                         If > 0 and < flow_steps, apply guidance only to last {partial_guidance} steps.

    Returns:
        A tuple containing the statistics and trajectories (no renders since they're saved immediately).
    """
    import os
    import imageio
    from collections import defaultdict
    
    # Create guided actor function
    def guided_actor_fn(observations, seed=None, temperature=eval_temperature):
        try:
            return agent.sample_action_with_guidance(
                observations=observations,
                energy_fn=energy_fn,
                seed=seed,
                guidance_coeff=guidance_coeff,
                temperature=temperature,
                partial_guidance=partial_guidance
            )
        except Exception as e:
            print(f"Error in guided actor function: {e}")
            # Fall back to regular sampling if guidance fails
            return agent.sample_actions(observations, seed=seed, temperature=temperature)
    
    # JIT-compile the guided actor function for better performance
    print("JIT-compiling guided actor function...")
    guided_actor_fn = jax.jit(guided_actor_fn)
    actor_fn = supply_rng(guided_actor_fn, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)
    energy_vals_by_episode = []


    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        if not render_always:
            should_render = i >= num_eval_episodes
        else:
            should_render = True

        observation, info = env.reset()
        done = False
        step = 0
        render = []
        energy_vals_list = []
        while not done:
            action, energy_vals = actor_fn(observations=observation, temperature=eval_temperature)
            energy_vals_list.append(energy_vals)

            action = np.array(action)
            action = np.clip(action, -1, 1)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            from utils.evaluation import add_to
            add_to(traj, transition)
            observation = next_observation
        
        # Save video immediately after episode completion if we have frames
        if should_render and len(render) > 0 and save_dir is not None:
            try:
                # Get success information if available
                success = False
                if success_info_getter is not None:
                    success = success_info_getter(traj)
                elif 'success' in info:
                    success = info['success']
                
                # Create video path with guidance info
                video_path = os.path.join(save_dir, f'guided_env_{i+1}_success{success}_coeff{guidance_coeff}.mp4')
                
                # Use imageio to save video
                video_writer = imageio.get_writer(video_path, fps=10)
                for frame in render:
                    video_writer.append_data(frame)
                video_writer.close()
                
                print(f"Guided episode {i+1} video saved immediately: {video_path}")
                
            except Exception as e:
                print(f"Error saving video for guided episode {i+1}: {e}")
        
        # Store trajectory data
        if i < num_eval_episodes and render_always:
            add_to(stats, flatten(info))
            trajs.append(traj)
        elif i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        energy_vals_by_episode.append(energy_vals_list)

    for k, v in stats.items():
        stats[k] = np.mean(v)

    
    energy_vals_by_episode = np.array(energy_vals_by_episode)

    return stats, trajs, energy_vals_by_episode


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
                   label=f'Guided Trajectory {i+1}')
            
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
            ax.set_title(f'Guided Agent Trajectory {i+1} in Maze Environment')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_aspect('equal')

            save_file_name = os.path.join(save_dir, f'guided_agent_trajectory_{i+1}.png')
            
            # Save the plot
            plt.tight_layout()
            plt.savefig(save_file_name, dpi=300, bbox_inches='tight')
            plt.close()  # Close the figure to free memory
            
            print(f"Guided trajectory {i+1} visualization saved as '{save_file_name}'")
    
    print(f"All guided trajectory visualizations saved as separate files.")


def plot_training_dataset_coverage(train_dataset, env_name, agent=None, q_percentile=0, plot_directions=True, save_dir="./"):
    """
    Plot the state coverage of the training dataset to visualize how well it covers the state space.
    Simplified version to avoid import conflicts.
    """
    import matplotlib.pyplot as plt

    save_file_name = os.path.join(save_dir, 'guided_training_dataset_coverage.png')
    
    print("Analyzing training dataset state coverage...")
    
    # Sample a subset of the dataset for visualization
    num_samples = min(1000, train_dataset.size)  # Limit to 1k samples for simplicity
    print(f"Sampling {num_samples} states from training dataset...")

    states = []
    for i in range(num_samples):
        try:
            batch = train_dataset.sample(1)
            if 'observations' in batch:
                obs = batch['observations'][0]
                if isinstance(obs, dict):
                    if 'state' in obs:
                        states.append(obs['state'])
                    elif 'xy' in obs:
                        states.append(obs['xy'])
                    else:
                        first_key = list(obs.keys())[0]
                        states.append(obs[first_key])
                else:
                    states.append(obs)
        except Exception as e:
            continue
    
    if not states:
        print("No states found in training dataset")
        return
    
    states = np.array(states)
    print(f"Extracted {len(states)} states with shape: {states.shape}")

    # Create simple visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: State distribution (first 2 dimensions if available)
    if states.shape[1] >= 2:
        axes[0].scatter(states[:, 0], states[:, 1], alpha=0.6, s=10)
        axes[0].set_title('Guided Training Dataset State Coverage (2D)')
        axes[0].set_xlabel('State Dimension 1')
        axes[0].set_ylabel('State Dimension 2')
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].hist(states.flatten(), bins=50, alpha=0.7, edgecolor='black')
        axes[0].set_title('Guided Training Dataset State Distribution (1D)')
        axes[0].set_xlabel('State Value')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Summary statistics
    axes[1].axis('off')
    summary_text = f"""Guided Training Dataset Analysis
    
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
    else:
        summary_text += f"""Range: [{states.min():.3f}, {states.max():.3f}]
Std: {states.std():.3f}
"""
    
    axes[1].text(0.05, 0.95, summary_text, transform=axes[1].transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_file_name, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Guided training dataset coverage plot saved: {save_file_name}")


def plot_q_function_drift_along_trajectory(agent, traj, env, env_name, save_dir):
    """
    Plot Q-function drift along the actual agent trajectory.
    Simplified version to avoid import conflicts.
    """
    import matplotlib.pyplot as plt

    save_file_name = os.path.join(save_dir, 'guided_q_function_drift_along_trajectory.png')
    
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
                num_action_samples = 5  # Reduced for simplicity
                sampled_actions = []
                
                # Sample actions using the agent
                for _ in range(num_action_samples):
                    action = actor_fn(observations=observation, temperature=0.1)
                    sampled_actions.append(action)
                
                sampled_actions = np.array(sampled_actions)
                
                # Get Q-values for sampled actions
                if hasattr(agent, 'network') and hasattr(agent.network, 'select'):
                    try:
                        batch_observation = np.repeat(observation[None, :], len(sampled_actions), axis=0)
                        q_vals = agent.network.select('critic')(batch_observation, actions=sampled_actions)
                        if isinstance(q_vals, (list, tuple)):
                            q_vals = q_vals[0]
                        q_values.append(np.array(q_vals))
                    except Exception as e:
                        print(f"Error getting Q-values from critic: {e}")
                        q_values.append(np.zeros(len(sampled_actions)))
                else:
                    q_values.append(np.zeros(len(sampled_actions)))
                
            except Exception as e:
                print(f"Error getting Q-values for step {i}: {e}")
                continue
        
        if len(q_values) == 0:
            continue
            
        q_values = np.array(q_values)
        
        # Create the plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Maximum Q-value over time
        max_q = np.max(q_values, axis=1) if len(q_values.shape) > 1 else q_values
        axes[0].plot(max_q, 'b-', linewidth=2, label='Max Q-value')
        axes[0].set_title('Guided Maximum Q-value over Trajectory')
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Max Q-value')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Trajectory with Q-value heatmap
        if len(xy_coords) > 0:
            # Ensure max_q has the same length as xy_coords
            if len(max_q) != len(xy_coords):
                min_len = min(len(max_q), len(xy_coords))
                xy_coords = xy_coords[:min_len]
                max_q = max_q[:min_len]
            
            # Ensure max_q is 1D
            if len(max_q.shape) > 1:
                max_q = max_q.flatten()
            
            scatter = axes[1].scatter(xy_coords[:, 0], xy_coords[:, 1], 
                                   c=max_q, cmap='viridis', s=50, alpha=0.8)
            axes[1].plot(xy_coords[:, 0], xy_coords[:, 1], 'k-', alpha=0.3, linewidth=1)
            axes[1].set_title('Guided Trajectory with Q-value Heatmap')
            axes[1].set_xlabel('X coordinate')
            axes[1].set_ylabel('Y coordinate')
            plt.colorbar(scatter, ax=axes[1], label='Max Q-value')
        
        plt.tight_layout()
        plt.savefig(save_file_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Guided Q-function drift plot saved: {save_file_name}")
        break  # Only plot first trajectory for simplicity
    
    print(f"Guided Q-function drift plots saved in {save_dir}")


def main(_):
    # Set up logger.
    agent_name = FLAGS.agent.agent_name
    env_name = FLAGS.env_name

    save_dir = os.path.join('plots', env_name, agent_name, FLAGS.save_tag)
    os.makedirs(save_dir, exist_ok=True)

    # Validate required arguments for guidance
    if FLAGS.lyapunov_model_path is None:
        raise ValueError("Must specify --lyapunov_model_path to load Lyapunov model for guidance")
    if FLAGS.lyapunov_model_step is None:
        raise ValueError("Must specify --lyapunov_model_step to load Lyapunov model for guidance")

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

    # Load Lyapunov model for guidance
    print("Loading Lyapunov model for guidance...")
    
    # Get state and action dimensions from environment
    state_dim = example_batch['observations'].shape[-1]
    action_dim = example_batch['actions'].shape[-1]
    
    # Create Lyapunov network
    hidden_dims = [int(x) for x in FLAGS.lyapunov_hidden_dims]
    lyapunov_fn = LatentLyapunovFunction(
        hidden_dim=hidden_dims,
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=FLAGS.latent_dim,
        layer_norm=FLAGS.lyapunov_layer_norm
    )
    
    # Load trained parameters
    lyapunov_params = load_lyapunov_network(
        FLAGS.lyapunov_model_path, 
        FLAGS.lyapunov_model_step, 
        lyapunov_fn, 
        state_dim, 
        action_dim
    )
    
    # Create energy function
    energy_fn = create_energy_function(lyapunov_fn, lyapunov_params)
    
    print(f"Lyapunov model loaded successfully!")
    print(f"Guidance coefficient: {FLAGS.guidance_coeff}")
    print(f"Network architecture: {state_dim + action_dim} -> {FLAGS.latent_dim} -> {hidden_dims} -> 1")

    # Plot training dataset coverage
    plot_training_dataset_coverage(replay_buffer, env_name, agent,
                                  q_percentile=FLAGS.q_percentile,
                                  plot_directions=FLAGS.plot_directions,
                                  save_dir=save_dir)

    # Evaluation with immediate video saving using guidance
    def get_success_from_traj(traj):
        """Extract success information from trajectory."""
        if 'info' in traj and len(traj['info']) > 0:
            return traj['info'][-1].get('success', False)
        return False
    
    eval_info, trajs, energy_vals_by_episode = evaluate_with_guidance_and_immediate_video_saving(
        agent=agent,
        env=eval_env,
        energy_fn=energy_fn,
        config=config,
        num_eval_episodes=FLAGS.eval_episodes,
        num_video_episodes=FLAGS.video_episodes,
        video_frame_skip=FLAGS.video_frame_skip,
        render_always=True,
        save_dir=save_dir,
        success_info_getter=get_success_from_traj,
        guidance_coeff=FLAGS.guidance_coeff,
        partial_guidance=FLAGS.partial_guidance,
    )
    
    print(f"All guided videos have been saved immediately during evaluation in {save_dir}")

    # Plot guided trajectories
    plot_traj(trajs, eval_env, env_name, save_dir)

    # Plot Q-function drift along trajectories
    plot_q_function_drift_along_trajectory(agent, trajs, eval_env, env_name, save_dir)

    # Save energy values:
    # np.save(os.path.join(save_dir, 'energy_vals_by_episode.npy'), energy_vals_by_episode)
    
    # Print evaluation results
    print("\n" + "="*60)
    print("GUIDED EVALUATION RESULTS")
    print("="*60)
    for key, value in eval_info.items():
        print(f"{key}: {value:.4f}")
    print(f"Guidance coefficient used: {FLAGS.guidance_coeff}")
    print("="*60)


if __name__ == '__main__':
    app.run(main)
