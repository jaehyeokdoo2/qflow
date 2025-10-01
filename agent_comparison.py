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
from utils.evaluation import evaluate, flatten, supply_rng
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-double-play-singletask-v0', 'Environment (dataset) name.')
flags.DEFINE_string('agent_1_path', None, 'Restore path for FQL agent.')
flags.DEFINE_string('agent_2_path', None, 'Restore path for IFQL agent.')
flags.DEFINE_integer('agent_1_epoch', None, 'Restore epoch for FQL agent.')
flags.DEFINE_integer('agent_2_epoch', None, 'Restore epoch for IFQL agent.')

flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')
flags.DEFINE_integer('balanced_sampling', 0, 'Whether to use balanced sampling for online fine-tuning.')

config_flags.DEFINE_config_file('fql_config', 'agents/fql.py', lock_config=False)
config_flags.DEFINE_config_file('ifql_config', 'agents/ifql.py', lock_config=False)


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


def compute_q_values(agent, states, actions, batch_size=256):
    """
    Compute Q-values for a batch of states and actions using the agent's critic network.

    Args:
        agent: The agent with critic network
        states: List of state observations
        actions: List of actions corresponding to the states
        batch_size: Batch size for efficient computation

    Returns:
        numpy array of Q-values
    """
    q_values = []

    for i in range(0, len(states), batch_size):
        batch_end = min(i + batch_size, len(states))
        batch_states = states[i:batch_end]
        batch_actions = actions[i:batch_end]

        try:
            if hasattr(agent, 'network') and hasattr(agent.network, 'select'):
                # Convert to numpy arrays for batch processing
                batch_states_np = np.array(batch_states)
                batch_actions_np = np.array(batch_actions)
                
                # Get Q-values from critic network with both states and actions
                q_vals = agent.network.select('critic')(batch_states_np, actions=batch_actions_np)

                if isinstance(q_vals, (list, tuple)):
                    if len(q_vals) > 1:
                        # Take minimum of multiple critics (conservative approach)
                        q_vals = np.minimum(q_vals[0], q_vals[1])
                    else:
                        q_vals = q_vals[0]

                # Convert JAX array to numpy and flatten
                q_vals_np = np.array(q_vals).flatten()
                q_values.extend(q_vals_np.tolist())
            else:
                q_values.extend([0.0] * (batch_end - i))
        except Exception as e:
            print(f"Error computing Q-values for batch {i//batch_size}: {e}")
            q_values.extend([0.0] * (batch_end - i))

    return np.array(q_values)


def plot_q_value_comparison(fql_trajs, ifql_trajs, fql_agent, ifql_agent, save_dir):
    """
    Plot Q-values over time (trajectory steps) for both agents.

    Args:
        fql_trajs: FQL trajectory data containing observations and actions
        ifql_trajs: IFQL trajectory data containing observations and actions
        fql_agent: FQL agent for computing Q-values
        ifql_agent: IFQL agent for computing Q-values
        save_dir: Directory to save the plot
    """
    import matplotlib.pyplot as plt

    if not fql_trajs or not ifql_trajs:
        print("Missing trajectory data for Q-value plotting")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Sample a few trajectories for visualization (to avoid clutter)
    max_trajectories = 5
    fql_sample_trajs = fql_trajs[:max_trajectories] if len(fql_trajs) >= max_trajectories else fql_trajs
    ifql_sample_trajs = ifql_trajs[:max_trajectories] if len(ifql_trajs) >= max_trajectories else ifql_trajs

    # Plot 1: FQL trajectories with FQL Q-values
    for i, traj in enumerate(fql_sample_trajs):
        if 'observation' in traj and 'action' in traj and len(traj['observation']) > 0:
            states = traj['observation']  # Use full trajectory
            actions = traj['action']
            
            traj_q_values = compute_q_values(fql_agent, states, actions)
            steps = np.arange(len(traj_q_values))
            axes[0, 0].plot(steps, traj_q_values, label=f'FQL Trajectory {i+1}', alpha=0.7)

    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Q-value')
    axes[0, 0].set_title('FQL Q-values Over Time (FQL Trajectories)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: IFQL trajectories with IFQL Q-values
    for i, traj in enumerate(ifql_sample_trajs):
        if 'observation' in traj and 'action' in traj and len(traj['observation']) > 0:
            states = traj['observation']  # Use full trajectory
            actions = traj['action']
            
            traj_q_values = compute_q_values(ifql_agent, states, actions)
            steps = np.arange(len(traj_q_values))
            axes[0, 1].plot(steps, traj_q_values, label=f'IFQL Trajectory {i+1}', alpha=0.7)

    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Q-value')
    axes[0, 1].set_title('IFQL Q-values Over Time (IFQL Trajectories)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: FQL trajectories with IFQL Q-values (cross-evaluation)
    for i, traj in enumerate(fql_sample_trajs):
        if 'observation' in traj and 'action' in traj and len(traj['observation']) > 0:
            states = traj['observation']  # Use full trajectory
            actions = traj['action']
            
            traj_q_values = compute_q_values(ifql_agent, states, actions)
            steps = np.arange(len(traj_q_values))
            axes[1, 0].plot(steps, traj_q_values, label=f'FQL Trajectory {i+1}', alpha=0.7)

    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Q-value')
    axes[1, 0].set_title('IFQL Q-values on FQL Trajectories (Cross-evaluation)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Q-value statistics summary
    all_fql_q = []
    all_ifql_q = []
    all_cross_q = []

    # Collect Q-values for statistics
    for traj in fql_trajs[:10]:  # Sample trajectories for stats
        if 'observation' in traj and 'action' in traj and len(traj['observation']) > 0:
            states = traj['observation']  # Use full trajectory for stats
            actions = traj['action']
            q_vals = compute_q_values(fql_agent, states, actions)
            all_fql_q.extend(q_vals)

    for traj in ifql_trajs[:10]:  # Sample trajectories for stats
        if 'observation' in traj and 'action' in traj and len(traj['observation']) > 0:
            states = traj['observation']  # Use full trajectory for stats
            actions = traj['action']
            q_vals = compute_q_values(ifql_agent, states, actions)
            all_ifql_q.extend(q_vals)

    for traj in fql_trajs[:10]:  # Cross-evaluation
        if 'observation' in traj and 'action' in traj and len(traj['observation']) > 0:
            states = traj['observation']  # Use full trajectory for stats
            actions = traj['action']
            q_vals = compute_q_values(ifql_agent, states, actions)
            all_cross_q.extend(q_vals)

    if all_fql_q and all_ifql_q and all_cross_q:
        stats_text = f"""Q-value Statistics Summary

FQL on FQL trajectories:
  Mean: {np.mean(all_fql_q):.3f}
  Std:  {np.std(all_fql_q):.3f}

IFQL on IFQL trajectories:
  Mean: {np.mean(all_ifql_q):.3f}
  Std:  {np.std(all_ifql_q):.3f}

IFQL on FQL trajectories:
  Mean: {np.mean(all_cross_q):.3f}
  Std:  {np.std(all_cross_q):.3f}

Difference (FQL - IFQL on FQL):
  Mean: {np.mean(all_fql_q) - np.mean(all_cross_q):.3f}
"""

        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                        verticalalignment='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[1, 1].set_title('Q-value Statistics Summary')
        axes[1, 1].axis('off')

    plt.tight_layout()

    # Save the comparison plot
    comparison_file = os.path.join(save_dir, 'q_value_trajectory_comparison.png')
    plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Q-value trajectory comparison plot saved: {comparison_file}")


def main(_):
    # Set up logger.
    env_name = FLAGS.env_name

    save_dir = os.path.join('plots', env_name, 'agent_comparison')
    os.makedirs(save_dir, exist_ok=True)

    # Make environment and datasets.
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
    example_batch = train_dataset.sample(1)

    # Create FQL agent
    fql_config = FLAGS.fql_config
    fql_agent_class = agents[fql_config['agent_name']]
    fql_agent = fql_agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        fql_config,
    )

    # Create IFQL agent
    ifql_config = FLAGS.ifql_config
    ifql_agent_class = agents[ifql_config['agent_name']]
    ifql_agent = ifql_agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        ifql_config,
    )

    # Restore FQL agent
    if FLAGS.agent_1_path and FLAGS.agent_1_epoch:
        fql_agent = restore_agent(fql_agent, FLAGS.agent_1_path, FLAGS.agent_1_epoch)

    # Restore IFQL agent
    if FLAGS.agent_2_path and FLAGS.agent_2_epoch:
        ifql_agent = restore_agent(ifql_agent, FLAGS.agent_2_path, FLAGS.agent_2_epoch)

    # Generate trajectories for both agents
    print("Generating trajectories for FQL agent...")
    fql_eval_info, fql_trajs, fql_renders = evaluate(
        agent=fql_agent,
        env=eval_env,
        config=fql_config,
        num_eval_episodes=FLAGS.eval_episodes,
        num_video_episodes=0,  # Skip video rendering for now
        video_frame_skip=FLAGS.video_frame_skip,
        render_always=False,
    )

    print("Generating trajectories for IFQL agent...")
    ifql_eval_info, ifql_trajs, ifql_renders = evaluate(
        agent=ifql_agent,
        env=eval_env,
        config=ifql_config,
        num_eval_episodes=FLAGS.eval_episodes,
        num_video_episodes=0,  # Skip video rendering for now
        video_frame_skip=FLAGS.video_frame_skip,
        render_always=False,
    )

    # Plot Q-value trajectories
    print("Creating Q-value trajectory comparison plots...")
    plot_q_value_comparison(fql_trajs, ifql_trajs, fql_agent, ifql_agent, save_dir)

    print("Agent comparison completed successfully!")


if __name__ == '__main__':
    app.run(main)
