import os
import platform

import json
import random
import time

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm, trange
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset, ReplayBuffer
from utils.evaluation import evaluate, flatten, supply_rng
from utils.flax_utils import restore_agent, save_agent

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Guidance_Evaluation', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-double-play-singletask-v0', 'Environment (dataset) name.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')
flags.DEFINE_string('agent_config', 'agents/fbrac.py', 'Path to agent config file.')

flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')

# Lyapunov model loading flags
flags.DEFINE_string('lyapunov_model_path', None, 'Path to saved Lyapunov model directory.')
flags.DEFINE_integer('lyapunov_model_step', None, 'Lyapunov model step to load (e.g., 1000000).')
flags.DEFINE_integer('lyapunov_latent_dim', 16, 'Latent dimension for Lyapunov function.')
flags.DEFINE_list('lyapunov_hidden_dims', [64, 64], 'Hidden layer dimensions for Lyapunov function.')
flags.DEFINE_boolean('lyapunov_layer_norm', False, 'Whether to use layer normalization in Lyapunov function.')
flags.DEFINE_boolean('use_cov', False, 'Whether to use covariance-based guidance coefficient estimation.')
flags.DEFINE_boolean('save_values', False, 'Whether to save energy, gradient, and lambda values.')

# Guidance parameters
flags.DEFINE_list('guidance_coeffs', ['0.0', '0.0001', '0.001', '0.01', '0.1'], 'List of guidance coefficients to test.')
flags.DEFINE_integer('partial_guidance', -1, 'If -1, apply guidance for entire sampling process. If > 0 and < flow_steps, apply guidance only to last {partial_guidance} steps.')

config_flags.DEFINE_config_file('agent', None, lock_config=False)


def evaluate_with_guidance(
    agent,
    env,
    energy_fn,
    config=None,
    num_eval_episodes=50,
    eval_temperature=0,
    guidance_coeff=1.0,
    partial_guidance=-1,
    seed=0,
    key=None,
    use_cov=False,
):
    """Evaluate the agent in the environment using guided sampling.

    Args:
        agent: Agent with sample_action_with_guidance method.
        env: Environment.
        energy_fn: Energy function for guidance.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        eval_temperature: Action sampling temperature.
        guidance_coeff: Guidance strength coefficient.
        partial_guidance: If -1, apply guidance for entire sampling process.
                         If > 0 and < flow_steps, apply guidance only to last {partial_guidance} steps.
        use_cov: Whether to use covariance-based guidance coefficient estimation.

    Returns:
        A tuple containing the statistics, trajectories, energy values, gradient values, and lambda values.
    """
    from collections import defaultdict
    
    # Create guided actor function
    def guided_actor_fn(observations, seed=None, temperature=eval_temperature):
        # Use regular fixed coefficient guidance method
        return agent.sample_action_with_guidance(
            observations=observations,
            energy_fn=energy_fn,
            seed=seed,
            guidance_coeff=guidance_coeff,
            temperature=temperature,
            partial_guidance=partial_guidance
        )
    
    # JIT-compile the guided actor function for better performance
    method_name = "covariance-based" if use_cov else "fixed coefficient"
    print(f"JIT-compiling {method_name} guided actor function for coeff={guidance_coeff}...")
    if key is None:
        key = jax.random.PRNGKey(int(seed))
    guided_actor_fn = jax.jit(guided_actor_fn)
    trajs = []
    stats = defaultdict(list)
    energy_vals_by_episode = []
    gradient_vals_by_episode = []
    lambda_vals_by_episode = []
    cosine_sim_vals_by_episode = []

    for i in trange(num_eval_episodes, desc=f"Evaluating coeff={guidance_coeff}"):
        key, ep_key = jax.random.split(key)
        traj = defaultdict(list)
        energy_vals_list = []
        gradient_vals_list = []
        lambda_vals_list = []
        cosine_sim_vals_list = []
        observation, info = env.reset(seed=i)
        done = False
        step = 0
        while not done:
            ep_key, action_key = jax.random.split(ep_key)
            result = guided_actor_fn(observations=observation, temperature=eval_temperature, seed=ep_key)
            # Handle different return formats based on guidance method
            if isinstance(result, tuple) and len(result) == 5:
                # Covariance method returns (action, energy_vals, gradient_vals, lambda_vals, cosine_sim_vals)
                action, energy_vals, gradient_vals, lambda_vals, cosine_sim_vals = result
                energy_vals_list.append(energy_vals)
                gradient_vals_list.append(gradient_vals)
                lambda_vals_list.append(lambda_vals)
                cosine_sim_vals_list.append(cosine_sim_vals)
            elif isinstance(result, tuple) and len(result) == 4:
                # Regular method returns (action, energy_vals, gradient_vals, cosine_sim_vals)
                action, energy_vals, gradient_vals, cosine_sim_vals = result
                energy_vals_list.append(energy_vals)
                gradient_vals_list.append(gradient_vals)
                cosine_sim_vals_list.append(cosine_sim_vals)
                # Create dummy lambda values if not provided
                lambda_vals_list.append([0.0])
            elif isinstance(result, tuple) and len(result) == 3:
                # Legacy format: (action, energy_vals, gradient_vals)
                action, energy_vals, gradient_vals = result
                energy_vals_list.append(energy_vals)
                gradient_vals_list.append(gradient_vals)
                # Create dummy lambda and cosine similarity values if not provided
                lambda_vals_list.append([0.0])
                cosine_sim_vals_list.append([0.0])
            elif isinstance(result, tuple) and len(result) == 2:
                action, energy_vals = result
                energy_vals_list.append(energy_vals)
                # Create dummy gradient, lambda, and cosine similarity values if not provided
                gradient_vals_list.append([jnp.zeros_like(observation)])
                lambda_vals_list.append([0.0])
                cosine_sim_vals_list.append([0.0])
            else:
                action = result
                # Create dummy energy, gradient, lambda, and cosine similarity values if not provided
                energy_vals_list.append(jnp.zeros(1))
                gradient_vals_list.append([jnp.zeros_like(observation)])
                lambda_vals_list.append([0.0])
                cosine_sim_vals_list.append([0.0])
            
            action = np.array(action)
            action = np.clip(action, -1, 1)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

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
        energy_vals_by_episode.append(energy_vals_list)
        gradient_vals_by_episode.append(gradient_vals_list)
        lambda_vals_by_episode.append(lambda_vals_list)
        cosine_sim_vals_by_episode.append(cosine_sim_vals_list)
        add_to(stats, flatten(info))
        trajs.append(traj)

    # Keep energy values as list of lists (no padding needed)
    # Convert JAX arrays to regular Python lists for consistency
    if energy_vals_by_episode and len(energy_vals_by_episode) > 0:
        converted_episodes = []
        for episode_vals in energy_vals_by_episode:
            # Convert JAX arrays to Python lists
            episode_list = []
            for val in episode_vals:
                if hasattr(val, 'tolist'):
                    # If it's a JAX array, convert to list
                    episode_list.append(val.tolist())
                elif isinstance(val, (list, tuple)):
                    # If it's already a list/tuple, keep as is
                    episode_list.append(list(val))
                else:
                    # If it's a single value, convert to float
                    episode_list.append(float(val))
            converted_episodes.append(episode_list)
        energy_vals_by_episode = converted_episodes
    else:
        # No energy values collected, keep as empty list
        energy_vals_by_episode = []

    # Convert gradient values to regular Python lists for consistency
    if gradient_vals_by_episode and len(gradient_vals_by_episode) > 0:
        converted_gradient_episodes = []
        for episode_vals in gradient_vals_by_episode:
            # Convert JAX arrays to Python lists
            episode_list = []
            for val in episode_vals:
                if hasattr(val, 'tolist'):
                    # If it's a JAX array, convert to list
                    episode_list.append(val.tolist())
                elif isinstance(val, (list, tuple)):
                    # If it's already a list/tuple, keep as is
                    episode_list.append(list(val))
                else:
                    # If it's a single value, convert to float
                    episode_list.append(float(val))
            converted_gradient_episodes.append(episode_list)
        gradient_vals_by_episode = converted_gradient_episodes
    else:
        # No gradient values collected, keep as empty list
        gradient_vals_by_episode = []

    # Convert lambda values to regular Python lists for consistency
    if lambda_vals_by_episode and len(lambda_vals_by_episode) > 0:
        converted_lambda_episodes = []
        for episode_vals in lambda_vals_by_episode:
            # Convert JAX arrays to Python lists
            episode_list = []
            for val in episode_vals:
                if hasattr(val, 'tolist'):
                    # If it's a JAX array, convert to list
                    episode_list.append(val.tolist())
                elif isinstance(val, (list, tuple)):
                    # If it's already a list/tuple, keep as is
                    episode_list.append(list(val))
                else:
                    # If it's a single value, convert to float
                    episode_list.append(float(val))
            converted_lambda_episodes.append(episode_list)
        lambda_vals_by_episode = converted_lambda_episodes
    else:
        # No lambda values collected, keep as empty list
        lambda_vals_by_episode = []

    # Convert cosine similarity values to regular Python lists for consistency
    if cosine_sim_vals_by_episode and len(cosine_sim_vals_by_episode) > 0:
        converted_cosine_sim_episodes = []
        for episode_vals in cosine_sim_vals_by_episode:
            # Convert JAX arrays to Python lists
            episode_list = []
            for val in episode_vals:
                if hasattr(val, 'tolist'):
                    # If it's a JAX array, convert to list
                    episode_list.append(val.tolist())
                elif isinstance(val, (list, tuple)):
                    # If it's already a list/tuple, keep as is
                    episode_list.append(list(val))
                else:
                    # If it's a single value, convert to float
                    episode_list.append(float(val))
            converted_cosine_sim_episodes.append(episode_list)
        cosine_sim_vals_by_episode = converted_cosine_sim_episodes
    else:
        # No cosine similarity values collected, keep as empty list
        cosine_sim_vals_by_episode = []

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, energy_vals_by_episode, gradient_vals_by_episode, lambda_vals_by_episode, cosine_sim_vals_by_episode


def main(_):
    # Set up logger.
    env_name = FLAGS.env_name

    # Load agent config
    if FLAGS.agent_config is None:
        raise ValueError("Must specify --agent_config to specify agent config file")
    
    # Load the config file
    import importlib.util
    spec = importlib.util.spec_from_file_location("agent_config", FLAGS.agent_config)
    agent_config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_config_module)
    config = agent_config_module.get_config()
    
    agent_name = config.agent_name
    print(config)
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, frame_stack=FLAGS.frame_stack)
    
    train_dataset = Dataset.create(**train_dataset)
    
    # Use the training dataset as the replay buffer.
    train_dataset = ReplayBuffer.create_from_initial_dataset(
        dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
    )
    replay_buffer = train_dataset

    os.environ['PYTHONHASHSEED'] = str(FLAGS.seed)
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    master_key = jax.random.PRNGKey(int(FLAGS.seed))

    # Create agent.
    example_batch = train_dataset.sample(1)
    agent_class = agents[config['agent_name']]
    print(agent_class)
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Restore agent.
    if FLAGS.restore_path is not None:
        exclude_modules = None
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch, exclude_modules=exclude_modules)
    
    # Create energy function
    energy_fn = None

    # Parse guidance coefficients
    guidance_coeffs = [float(x) for x in FLAGS.guidance_coeffs]
    print(f"Testing guidance coefficients: {guidance_coeffs}")
    print(f"Partial guidance: {FLAGS.partial_guidance}")
    
    # Store results
    all_results = {}
    
    # Evaluate with different guidance coefficients
    energy_vals_by_coeff = []
    gradient_vals_by_coeff = []
    lambda_vals_by_coeff = []
    cosine_sim_vals_by_coeff = []
    for coeff in guidance_coeffs:
        print(f"\n{'='*60}")
        print(f"EVALUATING GUIDANCE COEFFICIENT: {coeff}")
        print(f"{'='*60}")
        
        # Reset RNG state for each coefficient to ensure consistent evaluation
        random.seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)
        
        start_time = time.time()
        
        eval_info, trajs, energy_vals_by_episode, gradient_vals_by_episode, lambda_vals_by_episode, cosine_sim_vals_by_episode = evaluate_with_guidance(
            agent=agent,
            env=eval_env,
            energy_fn=energy_fn,
            config=config,
            num_eval_episodes=FLAGS.eval_episodes,
            eval_temperature=0,
            guidance_coeff=coeff,
            partial_guidance=FLAGS.partial_guidance,
            seed=FLAGS.seed,
            key=master_key,
            use_cov=FLAGS.use_cov,  # Use the flag value
        )
        energy_vals_by_coeff.append(energy_vals_by_episode)
        gradient_vals_by_coeff.append(gradient_vals_by_episode)
        lambda_vals_by_coeff.append(lambda_vals_by_episode)
        cosine_sim_vals_by_coeff.append(cosine_sim_vals_by_episode)
        end_time = time.time()
        eval_time = end_time - start_time
        
        # Store results
        all_results[coeff] = {
            'eval_info': eval_info,
            'eval_time': eval_time,
            'num_episodes': FLAGS.eval_episodes
        }
        
        # Print results
        print(f"\nGUIDANCE COEFFICIENT: {coeff}")
        print(f"Evaluation time: {eval_time:.2f} seconds")
        print(f"Time per episode: {eval_time/FLAGS.eval_episodes:.3f} seconds")
        print("\nResults:")
        for key, value in eval_info.items():
            print(f"  {key}: {value:.4f}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY OF ALL GUIDANCE COEFFICIENTS")
    print(f"{'='*80}")
    
    # Create a summary table
    print(f"{'Coeff':<10} {'Success Rate':<12} {'Avg Reward':<12} {'Time/Ep':<10} {'Total Time':<12}")
    print("-" * 60)
    
    for coeff in guidance_coeffs:
        results = all_results[coeff]
        eval_info = results['eval_info']
        eval_time = results['eval_time']
        time_per_ep = eval_time / FLAGS.eval_episodes
        
        # Extract key metrics (adjust these based on your environment's info keys)
        success_rate = eval_info.get('success', 0.0)
        avg_reward = eval_info.get('reward', 0.0)
        
        print(f"{coeff:<10} {success_rate:<12.4f} {avg_reward:<12.4f} {time_per_ep:<10.3f} {eval_time:<12.2f}")
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETED")
    print(f"{'='*80}")
    # Save summarized metrics per guidance coefficient (minimal persistence)
    try:
        # Save under the model directory: {restore_path or its parent}/eval_results
        model_base_dir = FLAGS.restore_path if os.path.isdir(FLAGS.restore_path) else os.path.dirname(FLAGS.restore_path)
        eval_dir = os.path.join(model_base_dir, 'eval_results')
        os.makedirs(eval_dir, exist_ok=True)
        summary_path = os.path.join(eval_dir, f"summary_{FLAGS.env_name}_sd{FLAGS.seed}_pg{FLAGS.partial_guidance}.json")
        summary_payload = {
            str(c): {
                **all_results[c]['eval_info'],
                'eval_time': all_results[c]['eval_time'],
                'num_episodes': all_results[c]['num_episodes'],
            }
            for c in guidance_coeffs
        }
        with open(summary_path, 'w') as f:
            json.dump(summary_payload, f, indent=2)
        print(f"Saved summary metrics to {summary_path}")
    except Exception as e:
        print(f"Warning: failed to save summary metrics: {e}")
    if FLAGS.save_values:
        # Create mapping of coefficient to energy values and save as JSON
        def convert_to_json_serializable(obj):
            """Recursively convert objects to JSON-serializable format."""
            if hasattr(obj, 'tolist'):
                # JAX/NumPy arrays
                return obj.tolist()
            elif isinstance(obj, (list, tuple)):
                # Lists and tuples
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                # Dictionaries
                return {str(k): convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (int, float, str, bool, type(None))):
                # Already JSON-serializable
                return obj
            else:
                # Try to convert to float, fallback to string
                try:
                    return float(obj)
                except (ValueError, TypeError):
                    return str(obj)
        
        energy_vals_by_coeff_map = {}
        for i, coeff in enumerate(guidance_coeffs):
            if i < len(energy_vals_by_coeff):
                # Convert all energy values to JSON-serializable format
                coeff_energy_vals = convert_to_json_serializable(energy_vals_by_coeff[i])
                energy_vals_by_coeff_map[str(coeff)] = coeff_energy_vals
            else:
                energy_vals_by_coeff_map[str(coeff)] = []
        
        gradient_vals_by_coeff_map = {}
        for i, coeff in enumerate(guidance_coeffs):
            if i < len(gradient_vals_by_coeff):
                # Convert all gradient values to JSON-serializable format
                coeff_gradient_vals = convert_to_json_serializable(gradient_vals_by_coeff[i])
                gradient_vals_by_coeff_map[str(coeff)] = coeff_gradient_vals
            else:
                gradient_vals_by_coeff_map[str(coeff)] = []
        
        lambda_vals_by_coeff_map = {}
        for i, coeff in enumerate(guidance_coeffs):
            if i < len(lambda_vals_by_coeff):
                # Convert all lambda values to JSON-serializable format
                coeff_lambda_vals = convert_to_json_serializable(lambda_vals_by_coeff[i])
                lambda_vals_by_coeff_map[str(coeff)] = coeff_lambda_vals
            else:
                lambda_vals_by_coeff_map[str(coeff)] = []

        cosine_sim_vals_by_coeff_map = {}
        for i, coeff in enumerate(guidance_coeffs):
            if i < len(cosine_sim_vals_by_coeff):
                # Convert all cosine similarity values to JSON-serializable format
                coeff_cosine_sim_vals = convert_to_json_serializable(cosine_sim_vals_by_coeff[i])
                cosine_sim_vals_by_coeff_map[str(coeff)] = coeff_cosine_sim_vals
            else:
                cosine_sim_vals_by_coeff_map[str(coeff)] = []
        
        # Save auxiliary values under the model's eval_results directory
        try:
            model_base_dir = FLAGS.restore_path if os.path.isdir(FLAGS.restore_path) else os.path.dirname(FLAGS.restore_path)
            save_dir = os.path.join(model_base_dir, 'eval_results')
            os.makedirs(save_dir, exist_ok=True)
        except Exception:
            save_dir = 'guidance_eval_results'
            os.makedirs(save_dir, exist_ok=True)
        
        # Save energy values as JSON file
        with open(os.path.join(save_dir, 'energy_vals_by_coeff.json'), 'w') as f:
            json.dump(energy_vals_by_coeff_map, f, indent=2)
        
        # Save gradient values as JSON file
        with open(os.path.join(save_dir, 'gradient_vals_by_coeff.json'), 'w') as f:
            json.dump(gradient_vals_by_coeff_map, f, indent=2)
        
        # Save lambda values as JSON file
        with open(os.path.join(save_dir, 'lambda_vals_by_coeff.json'), 'w') as f:
            json.dump(lambda_vals_by_coeff_map, f, indent=2)
        
        # Save cosine similarity values as JSON file
        with open(os.path.join(save_dir, 'cosine_sim_vals_by_coeff.json'), 'w') as f:
            json.dump(cosine_sim_vals_by_coeff_map, f, indent=2)
        
        print(f"Energy values saved to {save_dir}/energy_vals_by_coeff.json")
        print(f"Gradient values saved to {save_dir}/gradient_vals_by_coeff.json")
        print(f"Lambda values saved to {save_dir}/lambda_vals_by_coeff.json")
        print(f"Cosine similarity values saved to {save_dir}/cosine_sim_vals_by_coeff.json")

if __name__ == '__main__':
    app.run(main)
