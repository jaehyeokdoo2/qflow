import os
import json
import random
import time

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset, ReplayBuffer
from utils.evaluation import flatten
from utils.flax_utils import restore_agent

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'InitNoise_Evaluation', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-double-play-singletask-v0', 'Environment (dataset) name.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')

# Init noise parameters
flags.DEFINE_list('n_samples_list', ['8', '16', '32'], 'List of initial noise sample counts to test.')
flags.DEFINE_float('temperature', 1.0, 'Temperature for softmax selection (lower = more greedy).')
flags.DEFINE_boolean('save_summary', False, 'Whether to save summary metrics.')

config_flags.DEFINE_config_file('agent', None, lock_config=False)


def evaluate_with_init_noise(
    agent,
    env,
    num_eval_episodes=50,
    n_samples=16,
    temperature=1.0,
    seed=0,
    key=None,
):
    """Evaluate the agent using init noise sampling (select noise at t=0 via critic).

    Unlike SMC which resamples throughout, this method:
    1. Samples n_samples noise candidates
    2. Evaluates V(z, t=0) for each candidate using the time-conditioned critic
    3. Selects ONE noise via weighted categorical sampling
    4. Runs standard flow integration from the selected noise

    Args:
        agent: Agent with init_noise_sample_actions method.
        env: Environment.
        num_eval_episodes: Number of episodes to evaluate.
        n_samples: Number of noise candidates to consider.
        temperature: Temperature for softmax selection (lower = more greedy).
        seed: Random seed.
        key: JAX random key.

    Returns:
        A tuple containing the statistics and trajectories.
    """
    from collections import defaultdict
    from utils.evaluation import add_to

    # Create init noise actor function
    def init_noise_actor_fn(observations, seed=None):
        return agent.init_noise_sample_actions(
            observations=observations,
            seed=seed,
            n_samples=n_samples,
            temperature=temperature,
        )

    print(f"JIT-compiling init noise actor function with n_samples={n_samples}...")
    if key is None:
        key = jax.random.PRNGKey(int(seed))
    init_noise_actor_fn = jax.jit(init_noise_actor_fn)

    trajs = []
    stats = defaultdict(list)
    inference_costs = []

    for i in trange(num_eval_episodes, desc=f"Evaluating Init Noise (n_samples={n_samples})"):
        key, ep_key = jax.random.split(key)
        traj = defaultdict(list)
        observation, info = env.reset(seed=i)
        done = False

        while not done:
            ep_key, action_key = jax.random.split(ep_key)
            start = time.perf_counter()
            action = init_noise_actor_fn(observations=observation, seed=action_key)
            inference_cost = time.perf_counter() - start
            inference_costs.append(inference_cost)

            action = np.array(action)
            action = np.clip(action, -1, 1)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation

        add_to(stats, flatten(info))
        trajs.append(traj)

    for k, v in stats.items():
        stats[k] = np.mean(v)

    avg_inference_cost = np.mean(inference_costs)
    return stats, trajs, avg_inference_cost


def main(_):
    env_name = FLAGS.env_name

    # Load agent config from config_flags
    if FLAGS.agent is None:
        raise ValueError("Must specify --agent to specify agent config file")

    config = FLAGS.agent

    agent_name = config.agent_name
    print(config)

    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name, frame_stack=FLAGS.frame_stack
    )

    train_dataset = Dataset.create(**train_dataset)
    train_dataset = ReplayBuffer.create_from_initial_dataset(
        dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
    )

    # os.environ['PYTHONHASHSEED'] = str(FLAGS.seed)
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    master_key = jax.random.PRNGKey(int(FLAGS.seed))

    # Create agent
    example_batch = train_dataset.sample(1)
    agent_class = agents[config['agent_name']]
    print(agent_class)
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Restore agent
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Parse sample counts
    n_samples_list = [int(x) for x in FLAGS.n_samples_list]

    print(f"Testing sample counts: {n_samples_list}")

    all_results = {}

    for n_samples in n_samples_list:
        print(f"\n{'='*60}")
        print(f"EVALUATING INIT NOISE WITH {n_samples} SAMPLES")
        print(f"{'='*60}")

        random.seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)

        start_time = time.time()

        eval_info, trajs, avg_inference_cost = evaluate_with_init_noise(
            agent=agent,
            env=eval_env,
            num_eval_episodes=FLAGS.eval_episodes,
            n_samples=n_samples,
            temperature=FLAGS.temperature,
            seed=FLAGS.seed,
            key=master_key,
        )

        end_time = time.time()
        eval_time = end_time - start_time

        all_results[n_samples] = {
            'eval_info': eval_info,
            'eval_time': eval_time,
            'avg_inference_cost': avg_inference_cost,
            'num_episodes': FLAGS.eval_episodes,
        }

        print(f"\nInit Noise (n_samples={n_samples})")
        print(f"Evaluation time: {eval_time:.2f} seconds")
        print(f"Avg inference cost: {avg_inference_cost * 1000:.3f} ms")
        print(f"Time per episode: {eval_time / FLAGS.eval_episodes:.3f} seconds")
        print("\nResults:")
        for key, value in eval_info.items():
            print(f"  {key}: {value:.4f}")

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY OF INIT NOISE EVALUATION")
    print(f"{'='*80}")
    print(f"{'Samples':<12} {'Success Rate':<14} {'Avg Reward':<14} {'Infer (ms)':<12} {'Time/Ep':<10}")
    print("-" * 70)

    for n_samples in n_samples_list:
        results = all_results[n_samples]
        eval_info = results['eval_info']
        eval_time = results['eval_time']
        avg_inference_cost = results['avg_inference_cost']
        time_per_ep = eval_time / FLAGS.eval_episodes

        success_rate = eval_info.get('success', 0.0)
        avg_reward = eval_info.get('reward', 0.0)

        print(f"{n_samples:<12} {success_rate:<14.4f} {avg_reward:<14.4f} {avg_inference_cost*1000:<12.3f} {time_per_ep:<10.3f}")

    print(f"\n{'='*80}")
    print("INIT NOISE EVALUATION COMPLETED")
    print(f"{'='*80}")

    # Save summary
    if FLAGS.save_summary:
        try:
            model_base_dir = (
                FLAGS.restore_path
                if os.path.isdir(FLAGS.restore_path)
                else os.path.dirname(FLAGS.restore_path)
            )
            eval_dir = os.path.join(model_base_dir, 'eval_results')
            os.makedirs(eval_dir, exist_ok=True)

            summary_path = os.path.join(
                eval_dir, f"init_noise_summary_{FLAGS.env_name}_sd{FLAGS.seed:03d}.json"
            )
            summary_payload = {
                str(n): {
                    **all_results[n]['eval_info'],
                    'eval_time': all_results[n]['eval_time'],
                    'avg_inference_cost': all_results[n]['avg_inference_cost'],
                    'num_episodes': all_results[n]['num_episodes'],
                }
                for n in n_samples_list
            }
            with open(summary_path, 'w') as f:
                json.dump(summary_payload, f, indent=2)
            print(f"Saved summary metrics to {summary_path}")
        except Exception as e:
            print(f"Warning: failed to save summary metrics: {e}")


if __name__ == '__main__':
    app.run(main)

