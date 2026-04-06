from collections import defaultdict

import jax
import numpy as np
from tqdm import trange


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    agent,
    env,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    render_always=False,
    compute_eval_critic_loss=False,
    key=None,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    compute_eval_critic_loss=False
    if key is None:
        key = jax.random.PRNGKey(np.random.randint(0, 2**32))
    actor_fn = supply_rng(agent.sample_actions, rng=key)
    if compute_eval_critic_loss:
        eval_critic_loss_fn = supply_rng(agent.eval_critic_loss, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    
    trajs = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        if not render_always:
            should_render = i >= num_eval_episodes
        else:
            should_render = True

        observation, info = env.reset(seed=i)
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

    
    if compute_eval_critic_loss:    
        # Efficiently flatten all fields from variable-length trajectories for sampling
        eval_states = np.concatenate([np.asarray(traj['observation']) for traj in trajs], axis=0)
        eval_actions = np.concatenate([np.asarray(traj['action']) for traj in trajs], axis=0)
        eval_next_observations = np.concatenate([np.asarray(traj['next_observation']) for traj in trajs], axis=0)
        eval_rewards = np.concatenate([np.asarray(traj['reward']) for traj in trajs], axis=0)
        eval_masks = np.concatenate([np.asarray(traj['done']) for traj in trajs], axis=0) # if done is 1 mask is 0, if done is 0 mask is 1
        eval_masks = 1 - eval_masks
        eval_masks = eval_masks.reshape(-1)

        # Now sample using the flattened arrays
        eval_indices = np.random.choice(len(eval_states), 256, replace=False)

        eval_critic_loss, eval_critic_loss_info = eval_critic_loss_fn(batch=dict(
            observations=eval_states[eval_indices],
            actions=eval_actions[eval_indices],
            next_observations=eval_next_observations[eval_indices],
            rewards=eval_rewards[eval_indices],
            masks=eval_masks[eval_indices],
        ))
        for k, v in eval_critic_loss_info.items():
            stats[k] = v

    return stats, trajs, renders


def evaluate_init_noise(
    agent,
    env,
    n_samples=16,
    temperature_list=(0.1, 0.5, 1.0),
    num_eval_episodes=50,
    key=None,
):
    """Evaluate the agent using init noise sampling with multiple temperature values.
    
    This method efficiently evaluates init_noise sampling by:
    1. JIT-compiling actor functions for each temperature value once
    2. Running evaluation episodes for each temperature setting
    3. Returning aggregated metrics for all settings
    
    Args:
        agent: Agent with init_noise_sample_actions method.
        env: Environment.
        n_samples: Number of noise candidates to sample.
        temperature_list: List of temperature values to test (lower = more greedy).
        num_eval_episodes: Number of episodes to evaluate per temperature.
        key: JAX random key.
        
    Returns:
        all_stats: Dictionary mapping temperature to evaluation statistics.
    """
    if key is None:
        key = jax.random.PRNGKey(np.random.randint(0, 2**32))
    
    all_stats = {}
    
    # Pre-compile actor functions for each temperature value
    compiled_actors = {}
    for temperature in temperature_list:
        def make_init_noise_actor(temp):
            def init_noise_actor_fn(observations, seed=None):
                return agent.init_noise_sample_actions(
                    observations=observations,
                    seed=seed,
                    n_samples=n_samples,
                    temperature=temp,
                )
            return jax.jit(init_noise_actor_fn)
        compiled_actors[temperature] = make_init_noise_actor(temperature)
    
    # Evaluate each temperature setting
    for temperature in temperature_list:
        key, eval_key = jax.random.split(key)
        actor_fn = compiled_actors[temperature]
        
        stats = defaultdict(list)
        
        for i in range(num_eval_episodes):
            eval_key, ep_key = jax.random.split(eval_key)
            observation, info = env.reset(seed=i)
            done = False
            
            while not done:
                ep_key, action_key = jax.random.split(ep_key)
                action = actor_fn(observations=observation, seed=action_key)
                action = np.array(action)
                action = np.clip(action, -1, 1)
                
                next_observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                observation = next_observation
            
            add_to(stats, flatten(info))
        
        # Aggregate statistics
        for k, v in stats.items():
            stats[k] = np.mean(v)
        
        all_stats[temperature] = dict(stats)
    
    return all_stats


def evaluate_with_immediate_video_saving(
    agent,
    env,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    render_always=False,
    save_dir=None,
    success_info_getter=None,
    key=None,
):
    """Evaluate the agent in the environment and save videos immediately after each episode.

    Args:
        agent: Agent.
        env: Environment.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        render_always: Whether to render all episodes (including evaluation episodes).
        save_dir: Directory to save videos to. If None, videos are not saved.
        success_info_getter: Function to extract success information from trajectory info.

    Returns:
        A tuple containing the statistics and trajectories (no renders since they're saved immediately).
    """
    import os
    import imageio
    
    if key is None:
        key = jax.random.PRNGKey(np.random.randint(0, 2**32))
    actor_fn = supply_rng(agent.sample_actions, rng=key)
    trajs = []
    stats = defaultdict(list)

    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        if not render_always:
            should_render = i >= num_eval_episodes
        else:
            should_render = True

        observation, info = env.reset(seed=i)
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
                
                # Create video path
                video_path = os.path.join(save_dir, f'env_{i+1}_success{success}.mp4')
                
                # Use imageio to save video
                video_writer = imageio.get_writer(video_path, fps=10)
                for frame in render:
                    video_writer.append_data(frame)
                video_writer.close()
                
                print(f"Episode {i+1} video saved immediately: {video_path}")
                
            except Exception as e:
                print(f"Error saving video for episode {i+1}: {e}")
        
        # Store trajectory data
        if i < num_eval_episodes and render_always:
            add_to(stats, flatten(info))
            trajs.append(traj)
        elif i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs
