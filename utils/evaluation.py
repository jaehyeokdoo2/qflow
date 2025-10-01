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
    if key is None:
        key = jax.random.PRNGKey(np.random.randint(0, 2**32))
    actor_fn = supply_rng(agent.sample_actions, rng=key)
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

    return stats, trajs, renders


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
