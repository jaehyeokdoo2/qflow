import os
import platform

import json
import random
import time

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset, ReplayBuffer
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-double-play-singletask-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('train_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 10000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')

config_flags.DEFINE_config_file('agent', 'agents/idm.py', lock_config=False)


def evaluate_idm(agent, val_dataset, num_eval_batches=10):
    """Evaluate IDM performance on validation dataset."""
    total_loss = 0.0
    total_mae = 0.0
    total_cosine_sim = 0.0
    total_samples = 0
    
    for _ in range(num_eval_batches):
        batch = val_dataset.sample(agent.config['batch_size'])
        _, eval_info = agent.total_loss(batch, grad_params=None)
        
        batch_size = batch['observations'].shape[0]
        total_loss += eval_info['idm_loss'] * batch_size
        total_mae += eval_info['action_mae'] * batch_size
        total_cosine_sim += eval_info['action_cosine_similarity'] * batch_size
        total_samples += batch_size
    
    return {
        'idm_loss': total_loss / total_samples,
        'action_mae': total_mae / total_samples,
        'action_cosine_similarity': total_cosine_sim / total_samples,
    }


def main(_):
    # Set up logger.
    agent_name = FLAGS.agent.agent_name
    env_name = FLAGS.env_name
    exp_name = f"{agent_name}_{env_name}_{get_exp_name(FLAGS.seed)}"
    setup_wandb(project='idm', group=FLAGS.run_group, name=exp_name)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Make environment and datasets.
    config = FLAGS.agent
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, frame_stack=FLAGS.frame_stack)

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Set up datasets.
    train_dataset = Dataset.create(**train_dataset)
    epoch_steps = train_dataset.size // config['batch_size']
    print(f"Dataset Size: {train_dataset.size}")
    print(f"Epoch Steps: {epoch_steps}")
    train_dataset = ReplayBuffer.create_from_initial_dataset(
        dict(train_dataset), size=train_dataset.size + 1
    )

    FLAGS.train_steps = epoch_steps * 100
    FLAGS.save_interval = FLAGS.train_steps
    
    # Set p_aug and frame_stack.
    for dataset in [train_dataset, val_dataset]:
        if dataset is not None:
            dataset.p_aug = FLAGS.p_aug
            dataset.frame_stack = FLAGS.frame_stack

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
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()

    for i in tqdm.tqdm(range(1, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        # Sample batch for training
        batch = train_dataset.sample(config['batch_size'])
        
        # Update agent
        agent, update_info = agent.update(batch)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            
            # Validation evaluation
            if val_dataset is not None:
                val_metrics = evaluate_idm(agent, val_dataset)
                train_metrics.update({f'validation/{k}': v for k, v in val_metrics.items()})
            
            # Time metrics
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            
            # Log to wandb and csv
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if FLAGS.eval_interval != 0 and (i == 1 or i % FLAGS.eval_interval == 0):
            eval_metrics = {}
            
            # Comprehensive IDM evaluation
            if val_dataset is not None:
                eval_info = evaluate_idm(agent, val_dataset, num_eval_batches=20)
                for k, v in eval_info.items():
                    eval_metrics[f'evaluation/{k}'] = v
                
                # Additional IDM-specific metrics
                eval_batch = val_dataset.sample(config['batch_size'])
                current_obs = eval_batch['observations']
                next_obs = eval_batch['next_observations']
                true_actions = eval_batch['actions']
                
                # Predict actions
                predicted_actions = agent.predict_action(current_obs, next_obs)
                
                # Compute additional metrics
                action_distances = np.linalg.norm(predicted_actions - true_actions, axis=1)
                eval_metrics['evaluation/mean_action_distance'] = np.mean(action_distances)
                eval_metrics['evaluation/std_action_distance'] = np.std(action_distances)
                eval_metrics['evaluation/max_action_distance'] = np.max(action_distances)
                
                # Directional accuracy (cosine similarity)
                pred_norms = predicted_actions / (np.linalg.norm(predicted_actions, axis=1, keepdims=True) + 1e-8)
                true_norms = true_actions / (np.linalg.norm(true_actions, axis=1, keepdims=True) + 1e-8)
                directional_similarity = np.sum(pred_norms * true_norms, axis=1)
                eval_metrics['evaluation/mean_directional_similarity'] = np.mean(directional_similarity)
                
                # Action magnitude analysis
                pred_magnitudes = np.linalg.norm(predicted_actions, axis=1)
                true_magnitudes = np.linalg.norm(true_actions, axis=1)
                eval_metrics['evaluation/magnitude_ratio'] = np.mean(pred_magnitudes) / (np.mean(true_magnitudes) + 1e-8)
                eval_metrics['evaluation/magnitude_correlation'] = np.corrcoef(pred_magnitudes, true_magnitudes)[0, 1]

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
