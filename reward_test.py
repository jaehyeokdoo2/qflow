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
from utils.evaluation import evaluate, flatten
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-double-play-singletask-v0', 'Environment (dataset) name.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('online_steps', 0, 'Number of online steps.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('batch_size', 256, 'Batch size.')

def main(_):
    # Make environment and datasets.
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name)

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Set up datasets.
    train_dataset = Dataset.create(**train_dataset)

    # Use the training dataset as the replay buffer.
    train_dataset = ReplayBuffer.create_from_initial_dataset(
        dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
    )
    replay_buffer = train_dataset
    
    avg_rewards = []
    acc = 0

    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + FLAGS.online_steps + 1), smoothing=0.1, dynamic_ncols=True):
        if i <= FLAGS.offline_steps:
            batch = train_dataset.sample(FLAGS.batch_size)
            acc += np.mean(batch['rewards'])
            if i % FLAGS.log_interval == 0:
                avg_rewards.append(acc / FLAGS.log_interval)
                acc = 0
    
    # save in csv file
    with open(f'{FLAGS.env_name}_avg_rewards.csv', 'w') as f:
        for i, avg_reward in enumerate(avg_rewards):
            f.write(f'{i}, {avg_reward}\n')

if __name__ == '__main__':
    app.run(main)
