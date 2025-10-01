import os
import platform

import json
import random
import pickle

import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags
from ml_collections import config_flags

from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset
from utils.log_utils import get_exp_name
from utils.networks import LatentLyapunovFunction

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'LyapunovTraining', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-double-play-singletask-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('model_path', None, 'Path to saved model directory.')
flags.DEFINE_integer('model_step', None, 'Model step to load (e.g., 1000000).')

flags.DEFINE_integer('latent_dim', 16, 'Latent dimension.')
flags.DEFINE_list('hidden_dims', [64, 64], 'Hidden layer dimensions.')
flags.DEFINE_boolean('layer_norm', False, 'Whether to use layer normalization.')

flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')


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


def lyapunov_value(params, apply_fn, states, actions):
    """
    Get the actual Lyapunov value: -V(s,a) (negated for proper interpretation)
    Use this for evaluation and plotting where lower values should indicate safer actions
    """
    return -apply_fn(params, states, actions)


def main(_):
    print("=" * 60)
    print("LYAPUNOV FUNCTION ANALYSIS")
    print("=" * 60)
    
    # Validate required arguments
    if FLAGS.model_path is None:
        raise ValueError("Must specify --model_path to load saved model")
    if FLAGS.model_step is None:
        raise ValueError("Must specify --model_step to load saved model")
    
    # Make environment and datasets
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name, frame_stack=FLAGS.frame_stack
    )
    
    # Initialize random seeds
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    
    # Set up dataset
    train_dataset = Dataset.create(**train_dataset)
    train_dataset.p_aug = FLAGS.p_aug
    train_dataset.frame_stack = FLAGS.frame_stack
    
    # Get state and action dimensions from environment
    example_batch = train_dataset.sample(1)
    state_dim = example_batch['observations'].shape[-1]
    action_dim = example_batch['actions'].shape[-1]
    
    print(f"Environment: {FLAGS.env_name}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Dataset size: {train_dataset.size}")
    
    # Create Lyapunov network
    hidden_dims = [int(x) for x in FLAGS.hidden_dims]
    lyapunov_fn = LatentLyapunovFunction(
        hidden_dim=hidden_dims,
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=FLAGS.latent_dim,
        layer_norm=FLAGS.layer_norm
    )
    
    # Load trained parameters
    params = load_lyapunov_network(FLAGS.model_path, FLAGS.model_step, lyapunov_fn, state_dim, action_dim)
    apply_fn = lyapunov_fn.apply
    
    print(f"Network architecture: {state_dim + action_dim} -> {FLAGS.latent_dim} -> {hidden_dims} -> 1")
    print("Model loaded successfully!")
    print("Ready for analysis...")
    
    # TODO: Add analysis functions here


if __name__ == '__main__':
    app.run(main)
