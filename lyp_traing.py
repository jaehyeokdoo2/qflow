import os
import platform

import json
import random
import time

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import wandb
import optax
import flax.linen as nn
from flax.training import train_state
from absl import app, flags
from ml_collections import config_flags

from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb
from utils.networks import LatentLyapunovFunction

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'LyapunovTraining', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-double-play-singletask-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')

flags.DEFINE_integer('training_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('save_interval', 100000, 'Saving interval.')

flags.DEFINE_float('learning_rate', 3e-4, 'Learning rate.')
flags.DEFINE_integer('latent_dim', 16, 'Latent dimension.')
flags.DEFINE_list('hidden_dims', [64, 64], 'Hidden layer dimensions.')
flags.DEFINE_boolean('layer_norm', False, 'Whether to use layer normalization.')

# Lyapunov loss hyperparameters
flags.DEFINE_float('delta', 0.1, 'Margin for Lyapunov loss.')
flags.DEFINE_float('noise_std', 0.1, 'Standard deviation of noise for perturbed actions.')

flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')


class TrainState(train_state.TrainState):
    """Extended train state with additional metrics."""
    pass


def create_train_state(rng, lyapunov_fn, learning_rate, state_dim, action_dim):
    """Create initial training state."""
    # Initialize network with dummy inputs
    dummy_state = jnp.ones((1, state_dim))
    dummy_action = jnp.ones((1, action_dim))
    
    params = lyapunov_fn.init(rng, dummy_state, dummy_action)
    
    tx = optax.adam(learning_rate)
    return TrainState.create(
        apply_fn=lyapunov_fn.apply,
        params=params,
        tx=tx
    )


def lyapunov_loss(params, apply_fn, batch, delta=0.1, noise_std=0.1, rng_key=None):
    """
    Compute REVERSED Lyapunov loss: Expert actions should have HIGHER V(s,a) than perturbed actions
    L_V = E[max(0, V(s,a_perturbed) - V(s,a_expert) + δ)]
    
    This leverages the network's natural tendency to undervalue unseen regions:
    - Expert actions (seen in training) get HIGHER V values (network's natural preference)
    - Perturbed actions (more unseen) get LOWER V values (network's natural tendency)
    - At test time, we use -V(s,a) so expert actions appear as "lower energy/safer"
    
    Args:
        params: Network parameters
        apply_fn: Network apply function
        batch: Batch of data with 'observations' and 'actions'
        delta: Margin for the loss function
        noise_std: Standard deviation of Gaussian noise for perturbed actions
        rng_key: Random key for noise generation
    
    Returns:
        loss: Scalar loss value
        info: Dictionary with loss components
    """
    states = batch['observations']
    actions = batch['actions']
    
    # Generate random key if not provided
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    
    # Compute V(s,a) for expert actions
    v_expert = apply_fn(params, states, actions)
    
    # Generate perturbed actions by adding noise to expert actions
    noise = jax.random.normal(rng_key, actions.shape) * noise_std
    perturbed_actions = actions + noise
    
    # Compute V(s,a_perturbed) for the same states but with perturbed actions
    v_perturbed = apply_fn(params, states, perturbed_actions)
    
    # REVERSED Lyapunov constraint: V(s,a_expert) >= V(s,a_perturbed) + δ
    # Expert actions should have HIGHER V values than perturbed actions
    # Penalize when V(s,a_perturbed) > V(s,a_expert) - δ
    lyapunov_loss = jnp.mean(jax.nn.relu(v_perturbed - v_expert + delta))
    
    # Calculate difference (should be positive for expert > perturbed)
    v_difference_mean = jnp.mean(v_expert - v_perturbed)
    
    info = {
        'lyapunov_loss': lyapunov_loss,
        'v_expert_mean': jnp.mean(v_expert),
        'v_perturbed_mean': jnp.mean(v_perturbed),
        'v_difference_mean': v_difference_mean,  # Should be positive (expert > perturbed)
        'expert_advantage_ratio': jnp.mean(v_expert > v_perturbed),  # Fraction where expert > perturbed
        'mean_lyapunov_value': jnp.mean(v_expert),
        'std_lyapunov_value': jnp.std(v_expert),
        'min_lyapunov_value': jnp.min(v_expert),
        'max_lyapunov_value': jnp.max(v_expert),
    }
    
    return lyapunov_loss, info


@jax.jit
def train_step(state, batch, rng_key, delta=0.1, noise_std=0.1):
    """Perform a single training step."""
    def loss_fn(params):
        return lyapunov_loss(params, state.apply_fn, batch, delta, noise_std, rng_key)
    
    (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state, info


def lyapunov_value(params, apply_fn, states, actions):
    """
    Get the actual Lyapunov value: -V(s,a) (negated for proper interpretation)
    Use this for evaluation and plotting where lower values should indicate safer actions
    
    Args:
        params: Network parameters
        apply_fn: Network apply function
        states: State tensor
        actions: Action tensor
    
    Returns:
        Negated network output (proper Lyapunov values)
    """
    return -apply_fn(params, states, actions)


def save_lyapunov_network(state, save_dir, step):
    """Save the Lyapunov network parameters."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save parameters in a more JAX-friendly format
    params_path = os.path.join(save_dir, f'lyapunov_params_step_{step}.npz')
    flat_params, tree_def = jax.tree_util.tree_flatten(state.params)
    np.savez(params_path, *flat_params)
    
    # Save tree definition for reconstruction
    tree_path = os.path.join(save_dir, f'lyapunov_tree_step_{step}.pkl')
    import pickle
    with open(tree_path, 'wb') as f:
        pickle.dump(tree_def, f)
    
    # Save training state metadata
    metadata = {
        'step': step,
        'opt_state': state.opt_state,
    }
    metadata_path = os.path.join(save_dir, f'lyapunov_metadata_step_{step}.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Saved Lyapunov network to {save_dir} at step {step}")
    print(f"  Parameters: {params_path}")
    print(f"  Tree structure: {tree_path}")
    print(f"  Metadata: {metadata_path}")


def main(_):
    # Set up logger
    exp_name = f"lyapunov_{FLAGS.env_name}_{get_exp_name(FLAGS.seed)}"
    setup_wandb(project='lyapunov', group=FLAGS.run_group, name=exp_name)
    
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)
    
    # Make environment and datasets
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name, frame_stack=FLAGS.frame_stack
    )
    
    # Initialize random seeds
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    rng = jax.random.PRNGKey(FLAGS.seed)
    
    # Set up dataset
    train_dataset = Dataset.create(**train_dataset)
    train_dataset.p_aug = FLAGS.p_aug
    train_dataset.frame_stack = FLAGS.frame_stack
    
    # Get state and action dimensions from environment
    example_batch = train_dataset.sample(1)
    state_dim = example_batch['observations'].shape[-1]
    action_dim = example_batch['actions'].shape[-1]
    
    print(f"Environment dimensions:")
    print(f"  State dimension: {state_dim}")
    print(f"  Action dimension: {action_dim}")
    
    # Create Lyapunov network
    hidden_dims = [int(x) for x in FLAGS.hidden_dims]
    lyapunov_fn = LatentLyapunovFunction(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=FLAGS.latent_dim,
        hidden_dim=hidden_dims,
        layer_norm=FLAGS.layer_norm
    )
    
    # Create training state
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, lyapunov_fn, FLAGS.learning_rate, state_dim, action_dim)
    
    # Set up logging
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    first_time = time.time()
    last_time = time.time()
    
    print(f"Starting Lyapunov function training for {FLAGS.training_steps} steps...")
    print(f"Network architecture: {state_dim + action_dim} -> {FLAGS.latent_dim} -> {hidden_dims} -> 1")
    print(f"Lyapunov loss hyperparameters:")
    print(f"  Delta (margin): {FLAGS.delta}")
    print(f"  Noise std: {FLAGS.noise_std}")
    print(f"  Learning rate: {FLAGS.learning_rate}")
    print(f"  Batch size: {FLAGS.batch_size}")
    print(f"  Layer normalization: {FLAGS.layer_norm}")
    print(f"Dataset size: {train_dataset.size}")
    
    # Training loop
    for i in tqdm.tqdm(range(1, FLAGS.training_steps + 1), smoothing=0.1, dynamic_ncols=True):
        # Sample batch from dataset
        batch = train_dataset.sample(FLAGS.batch_size)
        
        # Generate random key for noise
        rng, step_rng = jax.random.split(rng)
        
        # Perform training step
        state, update_info = train_step(state, batch, step_rng, FLAGS.delta, FLAGS.noise_std)
        
        # Log metrics
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            
            # Add validation metrics if validation dataset exists
            if val_dataset is not None:
                val_batch = val_dataset.sample(FLAGS.batch_size)
                rng, val_rng = jax.random.split(rng)
                _, val_info = lyapunov_loss(state.params, state.apply_fn, val_batch, FLAGS.delta, FLAGS.noise_std, val_rng)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            train_metrics['training/step'] = i
            
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)
        
        # Save network
        if i % FLAGS.save_interval == 0:
            save_lyapunov_network(state, FLAGS.save_dir, i)
    
    # Save final network
    save_lyapunov_network(state, FLAGS.save_dir, FLAGS.training_steps)
    
    train_logger.close()
    print("Training completed!")


if __name__ == '__main__':
    app.run(main)
