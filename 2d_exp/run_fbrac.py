"""
Run FBRAC (Flow-Based Actor-Critic, outer critic only) toy 2D experiment.

Usage:
    python run_fbrac.py --dataset two_spirals --alphas 0.5 0.3
"""
import argparse
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from datasets import sample_from_dataset, plot_dataset, DOMAIN_MIN, DOMAIN_MAX
from models import integrate_flow
from train.fbrac import train_fbrac


def parse_args():
    p = argparse.ArgumentParser(description="FBRAC 2D Toy Experiment")
    p.add_argument('--dataset', type=str, default='two_spirals',
                   choices=['two_spirals', 'moons', 'swissroll', 'eight_gaussians'])
    p.add_argument('--n_data', type=int, default=10000)
    p.add_argument('--alphas', type=float, nargs='+', default=[0.5, 0.3])
    p.add_argument('--bc_epochs', type=int, default=2000)
    p.add_argument('--offline_epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--batch_size', type=int, default=4096)
    p.add_argument('--flow_steps', type=int, default=25)
    p.add_argument('--num_layers', type=int, default=4)
    p.add_argument('--hidden_dim', type=int, default=512)
    p.add_argument('--num_evals', type=int, default=512)
    p.add_argument('--num_plots', type=int, default=10)
    p.add_argument('--use_target_network', action='store_true')
    p.add_argument('--tau', type=float, default=0.005)
    p.add_argument('--save_dir', type=str, default='toy_experiment')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    np.random.seed(args.seed)
    epochs = args.bc_epochs + args.offline_epochs
    plot_step = epochs // args.num_plots if args.num_plots > 0 else None

    print(f"Loading dataset: {args.dataset} (n={args.n_data})")
    x_1_data, rewards_data = sample_from_dataset(args.n_data, args.dataset)
    rewards_data = np.array(rewards_data).flatten()

    results = []
    for alpha in args.alphas:
        print(f"\n--- FBRAC alpha={alpha} ---")
        flow_params, critic_params, bc_policy_params, history = train_fbrac(
            x_1_data, rewards_data,
            epochs=epochs,
            lr=args.lr,
            alpha=alpha,
            flow_steps=args.flow_steps,
            hidden=args.hidden_dim,
            num_layers=args.num_layers,
            plot_every=plot_step,
            use_target_network=args.use_target_network,
            tau=args.tau,
            batch_size=args.batch_size,
            bc_epochs=args.bc_epochs,
            offline_epochs=args.offline_epochs,
            num_evals=args.num_evals,
            save_dir=args.save_dir,
            dataset_type=args.dataset,
            n_data=args.n_data,
        )
        results.append((alpha, flow_params, bc_policy_params))

    # Final comparison plot
    results_sorted = sorted(results, key=lambda x: x[0], reverse=True)
    n_cols = 1 + len(results_sorted)
    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3))
    if n_cols == 1:
        axes = [axes]

    z_eval = jrandom.normal(jrandom.PRNGKey(999), (args.num_evals, 2))

    bc_flow_params = results_sorted[0][2]
    bc_actions = np.array(integrate_flow(bc_flow_params, z_eval, steps=args.flow_steps))
    axes[0].scatter(bc_actions[:, 0], bc_actions[:, 1], s=10, alpha=0.6, edgecolors='none')
    axes[0].set_title("BC", fontsize=15, fontstyle='italic')
    axes[0].set_xlim(DOMAIN_MIN, DOMAIN_MAX); axes[0].set_ylim(DOMAIN_MIN, DOMAIN_MAX)
    axes[0].set_aspect('equal')

    for i, (alpha, flow_params, _) in enumerate(results_sorted):
        actions = np.array(integrate_flow(flow_params, z_eval, steps=args.flow_steps))
        axes[i + 1].scatter(actions[:, 0], actions[:, 1], s=10, alpha=0.6, edgecolors='none')
        axes[i + 1].set_title(f'α={alpha}', fontsize=15, fontstyle='italic')
        axes[i + 1].set_xlim(DOMAIN_MIN, DOMAIN_MAX); axes[i + 1].set_ylim(DOMAIN_MIN, DOMAIN_MAX)
        axes[i + 1].set_aspect('equal')

    plt.tight_layout()
    out_dir = f'{args.save_dir}/{args.dataset}/fbrac'
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f'{out_dir}/final_comparison.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"\nSaved final comparison to {out_dir}/final_comparison.png")


if __name__ == '__main__':
    main()
