#!/bin/bash

set -e

task_idx=${1:-1}

# env_name="antmaze-large-navigate-singletask-task${task_idx}-v0"
env_name="humanoidmaze-medium-navigate-singletask-task${task_idx}-v0"

# double check with the user if the pased argument is desired
read -p "Are you sure you want to run FBRAC agent on task ${env_name}? (y/n): " confirm

if [ "$confirm" != "y" ]; then
    echo "Aborting..."
    exit 1
fi

echo "Running FBRAC agent on task ${env_name}"

offline_steps=1000000
eval_interval=50000

use_value_layer_norm=true
use_actor_layer_norm=false
actor_weight_decay=0.0
alpha=15
q_agg=mean
reward_discount=0.995
value_jacobian_reg=0.0
weight_value_jacobian_reg=false # Weight value Jacobian regularization.

start_seed=0
end_seed=0

actor_loss='pgbc'
lyapunov_reg=5.0
lyapunov_latent_dim=32
lyapunov_hidden_dims="512,512,512"
lyapunov_layer_norm=false

# lyapunov_path="/mnt/nas/jaehyeok/fql/exp/lyapunov/LyapunovTraining/lyapunov_antmaze-large-navigate-singletask-task1-v0_sd000_20250916_202904"
lyapunov_path="/mnt/nas/jaehyeok/fql/exp/lyapunov/LyapunovTraining/lyapunov_humanoidmaze-medium-navigate-singletask-task1-v0_sd000_20250916_200152"
lyapunov_step=1000000


for seed in $(seq $start_seed $end_seed); do
    python lyp_fbrac.py \
        --seed=$seed \
        --num_updates=1 \
        --eval_interval=$eval_interval \
        --offline_steps=$offline_steps \
        --env_name=$env_name \
        --lyapunov_path=$lyapunov_path \
        --lyapunov_step=$lyapunov_step \
        --agent=agents/lyapunov_fbrac.py \
        --agent.actor_weight_decay=$actor_weight_decay \
        --agent.actor_loss=$actor_loss \
        --agent.q_agg=$q_agg \
        --agent.alpha=$alpha \
        --agent.discount=$reward_discount \
        --agent.reward_scale=1 \
        --agent.normalize_q_loss=false \
        --agent.layer_norm=$use_value_layer_norm \
        --agent.actor_layer_norm=$use_actor_layer_norm \
        --agent.value_jacobian_reg=$value_jacobian_reg \
        --agent.weight_value_jacobian_reg=$weight_value_jacobian_reg \
        --agent.lyapunov_reg=$lyapunov_reg \
        --agent.lyapunov_latent_dim=$lyapunov_latent_dim \
        --agent.lyapunov_hidden_dims=$lyapunov_hidden_dims \
        --agent.lyapunov_layer_norm=$lyapunov_layer_norm
done
