#!/bin/bash

set -e

task_idx=${1:-1}

env_name="antmaze-large-navigate-singletask-task${task_idx}-v0"
# env_name="humanoidmaze-medium-navigate-singletask-task${task_idx}-v0"

# double check with the user if the pased argument is desired
read -p "Are you sure you want to run DQL_TDQ agent on task ${env_name}? (y/n): " confirm

if [ "$confirm" != "y" ]; then
    echo "Aborting..."
    exit 1
fi

echo "Running DQL_TDQ agent on task ${env_name}"

offline_steps=1000000
eval_interval=50000

use_value_layer_norm=true
use_actor_layer_norm=false
alpha=50
q_agg=mean
reward_discount=0.99
q_t_weight=1.0

# Diffusion-specific parameters
diffusion_steps=10
beta_schedule=vp
beta_min=0.1
beta_max=10.0

start_seed=0
end_seed=0

for seed in $(seq $start_seed $end_seed); do
    python main.py \
        --seed=$seed \
        --eval_interval=$eval_interval \
        --offline_steps=$offline_steps \
        --env_name=$env_name \
        --agent=agents/dql_tdq.py \
        --agent.q_agg=$q_agg \
        --agent.alpha=$alpha \
        --agent.discount=$reward_discount \
        --agent.reward_scale=1 \
        --agent.normalize_q_loss=false \
        --agent.layer_norm=$use_value_layer_norm \
        --agent.actor_layer_norm=$use_actor_layer_norm \
        --agent.q_t_weight=$q_t_weight \
        --agent.diffusion_steps=$diffusion_steps \
        --agent.beta_schedule=$beta_schedule \
        --agent.beta_min=$beta_min \
        --agent.beta_max=$beta_max
done

