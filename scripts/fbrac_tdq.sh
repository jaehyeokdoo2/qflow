#!/bin/bash

set -e

task_idx=${1:-1}

# env_name="antmaze-large-navigate-singletask-task${task_idx}-v0"
env_name="humanoidmaze-medium-navigate-singletask-task${task_idx}-v0"

# double check with the user if the pased argument is desired
read -p "Are you sure you want to run FBRAC_TDQ agent on task ${env_name}? (y/n): " confirm

if [ "$confirm" != "y" ]; then
    echo "Aborting..."
    exit 1
fi

echo "Running FBRAC_TDQ agent on task ${env_name}"

offline_steps=1000000
eval_interval=50000

use_value_layer_norm=true
use_actor_layer_norm=false
alpha=60
q_agg=mean
reward_discount=0.995
q_t_weight=1.0

start_seed=0
end_seed=0

for seed in $(seq $start_seed $end_seed); do
    python main.py \
        --seed=$seed \
        --eval_interval=$eval_interval \
        --offline_steps=$offline_steps \
        --env_name=$env_name \
        --agent=agents/fbrac_tdq.py \
        --agent.q_agg=$q_agg \
        --agent.alpha=$alpha \
        --agent.discount=$reward_discount \
        --agent.reward_scale=1 \
        --agent.normalize_q_loss=false \
        --agent.layer_norm=$use_value_layer_norm \
        --agent.actor_layer_norm=$use_actor_layer_norm \
        --agent.q_t_weight=$q_t_weight
done
