#!/bin/bash

set -e

task_idx=${1:-1}

# env_name="antmaze-large-navigate-singletask-task${task_idx}-v0"
env_name="humanoidmaze-medium-navigate-singletask-task${task_idx}-v0"
env_name="antmaze-large-play-v2"

# double check with the user if the pased argument is desired
read -p "Are you sure you want to run FQL agent on task ${env_name}? (y/n): " confirm

if [ "$confirm" != "y" ]; then
    echo "Aborting..."
    exit 1
fi

echo "Running FQL agent on task ${env_name}"

offline_steps=1000000
eval_interval=50000

# FQL hyperparams
use_value_layer_norm=true
use_actor_layer_norm=false
alpha=3 # BC coeff
reward_discount=0.99
q_agg=mean

utd_ratio=1

start_seed=0
end_seed=0

value_jacobian_reg=0.0
weight_value_jacobian_reg=false # Weight value Jacobian regularization.


for i in $(seq $start_seed $end_seed); do
    python main.py \
        --num_updates=$utd_ratio \
        --seed=$i \
        --offline_steps=$offline_steps \
        --eval_interval=$eval_interval \
        --env_name=$env_name \
        --agent=agents/fql.py \
        --agent.discount=$reward_discount \
        --agent.alpha=$alpha \
        --agent.q_agg=$q_agg \
        --agent.layer_norm=$use_value_layer_norm \
        --agent.actor_layer_norm=$use_actor_layer_norm \
        --agent.value_jacobian_reg=$value_jacobian_reg \
        --agent.weight_value_jacobian_reg=$weight_value_jacobian_reg
done