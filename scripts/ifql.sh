#!/bin/bash

set -e

# Env
# env_name="antmaze-large-navigate-singletask-task2-v0"
env_name="antmaze-giant-navigate-singletask-task1-v0"

offline_steps=1000000
eval_interval=50000

# IFQL hyperparams
use_value_layer_norm=true
use_actor_layer_norm=false
num_samples=32
discount=0.99
expectile=0.9

for i in {0..0}; do
    python main.py \
        --seed=$i \
        --offline_steps=$offline_steps \
        --eval_interval=$eval_interval \
        --env_name=$env_name \
        --agent=agents/ifql.py \
        --agent.discount=$discount \
        --agent.expectile=$expectile \
        --agent.num_samples=$num_samples \
        --agent.layer_norm=$use_value_layer_norm \
        --agent.actor_layer_norm=$use_actor_layer_norm
done