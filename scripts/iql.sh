#!/bin/bash

set -e

# Env
env_name="antmaze-large-navigate-singletask-task2-v0"
# env_name="antmaze-giant-navigate-singletask-task2-v0"
# env_name="humanoidmaze-medium-navigate-singletask-task1-v0"
offline_steps=1000000
eval_interval=50000

# FQL hyperparams
use_value_layer_norm=true
use_actor_layer_norm=false
alpha=3 # BC coeff
reward_discount=0.99
expectile=0.9

# Loss configs
use_mse=false # BC loss type (log_prob or mse)
normalize_q_loss=false # Q loss normalization
actor_weight_decay=0.0 # Weight decay coefficient for actor network.
actor_jacobian_reg=0.0 # Jacobian regularization coefficient.
actor_hessian_reg=0.0 # Hessian regularization coefficient.
weight_jacobian_reg=false # Weight Jacobian regularization.
value_jacobian_reg=1e-5 # Value Jacobian regularization.
weight_value_jacobian_reg=true # Weight value Jacobian regularization.


for i in {0..0}; do
    python main.py \
        --seed=$i \
        --offline_steps=$offline_steps \
        --eval_interval=$eval_interval \
        --env_name=$env_name \
        --agent=agents/iql.py \
        --agent.discount=$reward_discount \
        --agent.alpha=$alpha \
        --agent.use_mse=$use_mse \
        --agent.normalize_q_loss=$normalize_q_loss \
        --agent.actor_loss=ddpgbc \
        --agent.expectile=$expectile \
        --agent.layer_norm=$use_value_layer_norm \
        --agent.actor_layer_norm=$use_actor_layer_norm \
        --agent.actor_weight_decay=$actor_weight_decay \
        --agent.actor_jacobian_reg=$actor_jacobian_reg \
        --agent.actor_hessian_reg=$actor_hessian_reg \
        --agent.weight_jacobian_reg=$weight_jacobian_reg \
        --agent.value_jacobian_reg=$value_jacobian_reg \
        --agent.weight_value_jacobian_reg=$weight_value_jacobian_reg
done