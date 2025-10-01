#!/bin/bash

## General training params
use_visual_encoder=false
p_aug=0.5
frame_stack=3

# task="antmaze-large-diverse-v2"
task="antmaze-large-navigate-singletask-task2-v0"

train_steps=500000
log_interval=5000
save_dir="exp/"

TRAIN_PARAMS="\
--env_name=$task \
--train_steps=$train_steps \
--log_interval=$log_interval \
--save_dir=$save_dir \
"

if $use_visual_encoder; then    
    # include p_aug and frame_stack
    TRAIN_PARAMS="$TRAIN_PARAMS --p_aug $p_aug --frame_stack $frame_stack"
fi

## IDM hyperparams
idm_agent="agents/idm.py"
lr=1e-4
layer_norm=true

IDM_HYPER_PARAMS="\
--agent=$idm_agent \
--agent.lr=$lr \
--agent.layer_norm=$layer_norm \
"

## Run training
RUN_CMD="python main_idm.py"
HYPER_PARAMS=$IDM_HYPER_PARAMS

echo "Starting IDM training with parameters:"
echo "Task: $task"
echo "Train steps: $train_steps"
echo "Learning rate: $lr"
echo "Layer norm: $layer_norm"
echo ""

$RUN_CMD $TRAIN_PARAMS $HYPER_PARAMS
