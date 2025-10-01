#!/bin/bash

## General training params
use_visual_encoder=true
p_aug=0.5
frame_stack=3

task="visual-cube-single-play-singletask-task1-v0"
offline_steps=500000
log_interval=5000
save_dir="exp/"

TRAIN_PARAMS="\
--env_name=$task \
--offline_steps=$offline_steps \
--log_interval=$log_interval \
--save_dir=$save_dir \
"

if $use_visual_encoder; then    
    # include p_aug and frame_stack
    TRAIN_PARAMS="$TRAIN_PARAMS --p_aug $p_aug --frame_stack $frame_stack"
fi

## FQL hyperparams
fql_agent="agents/fql.py"
alpha=300
discount=0.99
encoder="impala_small"

FQL_HYPER_PARAMS="\
--agent=$fql_agent \
--agent.alpha=$alpha \
--agent.encoder=$encoder \
--agent.discount=$discount \
"

## IFQL hyperparams
ifql_agent="agents/ifql.py"
expectile=0.9
num_samples=32 # For rejection sampling (during inference)
discount=0.99
encoder="impala_small"

IFQL_HYPER_PARAMS="\
--agent=$ifql_agent \
--agent.expectile=$expectile \
--agent.num_samples=$num_samples \
--agent.discount=$discount \
--agent.encoder=$encoder \
"

## Run training

RUN_CMD="python main.py"
HYPER_PARAMS=$IFQL_HYPER_PARAMS

$RUN_CMD $TRAIN_PARAMS $HYPER_PARAMS