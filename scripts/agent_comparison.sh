#!/bin/bash

set -e

# Environment
env_name="antmaze-large-navigate-singletask-task2-v0"

# Model paths from eval.sh
fql_restore_path="/mnt/nas/jaehyeok/fql/exp/fql/Debug/fql_antmaze-large-navigate-singletask-task2-v0_sd000_20250827_170146"
ifql_restore_path="/mnt/nas/jaehyeok/fql/exp/fql/Debug/ifql_antmaze-large-navigate-singletask-task2-v0_sd000_20250901_163207_utd-ratio1"

# Run agent comparison
python agent_comparison.py \
    --env_name=$env_name \
    --agent_1_path=$fql_restore_path \
    --agent_2_path=$ifql_restore_path \
    --agent_1_epoch=1000000 \
    --agent_2_epoch=1000000 \
    --eval_episodes=10 \
    --seed=0 \
