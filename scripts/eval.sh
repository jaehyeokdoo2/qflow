#!/bin/bash

set -e


# FQL hyperparams
use_value_layer_norm=true
use_actor_layer_norm=false

fql_antmaze_large_restore_path="/mnt/nas/jaehyeok/fql/exp/fql/Debug/fql_antmaze-large-navigate-singletask-task2-v0_sd000_20250827_170146"
ifql_antmaze_large_restore_path="/mnt/nas/jaehyeok/fql/exp/fql/Debug/ifql_antmaze-large-navigate-singletask-task2-v0_sd000_20250901_163207_utd-ratio1"
fql_separate_antmaze_large_restore_path="/mnt/nas/jaehyeok/fql/exp/fql_separate/Debug/fql_separate_antmaze-large-navigate-singletask-task2-v0_sd000_20250828_193438"
rebrac_antmaze_large_restore_path="/mnt/nas/jaehyeok/fql/exp/fql/Debug/rebrac_antmaze-large-navigate-singletask-task2-v0_sd000_20250902_003455_utd-ratio1"
iql_antmaze_large_restore_path="/mnt/nas/jaehyeok/fql/exp/fql/Debug/iql_antmaze-large-navigate-singletask-task2-v0_sd000_20250904_174856_utd-ratio1"
iql_antmaze_large_restore_path="/mnt/nas/jaehyeok/fql/exp/fql/Debug/iql_antmaze-large-navigate-singletask-task2-v0_sd000_20250905_164434_utd-ratio1"
iql_antmaze_large_restore_paht="/mnt/nas/jaehyeok/fql/exp/fql/Debug/iql_antmaze-large-navigate-singletask-task2-v0_sd000_20250904_195824_utd-ratio1"
fbrac_antmaze_large_restore_path="/mnt/nas/jaehyeok/fql/exp/fql/Debug/fbrac_antmaze-large-navigate-singletask-task1-v0_sd000_20250907_031541_utd-ratio1"
fbrac_antmaze_large_reg_path="/mnt/nas/jaehyeok/fql/exp/fql/Debug/fbrac_antmaze-large-navigate-singletask-task2-v0_sd000_20250908_164219_utd-ratio1"

lyapunov_fbrac_antmaze_large_restore_path="/mnt/nas/jaehyeok/fql/exp/lyapunov_fbrac/LyapunovFBRAC/lyapunov_fbrac_antmaze-large-navigate-singletask-task1-v0_sd000_20250916_210818_utd-ratio1"
lyapunov_lowest_restore_path="/mnt/nas/jaehyeok/fql/exp/lyapunov_fbrac/LyapunovFBRAC/lyapunov_fbrac_antmaze-large-navigate-singletask-task1-v0_sd002_20250917_031332_utd-ratio1"


fbrac_antmaze_large_task1_reg_path="/mnt/nas/jaehyeok/fql/exp/fql/Debug/fbrac_antmaze-large-navigate-singletask-task1-v0_sd002_20250908_200922_utd-ratio1"

fql_antmaze_giant_restore_path="/mnt/nas/jaehyeok/fql/exp/fql/Debug/fql_antmaze-giant-navigate-singletask-task2-v0_sd000_20250902_173255_utd-ratio1"
ifql_antmaze_giant_restore_path="/mnt/nas/jaehyeok/fql/exp/fql/Debug/ifql_antmaze-giant-navigate-singletask-task2-v0_sd000_20250902_170151_utd-ratio1"
rebrac_antmaze_giant_restore_path="/mnt/nas/jaehyeok/fql/exp/fql/Debug/rebrac_antmaze-giant-navigate-singletask-task2-v0_sd000_20250902_170356_utd-ratio1"

iql_humanoidmaze_medium_restore_path="/mnt/nas/jaehyeok/fql/exp/fql/Debug/iql_humanoidmaze-medium-navigate-singletask-task1-v0_sd000_20250906_195900_utd-ratio1"
fql_humanoidmaze_medium_restore_path="/mnt/nas/jaehyeok/fql/exp/fql/Debug/fql_humanoidmaze-medium-navigate-singletask-task2-v0_sd000_20250909_193638_utd-ratio1"
fql_humanoidmaze_medium_reg_path="/mnt/nas/jaehyeok/fql/exp/fql/Debug/fql_humanoidmaze-medium-navigate-singletask-task2-v0_sd000_20250909_191200_utd-ratio1"
fbrac_humanoidmaze_medium_restore_path="/mnt/nas/jaehyeok/fql/exp/fql/Debug/fbrac_humanoidmaze-medium-navigate-singletask-task1-v0_sd000_20250908_213249_utd-ratio1"
lyapunov_fbrac_humanoidmaze_medium_restore_path="/mnt/nas/jaehyeok/fql/exp/lyapunov_fbrac/LyapunovFBRAC/lyapunov_fbrac_humanoidmaze-medium-navigate-singletask-task1-v0_sd000_20250916_175932_utd-ratio1"

test="/mnt/nas/jaehyeok/fql/exp/fql/Debug/fbrac_antmaze-large-navigate-singletask-task2-v0_sd000_20250919_172341_utd-ratio1"

# Env
env_name="antmaze-large-navigate-singletask-task2-v0"
# env_name="antmaze-giant-navigate-singletask-task2-v0"
# env_name="humanoidmaze-medium-navigate-singletask-task1-v0"
restore_path=$test
restore_epoch=1000000

python eval.py \
    --q_percentile=100 \
    --eval_episodes=100 \
    --seed=0 \
    --env_name=$env_name \
    --agent=agents/fbrac.py \
    --save_tag="no_guidance_test" \
    --restore_path=$restore_path \
    --restore_epoch=$restore_epoch \
    --agent.layer_norm=$use_value_layer_norm \
    --agent.actor_layer_norm=$use_actor_layer_norm