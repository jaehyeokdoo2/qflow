#!/bin/bash

set -e

# FQL hyperparams
use_value_layer_norm=true
use_actor_layer_norm=false

# Model restore paths
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
fbrac_humanoidmaze_medium_restore_path="/mnt/nas/jaehyeok/fql/exp/fql/Debug/fbrac_humanoidmaze-medium-navigate-singletask-task1-v0_sd000_20250918_191439_utd-ratio1"
lyapunov_fbrac_humanoidmaze_medium_restore_path="/mnt/nas/jaehyeok/fql/exp/lyapunov_fbrac/LyapunovFBRAC/lyapunov_fbrac_humanoidmaze-medium-navigate-singletask-task1-v0_sd000_20250916_175932_utd-ratio1"

# Lyapunov model paths (for guidance)
lyapunov_model_antmaze_large_path="/mnt/nas/jaehyeok/fql/exp/lyapunov_fbrac/LyapunovFBRAC/lyapunov_fbrac_antmaze-large-navigate-singletask-task1-v0_sd000_20250916_210818_utd-ratio1"
lyapunov_model_humanoidmaze_medium_path="/mnt/nas/jaehyeok/fql/exp/lyapunov/LyapunovTraining/lyapunov_humanoidmaze-medium-navigate-singletask-task1-v0_sd000_20250916_200152"

# Environment and model selection
# env_name="antmaze-large-navigate-singletask-task1-v0"
# env_name="antmaze-giant-navigate-singletask-task2-v0"
env_name="humanoidmaze-medium-navigate-singletask-task1-v0"

# Set restore path and Lyapunov model path based on environment
if [ "$env_name" = "antmaze-large-navigate-singletask-task1-v0" ]; then
    restore_path=$fbrac_antmaze_large_restore_path
    lyapunov_model_path=$lyapunov_model_antmaze_large_path
elif [ "$env_name" = "humanoidmaze-medium-navigate-singletask-task1-v0" ]; then
    restore_path=$fbrac_humanoidmaze_medium_restore_path
    lyapunov_model_path=$lyapunov_model_humanoidmaze_medium_path
else
    echo "Unknown environment: $env_name"
    exit 1
fi

restore_epoch=1000000
lyapunov_model_step=1000000

# Guidance parameters
guidance_coeff=0.001
# guidance_coeff=0.5
# guidance_coeff=2.0
partial_guidance=3

# Lyapunov network architecture
latent_dim=32
lyapunov_hidden_dims="512,512,512"
lyapunov_layer_norm=false

echo "Running guided evaluation with:"
echo "  Environment: $env_name"
echo "  FBRAC model: $restore_path"
echo "  Lyapunov model: $lyapunov_model_path"
echo "  Guidance coefficient: $guidance_coeff"
echo "  Partial guidance: $partial_guidance"
echo "  Restore epoch: $restore_epoch"
echo "  Lyapunov model step: $lyapunov_model_step"
echo ""

python eval_guidance.py \
    --q_percentile=100 \
    --eval_episodes=10 \
    --seed=0 \
    --env_name=$env_name \
    --agent=agents/fbrac.py \
    --save_tag="guided_coeff${guidance_coeff}_partial${partial_guidance}_alpha30-q_mean-discount995" \
    --partial_guidance=$partial_guidance \
    --restore_path=$restore_path \
    --restore_epoch=$restore_epoch \
    --lyapunov_model_path=$lyapunov_model_path \
    --lyapunov_model_step=$lyapunov_model_step \
    --guidance_coeff=$guidance_coeff \
    --latent_dim=$latent_dim \
    --lyapunov_hidden_dims=$lyapunov_hidden_dims \
    --lyapunov_layer_norm=$lyapunov_layer_norm \
    --agent.layer_norm=$use_value_layer_norm \
    --agent.actor_layer_norm=$use_actor_layer_norm
