#!/bin/bash

set -e

model_path="exp/lyapunov/LyapunovTraining/lyapunov_antmaze-large-navigate-singletask-task1-v0_sd000_20250912_180927"
model_step=1000000
env_name="antmaze-large-navigate-singletask-task1-v0"

python lyp_analysis.py \
    --model_path=$model_path \
    --model_step=$model_step \
    --env_name=$env_name \

