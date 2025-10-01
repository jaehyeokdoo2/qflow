#!/bin/bash

set -e

task_idx=${1:-1}

env_name="antmaze-large-navigate-singletask-task${task_idx}-v0"
# env_name="humanoidmaze-medium-navigate-singletask-task${task_idx}-v0"

# double check with the user if the passed argument is desired
read -p "Are you sure you want to run Lyapunov training on task ${env_name}? (y/n): " confirm

if [ "$confirm" != "y" ]; then
    echo "Aborting..."
    exit 1
fi

echo "Running Lyapunov training on task ${env_name}"

# Training hyperparameters
training_steps=1000000
batch_size=256
log_interval=5000
save_interval=1000000

# Network architecture
latent_dim=16
hidden_dims="512,512,512"
encoder_hidden_dims="128,128"
decoder_hidden_dims="128,128"
layer_norm=false
module_layer_norm=false

# Lyapunov loss hyperparameters
delta=0.1
noise_std=0.1
learning_rate=3e-4
reconstruction_coeff=1.0

# Experiment settings
start_seed=0
end_seed=0

for i in $(seq $start_seed $end_seed); do
    python lyp_training_reconstruction.py \
        --seed=$i \
        --training_steps=$training_steps \
        --batch_size=$batch_size \
        --log_interval=$log_interval \
        --save_interval=$save_interval \
        --env_name=$env_name \
        --latent_dim=$latent_dim \
        --hidden_dims=$hidden_dims \
        --encoder_hidden_dims=$encoder_hidden_dims \
        --decoder_hidden_dims=$decoder_hidden_dims \
        --layer_norm=$layer_norm \
        --module_layer_norm=$module_layer_norm \
        --delta=$delta \
        --noise_std=$noise_std \
        --learning_rate=$learning_rate \
        --reconstruction_coeff=$reconstruction_coeff \
    
done
