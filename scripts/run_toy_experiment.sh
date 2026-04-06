#!/bin/bash
set -e

# Network
hidden_dim=512
num_layers=4

# Alg.
flow_steps=25

bc_epochs=5000
offline_epochs=500

# Optim.
learning_rate=3e-4
batch_size=4096

# Display
num_plots=10
num_evals=512

alphas="5 1 0.5 0.3 0.2"
group1="eight_gaussians two_spirals"
group2="swissroll moons"

# Env
for task in $group1
do

python toy_experiment.py  \
    --hidden_dim $hidden_dim \
    --num_layers $num_layers \
    --bc_epochs $bc_epochs \
    --offline_epochs $offline_epochs \
    --learning_rate $learning_rate \
    --batch_size $batch_size \
    --flow_steps $flow_steps \
    --tasks $task \
    --num_plots $num_plots \
    --alphas $alphas \
    --num_evals $num_evals \
    --save_dir "toy_experiment/activation_test" \
    --run_flow_tvl
done