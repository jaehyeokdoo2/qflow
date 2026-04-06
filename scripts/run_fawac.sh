#!/bin/bash

set -e

start_task_idx=${1:-1}
end_task_idx=${2:-1}
debug=${3:-false}

# If third argument is "debug" or "true", enable debug mode
if [ "$debug" = "debug" ] || [ "$debug" = "true" ] || [ "$debug" = "1" ]; then
    DEBUG=true
else
    DEBUG=false
fi

for task_idx in $(seq $start_task_idx $end_task_idx); do
    echo "Running FAWAC for task ${task_idx}"
    echo "=========================================="
    echo ""

    # Environment settings - uncomment the environment you want to run
    # env_name="antmaze-large-navigate-singletask-task${task_idx}-v0"
    # env_name="antmaze-giant-navigate-singletask-task${task_idx}-v0"
    # env_name="humanoidmaze-medium-navigate-singletask-task${task_idx}-v0"
    # env_name="humanoidmaze-large-navigate-singletask-task${task_idx}-v0"
    # env_name="antsoccer-arena-navigate-singletask-task${task_idx}-v0"
    # env_name="cube-single-play-singletask-task${task_idx}-v0"
    # env_name="cube-double-play-singletask-task${task_idx}-v0"
    # env_name="scene-play-singletask-task${task_idx}-v0"
    # env_name="puzzle-3x3-play-singletask-task${task_idx}-v0"
    # env_name="puzzle-4x4-play-singletask-task${task_idx}-v0"

    env_name="antmaze-umaze-v2"
    # env_name="antmaze-umaze-diverse-v2"
    # env_name="antmaze-medium-play-v2"
    # env_name="antmaze-medium-diverse-v2"
    # env_name="antmaze-large-play-v2"
    env_name="antmaze-large-diverse-v2"

    # env_name="antmaze-giant-stitch-singletask-task${task_idx}-v0"

    # Training settings
    offline_steps=1000000
    online_steps=0
    eval_interval=100000
    save_interval=$offline_steps

    # Hyperparameters
    flow_steps=10
    batch_size=256
    layer_width=256 # D4RL: 256
    discount=0.99
    tau=0.005
    inv_temp=3
    q_agg="mean"

    # Seeds to run
    if [ "$DEBUG" = "true" ]; then
        seeds="0"
        echo "DEBUG MODE: Using only seed 0"
    else
        seeds="0 1 2 3 4 5 6 7"
    fi

    for seed in $seeds; do
        run_group=${env_name}
        base_dir="fawac-reproduce"

        if [ "$offline_steps" = "2000000" ]; then
            base_dir="${base_dir}_scale"
        fi

        echo "=========================================="
        echo "FAWAC"
        echo "=========================================="
        echo "Task: ${task_idx}"
        echo "Seed: ${seed}"
        echo "Environment: ${env_name}"
        echo "=========================================="

        python main.py \
            --seed=$seed \
            --eval_interval=$eval_interval \
            --offline_steps=$offline_steps \
            --online_steps=$online_steps \
            --save_interval=$save_interval \
            --env_name=$env_name \
            --save_dir=$base_dir \
            --run_group=$run_group \
            --eval_init_noise=false \
            --agent=agents/fawac.py \
            --agent.flow_steps=$flow_steps \
            --agent.batch_size=$batch_size \
            --agent.value_hidden_dims="${layer_width},${layer_width},${layer_width},${layer_width}" \
            --agent.actor_hidden_dims="${layer_width},${layer_width},${layer_width},${layer_width}" \
            --agent.discount=$discount \
            --agent.tau=$tau \
            --agent.inv_temp=$inv_temp \
            --agent.q_agg=$q_agg
    done
done
