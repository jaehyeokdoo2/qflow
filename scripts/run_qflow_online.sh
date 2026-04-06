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

# Base directory where offline models were saved (must match base_dir in run_qflow.sh)
restore_base_dir="rebuttal"

for task_idx in $(seq $start_task_idx $end_task_idx); do
    echo "Running QFLOW ONLINE FINE-TUNING for task ${task_idx}"
    echo "=========================================="
    echo ""

    # Environment settings - must match run_qflow.sh
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
    env_name="antmaze-medium-play-v2"
    env_name="antmaze-medium-diverse-v2"
    env_name="antmaze-large-play-v2"
    # env_name="antmaze-large-diverse-v2"

    # Online fine-tuning settings
    online_steps=200000
    eval_interval=100000
    save_interval=5000000
    restore_epoch=1000000  # Must match offline_steps in run_qflow.sh

    # Hyperparameters - must match run_qflow.sh (override_params=True path)
    flow_steps=10
    batch_size=256
    num_ensembles=2
    layer_width=256
    eval_init_noise=false
    time_embed_dim=16
    alpha=0.1
    discount=0.99
    q_agg="mean"
    actor_loss_type="grad"  # Must match qflow agent default used during offline training

    # Seeds to run
    if [ "$DEBUG" = "true" ]; then
        seeds="0"
        echo "DEBUG MODE: Using only seed 0"
    else
        seeds="4 5 6 7"
    fi

    for seed in $seeds; do
        run_group=${env_name}

        # Reconstruct offline exp_name to locate the restore path.
        # Matches main.py qflow exp_name generation logic:
        #   env_short = env_name with '-singletask-task' -> '-t', '-singletask' removed, '-v0' removed
        #   exp_name = "qflow-{env_short}-a{alpha}-qagg{q_agg}-te{time_embed_dim}-al{actor_loss}-sd{seed}"
        env_short="${env_name//-singletask-task/-t}"
        env_short="${env_short//-singletask/}"
        env_short="${env_short//-v0/}"
        offline_exp_name="qflow-${env_short}-a${alpha}-qagg${q_agg}-te${time_embed_dim}-al${actor_loss_type}-sd$(printf '%d' $seed)"
        restore_path="${restore_base_dir}/qflow/${run_group}/${offline_exp_name}"

        echo "=========================================="
        echo "QFLOW ONLINE FINE-TUNING"
        echo "=========================================="
        echo "Task: ${task_idx}"
        echo "Seed: ${seed}"
        echo "Environment: ${env_name}"
        echo "Experiment: ${offline_exp_name}"
        echo "Restore path: ${restore_path}"
        echo "Restore epoch: ${restore_epoch}"
        echo "=========================================="

        if [ ! -d "${restore_path}" ]; then
            echo "ERROR: Restore path does not exist: ${restore_path}"
            echo "Please check that the offline model was trained and saved correctly."
            exit 1
        fi

        python online_finetuning.py \
            --seed=$seed \
            --eval_interval=$eval_interval \
            --online_steps=$online_steps \
            --save_interval=$save_interval \
            --env_name=$env_name \
            --save_dir="online_finetuning" \
            --run_group=$run_group \
            --exp_name="${offline_exp_name}_online" \
            --restore_path=$restore_path \
            --restore_epoch=$restore_epoch \
            --eval_init_noise=$eval_init_noise \
            --agent=agents/qflow.py \
            --agent.flow_steps=$flow_steps \
            --agent.num_ensembles=$num_ensembles \
            --agent.batch_size=$batch_size \
            --agent.value_hidden_dims="${layer_width},${layer_width},${layer_width},${layer_width}" \
            --agent.actor_hidden_dims="${layer_width},${layer_width},${layer_width},${layer_width}" \
            --agent.normalize_q_loss=false \
            --agent.use_time_embed=true \
            --agent.time_embed_dim=$time_embed_dim \
            --agent.discount=$discount \
            --agent.alpha=$alpha \
            --agent.q_agg=$q_agg \
            --agent.actor_loss_type=$actor_loss_type
    done
done
