#!/bin/bash

set -e

start_task_idx=${1:-1}
end_task_idx=${2:-1}

# Parse arguments
for task_idx in $(seq $start_task_idx $end_task_idx); do
    echo "Running INIT NOISE EVALUATION for task ${task_idx}"
    echo "=========================================="
    echo ""

    # Environment settings (should match training config in run_fbrac_vd.sh)
    # env_name="cube-double-play-singletask-task${task_idx}-v0"
    # env_name="puzzle-3x3-play-singletask-task${task_idx}-v0"
    # env_name="humanoidmaze-medium-navigate-singletask-task${task_idx}-v0"
    env_name="antsoccer-arena-navigate-singletask-task${task_idx}-v0"
    # env_name="humanoidmaze-large-navigate-singletask-task${task_idx}-v0"

    # env_name="antmaze-large-navigate-singletask-task${task_idx}-v0"
    # env_name="antmaze-giant-navigate-singletask-task${task_idx}-v0"
    # env_name="puzzle-3x3-play-singletask-task${task_idx}-v0"
    env_name="cube-double-play-singletask-task${task_idx}-v0"

    # Training hyperparameters (must match the trained model)
    alpha=20
    flow_steps=10
    q_agg=mean
    discount=0.99

    # time ablation
    use_time_embed=true
    time_embed_dim=64
    use_time_penalty=false
    max_time_penalty=0.0

    if [ "$use_time_embed" = true ]; then
        time_embed_str="time_embed${time_embed_dim}"
    else
        time_embed_str=""
    fi

    if [ "$use_time_penalty" = true ]; then
        time_penalty_str="time_penalty${max_time_penalty}"
    else
        time_penalty_str=""
    fi

    # Evaluation settings
    eval_episodes=50
    n_samples_list="1,16"
    temperature=0.3  # Temperature for softmax selection (lower = more greedy)

    # Base directory structure (must match training)
    if [ "$q_agg" = min ]; then
        base_dir="grad_align_clipped"
    else
        base_dir="regression_ablation"
    fi

    start_seed=0
    end_seed=0
    for seed in $(seq $start_seed $end_seed); do

        # Build experiment name (must match training)
        exp_name="${env_name}_alpha${alpha}-flow${flow_steps}-qagg${q_agg}-discount${discount}_${time_embed_str}_${time_penalty_str}_sd$(printf '%03d' $seed)"

        run_group=${env_name}
        # Final path: base_dir/fbrac_vd/run_group/exp_name
        restore_path="${base_dir}/fbrac_vd/${run_group}/${exp_name}"

        echo "=========================================="
        echo "INIT NOISE EVALUATION"
        echo "=========================================="
        echo "Task: ${task_idx}"
        echo "Seed: ${seed}"
        echo "Experiment: ${exp_name}"
        echo "Restore path: ${restore_path}"
        echo "Samples: ${n_samples_list}"
        echo "Temperature: ${temperature}"
        echo "=========================================="

        # Check if model exists
        if [ ! -d "${restore_path}" ]; then
            echo "Warning: Model directory not found: ${restore_path}"
            echo "Skipping..."
            continue
        fi

        echo ""
        echo "Running init noise evaluation..."
        python init_noise_evaluation.py \
            --seed=$seed \
            --env_name=$env_name \
            --restore_path=$restore_path \
            --restore_epoch=1000000 \
            --agent=agents/fbrac_vd.py \
            --eval_episodes=$eval_episodes \
            --n_samples_list=$n_samples_list \
            --temperature=$temperature \
            --save_summary=true \
            --agent.alpha=$alpha \
            --agent.flow_steps=10 \
            --agent.q_agg=$q_agg \
            --agent.discount=$discount \
            --agent.value_hidden_dims="256,256,256,256" \
            --agent.use_time_embed=$use_time_embed \
            --agent.time_embed_dim=$time_embed_dim \
            --agent.use_time_penalty=$use_time_penalty \
            --agent.max_time_penalty=$max_time_penalty

        echo ""
        echo "Init noise evaluation completed!"
        echo "Results saved to: ${restore_path}/eval_results/"
        echo ""
    done
done

