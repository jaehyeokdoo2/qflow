#!/bin/bash

set -e

# Parse arguments
task_idx=${1:-1}

# Environment settings
env_name="antmaze-large-navigate-singletask-task${task_idx}-v0"
env_name="humanoidmaze-medium-navigate-singletask-task${task_idx}-v0"
# env_name="antsoccer-arena-navigate-singletask-task${task_idx}-v0"
# env_name="cube-double-play-singletask-task${task_idx}-v0"
# env_name="puzzle-4x4-play-singletask-task${task_idx}-v0"

# Training hyperparameters
offline_steps=1000000
eval_interval=100000
save_interval=$offline_steps
alpha=30
flow_steps=10
q_agg=mean
discount=0.995

# Evaluation settings
lyapunov_model_path="/mnt/nas/jaehyeok/fql/exp/lyapunov/LyapunovTraining/lyapunov_antmaze-large-navigate-singletask-task1-v0_sd000_20250916_202904"
lyapunov_model_path="/mnt/nas/jaehyeok/fql/exp/lyapunov/LyapunovTraining/lyapunov_humanoidmaze-medium-navigate-singletask-task1-v0_sd000_20250916_200152"
lyapunov_model_step="1000000"
guidance_coeffs="0.0, 0.1, 0.5, 1.0, 1.5"
eval_episodes=50
save_summary=false
save_values=false

use_rejection_sampling=false


start_seed=0
end_seed=0
for seed in $(seq $start_seed $end_seed); do

    # Build experiment name
    exp_name="${env_name}_alpha${alpha}-flow${flow_steps}-qagg${q_agg}-discount${discount}_sd$(printf '%03d' $seed)"

    # Directory structure
    base_dir="exp"
    run_group=${env_name}
    # Final path will be: base_dir/fbrac/run_group/exp_name
    final_save_dir="${base_dir}/fbrac/${run_group}/${exp_name}"

    echo "=========================================="
    echo "FBRAC PIPELINE"
    echo "=========================================="
    echo "Task: ${task_idx}"
    echo "Seed: ${seed}"
    echo "Experiment: ${exp_name}"
    echo "Save dir: ${final_save_dir}"
    echo "=========================================="

    # Step 1: Training
    echo ""
    echo "Step 1: Training..."
    # python main.py \
    #     --seed=$seed \
    #     --eval_interval=$eval_interval \
    #     --offline_steps=$offline_steps \
    #     --save_interval=$save_interval \
    #     --env_name=$env_name \
    #     --save_dir=$base_dir \
    #     --run_group=$run_group \
    #     --exp_name=$exp_name \
    #     --agent=agents/fbrac.py \
    #     --agent.alpha=$alpha \
    #     --agent.flow_steps=$flow_steps \
    #     --agent.q_agg=$q_agg \
    #     --agent.discount=$discount \
    #     --agent.reward_scale=1 \
    #     --agent.normalize_q_loss=false \
    #     --agent.layer_norm=true

    echo ""
    echo "Training completed!"
    echo "Model saved to: ${final_save_dir}"
    echo ""

    # Step 2: Evaluation with guidance
    echo "Step 2: Evaluating with guidance..."
    eval_name="eval_${exp_name}_gc${coeff}"
    eval_dir="exp/evaluations/${eval_name}"
    
    echo ""
    echo "Evaluating with guidance coefficient: ${coeff}"
    echo "Results will be saved to: ${eval_dir}"
    
    mkdir -p "$eval_dir"

    if [ "$use_rejection_sampling" = true ]; then
        for select_quantile in 0.5 0.7 0.9 1.0; do
            python guidance_evaluation.py \
                --env_name="$env_name" \
                --agent_config="agents/fbrac.py" \
                --restore_path="$final_save_dir" \
                --restore_epoch="$save_interval" \
                --lyapunov_model_path="$lyapunov_model_path" \
                --lyapunov_model_step="$lyapunov_model_step" \
                --lyapunov_hidden_dims="512,512,512" \
                --lyapunov_latent_dim=16 \
                --lyapunov_layer_norm=false \
                --eval_episodes="$eval_episodes" \
                --guidance_coeffs="0.0" \
                --partial_guidance="0" \
                --seed="$seed" \
                --save_summary="$save_summary" \
                --save_values="$save_values" \
                --select_quantile="$select_quantile" \
                --use_rejection_sampling="$use_rejection_sampling" \
                2>&1 | tee "$eval_dir/evaluation.log"
        done
    else

        for partial_guidance in 1 3 5; do
            python guidance_evaluation.py \
                --env_name="$env_name" \
                --agent_config="agents/fbrac.py" \
                --restore_path="$final_save_dir" \
                --restore_epoch="$save_interval" \
                --lyapunov_model_path="$lyapunov_model_path" \
                --lyapunov_model_step="$lyapunov_model_step" \
                --lyapunov_hidden_dims="512,512,512" \
                --lyapunov_latent_dim=16 \
                --lyapunov_layer_norm=false \
                --eval_episodes="$eval_episodes" \
                --guidance_coeffs="$guidance_coeffs" \
                --partial_guidance="$partial_guidance" \
                --seed="$seed" \
                --save_summary="$save_summary" \
                --save_values="$save_values" \
                --use_rejection_sampling="$use_rejection_sampling" \
                2>&1 | tee "$eval_dir/evaluation.log"
        done
    fi

    echo ""
    echo "=========================================="
    echo "PIPELINE COMPLETED!"
    echo "=========================================="
    echo "Training saved to: ${final_save_dir}"
    echo "Evaluations saved to: exp/evaluations/eval_${exp_name}_gc*"
    echo "=========================================="
done