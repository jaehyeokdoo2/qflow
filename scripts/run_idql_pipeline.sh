#!/bin/bash

set -e

# Parse arguments
task_idx=${1:-1}

# Environment settings
env_name="antmaze-large-navigate-singletask-task${task_idx}-v0"
# env_name="humanoidmaze-medium-navigate-singletask-task${task_idx}-v0"
# env_name="antsoccer-arena-navigate-singletask-task${task_idx}-v0"

# Training hyperparameters
offline_steps=1000000
eval_interval=100000
save_interval=$offline_steps
expectile=0.9
num_samples=32
diffusion_steps=10
reward_discount=0.99
use_value_layer_norm=true
use_actor_layer_norm=false

save_summary=false
save_values=false

# Evaluation settings
lyapunov_model_path="/mnt/nas/jaehyeok/fql/exp/lyapunov/LyapunovTraining/lyapunov_antmaze-large-navigate-singletask-task1-v0_sd000_20250916_202904"
lyapunov_model_path="/mnt/nas/jaehyeok/fql/exp/lyapunov/LyapunovTraining/lyapunov_humanoidmaze-medium-navigate-singletask-task1-v0_sd000_20250916_200152"
lyapunov_model_step="1000000"
guidance_coeffs="0.1,0.5,1.0"
eval_episodes=50

start_seed=0
end_seed=0
for seed in $(seq $start_seed $end_seed); do

    # Build experiment name
    exp_name="${env_name}_expectile${expectile}-numsamples${num_samples}-diff${diffusion_steps}-discount${reward_discount}_sd$(printf '%03d' $seed)"

    # Directory structure
    base_dir="exp"
    run_group=${env_name}
    # Final path will be: base_dir/idql/run_group/exp_name
    final_save_dir="${base_dir}/idql/${run_group}/${exp_name}"

    echo "=========================================="
    echo "IDQL PIPELINE"
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
    #     --agent=agents/idql.py \
    #     --agent.expectile=$expectile \
    #     --agent.num_samples=$num_samples \
    #     --agent.diffusion_steps=$diffusion_steps \
    #     --agent.discount=$reward_discount \
    #     --agent.layer_norm=$use_value_layer_norm \
    #     --agent.actor_layer_norm=$use_actor_layer_norm

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
    for partial_guidance in 1 2 3 5; do
    
    python guidance_evaluation.py \
        --env_name="$env_name" \
        --agent_config="agents/idql.py" \
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
        2>&1 | tee "$eval_dir/evaluation.log"
    done

    echo ""
    echo "=========================================="
    echo "PIPELINE COMPLETED!"
    echo "=========================================="
    echo "Training saved to: ${final_save_dir}"
    echo "Evaluations saved to: exp/evaluations/eval_${exp_name}_gc*"
    echo "=========================================="
done



