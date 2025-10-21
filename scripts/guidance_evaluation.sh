#!/bin/bash

# Guidance Evaluation Script
# This script runs guidance evaluation with multiple coefficients

# Set default values
# ENV_NAME="humanoidmaze-medium-navigate-singletask-task3-v0"
ENV_NAME="antmaze-large-navigate-singletask-task1-v0"
AGENT_NAME="dql"
RESTORE_EPOCH="1000000"
LYAPUNOV_HIDDEN_DIMS="512,512,512"
LYAPUNOV_LATENT_DIM=16
LYAPUNOV_LAYER_NORM=false
LYAPUNOV_MODEL_STEP="1000000"
EVAL_EPISODES=20
SEED=0

# lyapunov models
lyp_antmaze_task1="/mnt/nas/jaehyeok/fql/exp/lyapunov/LyapunovTraining/lyapunov_antmaze-large-navigate-singletask-task1-v0_sd000_20250916_202904"
lyp_antmaze_task2="/mnt/nas/jaehyeok/fql/exp/lyapunov/LyapunovTraining/lyapunov_antmaze-large-navigate-singletask-task2-v0_sd000_20250919_172659"
lyp_antmaze_task3="/mnt/nas/jaehyeok/fql/exp/lyapunov/LyapunovTraining/lyapunov_antmaze-large-navigate-singletask-task3-v0_sd000_20250919_172726"

lyp_antmaze_task2_noisestd_2="/mnt/nas/jaehyeok/fql/exp/lyapunov/LyapunovTraining/lyapunov_antmaze-large-navigate-singletask-task2-v0_sd000_20250919_183640"
lyp_antmaze_task2_noisestd_05="/mnt/nas/jaehyeok/fql/exp/lyapunov/LyapunovTraining/lyapunov_antmaze-large-navigate-singletask-task2-v0_sd000_20250929_213435"
lyp_antmaze_task3_noisestd_3="/mnt/nas/jaehyeok/fql/exp/lyapunov/LyapunovTraining/lyapunov_antmaze-large-navigate-singletask-task3-v0_sd000_20250919_183704"
lyp_antmaze_task3_noisestd_05="/mnt/nas/jaehyeok/fql/exp/lyapunov/LyapunovTraining/lyapunov_antmaze-large-navigate-singletask-task3-v0_sd000_20250929_210522"
lyp_antmaze_task3_noisestd_01="/mnt/nas/jaehyeok/fql/exp/lyapunov/LyapunovTraining/lyapunov_antmaze-large-navigate-singletask-task3-v0_sd000_20250929_210736"

lyp_humanoid_task1="/mnt/nas/jaehyeok/fql/exp/lyapunov/LyapunovTraining/lyapunov_humanoidmaze-medium-navigate-singletask-task1-v0_sd000_20250916_200152"
lyp_humanoid_task2="/mnt/nas/jaehyeok/fql/exp/lyapunov/LyapunovTraining/lyapunov_humanoidmaze-medium-navigate-singletask-task2-v0_sd000_20250919_175109"
lyp_humanoid_task3="/mnt/nas/jaehyeok/fql/exp/lyapunov/LyapunovTraining/lyapunov_humanoidmaze-medium-navigate-singletask-task3-v0_sd000_20250919_175123"

# antmaze large
fbrac_antmaze_task1="/mnt/nas/jaehyeok/fql/exp/fql/Debug/fbrac_antmaze-large-navigate-singletask-task1-v0_sd002_20250908_200922_utd-ratio1"
fbrac_antmaze_task2="/mnt/nas/jaehyeok/fql/exp/fql/Debug/fbrac_antmaze-large-navigate-singletask-task2-v0_sd000_20250919_172341_utd-ratio1"
fbrac_antmaze_task3="/mnt/nas/jaehyeok/fql/exp/fql/Debug/fbrac_antmaze-large-navigate-singletask-task3-v0_sd000_20250919_172433_utd-ratio1"

fql_antmaze_task1="/mnt/nas/jaehyeok/fql/exp/fql/Debug/"
fql_antmaze_task2="/mnt/nas/jaehyeok/fql/exp/fql/Debug/fql_antmaze-large-navigate-singletask-task2-v0_sd000_20250919_192530_utd-ratio1"

ifql_antmaze_task1="/mnt/nas/jaehyeok/fql/exp/fql/Debug/ifql_antmaze-large-navigate-singletask-task1-v0_sd000_20251002_114231_utd-ratio1"

dql_antmaze_task1="/mnt/nas/jaehyeok/fql/exp/fql/Debug/dql_antmaze-large-navigate-singletask-task1-v0_sd000_20251014_104305_utd-ratio1"

# humanoid large
fbrac_humanoid_task1="/mnt/nas/jaehyeok/fql/exp/fql/Debug/fbrac_humanoidmaze-medium-navigate-singletask-task1-v0_sd000_20250918_191439_utd-ratio1"
fbrac_humanoid_task2="/mnt/nas/jaehyeok/fql/exp/fql/Debug/fbrac_humanoidmaze-medium-navigate-singletask-task2-v0_sd000_20250919_085828_utd-ratio1"
fbrac_humanoid_task3="/mnt/nas/jaehyeok/fql/exp/fql/Debug/fbrac_humanoidmaze-medium-navigate-singletask-task3-v0_sd000_20250919_085910_utd-ratio1"

fql_humanoid_task3="/mnt/nas/jaehyeok/fql/exp/fql/Debug/fql_humanoidmaze-medium-navigate-singletask-task3-v0_sd000_20250919_192658_utd-ratio1"

RESTORE_PATH=$dql_antmaze_task1
LYAPUNOV_MODEL_PATH=$lyp_antmaze_task1

# Guidance coefficients to test (can be modified)
GUIDANCE_COEFFS="0.0, 0.001, 0.01, 0.1, 0.5, 1.0"
PARTIAL_GUIDANCE=3

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --env_name)
            ENV_NAME="$2"
            shift 2
            ;;
        --agent_name)
            AGENT_NAME="$2"
            shift 2
            ;;
        --restore_path)
            RESTORE_PATH="$2"
            shift 2
            ;;
        --restore_epoch)
            RESTORE_EPOCH="$2"
            shift 2
            ;;
        --lyapunov_model_path)
            LYAPUNOV_MODEL_PATH="$2"
            shift 2
            ;;
        --lyapunov_model_step)
            LYAPUNOV_MODEL_STEP="$2"
            shift 2
            ;;
        --lyapunov_hidden_dims)
            LYAPUNOV_HIDDEN_DIMS="$2"
            shift 2
            ;;
        --lyapunov_latent_dim)
            LYAPUNOV_LATENT_DIM="$2"
            shift 2
            ;;
        --lyapunov_layer_norm)
            LYAPUNOV_LAYER_NORM="$2"
            shift 2
            ;;
        --eval_episodes)
            EVAL_EPISODES="$2"
            shift 2
            ;;
        --guidance_coeffs)
            GUIDANCE_COEFFS="$2"
            shift 2
            ;;
        --partial_guidance)
            PARTIAL_GUIDANCE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --env_name ENV_NAME              Environment name (default: $ENV_NAME)"
            echo "  --agent_name AGENT_NAME          Agent name (default: $AGENT_NAME)"
            echo "  --restore_path PATH              Path to restore agent from"
            echo "  --restore_epoch EPOCH            Epoch to restore"
            echo "  --lyapunov_model_path PATH       Path to Lyapunov model directory"
            echo "  --lyapunov_model_step STEP       Lyapunov model step to load"
            echo "  --eval_episodes N                Number of evaluation episodes (default: $EVAL_EPISODES)"
            echo "  --guidance_coeffs COEFFS         Comma-separated guidance coefficients (default: $GUIDANCE_COEFFS)"
            echo "  --partial_guidance N             Partial guidance steps (default: $PARTIAL_GUIDANCE)"
            echo "  --seed SEED                      Random seed (default: $SEED)"
            echo "  -h, --help                       Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --env_name humanoidmaze-medium-navigate-singletask-task1-v0 \\"
            echo "     --restore_path /path/to/agent \\"
            echo "     --restore_epoch 1000000 \\"
            echo "     --lyapunov_model_path /path/to/lyapunov \\"
            echo "     --lyapunov_model_step 1000000 \\"
            echo "     --guidance_coeffs 0.0,0.001,0.01 \\"
            echo "     --partial_guidance 10"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$RESTORE_PATH" ]; then
    echo "Error: --restore_path is required"
    exit 1
fi

if [ -z "$LYAPUNOV_MODEL_PATH" ]; then
    echo "Error: --lyapunov_model_path is required"
    exit 1
fi

if [ -z "$LYAPUNOV_MODEL_STEP" ]; then
    echo "Error: --lyapunov_model_step is required"
    exit 1
fi

# Print configuration
echo "=========================================="
echo "GUIDANCE EVALUATION CONFIGURATION"
echo "=========================================="
echo "Environment: $ENV_NAME"
echo "Agent: $AGENT_NAME"
echo "Restore Path: $RESTORE_PATH"
echo "Restore Epoch: $RESTORE_EPOCH"
echo "Lyapunov Model Path: $LYAPUNOV_MODEL_PATH"
echo "Lyapunov Model Step: $LYAPUNOV_MODEL_STEP"
echo "Evaluation Episodes: $EVAL_EPISODES"
echo "Guidance Coefficients: $GUIDANCE_COEFFS"
echo "Partial Guidance: $PARTIAL_GUIDANCE"
echo "Seed: $SEED"
echo "=========================================="

# Create output directory
OUTPUT_DIR="guidance_eval_results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Run the evaluation
echo "Starting guidance evaluation..."
echo "Results will be saved to: $OUTPUT_DIR"

python guidance_evaluation.py \
    --env_name="$ENV_NAME" \
    --agent_config="agents/${AGENT_NAME}.py" \
    --restore_path="$RESTORE_PATH" \
    --restore_epoch="$RESTORE_EPOCH" \
    --lyapunov_model_path="$LYAPUNOV_MODEL_PATH" \
    --lyapunov_model_step="$LYAPUNOV_MODEL_STEP" \
    --lyapunov_hidden_dims="$LYAPUNOV_HIDDEN_DIMS" \
    --lyapunov_latent_dim="$LYAPUNOV_LATENT_DIM" \
    --lyapunov_layer_norm="$LYAPUNOV_LAYER_NORM" \
    --eval_episodes="$EVAL_EPISODES" \
    --guidance_coeffs="$GUIDANCE_COEFFS" \
    --partial_guidance="$PARTIAL_GUIDANCE" \
    --seed="$SEED" \
    2>&1 | tee "$OUTPUT_DIR/evaluation.log"

# Check if the evaluation was successful
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "EVALUATION COMPLETED SUCCESSFULLY"
    echo "=========================================="
    echo "Results saved to: $OUTPUT_DIR"
    echo "Log file: $OUTPUT_DIR/evaluation.log"
else
    echo ""
    echo "=========================================="
    echo "EVALUATION FAILED"
    echo "=========================================="
    echo "Check the log file: $OUTPUT_DIR/evaluation.log"
    exit 1
fi
