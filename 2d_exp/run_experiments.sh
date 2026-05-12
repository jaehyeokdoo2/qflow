#!/usr/bin/env bash

# ── Configuration ─────────────────────────────────────────────────────────────
DATASET="swissroll"           # two_spirals | moons | swissroll | eight_gaussians
N_DATA=10000
ALPHAS="0.3"     # space-separated list passed to --alphas

BC_EPOCHS=2000
OFFLINE_EPOCHS=100
LR=3e-4
BATCH_SIZE=4096

FLOW_STEPS=25
NUM_LAYERS=4
HIDDEN_DIM=512

NUM_EVALS=512
NUM_PLOTS=0               # set >0 to save mid-training snapshots
SAVE_DIR="toy_experiment"
SEED=0

# ── Script directory ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── FQL ───────────────────────────────────────────────────────────────────────
# echo "=================n========================="
# echo " Running FQL"
# echo "=========================================="
# python "${SCRIPT_DIR}/run_fql.py" \
#     --dataset        "${DATASET}" \
#     --n_data         "${N_DATA}" \
#     --alphas         ${ALPHAS} \
#     --bc_epochs      "${BC_EPOCHS}" \
#     --offline_epochs "${OFFLINE_EPOCHS}" \
#     --lr             "${LR}" \
#     --batch_size     "${BATCH_SIZE}" \
#     --flow_steps     "${FLOW_STEPS}" \
#     --num_layers     "${NUM_LAYERS}" \
#     --hidden_dim     "${HIDDEN_DIM}" \
#     --num_evals      "${NUM_EVALS}" \
#     --num_plots      "${NUM_PLOTS}" \
#     --save_dir       "${SAVE_DIR}" \
#     --seed           "${SEED}"

# # ── FBRAC ─────────────────────────────────────────────────────────────────────
# echo ""
# echo "=========================================="
# echo " Running FBRAC"
# echo "=========================================="
# python "${SCRIPT_DIR}/run_fbrac.py" \
#     --dataset        "${DATASET}" \
#     --n_data         "${N_DATA}" \
#     --alphas         ${ALPHAS} \
#     --bc_epochs      "${BC_EPOCHS}" \
#     --offline_epochs "${OFFLINE_EPOCHS}" \
#     --lr             "${LR}" \
#     --batch_size     "${BATCH_SIZE}" \
#     --flow_steps     "${FLOW_STEPS}" \
#     --num_layers     "${NUM_LAYERS}" \
#     --hidden_dim     "${HIDDEN_DIM}" \
#     --num_evals      "${NUM_EVALS}" \
#     --num_plots      "${NUM_PLOTS}" \
#     --save_dir       "${SAVE_DIR}" \
#     --seed           "${SEED}"

# ── QFlow ─────────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo " Running QFlow"
echo "=========================================="
python "${SCRIPT_DIR}/run_qflow.py" \
    --dataset        "${DATASET}" \
    --n_data         "${N_DATA}" \
    --alphas         ${ALPHAS} \
    --bc_epochs      "${BC_EPOCHS}" \
    --offline_epochs "${OFFLINE_EPOCHS}" \
    --lr             "${LR}" \
    --batch_size     "${BATCH_SIZE}" \
    --flow_steps     "${FLOW_STEPS}" \
    --num_layers     "${NUM_LAYERS}" \
    --hidden_dim     "${HIDDEN_DIM}" \
    --num_evals      "${NUM_EVALS}" \
    --num_plots      "${NUM_PLOTS}" \
    --save_dir       "${SAVE_DIR}" \
    --seed           "${SEED}"

echo ""
echo "=========================================="
echo " Done. Results saved to ${SAVE_DIR}/${DATASET}/"
echo "=========================================="
