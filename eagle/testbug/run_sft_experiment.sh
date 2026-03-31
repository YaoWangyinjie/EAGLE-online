#!/usr/bin/env bash
# =============================================================================
# run_sft_experiment.sh
#
# End-to-end runner for the EAGLE3 draft-model SFT overfit experiment.
#
# Workflow
# --------
#   Step 1 (gen)   : Run target model on benchmark questions → sft training data
#   Step 2 (train) : Fine-tune EAGLE3 draft model on that data
#   Step 3 (eval)  : Evaluate baseline vs SFT model and print comparison report
#
# Each step can be run independently via the STEP variable or the --step flag.
#
# Usage examples
# --------------
#   # Full pipeline:
#   bash run_sft_experiment.sh
#
#   # Only generate data:
#   bash run_sft_experiment.sh --step gen
#
#   # Only run training (assumes data already generated):
#   bash run_sft_experiment.sh --step train
#
#   # Only run evaluation (assumes checkpoint exists):
#   bash run_sft_experiment.sh --step eval
#
#   # Override the number of SFT epochs:
#   SFT_EPOCHS=500 bash run_sft_experiment.sh --step train
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Paths  (edit these to match your environment)
# ---------------------------------------------------------------------------
BASE_MODEL_PATH="/root/paddlejob/workspace/env_run/ea/model_weight/DeepSeek-R1-Distill-Llama-8B"
EAGLE_MODEL_PATH="/root/paddlejob/workspace/env_run/ea/model_weight/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EAGLE_DATA_DIR="$(dirname "${SCRIPT_DIR}")/data"      # EAGLE/eagle/data
QUESTION_FILE="${EAGLE_DATA_DIR}/mt_bench/question.jsonl"

SFT_DATA_FILE="${SCRIPT_DIR}/data/mt_bench_sft.jsonl"
CHECKPOINT_DIR="${SCRIPT_DIR}/checkpoints"
EVAL_DIR="${SCRIPT_DIR}/eval_results"

# ---------------------------------------------------------------------------
# Hyper-parameters  (override via environment variables)
# ---------------------------------------------------------------------------
# Data generation
GEN_TEMPERATURE="${GEN_TEMPERATURE:-0.6}"       # temperature for target-model sampling
GEN_SAMPLES_PER_Q="${GEN_SAMPLES_PER_Q:-1}"     # samples per question; increase for more data
GEN_MAX_NEW_TOKENS="${GEN_MAX_NEW_TOKENS:-1024}"

# SFT training
SFT_EPOCHS="${SFT_EPOCHS:-200}"
SFT_LR="${SFT_LR:-3e-5}"
SFT_BATCH="${SFT_BATCH:-1}"
SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-2}"
SFT_MAX_LEN="${SFT_MAX_LEN:-2048}"
SFT_SAVE_EVERY="${SFT_SAVE_EVERY:-10}"
# Set SFT_USE_DEEPSPEED=1 to enable multi-GPU DeepSpeed training
SFT_USE_DEEPSPEED="${SFT_USE_DEEPSPEED:-0}"
SFT_NUM_GPUS="${SFT_NUM_GPUS:-1}"

# Evaluation
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.0}"
EVAL_TOTAL_TOKEN="${EVAL_TOTAL_TOKEN:-60}"
EVAL_DEPTH="${EVAL_DEPTH:-5}"
EVAL_TOP_K="${EVAL_TOP_K:-10}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-1024}"

# ---------------------------------------------------------------------------
# Parse --step argument
# ---------------------------------------------------------------------------
STEP="${STEP:-all}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --step) STEP="$2"; shift 2 ;;
        *)      echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo " EAGLE3 SFT Overfit Experiment"
echo " Step(s) to run : ${STEP}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Utility: locate the latest checkpoint directory produced by sft_main.py
# ---------------------------------------------------------------------------
find_latest_checkpoint() {
    local ckpt_dir="$1"
    # Looks for epoch_N (plain) or state_N (deepspeed) directories
    local latest=""
    local max_n=-1
    for d in "${ckpt_dir}"/epoch_* "${ckpt_dir}"/state_*; do
        [ -d "$d" ] || continue
        n=$(basename "$d" | grep -oE '[0-9]+$')
        if [ -n "$n" ] && [ "$n" -gt "$max_n" ]; then
            max_n=$n
            latest=$d
        fi
    done
    echo "${latest}"
}

# ===========================================================================
# Step 1 – Data generation
# ===========================================================================
run_gen() {
    echo ""
    echo "------------------------------------------------------------"
    echo " Step 1: Generating SFT data with target model"
    echo "------------------------------------------------------------"

    mkdir -p "$(dirname "${SFT_DATA_FILE}")"

    python "${SCRIPT_DIR}/gen_sft_data.py" \
        --model_path            "${BASE_MODEL_PATH}" \
        --question_file         "${QUESTION_FILE}" \
        --output_file           "${SFT_DATA_FILE}" \
        --num_samples_per_question "${GEN_SAMPLES_PER_Q}" \
        --max_new_tokens        "${GEN_MAX_NEW_TOKENS}" \
        --temperature           "${GEN_TEMPERATURE}" \
        --dtype                 float16

    echo " Data generation complete → ${SFT_DATA_FILE}"
}

# ===========================================================================
# Step 2 – SFT Training
# ===========================================================================
run_train() {
    echo ""
    echo "------------------------------------------------------------"
    echo " Step 2: SFT fine-tuning of EAGLE3 draft model"
    echo "------------------------------------------------------------"

    if [ ! -f "${SFT_DATA_FILE}" ]; then
        echo "ERROR: SFT data file not found: ${SFT_DATA_FILE}"
        echo "       Run with --step gen first."
        exit 1
    fi

    mkdir -p "${CHECKPOINT_DIR}"

    COMMON_ARGS=(
        --basepath      "${BASE_MODEL_PATH}"
        --eaglepath     "${EAGLE_MODEL_PATH}"
        --trainpath     "${SFT_DATA_FILE}"
        --savedir       "${CHECKPOINT_DIR}"
        --num_epochs    "${SFT_EPOCHS}"
        --lr            "${SFT_LR}"
        --batch_size    "${SFT_BATCH}"
        --grad_accum    "${SFT_GRAD_ACCUM}"
        --max_len       "${SFT_MAX_LEN}"
        --save_every    "${SFT_SAVE_EVERY}"
    )

    if [ "${SFT_USE_DEEPSPEED}" = "1" ]; then
        echo "  → Using DeepSpeed with ${SFT_NUM_GPUS} GPU(s)"
        deepspeed --num_gpus "${SFT_NUM_GPUS}" \
            "${SCRIPT_DIR}/sft_main.py" \
            --deepspeed_config "${SCRIPT_DIR}/sft_ds_config.json" \
            "${COMMON_ARGS[@]}"
    else
        echo "  → Using single-GPU plain PyTorch"
        python "${SCRIPT_DIR}/sft_main.py" "${COMMON_ARGS[@]}"
    fi

    echo " Training complete → checkpoints in ${CHECKPOINT_DIR}"
}

# ===========================================================================
# Step 3 – Evaluation
# ===========================================================================
run_eval() {
    echo ""
    echo "------------------------------------------------------------"
    echo " Step 3: Evaluating baseline vs SFT draft model"
    echo "------------------------------------------------------------"

    SFT_CKPT="$(find_latest_checkpoint "${CHECKPOINT_DIR}")"
    if [ -z "${SFT_CKPT}" ]; then
        echo "ERROR: No checkpoint found in ${CHECKPOINT_DIR}"
        echo "       Run with --step train first."
        exit 1
    fi
    echo "  → Using SFT checkpoint: ${SFT_CKPT}"

    mkdir -p "${EVAL_DIR}"

    python "${SCRIPT_DIR}/eval_sft.py" \
        --base_model_path   "${BASE_MODEL_PATH}" \
        --baseline_eagle    "${EAGLE_MODEL_PATH}" \
        --sft_eagle         "${SFT_CKPT}" \
        --question_file     "${QUESTION_FILE}" \
        --output_dir        "${EVAL_DIR}" \
        --temperature       "${EVAL_TEMPERATURE}" \
        --total_token       "${EVAL_TOTAL_TOKEN}" \
        --depth             "${EVAL_DEPTH}" \
        --top_k             "${EVAL_TOP_K}" \
        --max_new_tokens    "${EVAL_MAX_NEW_TOKENS}"

    echo " Evaluation complete → results in ${EVAL_DIR}"
}

# ===========================================================================
# Dispatch
# ===========================================================================
case "${STEP}" in
    all)
        run_gen
        run_train
        run_eval
        ;;
    gen)
        run_gen
        ;;
    train)
        run_train
        ;;
    eval)
        run_eval
        ;;
    *)
        echo "Unknown step '${STEP}'. Valid values: all | gen | train | eval"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo " Done."
echo "============================================================"
