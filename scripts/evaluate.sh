#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${QWEN_MODEL_PATH:-}" ]]; then
  echo "Set QWEN_MODEL_PATH to Qwen2.5-VL-7B-Instruct." >&2
  exit 2
fi

SETTING="${SETTING:-fixed}"
TASKS="${TASKS:-realworldqa,mmstar,chartqa}"
TOKEN_BUDGET="${TOKEN_BUDGET:-0.10}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/${SETTING}/budget_${TOKEN_BUDGET}}"

case "${SETTING}" in
  fixed)
    INPUT_MIN_PIXELS=$((2048 * 28 * 28))
    INPUT_MAX_PIXELS=$((2048 * 28 * 28))
    ;;
  dynamic)
    INPUT_MIN_PIXELS=$((256 * 28 * 28))
    INPUT_MAX_PIXELS=$((2048 * 28 * 28))
    ;;
  *)
    echo "SETTING must be fixed or dynamic, got ${SETTING}." >&2
    exit 2
    ;;
esac

export LMMS_EVAL_PLUGINS=pace_vlm
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

MODEL_ARGS="pretrained=${QWEN_MODEL_PATH},min_pixels=3136,max_pixels=1605632"
MODEL_ARGS+=",input_min_pixels=${INPUT_MIN_PIXELS},input_max_pixels=${INPUT_MAX_PIXELS}"
MODEL_ARGS+=",token_budget=${TOKEN_BUDGET},extraction_layer=2,fusion_temperature=0.5"
MODEL_ARGS+=",apc_preview_depth=1,apc_global_weight=0.6,apc_detail_fraction=0.1"
MODEL_ARGS+=",apc_detail_scale=1.5,attn_implementation=flash_attention_2"

accelerate launch --num_processes "${NUM_PROCESSES}" -m lmms_eval \
  --model pace_qwen2_5_vl \
  --model_args "${MODEL_ARGS}" \
  --tasks "${TASKS}" \
  --batch_size 1 \
  --log_samples \
  --output_path "${OUTPUT_PATH}"
