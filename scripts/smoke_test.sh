#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${QWEN_MODEL_PATH:-}" ]]; then
  echo "Set QWEN_MODEL_PATH to Qwen2.5-VL-7B-Instruct." >&2
  exit 2
fi

export LMMS_EVAL_PLUGINS=pace_vlm
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

python -m lmms_eval \
  --model pace_qwen2_5_vl \
  --model_args "pretrained=${QWEN_MODEL_PATH},min_pixels=3136,max_pixels=1605632,input_min_pixels=1605632,input_max_pixels=1605632,token_budget=0.10,extraction_layer=2,fusion_temperature=0.5,attn_implementation=flash_attention_2" \
  --tasks realworldqa \
  --batch_size 1 \
  --limit 4
