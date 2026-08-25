#!/usr/bin/env bash
set -euo pipefail

export SETTING=fixed
export TOKEN_BUDGET=0.10
export TASKS=realworldqa,mmstar,chartqa
export OUTPUT_PATH="${OUTPUT_PATH:-outputs/paper/fixed_10_percent}"

exec "$(dirname "$0")/evaluate.sh"
