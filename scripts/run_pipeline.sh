#!/usr/bin/env bash
# Full SeDeM pipeline for one config: Stage 1 -> Stage 2 -> selector -> evaluation (K sweep).
# usage: bash scripts/run_pipeline.sh <config>
#   SKIP_STAGE1 / SKIP_STAGE2 / SKIP_SELECTOR / SKIP_EVAL = 1 to skip a step (e.g. to resume).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
load_config "$1"
S="$(dirname "${BASH_SOURCE[0]}")"
[[ "${SKIP_STAGE1:-0}" == 1 ]]   || bash "$S/train_stage1.sh" "$1"
[[ "${SKIP_STAGE2:-0}" == 1 ]]   || bash "$S/train_stage2.sh" "$1"
[[ "${SKIP_SELECTOR:-0}" == 1 ]] || bash "$S/train_selector.sh" "$1"
if [[ "${SKIP_EVAL:-0}" != 1 ]]; then
  for K in ${K_SWEEP:-$TOP_K}; do bash "$S/evaluate.sh" "$1" "$K"; done
fi
echo "[sedem] pipeline finished: $RUN_DIR"
