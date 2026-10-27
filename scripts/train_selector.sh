#!/usr/bin/env bash
# Selector training: late-interaction top-K selector on a frozen Stage-2 (or Stage-1) checkpoint. Paper Sec. 3.2 / App. A.3.
# usage: bash scripts/train_selector.sh <config> [extra args]
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
load_config "$1"; shift || true
arch_args_argparse
OUT="$RUN_DIR/selector"; mkdir -p "$OUT"
BASE="${SELECTOR_BASE_CHECKPOINT:-$(stage_ckpt "$SELECTOR_BASE")}"
ARGS=(--selector_type late_interaction --dataset "$SELECTOR_DATASET" --base_checkpoint "$BASE" --output_dir "$OUT"
  "${ARCH_ARGS_AP[@]}" --compressor_type "$COMPRESSOR_TYPE" --compressor_no_global_context True
  --compressor_slot_position_embedding False --compressor_linear_projection "$LINEAR_PROJECTION"
  --num_epochs "$SELECTOR_EPOCHS" --lr "$SELECTOR_LR" --temperature "$SELECTOR_TEMPERATURE"
  --lambda_margin "$SELECTOR_LAMBDA_MARGIN" --margin "$SELECTOR_MARGIN" --selector_heads "$SELECTOR_HEADS"
  --max_val_questions "$SELECTOR_MAX_VAL_QUESTIONS" --log_every 50 --eval_every "$SELECTOR_EVAL_EVERY" --seed "$SEED")
[[ -n "$SELECTOR_MAX_TRAIN_QUESTIONS" ]] && ARGS+=(--max_train_questions "$SELECTOR_MAX_TRAIN_QUESTIONS")
cd "$SEDEM_ROOT"
run "$PYTHON" train_selector.py "${ARGS[@]}" "$@"
