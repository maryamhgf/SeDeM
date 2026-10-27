#!/usr/bin/env bash
# Joint Stage 2 (used for the cross-model HotpotQA cell): selector + compressor + decompressor + decoder LoRA
# trained together with an evidence-label curriculum for block selection. usage: bash scripts/train_stage2_joint.sh <config>
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
load_config "$1"; shift || true
arch_args_argparse
OUT="$RUN_DIR/stage2_joint"; mkdir -p "$OUT"
ARGS=(--dataset "${JOINT_DATASET:-hotpot}" --base_checkpoint "${JOINT_BASE_CHECKPOINT:-$(stage_ckpt stage1)}"
  --selector_checkpoint "${JOINT_SELECTOR_CHECKPOINT:-$RUN_DIR/selector/selector.safetensors}" --selector_type late_interaction
  --output_dir "$OUT" "${ARCH_ARGS_AP[@]}"
  --total_steps "${JOINT_TOTAL_STEPS:-6000}" --train_top_k "${JOINT_TRAIN_TOP_K:-4}"
  --label_warmup "${JOINT_LABEL_WARMUP:-1500}" --label_anneal "${JOINT_LABEL_ANNEAL:-3000}"
  --lambda_margin "${JOINT_LAMBDA_MARGIN:-0.5}" --margin "${JOINT_MARGIN:-4.0}" --temperature "${JOINT_TEMPERATURE:-1.0}"
  --seed "$SEED")
cd "$SEDEM_ROOT"
run "$PYTHON" train_stage2_joint.py "${ARGS[@]}" "$@"
