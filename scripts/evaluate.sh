#!/usr/bin/env bash
# Evaluate a trained run. EVAL_HARNESS=multihop (HotpotQA/2Wiki/MuSiQue paragraph blocks, eval_multihop.py)
# or EVAL_HARNESS=qa (QASPER / RULER / any HF dataset with a registered transform, eval_qa.py).
# usage: bash scripts/evaluate.sh <config> [K] [extra args]     (K overrides TOP_K)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
load_config "$1"; shift || true
K="${TOP_K}"; if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then K="$1"; shift; fi
OUT="$RUN_DIR/eval"; mkdir -p "$OUT"
BASE="${EVAL_BASE_CHECKPOINT:-$(stage_ckpt stage2)}"
SEL="${EVAL_SELECTOR_CHECKPOINT:-$RUN_DIR/selector/selector.safetensors}"
cd "$SEDEM_ROOT"
if [[ "$EVAL_HARNESS" == "multihop" ]]; then
  arch_args_argparse
  ARGS=(--base_checkpoint "$BASE" --selector_type late_interaction --selector_heads "$SELECTOR_HEADS"
    --mode "$EVAL_SELECTION_MODE" --dataset "$EVAL_DATASET" --top_k "$K" --max_new_tokens "$MAX_NEW_TOKENS"
    --output_file "$OUT/${EVAL_SELECTION_MODE}_k${K}.json" --seed 0 --context_max_length "$EVAL_CONTEXT_MAX_LENGTH" "${ARCH_ARGS_AP[@]}")
  [[ "$EVAL_SELECTION_MODE" == *learned* ]] && ARGS+=(--selector_checkpoint "$SEL")
  [[ -n "$EVAL_MAX_SAMPLES" ]] && ARGS+=(--max_eval_samples "$EVAL_MAX_SAMPLES")
  run "$PYTHON" eval_multihop.py "${ARGS[@]}" ${EVAL_EXTRA_ARGS:-} "$@"
else
  arch_args
  ARGS=("${ARCH_ARGS[@]}" --restore_from "$BASE"
    --lora_decoder "$LORA_DECODER" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" --lora_target_modules "$LORA_TARGETS"
    --dataset_name "$DATASET_NAME" --dataset_config "$DATASET_CONFIG" --dataset_transform "${EVAL_TRANSFORM:-$STAGE2_TRANSFORM}"
    --eval_split "$EVAL_SPLIT" --eval_output_file "$OUT/${EVAL_SELECTION_MODE}_k${K}.json"
    --selection_mode "$EVAL_SELECTION_MODE" --top_k "$K" --selection_seed 0
    --model_max_length "${EVAL_MODEL_MAX_LEN:-$STAGE2_MAX_LEN}" --context_max_length "$EVAL_CONTEXT_MAX_LENGTH" --max_new_tokens "$MAX_NEW_TOKENS"
    --bf16 True --output_dir "$OUT/hf_tmp" --report_to none)
  [[ "$EVAL_SELECTION_MODE" == *learned* ]] && ARGS+=(--selector_checkpoint "$SEL")
  [[ -n "$EVAL_MAX_SAMPLES" ]] && ARGS+=(--max_eval_samples "$EVAL_MAX_SAMPLES")
  [[ -n "$RULER_TASK" ]] && ARGS+=(--ruler_task "$RULER_TASK")
  run "$PYTHON" eval_qa.py "${ARGS[@]}" ${EVAL_EXTRA_ARGS:-} "$@"
fi
