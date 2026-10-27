#!/usr/bin/env bash
# Stage 1: reconstruction pretraining of the compressor + decompressor (paper Sec. 4, Eq. 12).
# usage: bash scripts/train_stage1.sh <config> [extra HF/trainer args]
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
load_config "$1"; shift || true
arch_args
OUT="$RUN_DIR/stage1"; mkdir -p "$OUT"
ARGS=("${ARCH_ARGS[@]}"
  --restore_from "${STAGE1_RESTORE_FROM:-}" --freeze_compressor False
  --ctx_token_loss_weight "$STAGE1_CTX_TOKEN_LOSS_WEIGHT" --rec_loss_weight "$STAGE1_REC_LOSS_WEIGHT"
  --distill_loss_weight "$STAGE1_DISTILL_LOSS_WEIGHT" --distill_temperature "$DISTILL_TEMPERATURE"
  --dataset_name "$DATASET_NAME" --dataset_config "$DATASET_CONFIG" --dataset_transform "$STAGE1_TRANSFORM"
  --dataset_streaming "$DATASET_STREAMING" --subset_manifest "$SUBSET_MANIFEST"
  --output_dir "$OUT" --model_max_length "$STAGE1_MAX_LEN" --pretrain_window_size "$PRETRAIN_WINDOW"
  --per_device_train_batch_size "$BATCH_SIZE" --per_device_eval_batch_size "$BATCH_SIZE" --gradient_accumulation_steps "$GRAD_ACCUM"
  --lora_decoder False --learning_rate "$STAGE1_LR" --lr_scheduler_type cosine --warmup_ratio 0.1 --weight_decay 0.01
  --logging_steps "$LOGGING_STEPS" --eval_strategy steps --eval_steps "$STAGE1_EVAL_STEPS" --save_strategy steps --save_steps "$STAGE1_SAVE_STEPS"
  --save_total_limit 3 --metric_for_best_model eval_loss --greater_is_better False --load_best_model_at_end True
  --cache_hidden_states "$CACHE_HIDDEN_STATES" --hidden_states_cache_dir "$SEDEM_CACHE_DIR/${RUN_NAME}_L${EXTRACTION_LAYER}"
  --bf16 True --report_to "$REPORT_TO" --run_name "${RUN_NAME}_stage1" --seed "$SEED")
[[ -n "$STAGE1_MAX_STEPS" ]] && ARGS+=(--max_steps "$STAGE1_MAX_STEPS") || ARGS+=(--num_train_epochs "$STAGE1_EPOCHS")
[[ -n "$MAX_TRAIN_EVAL_SAMPLES" ]] && ARGS+=(--max_train_eval_samples "$MAX_TRAIN_EVAL_SAMPLES")
cd "$SEDEM_ROOT"
launch train_stage1.py "${ARGS[@]}" "$@"
