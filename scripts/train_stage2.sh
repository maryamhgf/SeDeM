#!/usr/bin/env bash
# Stage 2: task training (answer generation through injected, decompressed memory; decoder LoRA). Paper Sec. 4, Eq. 14.
# usage: bash scripts/train_stage2.sh <config> [extra args]   (expects $RUN_DIR/stage1/model.safetensors unless STAGE2_RESTORE_FROM is set)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
load_config "$1"; shift || true
arch_args
OUT="$RUN_DIR/stage2"; mkdir -p "$OUT"
RESTORE="${STAGE2_RESTORE_FROM:-$(stage_ckpt stage1)}"
ARGS=("${ARCH_ARGS[@]}"
  --restore_from "$RESTORE"
  --ctx_token_loss_weight "$STAGE2_CTX_TOKEN_LOSS_WEIGHT" --rec_loss_weight "$STAGE2_REC_LOSS_WEIGHT"
  --distill_loss_weight "$STAGE2_DISTILL_LOSS_WEIGHT" --distill_temperature "$DISTILL_TEMPERATURE"
  --dataset_name "$DATASET_NAME" --dataset_config "$DATASET_CONFIG" --dataset_transform "$STAGE2_TRANSFORM"
  --subset_manifest "$SUBSET_MANIFEST"
  --output_dir "$OUT" --model_max_length "$STAGE2_MAX_LEN"
  --per_device_train_batch_size "$BATCH_SIZE" --per_device_eval_batch_size "$BATCH_SIZE" --gradient_accumulation_steps "$GRAD_ACCUM"
  --lora_decoder "$LORA_DECODER" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" --lora_target_modules "$LORA_TARGETS"
  --learning_rate "$STAGE2_LR" --num_train_epochs "$STAGE2_EPOCHS" --lr_scheduler_type cosine --warmup_ratio 0.05 --weight_decay 0.01
  --logging_steps "$LOGGING_STEPS" --eval_strategy steps --eval_steps "$STAGE2_EVAL_STEPS" --save_strategy steps --save_steps "$STAGE2_SAVE_STEPS"
  --save_total_limit 3 --load_best_model_at_end True --metric_for_best_model eval_loss --greater_is_better False
  --cache_hidden_states "$CACHE_HIDDEN_STATES" --hidden_states_cache_dir "$SEDEM_CACHE_DIR/${RUN_NAME}_L${EXTRACTION_LAYER}"
  --bf16 True --report_to "$REPORT_TO" --run_name "${RUN_NAME}_stage2" --seed "$SEED")
[[ -n "${STAGE2_DATA_PATH:-}" ]] && ARGS+=(--data_path "$STAGE2_DATA_PATH" --eval_data_path "${STAGE2_EVAL_DATA_PATH:?set STAGE2_EVAL_DATA_PATH}")
[[ -n "$MAX_TRAIN_EVAL_SAMPLES" ]] && ARGS+=(--max_train_eval_samples "$MAX_TRAIN_EVAL_SAMPLES")
cd "$SEDEM_ROOT"
launch train_stage2.py "${ARGS[@]}" "$@"
