#!/usr/bin/env bash
# Shared helpers for every launcher. Source this file; do not execute it.
#
# Environment knobs (all optional):
#   SEDEM_ROOT        repo root (auto-detected)
#   PYTHON            python interpreter (default: python)
#   SEDEM_OUTPUT_DIR  where runs are written (default: $SEDEM_ROOT/outputs)
#   SEDEM_CACHE_DIR   hidden-state caches (default: $SEDEM_ROOT/cache)
#   NUM_GPUS          GPUs for accelerate (default: 1); MASTER_PORT (default 29500)
#   REPORT_TO         HF Trainer reporting backend (default: none; use wandb if you want)
#   DRY_RUN=1         print the commands without running them
set -euo pipefail
SEDEM_ROOT="${SEDEM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
SEDEM_OUTPUT_DIR="${SEDEM_OUTPUT_DIR:-$SEDEM_ROOT/outputs}"
SEDEM_CACHE_DIR="${SEDEM_CACHE_DIR:-$SEDEM_ROOT/cache}"
NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"
REPORT_TO="${REPORT_TO:-none}"
DRY_RUN="${DRY_RUN:-0}"
SEED="${SEED:-42}"
export PYTHONPATH="$SEDEM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1   # loss / eval lines show up in redirected logs immediately
[[ "$REPORT_TO" == "none" ]] && export WANDB_MODE="${WANDB_MODE:-disabled}"

load_config() {
  # usage: load_config <name|path>   (configs/<name>.sh or an explicit file)
  local cfg="${1:?usage: $0 <config>}"
  if [[ -f "$cfg" ]]; then CONFIG_FILE="$cfg"
  elif [[ -f "$SEDEM_ROOT/configs/$cfg.sh" ]]; then CONFIG_FILE="$SEDEM_ROOT/configs/$cfg.sh"
  else echo "[sedem] config not found: $cfg (looked in configs/)" >&2; exit 2; fi
  # defaults shared by every config (a config may override any of them)
  COMPRESSOR_TYPE="pooled_linear"; MEMORY_SLOTS=32; COMPRESSION_FACTOR=4
  INJECT_DECOMPRESSED=True; LINEAR_PROJECTION=True; SLOT_MIXER=linear
  MIX_EXTRACTION_LAYERS=False; MIXING_LAYERS=""; MIXING_LEARNABLE=True
  DISTILL_TEMPERATURE=2.0; PRETRAIN_WINDOW=128; BATCH_SIZE=1; GRAD_ACCUM=8; LOGGING_STEPS=5
  CACHE_HIDDEN_STATES=False; DATASET_STREAMING=False; STAGE1_MAX_STEPS=""; SUBSET_MANIFEST=""
  MAX_TRAIN_EVAL_SAMPLES=""; DATASET_CONFIG=""
  STAGE1_EVAL_STEPS=500; STAGE1_SAVE_STEPS=500; STAGE2_EVAL_STEPS=500; STAGE2_SAVE_STEPS=500
  STAGE2_MAX_EVAL_SAMPLES=1000; LORA_DECODER=True; LORA_R=64; LORA_ALPHA=128; LORA_DROPOUT=0.05
  LORA_TARGETS="q_proj,k_proj,v_proj,o_proj"
  SELECTOR_BASE=stage2; SELECTOR_HEADS=4; SELECTOR_EPOCHS=3; SELECTOR_LR=1e-3; SELECTOR_TEMPERATURE=0.07
  SELECTOR_LAMBDA_MARGIN=0.0; SELECTOR_MARGIN=2.0; SELECTOR_MAX_TRAIN_QUESTIONS=""; SELECTOR_MAX_VAL_QUESTIONS=200
  SELECTOR_EVAL_EVERY=500
  EVAL_SELECTION_MODE=learned; EVAL_MAX_SAMPLES=""; MAX_NEW_TOKENS=64; EVAL_CONTEXT_MAX_LENGTH=4096
  EVAL_SPLIT=validation; EVAL_TRANSFORM=""; RULER_TASK=""; K_SWEEP=""
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
  # CONFIG_OVERRIDES="KEY=VAL KEY2=VAL2" applies after the config (used by the ablation launchers)
  for kv in ${CONFIG_OVERRIDES:-}; do declare -g "$kv"; done
  : "${RUN_NAME:=$(basename "${CONFIG_FILE%.sh}")}"
  RUN_DIR="${RUN_DIR:-$SEDEM_OUTPUT_DIR/$RUN_NAME}"
  : "${ENCODER_MODEL:=$MODEL}"; : "${DECODER_MODEL:=$MODEL}"
  : "${STAGE1_MAX_LEN:=$MAX_LEN}"; : "${STAGE2_MAX_LEN:=$MAX_LEN}"
  [[ -n "${SUBSET_MANIFEST}" && "${SUBSET_MANIFEST}" != /* ]] && SUBSET_MANIFEST="$SEDEM_ROOT/$SUBSET_MANIFEST"
  mkdir -p "$RUN_DIR"
}

run() {
  printf '+'; printf ' %q' "$@"; printf '\n'
  [[ "$DRY_RUN" == "1" ]] || "$@"
}

launch() {
  # usage: launch <entrypoint.py> [args...]
  # Uses the accelerate configs shipped in scripts/ so runs do not depend on a machine-level accelerate config.
  if [[ "$NUM_GPUS" -gt 1 ]]; then
    run accelerate launch --config_file "$SEDEM_ROOT/scripts/accelerate_multi_gpu.yaml" --num_processes "$NUM_GPUS" --main_process_port "$MASTER_PORT" "$@"
  else
    run accelerate launch --config_file "$SEDEM_ROOT/scripts/accelerate_single_gpu.yaml" --num_processes 1 --main_process_port "$MASTER_PORT" "$@"
  fi
}

arch_args() {
  # architecture flags shared by Stage 1 / Stage 2 / eval_qa (HfArgumentParser entrypoints)
  ARCH_ARGS=(
    --model_name_or_path "$DECODER_MODEL" --encoder_name_or_path "$ENCODER_MODEL" --decoder_name_or_path "$DECODER_MODEL"
    --compressor_type "$COMPRESSOR_TYPE" --compressor_hidden_dim "$COMPRESSOR_HIDDEN_DIM"
    --memory_slots_per_segment "$MEMORY_SLOTS" --compression_factor "$COMPRESSION_FACTOR"
    --extraction_layer "$EXTRACTION_LAYER" --injection_layer "$INJECTION_LAYER" --inject_decompressed "$INJECT_DECOMPRESSED"
    --compressor_no_global_context True --compressor_slot_position_embedding False --compressor_linear_projection "$LINEAR_PROJECTION"
    --slot_mixer_type "$SLOT_MIXER" --slot_mixer_attn_heads 4 --decompressor_type mlp
    --mix_extraction_layers "$MIX_EXTRACTION_LAYERS" --mixing_norm per_layer_ln --mixing_init uniform --mixing_learnable "$MIXING_LEARNABLE"
  )
  [[ -n "$MIXING_LAYERS" ]] && ARCH_ARGS+=(--mixing_layers "$MIXING_LAYERS")
  [[ -n "${EXTRA_ARCH_ARGS:-}" ]] && ARCH_ARGS+=($EXTRA_ARCH_ARGS)
  return 0
}

arch_args_argparse() {
  # subset understood by the argparse entrypoints (train_selector.py, eval_multihop.py, train_stage2_joint.py)
  ARCH_ARGS_AP=(
    --model_name_or_path "$DECODER_MODEL" --encoder_name_or_path "$ENCODER_MODEL" --decoder_name_or_path "$DECODER_MODEL"
    --extraction_layer "$EXTRACTION_LAYER" --injection_layer "$INJECTION_LAYER" --compressor_hidden_dim "$COMPRESSOR_HIDDEN_DIM"
    --memory_slots_per_segment "$MEMORY_SLOTS" --compression_factor "$COMPRESSION_FACTOR"
    --mix_extraction_layers "$MIX_EXTRACTION_LAYERS" --mixing_learnable "$MIXING_LEARNABLE"
  )
  [[ -n "$MIXING_LAYERS" ]] && ARCH_ARGS_AP+=(--mixing_layers "$MIXING_LAYERS")
  return 0
}

stage_ckpt() { echo "$RUN_DIR/$1/model.safetensors"; }
