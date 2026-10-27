# Out-of-distribution transfer (paper Sec. 7.3): SlimPajama Stage 1 -> Stage 2 on a multi-task QA mix with no QASPER
# (build it with: python data/prepare_ood_stage2_mix.py --output_dir $SEDEM_OUTPUT_DIR/ood_mix) -> zero-shot QASPER,
# selector disabled, full papers (context_max_length 24576).
#   STAGE2_RESTORE_FROM=<slimpajama_stage1_3b/stage1/model.safetensors> bash scripts/train_stage2.sh ood_transfer_qasper_3b
#   bash scripts/evaluate.sh ood_transfer_qasper_3b
MODEL=meta-llama/Llama-3.2-3B; COMPRESSOR_HIDDEN_DIM=3072
EXTRACTION_LAYER=14; INJECTION_LAYER=-999; MIX_EXTRACTION_LAYERS=False
DATASET_NAME=allenai/qasper; DATASET_CONFIG=default; STAGE1_TRANSFORM=qasper; STAGE2_TRANSFORM=""; EVAL_TRANSFORM=qasper
STAGE2_DATA_PATH="${STAGE2_DATA_PATH:-$SEDEM_OUTPUT_DIR/ood_mix/train.jsonl}"; STAGE2_EVAL_DATA_PATH="${STAGE2_EVAL_DATA_PATH:-$SEDEM_OUTPUT_DIR/ood_mix/val.jsonl}"
MAX_LEN=4096
STAGE1_EPOCHS=1; STAGE1_LR=3e-4; STAGE1_CTX_TOKEN_LOSS_WEIGHT=0.0; STAGE1_REC_LOSS_WEIGHT=1.0; STAGE1_DISTILL_LOSS_WEIGHT=0.5
STAGE2_EPOCHS=2; STAGE2_LR=5e-5; STAGE2_CTX_TOKEN_LOSS_WEIGHT=0.0; STAGE2_REC_LOSS_WEIGHT=0.25; STAGE2_DISTILL_LOSS_WEIGHT=0.0
STAGE2_EVAL_STEPS=200; STAGE2_SAVE_STEPS=200; LORA_R=128; LORA_ALPHA=128
SELECTOR_DATASET=qasper
EVAL_HARNESS=qa; EVAL_SELECTION_MODE=off; TOP_K=0; MAX_NEW_TOKENS=512; EVAL_CONTEXT_MAX_LENGTH=24576; EVAL_MODEL_MAX_LEN=24576
