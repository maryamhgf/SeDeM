# Cross-model SeDeM (1B encoder, 3B decoder) on HotpotQA-Distractor (paper Table 1, cross-model row: F1 44.39).
# Settings from the run behind that cell (May 2026): Stage 1 (3 epochs, single extraction layer 10 of the 1B encoder,
# injection at layer 10 of the 3B decoder), selector on the Stage-1 checkpoint, Stage 2 for 2 epochs, scored on the
# first 1,000 validation questions at K=2. (`scripts/train_stage2_joint.sh` is an alternative joint Stage 2; it was
# not used for the printed cell.)
ENCODER_MODEL=meta-llama/Llama-3.2-1B; DECODER_MODEL=meta-llama/Llama-3.2-3B; COMPRESSOR_HIDDEN_DIM=3072
EXTRACTION_LAYER=10; INJECTION_LAYER=10; MIX_EXTRACTION_LAYERS=False
DATASET_NAME=hotpotqa/hotpot_qa; DATASET_CONFIG=distractor
STAGE1_TRANSFORM=hotpotqa; STAGE2_TRANSFORM=hotpotqa; SUBSET_MANIFEST=data/manifests/hotpot_subset_10k1k.json
MAX_LEN=1536
STAGE1_EPOCHS=3; STAGE1_LR=3e-4; STAGE1_CTX_TOKEN_LOSS_WEIGHT=1.0; STAGE1_REC_LOSS_WEIGHT=0.25; STAGE1_DISTILL_LOSS_WEIGHT=0.5
STAGE2_EPOCHS=2; STAGE2_LR=5e-5; STAGE2_CTX_TOKEN_LOSS_WEIGHT=0.0; STAGE2_REC_LOSS_WEIGHT=0.25; STAGE2_DISTILL_LOSS_WEIGHT=0.0
SELECTOR_BASE=stage1; SELECTOR_DATASET=hotpot; SELECTOR_EPOCHS=3; SELECTOR_LR=1e-3; SELECTOR_TEMPERATURE=0.07; SELECTOR_LAMBDA_MARGIN=0.0
SELECTOR_MAX_TRAIN_QUESTIONS=10000; SELECTOR_MAX_VAL_QUESTIONS=200
EVAL_HARNESS=multihop; EVAL_DATASET=hotpot; TOP_K=2; K_SWEEP="2 4 6 8 10"; EVAL_MAX_SAMPLES=1000; MAX_NEW_TOKENS=64
