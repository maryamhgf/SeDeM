# Budget configuration: HotpotQA-Distractor, Llama-3.2-1B, Stage 1/2 on 3,000 training questions.
# A complete Stage 1 -> Stage 2 -> selector -> K-sweep run for checking an installation or a new idea; the paper
# cells use the full training set (hotpot_1b). Measured on one A100-40GB: Stage 1 60 min, Stage 2 17 min,
# selector 2 h, K sweep 25 min (about 3 h 45 min in total).
# Expected (first 1,000 validation questions): F1 about 43 / EM about 30 at K=2, decreasing with K; two independent
# runs of this recipe differ by 1-3 F1. The selector trains on SELECTOR_MAX_TRAIN_QUESTIONS questions drawn from the
# full training split, independent of the 3,000-question subset used for Stage 1/2.
MODEL=meta-llama/Llama-3.2-1B; COMPRESSOR_HIDDEN_DIM=2048
EXTRACTION_LAYER=8; INJECTION_LAYER=6; MIX_EXTRACTION_LAYERS=True; MIXING_LAYERS=6,8,10,12
DATASET_NAME=hotpotqa/hotpot_qa; DATASET_CONFIG=distractor
STAGE1_TRANSFORM=hotpotqa; STAGE2_TRANSFORM=hotpotqa
SUBSET_MANIFEST=data/manifests/hotpot_subset_3k1k.json       # 3,000 train / 1,000 validation questions (stratified)
MAX_LEN=1536
STAGE1_EPOCHS=3; STAGE1_LR=3e-4; STAGE1_CTX_TOKEN_LOSS_WEIGHT=1.0; STAGE1_REC_LOSS_WEIGHT=0.25; STAGE1_DISTILL_LOSS_WEIGHT=0.5
STAGE2_EPOCHS=2; STAGE2_LR=5e-5; STAGE2_CTX_TOKEN_LOSS_WEIGHT=1.0; STAGE2_REC_LOSS_WEIGHT=0.25; STAGE2_DISTILL_LOSS_WEIGHT=0.0
SELECTOR_DATASET=hotpot; SELECTOR_EPOCHS=3; SELECTOR_LR=5e-4; SELECTOR_TEMPERATURE=1.0; SELECTOR_LAMBDA_MARGIN=0.5; SELECTOR_MARGIN=2.0
SELECTOR_MAX_TRAIN_QUESTIONS=10000; SELECTOR_MAX_VAL_QUESTIONS=200
EVAL_HARNESS=multihop; EVAL_DATASET=hotpot; TOP_K=2; K_SWEEP="2 4 6 8 10"; EVAL_MAX_SAMPLES=1000; MAX_NEW_TOKENS=64
