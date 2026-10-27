# SeDeM on HotpotQA-Distractor, Llama-3.2-1B encoder+decoder (paper Table 1, 1B block: F1 50.93 / ROUGE-L 50.82).
# Settings recovered from the logs of the run behind that cell (May 2026). The cell was scored with the
# paragraph-block evaluator (`learned` mode) on the full 7,405-question validation split at K=2.
MODEL=meta-llama/Llama-3.2-1B; COMPRESSOR_HIDDEN_DIM=2048
EXTRACTION_LAYER=8; INJECTION_LAYER=6                       # 1B has 16 layers: extract mid-stack, inject below it
MIX_EXTRACTION_LAYERS=True; MIXING_LAYERS=6,8,10,12         # mixed extraction layers (App. A.12), as used for the 1B cells
DATASET_NAME=hotpotqa/hotpot_qa; DATASET_CONFIG=distractor
STAGE1_TRANSFORM=hotpotqa; STAGE2_TRANSFORM=hotpotqa        # Stage 2 trains on the two gold paragraphs; eval uses all 10 (eval_multihop)
SUBSET_MANIFEST=data/manifests/hotpot_subset_fulltrain_1kval.json   # full 90,447 train; 1k held-out val for checkpoint selection
MAX_LEN=1536
STAGE1_EPOCHS=1; STAGE1_LR=3e-4; STAGE1_CTX_TOKEN_LOSS_WEIGHT=1.0; STAGE1_REC_LOSS_WEIGHT=0.25; STAGE1_DISTILL_LOSS_WEIGHT=0.5
STAGE2_EPOCHS=1; STAGE2_LR=5e-5; STAGE2_CTX_TOKEN_LOSS_WEIGHT=0.0; STAGE2_REC_LOSS_WEIGHT=0.25; STAGE2_DISTILL_LOSS_WEIGHT=0.0
SELECTOR_DATASET=hotpot; SELECTOR_EPOCHS=3; SELECTOR_LR=5e-4; SELECTOR_TEMPERATURE=1.0; SELECTOR_LAMBDA_MARGIN=0.5; SELECTOR_MARGIN=2.0
SELECTOR_MAX_TRAIN_QUESTIONS=10000; SELECTOR_MAX_VAL_QUESTIONS=200
EVAL_HARNESS=multihop; EVAL_DATASET=hotpot; TOP_K=2; K_SWEEP="2 4 6 8 10"; EVAL_MAX_SAMPLES=7405; MAX_NEW_TOKENS=64
