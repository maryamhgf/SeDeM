# SeDeM on HotpotQA-Distractor, Llama-3.2-3B encoder+decoder (paper Table 1, 3B block: F1 58.30 / ROUGE-L 58.16).
# Settings from the logs of the run behind that cell (June/July 2026). NOTE: this cell was scored in `native_learned`
# mode (the K selected paragraphs are re-compressed from raw text, no per-paragraph padding) on the full 7,405
# validation questions; the 1B cell used the padded `learned` mode. Both modes are available in eval_multihop.py.
MODEL=meta-llama/Llama-3.2-3B; COMPRESSOR_HIDDEN_DIM=3072
EXTRACTION_LAYER=14; INJECTION_LAYER=10                     # 3B has 28 layers
MIX_EXTRACTION_LAYERS=True; MIXING_LAYERS=10,12,14,16
DATASET_NAME=hotpotqa/hotpot_qa; DATASET_CONFIG=distractor
STAGE1_TRANSFORM=hotpotqa; STAGE2_TRANSFORM=hotpotqa
SUBSET_MANIFEST=data/manifests/hotpot_subset_fulltrain_1kval.json
MAX_LEN=1536
STAGE1_EPOCHS=1; STAGE1_LR=3e-4; STAGE1_CTX_TOKEN_LOSS_WEIGHT=1.0; STAGE1_REC_LOSS_WEIGHT=0.25; STAGE1_DISTILL_LOSS_WEIGHT=0.5
STAGE2_EPOCHS=1; STAGE2_LR=5e-5; STAGE2_CTX_TOKEN_LOSS_WEIGHT=0.0; STAGE2_REC_LOSS_WEIGHT=0.25; STAGE2_DISTILL_LOSS_WEIGHT=0.0
SELECTOR_DATASET=hotpot; SELECTOR_EPOCHS=3; SELECTOR_LR=5e-4; SELECTOR_TEMPERATURE=1.0; SELECTOR_LAMBDA_MARGIN=0.5; SELECTOR_MARGIN=2.0
SELECTOR_MAX_TRAIN_QUESTIONS=10000; SELECTOR_MAX_VAL_QUESTIONS=200
EVAL_HARNESS=multihop; EVAL_DATASET=hotpot; TOP_K=2; K_SWEEP="2 4 6 8 10"; EVAL_MAX_SAMPLES=7405; MAX_NEW_TOKENS=64
EVAL_SELECTION_MODE=native_learned
