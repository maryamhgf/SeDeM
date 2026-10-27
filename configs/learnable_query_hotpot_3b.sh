# App. A.11: learnable-query (Perceiver-style) compressor instead of the pooled-linear compressor, HotpotQA, 3B.
# Mixed extraction layers with frozen uniform mixing weights, as launched. Evaluate selector-disabled.
MODEL=meta-llama/Llama-3.2-3B; COMPRESSOR_HIDDEN_DIM=3072
COMPRESSOR_TYPE=learnable_query; EXTRA_ARCH_ARGS="--out_proj_mode identity --compressor_blend_mode content_first --gate_init 0.1"
EXTRACTION_LAYER=14; INJECTION_LAYER=10; MIX_EXTRACTION_LAYERS=True; MIXING_LAYERS=10,12,14,16; MIXING_LEARNABLE=False
DATASET_NAME=hotpotqa/hotpot_qa; DATASET_CONFIG=distractor
STAGE1_TRANSFORM=hotpotqa; STAGE2_TRANSFORM=hotpotqa; EVAL_TRANSFORM=hotpotqa_full
SUBSET_MANIFEST=data/manifests/hotpot_subset_10k1k.json; MAX_LEN=1536
STAGE1_EPOCHS=3; STAGE1_LR=3e-4; STAGE1_CTX_TOKEN_LOSS_WEIGHT=0.0; STAGE1_REC_LOSS_WEIGHT=1.0; STAGE1_DISTILL_LOSS_WEIGHT=0.5
STAGE2_EPOCHS=3; STAGE2_LR=5e-5; STAGE2_CTX_TOKEN_LOSS_WEIGHT=0.0; STAGE2_REC_LOSS_WEIGHT=0.25; STAGE2_DISTILL_LOSS_WEIGHT=0.0
SELECTOR_DATASET=hotpot
EVAL_HARNESS=qa; EVAL_SELECTION_MODE=off; TOP_K=0; MAX_NEW_TOKENS=64; EVAL_MAX_SAMPLES=1000
