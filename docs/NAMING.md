# Naming

The code uses the paper's vocabulary. For readers of the pre-release research code, the main renames:

| Paper term (release) | Old name |
|---|---|
| `SeDeMModel` (`sedem/modeling.py`) | `HMEM_HiddenState` |
| `SegmentCompressor` / `Decompressor` (`sedem/compressor.py`) | `ChunkedGatedCompressor` / `ChunkedDecompressor` |
| `LearnableQueryCompressor` (App. A.11 ablation) | `MLPPoolingCompressor` ("Perceiver") |
| `ExtractionLayerMixer` (App. A.12) | `MultiLayerMixer` (M0 / M1) |
| `LateInteractionSelector`, `LateInteractionScorer` (`sedem/selector.py`) | `R2Head`, `CrossAttentionScorer` ("retriever") |
| `PooledQuerySelector` | `R1Head` |
| selection mode `learned` / `label` / `random` / `off` | `r2_learned` / `label` ("oracle") / `random` / `off` |
| `--extraction_layer`, `--injection_layer`, `--inject_decompressed` | `--hidden_state_layer`, `--decompress_inject_layer`, `--decompress_inject` |
| `--memory_slots_per_segment`, `--compression_factor` | `--fixed_mem_size`, `--mean_compression_rate` |
| `--compressor_type pooled_linear` / `learnable_query` | `--compressor_variant chunked_gated` / `mlp_pooling` |
| `--ctx_token_loss_weight`, `--rec_loss_weight`, `--distill_loss_weight` | `--tok_weight`, `--decomp_weight`, `--distill_weight` |
| `--mix_extraction_layers`, `--mixing_layers` | `--use_layer_mixing`, `--mix_source_layers` |
| `--selection_mode`, `--selector_checkpoint`, `--top_k` | `--retrieval_mode`, `--retriever_checkpoint`, `--e1_top_k` |
| `selector.safetensors`, `selector_config.json` | `retriever.safetensors`, `retriever_config.json` (still readable) |
| `train_stage1.py`, `train_stage2.py`, `train_selector.py`, `train_stage2_joint.py` | `pretrain_hidden_state.py`, `finetune_qa_hidden_state.py`, `train_retriever.py`, `train_end_to_end.py` |
| `eval_multihop.py`, `eval_qa.py` | `eval_qa_hotpot_test.py`, `eval_qa.py` |

Checkpoint tensor names (`compressor.*`, `decompressor.*`, `mixer.*`, `mem_norm`, `mem_scale`, LoRA) are unchanged, so
checkpoints trained with the old code load directly.
