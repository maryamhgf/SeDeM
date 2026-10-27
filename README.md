# SeDeM: Selective Decompression of Hidden-State Memories for Long-Context Question Answering

Official code for **SeDeM: Selective Decompression of Hidden-State Memories for Long-Context Question Answering**
(Maryam Haghifam, Jason Cong, Yizhou Sun; EMNLP 2026). Paper: https://arxiv.org/abs/2608.00311

SeDeM stores a long context as compact hidden-state memory blocks, selects the query-relevant blocks, decompresses
only those into decoder-compatible hidden states, and injects them at an intermediate decoder layer.

```
context segments --frozen LLM (layer l_extract)--> hidden states --compressor--> memory blocks  (query independent)
query --> selector (top-K blocks over the memory bank) --> decompressor --> reconstructed states --> inject at l_inject --> answer
```

| Paper component | Code |
|---|---|
| Compressor (mean-pool over C tokens + shared projection W_in), Sec. 3.1 | `sedem/compressor.py` `SegmentCompressor` |
| Top-K selector (late interaction, R scoring heads), Sec. 3.2 | `sedem/selector.py` `LateInteractionSelector` |
| Decompressor (two-layer MLP, one memory vector -> C hidden states), Sec. 3.3 | `sedem/compressor.py` `Decompressor` |
| Two-phase decoder forward with injection at l_inject | `sedem/modeling.py` `SeDeMModel.forward_with_injection` |
| Stage 1 reconstruction pretraining (L_ctx + L_distill + L_rec), Sec. 4 | `train_stage1.py` |
| Stage 2 task training with decoder LoRA, Sec. 4 | `train_stage2.py` (+ `train_selector.py` for L_ret) |
| Evaluation (token-F1, ROUGE-L, EM, online TTFT, decode throughput) | `eval_multihop.py`, `eval_qa.py` |

## Installation

```bash
git clone https://github.com/maryamhgf/SeDeM.git && cd SeDeM
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # torch 2.6 / transformers 5.3 / peft 0.18 (tested); flash-attn optional
huggingface-cli login                       # Llama-3.2 weights are gated
```

One A100-40GB (or any 40 GB GPU) trains and evaluates the 1B and 3B configurations with batch size 1 and
gradient accumulation 8. Multi-GPU: set `NUM_GPUS` (the launchers use `accelerate`).

## Quick start

```bash
bash scripts/run_pipeline.sh smoke_hotpot_1b

bash scripts/run_pipeline.sh hotpot_1b_3k

bash scripts/run_pipeline.sh hotpot_1b

bash scripts/train_stage1.sh  hotpot_1b
bash scripts/train_stage2.sh  hotpot_1b
bash scripts/train_selector.sh hotpot_1b
bash scripts/evaluate.sh      hotpot_1b 2
```

Every launcher takes a config name from `configs/` (or a path to your own file). `SEDEM_OUTPUT_DIR` (default
`outputs/`) and `SEDEM_CACHE_DIR` (default `cache/`) control where things are written; `DRY_RUN=1` prints the exact
Python commands without running them; `REPORT_TO=wandb` enables Weights & Biases logging.

## Reproducing the paper

| Result | Command |
|---|---|
| Table 2, same-backbone cells | `bash scripts/run_pipeline.sh {hotpot,twowiki,musique,qasper}_{1b,3b}` |
| Table 2, cross-model row (1B encoder, 3B decoder) | `bash scripts/run_pipeline.sh crossmodel_{twowiki,musique,qasper}`; HotpotQA: see `configs/crossmodel_hotpot.sh` (joint Stage 2) |
| Table 3, online TTFT and decode throughput | `bash scripts/timing.sh <config> [K]` on an idle GPU (p50 over 400 examples) |
| Table 4, no-decompression ablation | `ARM=linear bash scripts/ablation_no_decompression.sh {hotpot,qasper}_3b` and `ARM=mlp ...` |
| Table 5 / 16, selection budget K | `bash scripts/evaluate.sh hotpot_1b <K>`, `bash scripts/evaluate.sh twowiki_1b <K>` |
| Table 6, decoder LoRA ablation | `CONFIG_OVERRIDES="LORA_DECODER=False EVAL_SELECTION_MODE=off EVAL_HARNESS=qa" bash scripts/run_pipeline.sh qasper_3b` |
| Sec. 7.3, out-of-distribution transfer | `bash scripts/train_stage1.sh slimpajama_stage1_3b`, then `configs/ood_transfer_qasper_3b.sh` (header has the commands) |
| App. A.9 parameters and FLOPs | `python analysis/compute_params_flops.py` |
| App. A.10 effective rank | `python analysis/measure_effective_rank.py --help` |
| App. A.12 layer diagnostics | `python analysis/layer_causal_patching.py --help` |
| App. A.15 segment length / compression factor | `bash scripts/granularity_sweep.sh qasper_1b` |
| App. A.16 RULER qa_2 | `EVAL_BASE_CHECKPOINT=<stage2 ckpt> bash scripts/evaluate.sh ruler_qa2_3b` |

Each config file documents the values it carries over from the runs behind the paper; see `docs/REPRODUCTION_NOTES.md`
for per-cell notes (number of evaluation examples, K, and the few places where the launch settings differ from the paper text).

## Using SeDeM with your own model or data

**Backbone.** Any decoder-only HuggingFace causal LM works for the encoder and the decoder (tested: Llama-3.2-1B/3B,
Qwen3-4B). Set `MODEL` (or `ENCODER_MODEL` / `DECODER_MODEL` for the cross-model setting), the extraction and injection
layers, and `COMPRESSOR_HIDDEN_DIM` to the decoder hidden size:

```bash
# configs/my_model.sh
MODEL=Qwen/Qwen3-4B-Base; COMPRESSOR_HIDDEN_DIM=2560
EXTRACTION_LAYER=18; INJECTION_LAYER=12            # 36-layer model: extract mid-stack, inject a few layers below
MIX_EXTRACTION_LAYERS=False
DATASET_NAME=hotpotqa/hotpot_qa; DATASET_CONFIG=distractor; STAGE1_TRANSFORM=hotpotqa; STAGE2_TRANSFORM=hotpotqa
MAX_LEN=1536; STAGE1_EPOCHS=1; STAGE1_LR=3e-4; STAGE1_CTX_TOKEN_LOSS_WEIGHT=1.0; STAGE1_REC_LOSS_WEIGHT=0.25; STAGE1_DISTILL_LOSS_WEIGHT=0.5
STAGE2_EPOCHS=1; STAGE2_LR=5e-5; STAGE2_CTX_TOKEN_LOSS_WEIGHT=1.0; STAGE2_REC_LOSS_WEIGHT=0.25; STAGE2_DISTILL_LOSS_WEIGHT=0.0
SELECTOR_DATASET=hotpot; EVAL_HARNESS=multihop; EVAL_DATASET=hotpot; TOP_K=2
```

**Memory geometry.** `MEMORY_SLOTS` (N_M, default 32) and `COMPRESSION_FACTOR` (C, default 4) give segment length
T = N_M * C = 128 tokens and one memory block per segment. `TOP_K` is the selection budget K.

**Dataset.** A dataset is a pair of transform functions registered in `sedem/data.py`: a Stage-1 transform that yields
raw text, and a Stage-2 / evaluation transform that yields `input` (context), `prompt` (question), `answer`, and, when
available, per-segment evidence labels for selector training. See `docs/ADDING_A_DATASET.md`. Datasets without evidence
labels can run SeDeM in selector-disabled mode (`EVAL_SELECTION_MODE=off`, all blocks decompressed).

## Checkpoints

`train_stage1.py` and `train_stage2.py` save `model.safetensors` with the trainable modules only (compressor,
decompressor, layer mixer, memory norm, LoRA); the frozen backbone is re-loaded from the HuggingFace id.
`train_selector.py` saves `selector.safetensors` + `selector_config.json`. Checkpoints trained with the pre-release
code load unchanged (legacy file and key names are recognised).

## Citation

```bibtex
@inproceedings{haghifam2026sedem,
  title     = {SeDeM: Selective Decompression of Hidden-State Memories for Long-Context Question Answering},
  author    = {Haghifam, Maryam and Cong, Jason and Sun, Yizhou},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2608.00311}
}
```

## Acknowledgements

The training scaffolding (`run_training`, `sample_context_window`) is adapted from the
[ICAE codebase](https://github.com/getao/icae) (Ge et al., ICLR 2024; CC0-1.0).

## License

MIT. See `LICENSE`.
