# Reproduction notes

The configs in `configs/` carry the hyper-parameters of the runs behind each paper cell. A few things to know
before comparing numbers.

## Scope

This release contains SeDeM only. The baselines and reference rows of the paper (ICAE, ComprExIT, and the frozen
and fine-tuned full-context references) are not part of it; their numbers in Table 1 and Table 2 come from the
respective published code.

## Table 1 (SeDeM rows): what produced each printed cell

Settings in each config were recovered from the logs and evaluation records of the run behind the printed value.
"Harness" is the script that scored the cell; "n" the number of validation questions.

| Cell | Printed F1 | Config | Mode / K | n | Harness |
|---|---|---|---|---|---|
| HotpotQA 1B | 50.93 | `hotpot_1b` | learned, K=2 | 7,405 (full) | `eval_multihop.py` (paragraph blocks) |
| HotpotQA 3B | 58.30 | `hotpot_3b` | native_learned, K=2 | 7,405 (full) | `eval_multihop.py` (selected paragraphs re-compressed from raw text) |
| 2Wiki 1B | 55.69 | `twowiki_1b` | learned, K=2 | 1,000 | `eval_multihop.py` (full split with the same checkpoint: 54.71) |
| 2Wiki 3B | 67.25 | `twowiki_3b` | learned, K=4 | 1,000 | `eval_multihop.py`; the original record was lost, the same checkpoint re-scored gives 67.04 at K=4 and 62.52 at K=2 |
| MuSiQue 1B | 18.63 | `musique_1b` | selector disabled | 2,417 (full) | `eval_qa.py`; checkpoints not preserved |
| MuSiQue 3B | 21.85 | `musique_3b` | learned, K=10 | 1,000 | `eval_multihop.py`; checkpoints not preserved (full split, K=8: 22.18) |
| QASPER 1B | 23.82 | `qasper_1b` | learned, K=32 | 945 (full) | `eval_qa.py` |
| QASPER 3B | 26.74 | `qasper_3b` | selector disabled | 945 (full) | `eval_qa.py` |
| Cross-model HotpotQA | 44.39 | `crossmodel_hotpot` | learned, K=2 | 1,000 | `eval_multihop.py` (record: 44.36) |
| Cross-model 2Wiki | 63.44 | `crossmodel_twowiki` | learned, K=2 | 1,000 | `eval_multihop.py`; Stage-2 checkpoint not preserved |
| Cross-model MuSiQue | 19.24 | `crossmodel_musique` | learned, K=8 | 1,000 | `eval_multihop.py` |
| Cross-model QASPER | 20.23 | `crossmodel_qasper` | learned, K=32 | 945 (full) | `eval_qa.py` |

Note the two HotpotQA cells use different evaluation modes: the 1B cell scores the padded per-paragraph memory
blocks (`learned`), the 3B cell re-compresses the selected paragraphs from raw text (`native_learned`, which gives
52.7 for the 1B checkpoint). Both modes are available in `eval_multihop.py` (`--mode`). The paper text states K=2 /
K=4 / K=8 for the multi-hop / MuSiQue / QASPER datasets; the K that produced each printed value is listed above and
set in the configs. `EVAL_MAX_SAMPLES=""` evaluates a full split; `scripts/evaluate.sh <config> <K>` sweeps K.

## Re-scoring the paper checkpoints with this release

Every surviving checkpoint behind a Table 1 SeDeM cell was re-scored with this release's evaluators
(`scripts/evaluate.sh <config> <K>` with `EVAL_BASE_CHECKPOINT` / `EVAL_SELECTOR_CHECKPOINT` pointing at the original
files; one A100-40GB per cell). "Identical" counts the validation questions whose generated answer is byte-identical to
the answer stored in the original evaluation record.

| Cell | Config, mode, K | n | This release F1 / EM | Original record F1 | Identical predictions | Wall time |
|---|---|---|---|---|---|---|
| HotpotQA 1B | `hotpot_1b`, learned, K=2 | 7,405 | 50.93 / 38.42 | 50.93 | 7,405 / 7,405 | 63 min |
| HotpotQA 3B | `hotpot_3b`, native_learned, K=2 | 7,405 | 58.29 / 46.02 | 58.30 | 7,235 / 7,405 | 103 min |
| 2Wiki 1B | `twowiki_1b`, learned, K=2 | 1,000 | 55.69 / 49.40 | 55.69 | 1,000 / 1,000 | 8 min |
| 2Wiki 3B | `twowiki_3b`, learned, K=4 | 1,000 | 66.98 / 61.10 | 67.04 (re-score of the checkpoint; printed 67.25) | 987 / 1,000 | 16 min |
| QASPER 1B | `qasper_1b`, learned, K=32 | 945 | 23.82 / 11.11 | 23.82 | 945 / 945 | 37 min |
| QASPER 3B | `qasper_3b`, selector disabled | 945 | 26.49 / 11.75 | 26.74 | 789 / 945 | 111 min |
| Cross-model HotpotQA | `crossmodel_hotpot`, learned, K=2 | 1,000 | 44.36 / 31.50 | 44.36 (printed 44.39) | 1,000 / 1,000 | 8 min |
| Cross-model MuSiQue | `crossmodel_musique`, learned, K=8 | 1,000 | 19.24 / 11.90 | 19.24 | 1,000 / 1,000 | 22 min |
| Cross-model QASPER | `crossmodel_qasper`, learned, K=32 | 945 | 20.23 / 10.90 | 20.23 | 945 / 945 | 56 min |

Six cells reproduce their record answer for answer. The two 3B multi-hop cells differ on 1-2% of the questions with F1
within 0.1 of the record: the 3B backbone runs in bf16, whose matrix products are not bit-reproducible across runs, so
selector scores that are nearly tied can pick different blocks. On 2Wiki 3B the 13 questions with a different answer are
exactly the 13 whose selected blocks differ. The QASPER 3B record was written in May 2026; the pre-release evaluator as
of October 2026 scores the same checkpoint at 26.49 with answers identical to this release on all 945 questions, so the
0.25 gap is drift of the pre-release code between those dates, not something introduced by the release.

The MuSiQue 1B and 3B checkpoints and the cross-model 2Wiki Stage-2 checkpoint were not preserved; those cells can only
be reproduced by retraining with their configs. The 2Wiki 3B value printed in the paper (67.25) came from an evaluation
record that was lost; the surviving checkpoint scores 67.04 under the pre-release code and 66.98 here.

## Where the launch settings differ from the paper text

* **Stage 1 data.** Appendix A.6.2 describes Stage 1 on 300M SlimPajama tokens. The Table 2 runs pretrain the
  compressor and decompressor on each dataset's own contexts (`STAGE1_TRANSFORM` in each config). The SlimPajama Stage 1
  (`slimpajama_stage1_3b`) is the one used for the transfer test of Sec. 7.3.
* **Selection loss.** A.6.2 lists temperature 0.07 with margin weight 0.5. The hotpot configs use temperature 1.0 with the
  margin term; the 2Wiki/MuSiQue/QASPER-3B configs use temperature 0.07 without the margin term. The values in each
  config are the ones that produced that cell.
* **Reconstruction loss weight.** The configs use `REC_LOSS_WEIGHT=0.25` in Stage 2 (A.6.2 says 0.1).
* **QASPER generation length.** The launchers use `MAX_NEW_TOKENS=512`; the paper text says 128. EOS stops generation early
  in practice.
* **Stage 2 and the selector.** `train_stage2.py` trains the compressor, decompressor and decoder LoRA with all memory
  blocks decompressed (no selection); `train_selector.py` then trains the selector on the frozen Stage 2 model with the
  selection loss. The joint variant (`train_stage2_joint.py`) was used for the cross-model HotpotQA cell only.
* **Compressor details not in the paper text.** The pooled-linear compressor applies LayerNorm before and after the
  projection and a learned slot-to-slot linear mixer (`SLOT_MIXER=linear`, identity-initialised); the decompressed states
  pass through a LayerNorm and a learned scalar before injection. Set `SLOT_MIXER=none` to disable the mixer.
* **Answer normalisation.** `eval_multihop.py` strips leading articles only; `eval_qa.py` uses full SQuAD normalisation.

## Runtime (one A100-40GB)

Stage 1 on HotpotQA (90k contexts, 1 epoch): about 6 h (1B) / 12 h (3B). Stage 2: similar. Selector: 1 h.
Evaluation on the full HotpotQA validation set: 30-40 min. Set `CACHE_HIDDEN_STATES=True` to cache extraction-layer
states on disk when the same contexts are reused across epochs (hundreds of GB for the 3B full-train runs).

## Timing protocol (Table 3)

`scripts/timing.sh` evaluates 400 examples with the bit-exact inference optimisations enabled (parallel segment
compression, batched compressor, early exit at the extraction layer, `torch.compile`) and reports p50 online TTFT
(segmentation + extraction + compression + selection + decompression + decoder prefill) and decode tokens/s. Use an
otherwise idle GPU; the first example pays the compile warm-up and is excluded from the p50.

## Full-recipe retrain of a paper cell with this release

`bash scripts/run_pipeline.sh hotpot_1b` (HotpotQA-Distractor, Llama-3.2-1B, the full training set, one A100-40GB) retrained
the HotpotQA 1B cell from scratch with this release: Stage 1 10 h 46 min, Stage 2 3 h 58 min, selector about 1 h 58 min, then
the K sweep on all 7,405 validation questions (about 33 min per K). The paper checkpoint was scored with the same evaluator.

| | Stage 2 val. loss | Selector recall@2 / @4 / @6 / @8 (200 held-out q.) |
|---|---|---|
| This release | 0.685 | 0.635 / 0.825 / 0.918 / 0.968 |
| Paper run | 0.699 | 0.6275 / 0.845 / 0.9275 / 0.975 |

| K | This release: F1 / EM / recall@K | Paper checkpoint: F1 / EM / recall@K |
|---|---|---|
| 2 (Table 1 cell) | 50.72 / 38.81 / 0.665 | 50.93 / 38.42 / 0.686 |
| 4 | 51.23 / 38.97 / 0.858 | 50.83 / 38.34 / 0.875 |
| 6 | 49.08 / 36.87 / 0.938 | 49.20 / 36.69 / 0.951 |
| 8 | 47.87 / 35.64 / 0.979 | 47.47 / 35.15 / 0.983 |
| 10 | 46.04 / 34.14 / 1.000 | 45.65 / 33.76 / 1.000 |

With 7,405 questions the standard error of F1 is about 0.5 points, so the retrained model matches the printed value
(50.93 at K=2) and the paper checkpoint at every K within run-to-run noise.

## Verified end-to-end run of this release (small budget)

`bash scripts/run_pipeline.sh hotpot_1b_3k` (HotpotQA-Distractor, Llama-3.2-1B, 3,000 training questions, one A100-40GB,
3 h 45 min), compared with a run of the same recipe by the pre-release research code. Both rows are scored with
`eval_multihop.py` on the first 1,000 validation questions; all schedules ran to the end (3 / 2 / 3 epochs).

| | Stage 1 val. loss | Stage 2 val. loss | Selector recall@2 / @8 (200 q.) |
|---|---|---|---|
| This release | 6.906 | 6.618 | 0.68 / 0.98 |
| Pre-release code | 6.894 | 6.645 | 0.64 / 0.98 |

| K | This release: F1 / EM / recall@K | Pre-release checkpoint: F1 / EM / recall@K |
|---|---|---|
| 2 | 43.0 / 30.4 / 0.69 | 39.8 / 27.7 / 0.64 |
| 4 | 41.5 / 28.8 / 0.88 | 40.5 / 28.1 / 0.84 |
| 6 | 39.2 / 26.6 / 0.95 | 38.4 / 25.9 / 0.93 |
| 8 | 36.5 / 24.2 / 0.99 | 36.3 / 24.2 / 0.98 |
| 10 | 33.3 / 22.2 / 1.00 | 34.8 / 23.2 / 1.00 |

With 1,000 questions the standard error of F1 is about 1.5 points, so the two runs agree within run-to-run noise except
at K=2, where the release run's selector recall is higher. On the pre-release checkpoint itself the release evaluator and
the pre-release evaluators produce identical predictions on all 1,000 questions. This is a code-parity check on a small
training budget, not a paper result; no baseline was trained for this 3,000-question recipe.
