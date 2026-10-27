#!/usr/bin/env bash
# Online TTFT / decode-throughput measurement (paper Table 3 protocol): batch size 1, bf16, the
# inference optimisations that are bit-exact (parallel segment compression, batched compressor,
# early exit at the extraction layer, torch.compile). Run on an otherwise idle GPU; numbers are p50 over N examples.
# usage: bash scripts/timing.sh <config> [K]
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
load_config "$1"
N="${TIMING_EXAMPLES:-400}"
if [[ "$EVAL_HARNESS" == "multihop" ]]; then
  EVAL_EXTRA_ARGS="--parallel --batched_compressor_fast --compile_decoder --backbone_early_exit --compile_compressor"
else
  EVAL_EXTRA_ARGS="--parallel True --batched_compressor True --backbone_early_exit True --compile_decoder True --compile_compressor True"
fi
EVAL_MAX_SAMPLES="$N" EVAL_EXTRA_ARGS="$EVAL_EXTRA_ARGS" RUN_DIR="$RUN_DIR" bash "$(dirname "${BASH_SOURCE[0]}")/evaluate.sh" "$@"
