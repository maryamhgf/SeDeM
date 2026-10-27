#!/usr/bin/env bash
# Appendix A.15: segment length T = N_M * C and compression factor C sweep on QASPER (selector disabled).
# Left panel: vary T at C=4 (fixed memory budget). Right panel: vary C at T=128.
# usage: bash scripts/granularity_sweep.sh qasper_1b
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
S="$(dirname "${BASH_SOURCE[0]}")"; CFG="$1"
for T in ${SEGMENT_LENGTHS:-64 128 256 512 1024}; do
  M=$((T / 4))
  RUN_DIR="$SEDEM_OUTPUT_DIR/$(basename "${CFG%.sh}")_T${T}_C4" CONFIG_OVERRIDES="MEMORY_SLOTS=$M COMPRESSION_FACTOR=4 EVAL_SELECTION_MODE=off" \
    SKIP_SELECTOR=1 bash "$S/run_pipeline.sh" "$CFG"
done
for C in ${COMPRESSION_FACTORS:-2 8 16}; do
  M=$((128 / C))
  RUN_DIR="$SEDEM_OUTPUT_DIR/$(basename "${CFG%.sh}")_T128_C${C}" CONFIG_OVERRIDES="MEMORY_SLOTS=$M COMPRESSION_FACTOR=$C EVAL_SELECTION_MODE=off" \
    SKIP_SELECTOR=1 bash "$S/run_pipeline.sh" "$CFG"
done
