#!/usr/bin/env bash
# Table 4 ablation ("No decomp."): no decompressor and no injection; the memory slots are prepended
# to the decoder input as a soft prefix. ARM=linear (default) or ARM=mlp selects the compressor projection.
# usage: ARM=linear bash scripts/ablation_no_decompression.sh <config>
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ARM="${ARM:-linear}"
case "$ARM" in linear) LP=True ;; mlp) LP=False ;; *) echo "ARM must be linear|mlp" >&2; exit 2 ;; esac
export RUN_DIR="${RUN_DIR:-$SEDEM_OUTPUT_DIR/$(basename "${1%.sh}")_nodecomp_${ARM}}"
export CONFIG_OVERRIDES="INJECT_DECOMPRESSED=False LINEAR_PROJECTION=$LP INJECTION_LAYER=-999 \
STAGE1_CTX_TOKEN_LOSS_WEIGHT=0.0 STAGE1_REC_LOSS_WEIGHT=0.0 STAGE1_DISTILL_LOSS_WEIGHT=0.0 \
STAGE2_CTX_TOKEN_LOSS_WEIGHT=0.0 STAGE2_REC_LOSS_WEIGHT=0.0 STAGE2_DISTILL_LOSS_WEIGHT=0.0 \
EVAL_HARNESS=qa EVAL_SELECTION_MODE=off ${CONFIG_OVERRIDES:-}"
SKIP_SELECTOR=1 bash "$(dirname "${BASH_SOURCE[0]}")/run_pipeline.sh" "$1"
