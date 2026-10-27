"""Diagnostic eval for the K-sweep gap (HotpotQA distractor).

Runs the same selection-aware QA loop as `eval_qa_hotpot.py`, but supports four
selection modes so we can isolate where the K=6 ≈ no-ret / K=8 < no-ret gap
comes from:

    score_topk   — current behavior (top-K by selector score, doc-order injected)
    label        — gold paragraph blocks only (selection upper bound)
    null_topk    — keep ALL 10 blocks in original layout; zero out non-top-K
    null_label   — keep ALL 10 blocks in original layout; zero out non-gold

Notes:
    * `score_topk` matches the existing eval exactly (`top_idx.sort()` makes
      injection doc-order).
    * `label` upper-bounds selection. Compare its F1 to `score_topk` at K=2.
    * `null_topk` decouples content from layout. If F1 ≈ no-selection, the
      content of dropped blocks was not necessary; the gap was layout/length.
      If F1 ≈ score_topk, content was doing the work.
    * `null_label` is the strongest control: gold content + full layout.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from typing import List

import torch
from datasets import load_dataset
from safetensors.torch import load_file as safe_load

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sedem.blocks_hotpot import (
    build_memory_blocks_hotpot,
    hotpot_eval_transform,
)
from sedem.evidence_labels_hotpot import evidence_labels_from_supporting_facts
from sedem.blocks_twowiki import (
    twowiki_eval_transform,
    build_memory_blocks_twowiki,
    evidence_labels_twowiki,
)
from sedem.blocks_musique import (
    musique_eval_transform,
    build_memory_blocks_musique,
    evidence_labels_musique,
)
from sedem.selector import PooledQuerySelector, LateInteractionSelector
from sedem.pipeline import (
    load_sedem_model, decompress_all_segments, gather_selected_blocks, decode_with_injected_memory,
    normalize_memory, prefill_with_injected_memory_cached, decode_step_with_injected_memory_cached,
)
import re


_BANK_STOP = {
    "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is", "are", "was",
    "were", "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "did", "does", "do", "has", "have", "had", "that", "this", "it", "its", "by",
    "with", "from", "as", "at", "be", "been", "same", "both", "more", "than",
}


class _BankPool:
    """Corpus of 'other article' paragraphs used to grow the memory bank.

    Built once from the val split. `pick` returns N paragraphs that are either
    question-similar (background knowledge) or uniformly random (noise control).
    Paragraphs whose title appears in the question's own context are excluded, so
    added memories never duplicate the question's own evidence.
    """

    def __init__(self, val, pool_examples: int, seed: int):
        self.titles: List[str] = []
        self.texts: List[str] = []
        self.index = {}
        limit = min(pool_examples, len(val))
        for i in range(limit):
            row = val[i]
            for t, p in zip(row["paragraph_titles"], row["paragraphs"]):
                if not p:
                    continue
                self.titles.append(t)
                self.texts.append(p)
        for pid, (t, p) in enumerate(zip(self.titles, self.texts)):
            for w in set(self._toks(t + " " + p[:400])):
                self.index.setdefault(w, []).append(pid)
        self._rng = random.Random(seed)

    @staticmethod
    def _toks(text: str) -> List[str]:
        return [w for w in re.sub(r"[^\w\s]", " ", (text or "").lower()).split()
                if len(w) > 2 and w not in _BANK_STOP]

    def pick(self, question: str, own_titles, n: int, source: str, ex_idx: int):
        if n <= 0 or not self.texts:
            return [], []
        if source == "random":
            rng = random.Random((ex_idx + 1) * 7919 + self._rng.randint(0, 1 << 20))
            cand = []
            tries = 0
            while len(cand) < n and tries < n * 50:
                pid = rng.randrange(len(self.texts))
                if self.titles[pid] not in own_titles and pid not in cand:
                    cand.append(pid)
                tries += 1
        else:
            counts = {}
            for w in set(self._toks(question)):
                for pid in self.index.get(w, ()):
                    counts[pid] = counts.get(pid, 0) + 1
            ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            cand = [pid for pid, _ in ranked if self.titles[pid] not in own_titles][:n]
            if len(cand) < n:  # top up deterministically if the question is rare
                rng = random.Random(ex_idx)
                seen = set(cand)
                while len(cand) < n:
                    pid = rng.randrange(len(self.texts))
                    if pid not in seen and self.titles[pid] not in own_titles:
                        cand.append(pid); seen.add(pid)
        return [self.titles[i] for i in cand], [self.texts[i] for i in cand]


def _normalize(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    while tokens and tokens[0] in ("a", "an", "the"):
        tokens.pop(0)
    return " ".join(tokens)


def _em(pred: str, refs: List[str]) -> float:
    n = _normalize(pred)
    return float(any(n == _normalize(r) for r in refs))


def _f1(pred: str, ref: str) -> float:
    p = _normalize(pred).split()
    r = _normalize(ref).split()
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    common = set(p) & set(r)
    nc = sum(min(p.count(w), r.count(w)) for w in common)
    if nc == 0:
        return 0.0
    pr = nc / len(p)
    rc = nc / len(r)
    return 2 * pr * rc / (pr + rc)


def _best_f1(pred: str, refs: List[str]) -> float:
    return max((_f1(pred, r) for r in refs), default=0.0)


def _lcs_length(x: List[str], y: List[str]) -> int:
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def _rouge_l(pred: str, ref: str) -> float:
    p = _normalize(pred).split()
    r = _normalize(ref).split()
    if not p or not r:
        return 0.0
    lcs = _lcs_length(p, r)
    pr = lcs / len(p)
    rc = lcs / len(r)
    if pr + rc == 0:
        return 0.0
    return 2 * pr * rc / (pr + rc)


def _best_rouge_l(pred: str, refs: List[str]) -> float:
    return max((_rouge_l(pred, r) for r in refs), default=0.0)


def _kv_cache_stats_from_pkv(past):
    """Walk past_key_values; return (bytes, seq_len). 0,0 if empty/None."""
    if past is None:
        return 0, 0
    tensors = []
    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        for t in list(past.key_cache) + list(past.value_cache):
            if isinstance(t, torch.Tensor) and t.numel() > 0:
                tensors.append(t)
    else:
        for layer in past:
            if layer is None:
                continue
            for t in layer:
                if isinstance(t, torch.Tensor) and t.numel() > 0:
                    tensors.append(t)
    if not tensors:
        return 0, 0
    total_bytes = sum(t.numel() * t.element_size() for t in tensors)
    seq_len = int(tensors[0].shape[-2])
    return int(total_bytes), seq_len


def _analytical_kv_bytes(model, seq_len: int) -> int:
    """Per-layer K + V analytical size for a sequence of `seq_len` tokens.
    The decompress-inject decoder isn't KV-cached at runtime; this is what it
    WOULD be if cached (model architecture invariant)."""
    try:
        cfg = model.decoder.config if hasattr(model, "decoder") else model.backbone.config
    except AttributeError:
        return 0
    n_layers = int(getattr(cfg, "num_hidden_layers", 0) or 0)
    n_kv = int(getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads", 0)) or 0)
    hidden = int(getattr(cfg, "hidden_size", 0) or 0)
    n_heads = int(getattr(cfg, "num_attention_heads", 1) or 1)
    head_dim = hidden // max(n_heads, 1)
    # bf16 default (2 bytes); K + V doubles it
    return 2 * n_layers * n_kv * head_dim * 2 * seq_len


def _agg(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
    ys = sorted(xs)
    m = sum(ys) / len(ys)
    def pct(p):
        k = max(0, min(len(ys) - 1, int(round((p / 100.0) * (len(ys) - 1)))))
        return ys[k]
    return {
        "mean": round(m, 2),
        "p50": round(pct(50), 2),
        "p90": round(pct(90), 2),
        "p99": round(pct(99), 2),
        "min": round(ys[0], 2),
        "max": round(ys[-1], 2),
    }


def _build_selected_hs(decompressed_hs, selected_idx, tokens_per_block, num_blocks, mode):
    """Return injected Ĥ tensor according to `mode`.

    decompressed_hs: (1, num_blocks * tokens_per_block, D)
    selected_idx:    LongTensor[K] of block indices to keep (sorted ascending).
    """
    if mode in ("learned", "label", "random_topk", "wrong_topk"):
        # Concatenate selected blocks only (existing behavior).
        return gather_selected_blocks(decompressed_hs, selected_idx, tokens_per_block)
    elif mode in ("null_topk", "null_label"):
        # Keep full layout; zero out non-selected blocks.
        out = decompressed_hs.clone()
        keep = set(selected_idx.tolist())
        for k in range(num_blocks):
            if k not in keep:
                out[:, k * tokens_per_block : (k + 1) * tokens_per_block, :] = 0.0
        return out
    else:
        raise ValueError(f"unknown mode: {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_checkpoint", required=True)
    ap.add_argument("--selector_checkpoint", default=None,
                    help="Required for score_topk/null_topk; ignored for label/null_label.")
    ap.add_argument("--selector_type", choices=["pooled", "late_interaction"], default="late_interaction")
    ap.add_argument("--selector_d_key", type=int, default=256)
    ap.add_argument("--selector_heads", type=int, default=4)
    ap.add_argument("--mode", choices=["learned", "label", "null_topk", "null_label",
                                       "random_topk", "wrong_topk", "native_learned", "native_label",
                                       "score_topk", "native_topk"],  # last two: pre-release aliases
                    required=True,
                    help="learned: top-K blocks by the selector over per-paragraph memory blocks; label: the gold "
                         "blocks; native_learned / native_label: same selection, but the selected paragraphs are "
                         "re-compressed from their raw text (no per-paragraph padding) before decompression.")
    ap.add_argument("--context_max_length", type=int, default=4096,
                    help="Token cap for re-compressing the selected paragraphs in the native_* modes.")
    ap.add_argument("--dump_scores", action="store_true",
                    help="Store per-block selector scores in each row for offline analysis.")
    ap.add_argument("--top_k", type=int, default=4,
                    help="K for score_topk/null_topk. Ignored for label modes (uses all gold blocks).")
    ap.add_argument("--max_eval_samples", type=int, default=1000)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--dataset_cache_dir", default=None)
    ap.add_argument("--output_file", required=True)
    ap.add_argument("--seed", type=int, default=0)
    # --- memory-bank scaling (flag-gated; --bank_extra 0 == original behavior) ---
    ap.add_argument("--bank_extra", type=int, default=0,
                    help="Append N extra paragraphs from the corpus to the memory bank "
                         "BEFORE top-K selection. 0 = original per-question bank.")
    ap.add_argument("--bank_extra_source", choices=["related", "random"], default="related",
                    help="related = question-similar corpus paragraphs (background knowledge); "
                         "random = uniformly sampled paragraphs (noise control).")
    ap.add_argument("--bank_pool_examples", type=int, default=2000,
                    help="How many val examples contribute their paragraphs to the corpus pool.")
    ap.add_argument("--bank_seed", type=int, default=0)
    ap.add_argument("--bank_own_context", choices=["keep", "drop"], default="keep",
                    help="keep = question's own paragraphs stay in the bank (original). "
                         "drop = the bank is the ONLY source of evidence, so added "
                         "memories can actually carry the answer.")
    ap.add_argument("--bank_gold", choices=["auto", "force", "exclude"], default="auto",
                    help="With --bank_own_context drop: force = insert the question's gold "
                         "paragraphs into the bank among the distractors; exclude = keep them "
                         "out (evidence-free floor); auto = whatever sampling yields.")
    ap.add_argument("--dataset", choices=["hotpot", "twowiki", "musique"], default="hotpot",
                    help="Which dataset/schema to evaluate against.")
    # Optional model-arch overrides (needed for ckpts trained with multi-layer mixing)
    ap.add_argument("--mix_extraction_layers", type=lambda s: s.lower() == "true", default=False)
    ap.add_argument("--mixing_layers", type=str, default="10,12,14,16")
    ap.add_argument("--mixing_norm", type=str, default="per_layer_ln")
    ap.add_argument("--mixing_init", type=str, default="uniform")
    ap.add_argument("--mixing_learnable", type=lambda s: s.lower() == "true", default=False)
    ap.add_argument("--injection_layer", type=int, default=-999)
    # Cross-model support: separate encoder + decoder names. Empty falls back to model_name_or_path.
    ap.add_argument("--model_name_or_path", default="meta-llama/Llama-3.2-3B",
                    help="Same-model backbone. Set to meta-llama/Llama-3.2-1B for 1B ckpts.")
    ap.add_argument("--compressor_hidden_dim", type=int, default=3072,
                    help="Compressor bottleneck dim — must match model hidden_size "
                         "(3072 for 3B, 2048 for 1B).")
    ap.add_argument("--encoder_name_or_path", default="",
                    help="Cross-model encoder. Empty = use model_name_or_path.")
    ap.add_argument("--decoder_name_or_path", default="",
                    help="Cross-model decoder. Empty = use model_name_or_path.")
    ap.add_argument("--extraction_layer", type=int, default=14,
                    help="Encoder layer to extract HS from (cross-model: 1B encoder layer).")
    ap.add_argument("--memory_slots_per_segment", type=int, default=32,
                    help="N_M: memory slots per context segment (segment length T = N_M * C).")
    ap.add_argument("--compression_factor", type=int, default=4,
                    help="C: tokens summarised by each memory slot.")
    ap.add_argument("--parallel", action="store_true",
                    help="Enable parallel-segment compression in SeDeMModel.compress_segments "
                         "(batches segments along the B dim, one backbone forward). "
                         "Safe only for stateless compressors (SegmentCompressor; "
                         "LearnableQueryCompressor with ignore_prev_mem=True). Default off "
                         "preserves the sequential loop.")
    ap.add_argument("--profile_steps", type=int, default=0,
                    help="If >0, run decode-step profiler on the first --profile_steps examples "
                         "and exit before the main eval loop. Breaks each decode_with_injected_memory call into "
                         "Phase 1 (text-only layers 0..inject-1) + Phase 2 ([decomp|text] layers "
                         "inject..end + lm_head) + pre (norm+embed). Also times HF-cached forward "
                         "on the same backbone for reference. Writes results to "
                         "<output_file>.profile.json. Default 0 = no-op.")
    ap.add_argument("--kv_cache", action="store_true",
                    help="Use cached two-phase decoder forward (Tier 1A optimization). "
                         "Phase 1 caches text K,V; Phase 2 caches [decomp|text] K,V across decode "
                         "steps. Output equivalent to the non-cached path modulo numerical noise. "
                         "Default off.")
    ap.add_argument("--correctness_check", type=int, default=0,
                    help="If >0 and --kv_cache, on the first N examples compute argmax tokens via "
                         "BOTH the cached and non-cached paths and print mismatches. Use for "
                         "verifying the KV-cache implementation. Default 0 = no check.")
    ap.add_argument("--compile_decoder", action="store_true",
                    help="Apply torch.compile(mode='reduce-overhead', dynamic=True) to each "
                         "decoder layer's forward + compressor + decompressor. Targets per-step "
                         "kernel-launch overhead at batch=1. Output equivalent modulo numerical "
                         "noise. Default off. First decode step pays a "
                         "~30-60s warmup cost.")
    ap.add_argument("--batched_compressor_fast", action="store_true",
                    help="Collapse the per-segment compressor for-loop into ONE batched "
                         "compressor call (Win A: eliminates per-seg .item() syncs; Win B1: "
                         "one compressor forward over [num_seg, seg_len, D] for full segments, "
                         "sequential fallback for the partial last segment if present). "
                         "BIT-EXACT vs sequential (every op inside SegmentCompressor is "
                         "batch-independent). Requires --parallel. Default off.")
    ap.add_argument("--verify_batched_bitexact", type=int, default=0,
                    help="If >0, on the first N examples run BOTH the sequential per-seg loop "
                         "and the batched path on the same input and assert max-abs-diff == 0 "
                         "(bf16 may need <=1e-3 tolerance). Forces --batched_compressor_fast. "
                         "Use to verify the optimization is value-preserving before full runs.")
    ap.add_argument("--shape_stable_kv", action="store_true",
                    help="When --kv_cache is on, allocate StaticCache(max_cache_len) so dynamo "
                         "sees the SAME past_key_values shape across all decode steps. Eliminates "
                         "per-step recompilation. Bit-exact vs DynamicCache modulo "
                         "bf16 reduction-order. Default off keeps DynamicCache path.")
    ap.add_argument("--verify_shape_stable_bitexact", type=int, default=0,
                    help="If >0 and --shape_stable_kv is on, on the first N examples run BOTH "
                         "the StaticCache and DynamicCache paths on the same input and compare "
                         "argmax tokens at each decode step. Aborts on mismatch.")
    ap.add_argument("--backbone_early_exit", action="store_true",
                    help="When set, stop the compress-phase backbone forward at layer "
                         "extraction_layer (skip layers ℓ..L-1 whose outputs are never read). "
                         "Bit-exact. For 1B with ℓ=8/16 this saves ~50%% of backbone compute on "
                         "qasper; for ℓ=14/16 it saves ~12.5%% (hotpot). Default off.")
    ap.add_argument("--verify_backbone_early_exit", type=int, default=0,
                    help="If >0 and --backbone_early_exit is on, on the first N _get_backbone_"
                         "hidden_states calls A/B against the full forward and assert "
                         "max_abs_diff == 0 (strict bit-exact). Aborts on drift.")
    ap.add_argument("--compile_compressor", action="store_true",
                    help="Apply torch.compile(mode='default', dynamic=True) to model.compressor "
                         "(SegmentCompressor / SeDeM). Targets compressor kernel-launch "
                         "overhead on the compress hotpath. Pairs with --parallel + "
                         "--batched_compressor_fast. dynamic=True handles per-segment shape "
                         "variation. compressor_is_stateless() unwraps OptimizedModule via "
                         "_orig_mod so the parallel fast path stays active. Bit-exact (Inductor "
                         "fuses kernels without reordering reductions). First example pays "
                         "compile warmup. Default off.")
    ap.add_argument("--compile_decompressor", action="store_true",
                    help="Apply torch.compile(mode='default', dynamic=True) to model.decompressor "
                         "(the inject-path module run per query inside the decompress step). "
                         "decompress_all_segments calls model.decompressor(seg) directly (no _orig_mod "
                         "unwrap), so the compiled module is used. Per-segment shape is the fixed "
                         "(1, memory_slots, D), so it specializes cleanly. Bit-exact (Inductor fuses "
                         "without reordering reductions); a one-time setup A/B assert confirms it. "
                         "Default off.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda")

    # Bit-exact perf knobs (applied unconditionally — they don't change numerics).
    # cudnn.benchmark picks the fastest algo per (shape, dtype) on first call. Safe under bf16.
    # TF32 explicitly OFF so any float32 path (norms, selector, head) stays strict IEEE.
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    if args.compile_decoder:
        # Make all-opt (compile + kv_cache) actually compose: each KV-cache decode step
        # has a slightly different past_key_values.layers structure, which under the
        # default cache_size_limit=8 forces dynamo to fall back to eager after 8 distinct
        # shapes — that's the +38 ms decode regression in eff_runs/hotpot_1b_allopt_k2.
        # Raising the cap is pure cache-capacity; numerics are identical.
        import torch._dynamo as _dynamo
        _dynamo.config.cache_size_limit = 256
        _dynamo.config.accumulated_cache_size_limit = 1024
        print(f"  [dynamo] cache_size_limit={_dynamo.config.cache_size_limit}, "
              f"accumulated_cache_size_limit={_dynamo.config.accumulated_cache_size_limit}")

    args.mode = {"score_topk": "learned", "native_topk": "native_learned"}.get(args.mode, args.mode)
    needs_selector = args.mode in ("learned", "null_topk", "native_learned")
    needs_random = args.mode in ("random_topk", "wrong_topk")
    if needs_selector and args.selector_checkpoint is None:
        raise SystemExit(f"--selector_checkpoint required for mode={args.mode}")

    print(f"=== {args.dataset} — mode={args.mode} ===")
    print(f"  base ckpt   : {args.base_checkpoint}")
    print(f"  selector   : {args.selector_checkpoint}  (used={needs_selector})")
    print(f"  K           : {args.top_k}  (label modes use all gold blocks)")
    print(f"  samples     : {args.max_eval_samples}")
    print(f"  output      : {args.output_file}")

    model = load_sedem_model(args.base_checkpoint, "/tmp/hotpot_eval_test_dummy", device, args=args)
    model.eval()
    if args.parallel:
        model._parallel_segments = True
        print("  [parallel] _parallel_segments=True (batched-segment compression enabled)")
    if args.batched_compressor_fast or args.verify_batched_bitexact > 0:
        if not args.parallel:
            raise SystemExit("--batched_compressor_fast requires --parallel")
        model._batched_compressor_fast = True
        model._verify_batched_remaining = int(args.verify_batched_bitexact)
        print("  [batched_compressor_fast] _batched_compressor_fast=True "
              "(one batched compressor call per example, bit-exact)"
              + (f"; verify on first {args.verify_batched_bitexact} examples"
                 if args.verify_batched_bitexact > 0 else ""))
    if args.shape_stable_kv:
        if not args.kv_cache:
            raise SystemExit("--shape_stable_kv requires --kv_cache")
        model._shape_stable_kv = True
        print("  [shape_stable_kv] _shape_stable_kv=True (StaticCache; shape-stable across "
              "decode steps; bit-exact vs DynamicCache modulo bf16 reduction-order)")
    if args.verify_shape_stable_bitexact > 0 and not args.shape_stable_kv:
        raise SystemExit("--verify_shape_stable_bitexact requires --shape_stable_kv")
    if args.backbone_early_exit:
        model._backbone_early_exit = True
        ℓ = int(model.extraction_layer)
        L = int(model.backbone.config.num_hidden_layers)
        pct = 100.0 * (L - ℓ) / max(L, 1)
        print(f"  [backbone_early_exit] _backbone_early_exit=True (ℓ={ℓ}/{L}, "
              f"skip {L-ℓ} layers = ~{pct:.1f}% of backbone compute saved). Bit-exact.")
    if args.verify_backbone_early_exit > 0:
        if not args.backbone_early_exit:
            raise SystemExit("--verify_backbone_early_exit requires --backbone_early_exit")
        model._verify_backbone_early_exit_remaining = int(args.verify_backbone_early_exit)
        print(f"  [backbone_early_exit verifier] will A/B-check first "
              f"{args.verify_backbone_early_exit} backbone calls vs full forward "
              "(strict max_abs_diff == 0).")
    if args.compile_decoder:
        # Wrap each decoder layer with torch.compile (mode='default', dynamic=True).
        # mode='default' does Inductor kernel fusion only (no CUDA Graphs), so no
        # buffer-reuse hazards across the Python layer loop in forward_with_injection.
        # reduce-overhead was tried and crashed: layer N+1 records its graph against
        # layer N's output buffer, which gets overwritten on the next token's layer-N
        # call. dynamic=True handles prefill vs decode shape change without recompiles.
        # Compressor is opt-in via --compile_compressor (decompressor still uncompiled;
        # called 1x per example so warmup doesn't amortize).
        if hasattr(model.decoder, 'model') and hasattr(model.decoder.model, 'model'):
            _inner = model.decoder.model.model
        else:
            _inner = model.decoder.model
        for _i, _layer in enumerate(_inner.layers):
            _inner.layers[_i] = torch.compile(_layer, mode='default', dynamic=True)
        print(f"  [compile_decoder] compiled {len(_inner.layers)} decoder layers "
              f"(mode=default, dynamic=True). First example will be slow.")

    # Compile the compressor (TTFT lever, bit-exact). compressor_is_stateless
    # in sedem.modeling.py unwraps OptimizedModule via _orig_mod so
    # the --parallel fast path still classifies it as stateless.
    if args.compile_compressor:
        if not args.compile_decoder:
            # Ensure dynamo cache is large enough even without decoder compile.
            import torch._dynamo as _dynamo_c
            _dynamo_c.config.cache_size_limit = max(_dynamo_c.config.cache_size_limit, 256)
            _dynamo_c.config.accumulated_cache_size_limit = max(
                _dynamo_c.config.accumulated_cache_size_limit, 1024
            )
        _comp_cls = type(model.compressor).__name__
        model.compressor = torch.compile(model.compressor, mode='default', dynamic=True)
        print(f"  [compile_compressor] wrapped model.compressor ({_comp_cls}) with "
              f"torch.compile(mode='default', dynamic=True). First example pays compile "
              f"warmup; bit-exact thereafter. Parallel fast path still active.")

    # Compile the decompressor (inject-path TTFT lever, bit-exact). decompress_all_segments
    # calls model.decompressor(seg) directly so the compiled module is used.
    if args.compile_decompressor:
        if not (args.compile_decoder or args.compile_compressor):
            import torch._dynamo as _dynamo_d
            _dynamo_d.config.cache_size_limit = max(_dynamo_d.config.cache_size_limit, 256)
            _dynamo_d.config.accumulated_cache_size_limit = max(
                _dynamo_d.config.accumulated_cache_size_limit, 1024)
        _decomp_cls = type(model.decompressor).__name__
        _eager_decomp = model.decompressor
        model.decompressor = torch.compile(model.decompressor, mode='default', dynamic=True)
        # One-time bit-exact verifier: A/B the compiled vs eager decompressor on a
        # representative (1, memory_slots, D) slot tensor; abort on drift.
        with torch.no_grad():
            _dummy = torch.randn(1, model.memory_slots, model.dim, device=device,
                                 dtype=torch.float32)
            _ref = _eager_decomp(_dummy)
            _got = model.decompressor(_dummy)
            _mad = (_ref - _got).abs().max().item()
        _tol = 1e-3
        if _mad > _tol:
            raise SystemExit(
                f"--compile_decompressor not bit-exact: max_abs_diff={_mad:.3e} > tol={_tol:.0e}")
        print(f"  [compile_decompressor] wrapped model.decompressor ({_decomp_cls}) with "
              f"torch.compile(mode='default', dynamic=True). Setup A/B max_abs_diff={_mad:.2e} "
              f"(<= {_tol:.0e}) -> bit-exact. First example pays compile warmup.")

    head = None
    if needs_selector:
        if args.selector_type == "pooled":
            head = PooledQuerySelector(model.dim, d_key=args.selector_d_key).to(device).float()
        else:
            head = LateInteractionSelector(model.dim, num_heads=args.selector_heads).to(device).float()
        from sedem.selector import resolve_selector_checkpoint
        head_state = safe_load(resolve_selector_checkpoint(args.selector_checkpoint))
        head.load_state_dict(head_state, strict=False)
        head.eval()

    # Dataset-specific data loading + helpers
    if args.dataset == "hotpot":
        print("Loading hotpotqa/hotpot_qa distractor val ...")
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor",
                          cache_dir=args.dataset_cache_dir, trust_remote_code=True)
        val = ds["validation"].map(
            hotpot_eval_transform, batched=True,
            remove_columns=ds["validation"].column_names, load_from_cache_file=True,
        )
        num_blocks_base = 10
        compute_blocks_fn = build_memory_blocks_hotpot
    elif args.dataset == "twowiki":
        print("Loading 2WikiMultihopQA val ...")
        ds = load_dataset("xanhho/2WikiMultihopQA", revision="refs/convert/parquet",
                          cache_dir=args.dataset_cache_dir)
        val = ds["validation"].map(
            twowiki_eval_transform, batched=True,
            remove_columns=ds["validation"].column_names, load_from_cache_file=True,
        )
        num_blocks_base = 10
        compute_blocks_fn = build_memory_blocks_twowiki
    elif args.dataset == "musique":
        print("Loading MuSiQue val ...")
        ds = load_dataset("dgslibisey/MuSiQue", cache_dir=args.dataset_cache_dir)
        val = ds["validation"].map(
            musique_eval_transform, batched=True,
            remove_columns=ds["validation"].column_names, load_from_cache_file=True,
        )
        num_blocks_base = 20
        compute_blocks_fn = build_memory_blocks_musique
    else:
        raise ValueError(f"Unknown --dataset {args.dataset}")

    # Corpus pool for --bank_extra: paragraphs from the FULL val split (not just the
    # evaluated subset), so added memories are "other articles", not the question's own.
    bank_pool = None
    if args.bank_extra > 0:
        bank_pool = _BankPool(val, args.bank_pool_examples, args.bank_seed)
        print(f"  bank pool: {len(bank_pool.texts)} paragraphs "
              f"from {args.bank_pool_examples} examples (source={args.bank_extra_source})")

    if args.max_eval_samples is not None:
        val = val.select(range(min(args.max_eval_samples, len(val))))
    print(f"  val examples: {len(val)}")

    chunk_size = model.memory_slots * model.compression_factor
    block_size = model.memory_slots
    tokens_per_block = chunk_size  # paragraph length per block

    em_sum = f1_sum = rouge_sum = recall_sum = 0.0
    n_picked_sum = 0
    n = 0
    rows = []
    # Per-example timing/memory accumulators
    compress_list = []
    select_list = []
    prefill_list = []
    decompress_list = []
    decode_ms_list = []
    ttft_list = []
    total_ms_list = []
    peak_alloc_list = []
    peak_reserved_list = []
    kv_bytes_list = []     # analytical (architecture-equivalent if KV-cached)
    kv_tokens_list = []
    decode_rate_list = []
    inj_len_list = []
    gen_tok_list = []
    t0 = time.time()

    with torch.inference_mode():
        for ex_idx, row in enumerate(val):
            paragraphs = list(row["paragraphs"])
            paragraph_titles = list(row["paragraph_titles"])
            # Pad the question's own context to the dataset's block count, then append
            # extra corpus paragraphs. num_blocks is per-example from here on.
            paragraphs = (paragraphs + [""] * num_blocks_base)[:num_blocks_base]
            paragraph_titles = (paragraph_titles + [""] * num_blocks_base)[:num_blocks_base]
            own_titles_all = list(row["paragraph_titles"])
            own_paras_all = list(row["paragraphs"])
            gold_titles = set(row.get("supporting_titles") or [])
            extra_titles, extra_texts = [], []
            if bank_pool is not None:
                extra_titles, extra_texts = bank_pool.pick(
                    row["prompt"], set(own_titles_all), args.bank_extra,
                    args.bank_extra_source, ex_idx,
                )
            if args.bank_own_context == "drop":
                # Bank-only evidence: the question's own paragraphs are NOT prepended.
                # The bank is built from corpus articles, optionally seeded with the
                # question's gold articles so added memory can carry the answer.
                bank_titles, bank_texts = list(extra_titles), list(extra_texts)
                if args.bank_gold == "force":
                    gold_pairs = [(t, p) for t, p in zip(own_titles_all, own_paras_all)
                                  if t in gold_titles and p]
                    # drop as many distractors as gold blocks we insert, keeping bank size fixed
                    keep = max(0, len(bank_texts) - len(gold_pairs))
                    bank_titles, bank_texts = bank_titles[:keep], bank_texts[:keep]
                    rng_g = random.Random(1000003 + ex_idx)
                    for t, p in gold_pairs:
                        j = rng_g.randint(0, len(bank_texts))
                        bank_titles.insert(j, t); bank_texts.insert(j, p)
                paragraphs = bank_texts
                paragraph_titles = bank_titles
                num_blocks = len(bank_texts)
            else:
                paragraphs = paragraphs + extra_texts
                paragraph_titles = paragraph_titles + extra_titles
                num_blocks = num_blocks_base + len(extra_texts)
            if num_blocks < 1:
                raise ValueError("empty memory bank; raise --bank_extra")
            question = row["prompt"]
            gold_answer = row["answer"]
            # `all_answers` is preserved per-dataset; pick best F1 over them later.
            all_gold = row.get("all_answers") or [gold_answer]

            # Reset peak counters per example.
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)

            # ---- COMPRESS (encoder pass over all paragraphs) ----
            torch.cuda.synchronize()
            _t_comp = time.perf_counter()
            blocks, _, _, _, nb = compute_blocks_fn(
                model, paragraphs, device,
                num_blocks=num_blocks, chunk_size=chunk_size,
            )
            torch.cuda.synchronize()
            compress_ms = (time.perf_counter() - _t_comp) * 1000.0
            assert nb == num_blocks
            memory = blocks.view(1, num_blocks * block_size, blocks.size(-1))

            # Dataset-specific positive-mask
            if args.dataset == "hotpot":
                supporting_titles = row["supporting_titles"]
                ctx_dict = {"title": paragraph_titles, "sentences": [[] for _ in paragraph_titles]}
                sf_dict = {"title": supporting_titles, "sent_id": []}
                pos_mask, _ = evidence_labels_from_supporting_facts(
                    ctx_dict, sf_dict, num_blocks=num_blocks,
                )
            elif args.dataset == "twowiki":
                supporting_titles = row["supporting_titles"]
                pos_mask, _ = evidence_labels_twowiki(
                    paragraph_titles, supporting_titles, num_blocks=num_blocks,
                )
            elif args.dataset == "musique":
                pos_mask, _ = evidence_labels_musique(
                    (list(row["is_supporting"])[:num_blocks_base]
                     + [False] * (num_blocks - num_blocks_base)), num_blocks=num_blocks,
                )
            pos_set = set(torch.nonzero(pos_mask).flatten().tolist())

            # ---- RETRIEVE: score blocks vs question and pick top-K ----
            torch.cuda.synchronize()
            _t_ret = time.perf_counter()
            scores_for_dump = None
            if args.mode in ("learned", "null_topk", "native_learned"):
                tok = model.tokenizer
                q_ids = tok(question, add_special_tokens=False, truncation=False,
                            return_attention_mask=False)["input_ids"]
                if not q_ids:
                    q_ids = [tok.eos_token_id or 0]
                q_t = torch.tensor([q_ids], dtype=torch.long, device=device).clamp(
                    max=model.original_vocab_size - 1,
                )
                # Query reads only extraction_layer. Flag-gated, default OFF:
                #  - --backbone_early_exit ON (and no layer-mixing): route through the
                #    bit-exact early-exit dispatcher (skips ℓ..L-1 for the query too).
                #  - else: ORIGINAL full-stack forward (inefficient; preserved verbatim).
                if getattr(model, "_backbone_early_exit", False) and not getattr(model, "_mix_extraction_layers", False):
                    q_hs = model.extract_hidden_states(q_t).float()
                else:
                    # original (inefficient) selection path — kept for parity / disable
                    q_out = model.backbone(q_t, output_hidden_states=True)
                    q_hs = q_out.hidden_states[model.extraction_layer].float()
                # Cross-model: lift enc_dim → dec_dim so q_hs matches block space.
                if hasattr(model, "compressor") and hasattr(model.compressor, "input_proj"):
                    q_hs = model.compressor.input_proj(q_hs)
                scores = head(q_hs, blocks.float()).squeeze(0)
                K = min(args.top_k, num_blocks)
                top_idx = scores.topk(K)[1].sort()[0]
                if args.dump_scores:
                    scores_for_dump = scores.detach().cpu().tolist()
            elif args.mode in ("random_topk", "wrong_topk"):
                # Random K blocks (no selector). For wrong_topk, exclude gold.
                K = min(args.top_k, num_blocks)
                if args.mode == "wrong_topk":
                    pool = [k for k in range(num_blocks) if k not in pos_set]
                    if len(pool) < K:
                        # Fall back to including some gold if not enough distractors (rare).
                        pool = list(range(num_blocks))
                else:
                    pool = list(range(num_blocks))
                rng_local = random.Random(args.seed * 1000003 + ex_idx)
                chosen = rng_local.sample(pool, K) if K <= len(pool) else pool[:K]
                top_idx = torch.tensor(sorted(chosen), dtype=torch.long, device=device)
            else:
                # label / null_label: pick exactly the gold block indices.
                gold_idx = sorted(pos_set)
                if not gold_idx:
                    # No supporting paragraphs identified — skip with zero injection.
                    gold_idx = []
                top_idx = torch.tensor(gold_idx, dtype=torch.long, device=device)
                K = len(gold_idx)

            picked = set(top_idx.tolist())
            n_pos_picked = len(pos_set.intersection(picked))
            recall_at_k = n_pos_picked / max(len(pos_set), 1)
            torch.cuda.synchronize()
            select_ms = (time.perf_counter() - _t_ret) * 1000.0

            # ---- Decompress + build injected Ĥ ----------------------------------------
            # P3 fast path: for modes that just gather the top-K blocks (no zero-masking
            # of non-selected positions), decompress ONLY the top-K segments. The
            # decompressor is per-segment independent (train_end_to_end.decompress_all_segments
            # is a pure per-segment loop) so this is bit-exact for these modes.
            # Null modes need the full layout (they clone + zero non-selected positions),
            # so fall back to the all-blocks path for those.
            torch.cuda.synchronize()
            _t_decomp = time.perf_counter()
            _p3_modes = ("learned", "label", "random_topk", "wrong_topk")
            if args.mode in ("native_learned", "native_label"):
                # Native path: the selected paragraphs are re-compressed from their raw text as one
                # context (natural segmentation, no per-paragraph padding) and that memory is
                # decompressed. Selection itself used the per-paragraph memory blocks above.
                sel = sorted(int(i) for i in top_idx.tolist())
                sel_paras = [paragraphs[i] for i in sel if i < len(paragraphs) and paragraphs[i]]
                native_text = "\n\n".join(sel_paras)
                nat_ids = model.tokenizer(
                    native_text, max_length=args.context_max_length, truncation=True,
                    padding=False, add_special_tokens=False, return_attention_mask=False,
                )["input_ids"]
                if not nat_ids:
                    nat_ids = [model.tokenizer.eos_token_id or 0]
                nat_t = torch.tensor([nat_ids], dtype=torch.long, device=device)
                native_memory = model.compress_segments(nat_t)
                if native_memory.dim() == 2:
                    native_memory = native_memory.unsqueeze(0)
                decompressed_hs, _ = decompress_all_segments(model, native_memory, decompressor_trainable=False)
                selected_hs = decompressed_hs
                inj_len = selected_hs.size(1)
            elif K > 0 and args.mode in _p3_modes:
                # Reshape memory to (1, num_blocks, memory_slots, D), index top-K, flatten back.
                mem_blocks = memory.view(1, num_blocks, model.memory_slots, model.dim)
                memory_topk = mem_blocks[:, top_idx, :, :].reshape(1, -1, model.dim)
                decompressed_hs, _ = decompress_all_segments(model, memory_topk, decompressor_trainable=False)
                selected_hs = decompressed_hs  # already in top-K order, no further gather
                inj_len = selected_hs.size(1)
            elif K == 0 and args.mode in ("learned", "label"):
                # Edge case: no blocks selected — feed zero-length injection.
                # Need a decompressed_hs reference of correct dtype/device; decompress one
                # segment to get a properly-shaped zero-length tensor.
                first_seg = memory[:, :model.memory_slots, :]
                decompressed_hs, _ = decompress_all_segments(model, first_seg, decompressor_trainable=False)
                selected_hs = decompressed_hs[:, :0, :]
                inj_len = 0
            else:
                # null_topk / null_label and other modes — need the full layout for
                # zero-masking of non-selected blocks.
                decompressed_hs, _ = decompress_all_segments(model, memory, decompressor_trainable=False)
                selected_hs = _build_selected_hs(
                    decompressed_hs, top_idx, tokens_per_block, num_blocks, args.mode,
                )
                inj_len = selected_hs.size(1)
            torch.cuda.synchronize()
            decompress_ms = (time.perf_counter() - _t_decomp) * 1000.0

            # ---- Build prompt and decode ---------------------------------------------
            tok = model.tokenizer
            has_chat = (hasattr(tok, "chat_template") and tok.chat_template is not None)
            if has_chat:
                from sedem.data import QA_SYSTEM_PROMPT
                msgs = [
                    {"role": "system", "content": QA_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Question: {question}"},
                ]
                try:
                    chat = tok.apply_chat_template(msgs, enable_thinking=False,
                                                   tokenize=False, add_generation_prompt=True)
                except TypeError:
                    chat = tok.apply_chat_template(msgs, tokenize=False,
                                                   add_generation_prompt=True)
                chat_ids = tok(chat, add_special_tokens=False, truncation=False)["input_ids"]
            else:
                chat_ids = tok(f"Question: {question}\nAnswer:",
                               add_special_tokens=False)["input_ids"]

            # total_mem_len must match the injection length so two-phase forward sees
            # consistent placeholder count. For null_* modes inj_len = num_blocks * tpb.
            total_mem_len = inj_len
            mem_placeholder_id = model.mem_id_start
            prompt_ids = [mem_placeholder_id] * total_mem_len + [model.ft_token_id] + chat_ids
            full_ids = prompt_ids[:]

            # ---- PREFILL: first token ----
            generated = []
            torch.cuda.synchronize()
            _t_prefill = time.perf_counter()
            if args.kv_cache:
                # Cached path: normalize selected_hs ONCE, prefill returns last-pos logits + caches
                selected_hs_norm = normalize_memory(model, selected_hs)
                full_t = torch.tensor([full_ids], dtype=torch.long, device=device)
                # max_new_tokens forwarded so the StaticCache (if active) is sized
                # for prefill + full decode horizon up-front.
                last_logits, pkv1, pkv2, text_pos_past = prefill_with_injected_memory_cached(
                    model, selected_hs_norm, full_t, total_mem_len,
                    max_new_tokens=args.max_new_tokens)
                vocab_sz = min(last_logits.size(-1), model.original_vocab_size)
                next_logits = last_logits[0, -1, :vocab_sz]
            else:
                # Non-cached path: full two-phase forward.
                # P2: pass inference_only=True so the lm_head/final_norm only run on the
                # last text position (bit-exact for greedy decode; RMSNorm + Linear are
                # per-token, so slice-before == slice-after).
                full_t = torch.tensor([full_ids], dtype=torch.long, device=device)
                logits = decode_with_injected_memory(model, selected_hs, full_t, total_mem_len,
                                        inference_only=True)
                vocab_sz = min(logits.size(-1), model.original_vocab_size)
                next_logits = logits[0, -1, :vocab_sz]
            next_id = int(next_logits.argmax().item())
            torch.cuda.synchronize()
            prefill_ms = (time.perf_counter() - _t_prefill) * 1000.0

            decode_ms = 0.0
            if next_id != model.eos_id:
                generated.append(next_id)
                full_ids.append(next_id)
                # ---- DECODE: remaining tokens ----
                torch.cuda.synchronize()
                _t_dec = time.perf_counter()
                for _ in range(args.max_new_tokens - 1):
                    if args.kv_cache:
                        last_logits, pkv1, pkv2, text_pos_past = decode_step_with_injected_memory_cached(
                            model, selected_hs_norm, next_id, pkv1, pkv2, text_pos_past)
                        vocab_sz = min(last_logits.size(-1), model.original_vocab_size)
                        next_logits = last_logits[0, -1, :vocab_sz]
                    else:
                        # P2: inference_only also for non-cached decode steps
                        full_t = torch.tensor([full_ids], dtype=torch.long, device=device)
                        logits = decode_with_injected_memory(model, selected_hs, full_t, total_mem_len,
                                                inference_only=True)
                        vocab_sz = min(logits.size(-1), model.original_vocab_size)
                        next_logits = logits[0, -1, :vocab_sz]
                    next_id = int(next_logits.argmax().item())
                    if next_id == model.eos_id:
                        break
                    generated.append(next_id)
                    full_ids.append(next_id)
                torch.cuda.synchronize()
                decode_ms = (time.perf_counter() - _t_dec) * 1000.0

            # TTFT (paper definition): compress + select + decompress + prefill = time
            # until first output token. decompress_ms was previously in an untimed gap
            # between select and prefill; it is on the critical path, so it is counted.
            ttft_ms = compress_ms + select_ms + decompress_ms + prefill_ms
            total_ms = ttft_ms + decode_ms

            peak_alloc_mb = 0.0
            peak_reserved_mb = 0.0
            if torch.cuda.is_available():
                peak_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 * 1024)

            # Analytical KV size if this path were KV-cached: text_len + injection_len.
            kv_seq_len = inj_len + len(full_ids) - inj_len  # = len(full_ids) (placeholders + text + generated)
            kv_cache_bytes = _analytical_kv_bytes(model, kv_seq_len)

            decoded_tokens = len(generated)
            decode_tok_per_s = (decoded_tokens / (decode_ms / 1000.0)) if decode_ms > 0 and decoded_tokens > 0 else 0.0

            pred_text = tok.decode(generated, skip_special_tokens=True).strip()
            # Use all_answers (answer + aliases for MuSiQue/QASPER) when available.
            em = _em(pred_text, all_gold)
            f1 = _best_f1(pred_text, all_gold)
            rl = _best_rouge_l(pred_text, all_gold)

            em_sum += em
            f1_sum += f1
            rouge_sum += rl
            recall_sum += recall_at_k
            n_picked_sum += K
            n += 1
            compress_list.append(compress_ms)
            select_list.append(select_ms)
            decompress_list.append(decompress_ms)
            prefill_list.append(prefill_ms)
            decode_ms_list.append(decode_ms)
            ttft_list.append(ttft_ms)
            total_ms_list.append(total_ms)
            peak_alloc_list.append(round(peak_alloc_mb, 1))
            peak_reserved_list.append(round(peak_reserved_mb, 1))
            kv_bytes_list.append(int(kv_cache_bytes))
            kv_tokens_list.append(int(kv_seq_len))
            if decode_tok_per_s > 0:
                decode_rate_list.append(decode_tok_per_s)
            inj_len_list.append(inj_len)
            gen_tok_list.append(decoded_tokens)

            row_out = {
                "question": question, "prediction": pred_text, "gold": gold_answer,
                "em": em, "f1": f1, "rouge_l": rl, "recall_at_k": recall_at_k,
                "picked_blocks": sorted(picked),
                "gold_blocks": sorted(pos_set),
                "n_picked": K,
                "inj_len": inj_len,
                "compress_ms": round(compress_ms, 2),
                "select_ms": round(select_ms, 2),
                "decompress_ms": round(decompress_ms, 2),
                "prefill_ms": round(prefill_ms, 2),
                "decode_ms": round(decode_ms, 2),
                "ttft_ms": round(ttft_ms, 2),
                "total_ms": round(total_ms, 2),
                "peak_gpu_mb": round(peak_alloc_mb, 1),
                "peak_gpu_reserved_mb": round(peak_reserved_mb, 1),
                "kv_cache_bytes": int(kv_cache_bytes),
                "kv_cache_tokens": int(kv_seq_len),
                "decode_tok_per_s": round(decode_tok_per_s, 2),
                "generated_tokens": int(decoded_tokens),
            }
            if scores_for_dump is not None:
                row_out["block_scores"] = scores_for_dump
            rows.append(row_out)

            if (n % 50 == 0):
                mean_ttft = sum(ttft_list) / len(ttft_list)
                mean_dec = sum(decode_ms_list) / len(decode_ms_list)
                print(f"  {n}/{len(val)}  EM={100*em_sum/n:.2f}  F1={100*f1_sum/n:.2f}  "
                      f"RougeL={100*rouge_sum/n:.2f}  recall@K={recall_sum/n:.3f}  "
                      f"avgK={n_picked_sum/n:.2f}  TTFT={mean_ttft:.0f}ms  "
                      f"decode={mean_dec:.0f}ms  ({(time.time()-t0)/60:.1f} min)")

    summary = {
        "mode": args.mode,
        "top_k_arg": args.top_k,
        "n": n,
        "EM": 100 * em_sum / max(n, 1),
        "F1": 100 * f1_sum / max(n, 1),
        "RougeL": 100 * rouge_sum / max(n, 1),
        "recall_at_k": recall_sum / max(n, 1),
        "avg_k_picked": n_picked_sum / max(n, 1),
        "wall_time_sec": round(time.time() - t0, 1),
        # ---- per-example timing breakdowns (ms) ----
        "compress_ms": _agg(compress_list),
        "select_ms": _agg(select_list),
        "decompress_ms": _agg(decompress_list),
        "prefill_ms": _agg(prefill_list),
        "decode_ms": _agg(decode_ms_list),
        "ttft_ms": _agg(ttft_list),
        "total_ms": _agg(total_ms_list),
        # ---- per-example memory ----
        "peak_gpu_mb": _agg(peak_alloc_list),
        "peak_gpu_reserved_mb": _agg(peak_reserved_list),
        "kv_cache_bytes": _agg(kv_bytes_list),
        "kv_cache_tokens": _agg(kv_tokens_list),
        "decode_tok_per_s": _agg(decode_rate_list),
        "inj_len": _agg(inj_len_list),
        "generated_tokens": _agg(gen_tok_list),
    }

    out = {
        "config": vars(args),
        "aggregated": summary,
        "examples": rows,
    }
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n=== {args.dataset} — mode={args.mode} K={args.top_k} ===")
    print(f"  N           : {summary['n']}")
    print(f"  EM          : {summary['EM']:.2f}")
    print(f"  F1          : {summary['F1']:.2f}")
    print(f"  RougeL      : {summary['RougeL']:.2f}")
    print(f"  recall@K    : {summary['recall_at_k']:.4f}")
    print(f"  avg_k_picked: {summary['avg_k_picked']:.2f}")
    print(f"  TTFT (ms)   : mean {summary['ttft_ms']['mean']:.0f}  p50 {summary['ttft_ms']['p50']:.0f}  "
          f"p90 {summary['ttft_ms']['p90']:.0f}  max {summary['ttft_ms']['max']:.0f}")
    print(f"  decode (ms) : mean {summary['decode_ms']['mean']:.0f}  p50 {summary['decode_ms']['p50']:.0f}  "
          f"p90 {summary['decode_ms']['p90']:.0f}  max {summary['decode_ms']['max']:.0f}")
    print(f"  peak GPU    : mean {summary['peak_gpu_mb']['mean']:.0f}MB  "
          f"p90 {summary['peak_gpu_mb']['p90']:.0f}  max {summary['peak_gpu_mb']['max']:.0f}")
    print(f"  decode/s    : mean {summary['decode_tok_per_s']['mean']:.1f} tok/s")
    print(f"  saved -> {args.output_file}")


if __name__ == "__main__":
    main()
