import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
"""
Per-segment output-erank diagnostic.

For each example, compute the effective rank of the COMPRESSED memory
M_seg in R^{32 x D} per segment:
    erank(M_seg) = exp(-sum_i p_i log p_i),   p_i = sigma_i / sum_j sigma_j
where sigma_i are the singular values of M_seg.

  - erank(M_seg) ~ 32 means the compressor produced 32 nearly orthogonal
    directions for that segment -- it's using its full budget.
  - erank(M_seg) << 32 means many slots are linearly dependent -- the
    segment isn't using its budget; budget could be shrunk for free.

We cross-correlate per-segment output erank with `is_supporting`
(does the segment contain an answer-bearing sentence?) so the question
"is output erank a routing signal for variable-M?" gets a direct answer.

Usage (same architecture flags as evaluation):
  python analysis/measure_effective_rank.py --restore_from <stage2 checkpoint> \\
      --output_dir <dir> --n_samples 200 <architecture flags>
"""

import transformers
import transformers.utils.import_utils
import transformers.trainer

import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from sedem.modeling import (
    SeDeMModel, ModelArguments, DataArguments, TrainingArguments,
)
from analysis.common import (
    load_hotpotqa_with_evidence, build_evidence_mask,
    compress_and_capture_attention, load_model, stratified_pool,
)


# =====================================================================
@dataclass
class OutputErankArguments:
    n_samples: int = field(default=200)
    diag_seed: int = field(default=42)
    max_context_tokens: int = field(default=4096)
    output_file: str = field(default="output_erank.json")


# =====================================================================
def output_erank_per_segment(memory_raw: torch.Tensor, memory_slots: int,
                              num_segments: int) -> np.ndarray:
    """SVD-based effective rank for each segment's 32x3072 memory block.
    Returns array of shape (num_segments,)."""
    out = np.zeros(num_segments, dtype=np.float64)
    for s in range(num_segments):
        M_s = memory_raw[0, s * memory_slots : (s + 1) * memory_slots, :].float()
        sv = torch.linalg.svdvals(M_s)
        p = sv / (sv.sum() + 1e-12)
        # entropy in nats; exp -> effective number of directions
        ent = -(p * (p + 1e-12).log()).sum()
        out[s] = float(torch.exp(ent).item())
    return out


# =====================================================================
def analyze_one(model, example, max_context_tokens, device):
    tok = model.tokenizer
    context_ids = tok(example["paper_text"], max_length=max_context_tokens,
                      truncation=True, padding=False, add_special_tokens=False,
                      return_attention_mask=False)["input_ids"]
    evidence_mask = build_evidence_mask(tok, context_ids, example["evidence_texts"])
    if evidence_mask.sum() == 0:
        return None
    input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
    memory_raw, _attn, boundaries = compress_and_capture_attention(model, input_ids)
    num_segments = len(boundaries)
    memory_slots = model.memory_slots

    # per-segment evidence label (any-overlap) and fraction
    is_supporting = np.zeros(num_segments, dtype=np.int32)
    ev_frac = np.zeros(num_segments, dtype=np.float64)
    for s, (st, en) in enumerate(boundaries):
        if en > st:
            ev_frac[s] = float(evidence_mask[st:en].mean())
            is_supporting[s] = int(evidence_mask[st:en].any())

    # output erank per segment
    erank_seg = output_erank_per_segment(memory_raw, memory_slots, num_segments)

    # also useful: norm per segment (for sanity / cross-check vs memory norm)
    mem_norm = np.zeros(num_segments, dtype=np.float64)
    for s in range(num_segments):
        seg = memory_raw[0, s * memory_slots : (s + 1) * memory_slots, :].float()
        mem_norm[s] = float(seg.norm(dim=-1).mean().item())

    return {
        "paper_id": example["paper_id"],
        "answer_type": example["answer_type"],
        "question": example["question"],
        "answer": example["answer"],
        "context_tokens": len(context_ids),
        "num_segments": num_segments,
        "output_erank_per_segment": erank_seg.tolist(),
        "mem_norm_per_segment": mem_norm.tolist(),
        "is_supporting_per_segment": is_supporting.tolist(),
        "evidence_frac_per_segment": ev_frac.tolist(),
    }


# =====================================================================
def histogram_text(values, bins=20, lo=None, hi=None, width=50):
    """ASCII histogram for terminal output."""
    a = np.asarray(values, dtype=float)
    if lo is None: lo = float(a.min())
    if hi is None: hi = float(a.max())
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(a, bins=edges)
    peak = max(counts.max(), 1)
    out = []
    for i, c in enumerate(counts):
        bar = "#" * int(width * c / peak)
        out.append(f"  [{edges[i]:6.2f}, {edges[i+1]:6.2f})  {c:5d}  {bar}")
    return "\n".join(out)


def spearman(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.size < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


# =====================================================================
def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, OutputErankArguments)
    )
    model_args, data_args, training_args, ea = parser.parse_args_into_dataclasses()

    torch.manual_seed(ea.diag_seed)
    np.random.seed(ea.diag_seed)
    random.seed(ea.diag_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[erank] loading model from: {training_args.restore_from}")
    model = load_model(model_args, training_args, device)

    print(f"[erank] loading HotpotQA (full context, gold + distractor)...")
    all_examples = load_hotpotqa_with_evidence(
        data_args.dataset_name, data_args.dataset_cache_dir,
    )
    print(f"[erank] {len(all_examples)} examples available")

    pools = stratified_pool(all_examples, ea.n_samples, ea.diag_seed,
                             oversample_factor=4)
    quota_str = ", ".join(f"{t}:{q}" for t, (q, _) in pools.items())
    print(f"[erank] per-type quotas: {quota_str}  (per-paper cap=2)")

    results, skipped = [], 0
    seen_per_paper = {}
    pbar = tqdm(total=ea.n_samples, desc="output_erank")
    for t, (quota, pool) in pools.items():
        taken = 0
        for ex in pool:
            if taken >= quota: break
            pid = ex["paper_id"]
            if seen_per_paper.get(pid, 0) >= 2: continue
            r = analyze_one(model, ex, ea.max_context_tokens, device)
            if r is None:
                skipped += 1; continue
            results.append(r); seen_per_paper[pid] = seen_per_paper.get(pid, 0) + 1
            taken += 1; pbar.update(1)
    pbar.close()
    print(f"[erank] analyzed={len(results)}  skipped={skipped}")

    # ---- pool per-segment stats ---------------------------------------
    all_erank, all_supporting, all_norm = [], [], []
    per_example_var, per_example_mean, per_example_min, per_example_max = [], [], [], []
    for r in results:
        e = np.array(r["output_erank_per_segment"])
        s = np.array(r["is_supporting_per_segment"])
        n = np.array(r["mem_norm_per_segment"])
        all_erank.extend(e.tolist())
        all_supporting.extend(s.tolist())
        all_norm.extend(n.tolist())
        if e.size >= 2:
            per_example_var.append(float(e.std()))
            per_example_mean.append(float(e.mean()))
            per_example_min.append(float(e.min()))
            per_example_max.append(float(e.max()))

    erank_arr = np.array(all_erank)
    sup_arr = np.array(all_supporting)
    norm_arr = np.array(all_norm)

    n_segs = erank_arr.size
    print("\n" + "=" * 66)
    print(f" OUTPUT ERANK PER SEGMENT  (n={n_segs} pooled segments, {len(results)} examples)")
    print("=" * 66)

    print(f"\n Pooled distribution of output_erank_s (max possible = {model.memory_slots}):")
    print(f"   mean   = {erank_arr.mean():.3f}")
    print(f"   std    = {erank_arr.std():.3f}")
    print(f"   min/max= {erank_arr.min():.3f} / {erank_arr.max():.3f}")
    qs = np.quantile(erank_arr, [0.05, 0.25, 0.5, 0.75, 0.95])
    print(f"   quantiles (5,25,50,75,95) = {qs.round(3).tolist()}")
    print()
    print(" Histogram (pooled segments):")
    print(histogram_text(erank_arr, bins=25,
                          lo=max(1.0, float(erank_arr.min()) - 0.5),
                          hi=float(model.memory_slots)))

    print(f"\n Per-example variation (n_examples with >=2 segs = {len(per_example_var)}):")
    print(f"   mean(std within example)  = {np.mean(per_example_var):.3f}")
    print(f"   mean(max - min within ex) = "
          f"{np.mean([mx - mn for mx, mn in zip(per_example_max, per_example_min)]):.3f}")

    # supporting vs non-supporting comparison
    erank_sup = erank_arr[sup_arr == 1]
    erank_non = erank_arr[sup_arr == 0]
    print(f"\n Supporting vs non-supporting segments:")
    print(f"   supporting    : n={erank_sup.size:4d}   mean erank={erank_sup.mean():.3f}   std={erank_sup.std():.3f}")
    print(f"   non-supporting: n={erank_non.size:4d}   mean erank={erank_non.mean():.3f}   std={erank_non.std():.3f}")
    if erank_sup.size > 0 and erank_non.size > 0:
        delta = float(erank_sup.mean() - erank_non.mean())
        # bootstrap 95% CI of the unpaired delta
        rng = np.random.default_rng(ea.diag_seed)
        boot = []
        for _ in range(2000):
            a = rng.choice(erank_sup, size=erank_sup.size, replace=True).mean()
            b = rng.choice(erank_non, size=erank_non.size, replace=True).mean()
            boot.append(a - b)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        print(f"   delta (sup - non-sup) = {delta:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]")

    # rank correlation: erank vs is_supporting; erank vs mem_norm
    rho_sup = spearman(erank_arr, sup_arr.astype(float))
    rho_norm = spearman(erank_arr, norm_arr)
    print(f"\n Rank correlations:")
    print(f"   rho(output_erank, is_supporting) = "
          f"{rho_sup if rho_sup is None else f'{rho_sup:+.3f}'}")
    print(f"   rho(output_erank, mem_norm)      = "
          f"{rho_norm if rho_norm is None else f'{rho_norm:+.3f}'}")

    # ---- decision rule ------------------------------------------------
    print("\n Decision rule (output erank as variable-M routing signal):")
    if rho_sup is not None and abs(rho_sup) >= 0.15:
        print(f"   -> output erank {'TRACKS' if rho_sup > 0 else 'INVERSELY tracks'} "
              "supporting label; usable as a free routing signal at compression time.")
    elif erank_arr.std() < 1.0:
        print("   -> output erank is near-uniform across segments "
              f"(pooled std = {erank_arr.std():.2f}); the compressor uses its full "
              "budget regardless of segment content. Shrinking by output erank will not work.")
    else:
        print("   -> output erank varies but does not track is_supporting "
              f"(rho = {rho_sup}); informative but not sufficient on its own.")

    # ---- save ---------------------------------------------------------
    out_path = ea.output_file
    if not os.path.isabs(out_path):
        out_path = os.path.join(training_args.output_dir, out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    payload = {
        "config": {
            "checkpoint": training_args.restore_from,
            "compressor_type": getattr(model_args, "compressor_type", None),
            "memory_slots": training_args.memory_slots_per_segment,
            "compression_factor": training_args.compression_factor,
            "n_samples_requested": ea.n_samples,
            "max_context_tokens": ea.max_context_tokens,
            "seed": ea.diag_seed,
        },
        "summary": {
            "n_examples": len(results),
            "n_segments_pooled": int(n_segs),
            "memory_slots": model.memory_slots,
            "pooled_mean": float(erank_arr.mean()),
            "pooled_std": float(erank_arr.std()),
            "pooled_min": float(erank_arr.min()),
            "pooled_max": float(erank_arr.max()),
            "pooled_quantiles_5_25_50_75_95": qs.tolist(),
            "supporting_mean": float(erank_sup.mean()) if erank_sup.size else None,
            "non_supporting_mean": float(erank_non.mean()) if erank_non.size else None,
            "supporting_minus_non_supporting": (float(erank_sup.mean() - erank_non.mean())
                                                  if erank_sup.size and erank_non.size else None),
            "spearman_erank_vs_is_supporting": rho_sup,
            "spearman_erank_vs_mem_norm": rho_norm,
        },
        "per_example": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[erank] wrote {out_path}")


if __name__ == "__main__":
    main()
