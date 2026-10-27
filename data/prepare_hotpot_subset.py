import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
"""
Build a stratified subset of HotpotQA for reproducible experiments.

Stratifies by (question_type, answer_type, context_length_bin):
  - question_type : "bridge" or "comparison"    (from example["type"])
  - answer_type   : "yes_no" or "span"          (from answer.lower() in {yes, no})
  - length_bin    : "short" / "medium" / "long" (from context word-count tertiles
                                                 computed on the train split)

Samples proportionally from each stratum so the subset preserves the original
distribution of question types, answer types, and context-length buckets.

Deterministic (fixed seed) and idempotent: re-running with the same arguments
produces the same indices. Saves the selected indices and example IDs to a JSON
manifest so the filter runs once and every downstream training/eval run uses
exactly the same examples.

Usage:
    python prepare_hotpot_subset.py \\
        --output data/hotpot_subset_3k1k.json \\
        --train-size 3000 \\
        --val-size 1000 \\
        --seed 42
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from typing import Any

from datasets import load_dataset


def _context_word_count(example: dict[str, Any]) -> int:
    """Total word count across all paragraphs in the HotpotQA context (distractor + gold)."""
    ctx = example["context"]
    total = 0
    for title, sents in zip(ctx["title"], ctx["sentences"]):
        total += len(title.split()) + sum(len(s.split()) for s in sents)
    return total


def _answer_type(answer: str) -> str:
    a = answer.strip().lower()
    return "yes_no" if a in ("yes", "no") else "span"


def _compute_length_bins(word_counts: list[int]) -> tuple[int, int]:
    """Compute tertile boundaries from a list of context word counts."""
    sorted_counts = sorted(word_counts)
    n = len(sorted_counts)
    lo = sorted_counts[n // 3]
    hi = sorted_counts[(2 * n) // 3]
    return lo, hi


def _bucket_length(n_words: int, lo: int, hi: int) -> str:
    if n_words < lo:
        return "short"
    if n_words < hi:
        return "medium"
    return "long"


def _stratum_key(example: dict[str, Any], lo: int, hi: int) -> str:
    qtype = example.get("type", "unknown")
    atype = _answer_type(example["answer"])
    lbucket = _bucket_length(_context_word_count(example), lo, hi)
    return f"{qtype}|{atype}|{lbucket}"


def _stratified_sample(
    dataset,
    target_size: int,
    lo: int,
    hi: int,
    rng: random.Random,
) -> tuple[list[int], list[str], dict[str, int]]:
    """Return (selected_indices, selected_ids, stratum_distribution)."""
    strata: dict[str, list[int]] = defaultdict(list)
    for i, ex in enumerate(dataset):
        strata[_stratum_key(ex, lo, hi)].append(i)

    total = sum(len(v) for v in strata.values())
    assert total == len(dataset), f"stratum count mismatch: {total} vs {len(dataset)}"

    # Proportional target per stratum, with rounding adjustment so the total
    # matches the requested subset size exactly.
    targets: dict[str, int] = {}
    for key, indices in strata.items():
        share = len(indices) / total
        targets[key] = min(len(indices), int(round(share * target_size)))

    # Correct rounding drift by adding/removing from the largest strata.
    drift = target_size - sum(targets.values())
    ordered_keys = sorted(strata.keys(), key=lambda k: -len(strata[k]))
    i = 0
    while drift != 0 and i < len(ordered_keys) * 10:
        key = ordered_keys[i % len(ordered_keys)]
        available = len(strata[key]) - targets[key]
        if drift > 0 and available > 0:
            targets[key] += 1
            drift -= 1
        elif drift < 0 and targets[key] > 0:
            targets[key] -= 1
            drift += 1
        i += 1
    assert drift == 0, f"could not reconcile stratum rounding: drift={drift}"

    # Sample without replacement from each stratum.
    selected_indices: list[int] = []
    for key in sorted(strata.keys()):
        pool = strata[key]
        k = targets[key]
        if k == 0:
            continue
        picks = rng.sample(pool, k=k)
        selected_indices.extend(picks)

    selected_indices.sort()
    selected_ids = [dataset[i]["id"] for i in selected_indices]
    return selected_indices, selected_ids, dict(targets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="hotpotqa/hotpot_qa")
    parser.add_argument("--config", default="distractor")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output", required=True, help="Path to save the JSON manifest.")
    parser.add_argument("--train-size", type=int, default=3000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Overwrite output file if it exists.")
    args = parser.parse_args()

    if os.path.exists(args.output) and not args.force:
        print(f"[prepare_hotpot_subset] {args.output} already exists; use --force to overwrite.")
        with open(args.output) as f:
            manifest = json.load(f)
        meta = manifest["meta"]
        print(f"  Train: {len(manifest['train']['indices'])} examples (target {meta['target_train']})")
        print(f"  Val:   {len(manifest['validation']['indices'])} examples (target {meta['target_val']})")
        return

    print(f"[prepare_hotpot_subset] Loading {args.dataset} ({args.config})...")
    dataset = load_dataset(
        args.dataset,
        args.config,
        cache_dir=args.cache_dir,
    )
    train = dataset["train"]
    val = dataset["validation"]
    print(f"  Full train: {len(train)} examples")
    print(f"  Full val:   {len(val)} examples")

    # Compute length-bin boundaries from the full train split so train and val
    # share the same bucketing definition.
    print("[prepare_hotpot_subset] Computing context-length tertiles on train...")
    train_lengths = [_context_word_count(ex) for ex in train]
    lo, hi = _compute_length_bins(train_lengths)
    print(f"  Length tertiles (words): short<{lo}, medium<{hi}, long>={hi}")

    rng = random.Random(args.seed)
    print(f"[prepare_hotpot_subset] Stratified sampling (seed={args.seed})...")
    train_idx, train_ids, train_strata = _stratified_sample(
        train, args.train_size, lo, hi, rng
    )
    val_idx, val_ids, val_strata = _stratified_sample(
        val, args.val_size, lo, hi, rng
    )

    # Report stratum distribution
    print(f"\n[prepare_hotpot_subset] Train strata ({len(train_idx)} examples):")
    for key in sorted(train_strata.keys()):
        print(f"  {key:40s} {train_strata[key]:5d}")
    print(f"\n[prepare_hotpot_subset] Val strata ({len(val_idx)} examples):")
    for key in sorted(val_strata.keys()):
        print(f"  {key:40s} {val_strata[key]:5d}")

    # Sanity: verify ratios are roughly preserved for the headline dims
    def _ratio_report(split_name: str, dataset_split, selected_indices: list[int]):
        def counts(subset_indices):
            qt = Counter()
            at = Counter()
            for i in subset_indices:
                ex = dataset_split[i]
                qt[ex.get("type", "unknown")] += 1
                at[_answer_type(ex["answer"])] += 1
            return qt, at

        full_qt, full_at = counts(range(len(dataset_split)))
        sub_qt, sub_at = counts(selected_indices)
        n_full = len(dataset_split)
        n_sub = len(selected_indices)
        print(f"\n[sanity] {split_name} — ratios (full → subset)")
        for k in sorted(full_qt.keys()):
            print(f"  type={k:12s}  {full_qt[k] / n_full:6.3f} → {sub_qt[k] / n_sub:6.3f}")
        for k in sorted(full_at.keys()):
            print(f"  answer={k:10s}  {full_at[k] / n_full:6.3f} → {sub_at[k] / n_sub:6.3f}")

    _ratio_report("train", train, train_idx)
    _ratio_report("val", val, val_idx)

    manifest = {
        "meta": {
            "dataset": args.dataset,
            "config": args.config,
            "seed": args.seed,
            "target_train": args.train_size,
            "target_val": args.val_size,
            "actual_train": len(train_idx),
            "actual_val": len(val_idx),
            "length_bin_edges_words": [lo, hi],
            "strata_train": train_strata,
            "strata_val": val_strata,
            "source_train_size": len(train),
            "source_val_size": len(val),
        },
        "train": {
            "indices": train_idx,
            "ids": train_ids,
        },
        "validation": {
            "indices": val_idx,
            "ids": val_ids,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[prepare_hotpot_subset] Saved manifest to {args.output}")
    print(f"  {len(train_idx)} train + {len(val_idx)} val = {len(train_idx) + len(val_idx)} total examples")


if __name__ == "__main__":
    main()
