import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
"""Build a stratified subset of 2WikiMultihopQA.

Produces a manifest in the SAME format as prepare_hotpot_subset.py
(consumable by training_utils.apply_subset_manifest), so existing
training/eval code paths Just Work — flip --subset_manifest and go.

Two-mode stratification:
  - train: TYPE-BALANCED (equal allocation per question type, length-bin
    stratified within type). Ensures rare types (compositional, inference)
    aren't undertrained.
  - val: PROPORTIONAL stratified (matches natural test distribution so the
    F1 number is comparable to a full-val run).

Length bins are tertiles of context word-count computed on the full train
split (shared with val for consistent bucketing).

Source: xanhho/2WikiMultihopQA at revision='refs/convert/parquet'.
The context field comes back as a JSON-string in this revision; we parse it.

Usage:
    python prepare_2wiki_subset.py \\
        --output data/2wiki_subset_6k1k.json \\
        --train-size 6000 \\
        --val-size 1000 \\
        --seed 42
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from typing import Any, List, Tuple

from datasets import load_dataset


def _parse_context(val):
    """2Wiki parquet stores context as JSON string."""
    if isinstance(val, str):
        return json.loads(val)
    return val


def _context_word_count(example: dict[str, Any]) -> int:
    ctx = _parse_context(example["context"])
    total = 0
    for title, sents in ctx:
        total += len(title.split()) + sum(len(s.split()) for s in sents)
    return total


def _compute_length_bins(word_counts: list[int]) -> tuple[int, int]:
    s = sorted(word_counts)
    n = len(s)
    return s[n // 3], s[(2 * n) // 3]


def _bucket_length(n: int, lo: int, hi: int) -> str:
    if n < lo:
        return "short"
    if n < hi:
        return "medium"
    return "long"


def _stratum_key(example, lo, hi):
    qtype = example.get("type", "unknown")
    lbucket = _bucket_length(_context_word_count(example), lo, hi)
    return f"{qtype}|{lbucket}"


def _proportional_sample(
    dataset, target_size: int, lo: int, hi: int, rng: random.Random,
) -> Tuple[List[int], List[str], dict]:
    """Stratified-proportional sampling: each stratum gets share matching natural rate."""
    strata: dict[str, list[int]] = defaultdict(list)
    for i, ex in enumerate(dataset):
        strata[_stratum_key(ex, lo, hi)].append(i)
    total = sum(len(v) for v in strata.values())
    targets = {k: min(len(v), int(round(len(v) / total * target_size)))
               for k, v in strata.items()}
    drift = target_size - sum(targets.values())
    keys = sorted(strata.keys(), key=lambda k: -len(strata[k]))
    i = 0
    while drift != 0 and i < len(keys) * 10:
        k = keys[i % len(keys)]
        avail = len(strata[k]) - targets[k]
        if drift > 0 and avail > 0:
            targets[k] += 1; drift -= 1
        elif drift < 0 and targets[k] > 0:
            targets[k] -= 1; drift += 1
        i += 1
    sel = []
    for k in sorted(strata):
        if targets[k] == 0:
            continue
        sel.extend(rng.sample(strata[k], targets[k]))
    sel.sort()
    ids = [str(dataset[i]["_id"]) for i in sel]
    return sel, ids, dict(targets)


def _type_balanced_sample(
    dataset, train_size: int, lo: int, hi: int, rng: random.Random,
) -> Tuple[List[int], List[str], dict]:
    """Equal allocation per question type; length-stratified within type."""
    by_type: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for i, ex in enumerate(dataset):
        qtype = ex.get("type", "unknown")
        lbucket = _bucket_length(_context_word_count(ex), lo, hi)
        by_type[qtype][lbucket].append(i)

    n_types = len(by_type)
    per_type = train_size // n_types
    print(f"[2wiki/type-balanced] {n_types} types × {per_type} each = {n_types * per_type} (target {train_size})")

    sel: list[int] = []
    targets: dict[str, int] = {}
    for qtype in sorted(by_type.keys()):
        # length-proportional within type
        len_strata = by_type[qtype]
        type_total = sum(len(v) for v in len_strata.values())
        for lbucket in sorted(len_strata.keys()):
            pool = len_strata[lbucket]
            share = len(pool) / type_total
            k = min(len(pool), int(round(share * per_type)))
            if k > 0:
                sel.extend(rng.sample(pool, k))
                targets[f"{qtype}|{lbucket}"] = k
    sel.sort()
    ids = [str(dataset[i]["_id"]) for i in sel]
    return sel, ids, targets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="xanhho/2WikiMultihopQA")
    p.add_argument("--revision", default="refs/convert/parquet")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--train-size", type=int, default=6000)
    p.add_argument("--val-size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if os.path.exists(args.output) and not args.force:
        with open(args.output) as f:
            m = json.load(f)
        meta = m["meta"]
        print(f"[prepare_2wiki_subset] {args.output} exists; train={meta['actual_train']}, val={meta['actual_val']}")
        return

    print(f"[prepare_2wiki_subset] Loading {args.dataset} (rev={args.revision})...")
    ds = load_dataset(args.dataset, revision=args.revision, cache_dir=args.cache_dir)
    train, val = ds["train"], ds["validation"]
    print(f"  train: {len(train)}, val: {len(val)}")

    print(f"[prepare_2wiki_subset] Type counts (train): "
          f"{dict(Counter(ex['type'] for ex in train))}")

    print("[prepare_2wiki_subset] Computing length tertiles on train...")
    train_lengths = [_context_word_count(ex) for ex in train]
    lo, hi = _compute_length_bins(train_lengths)
    print(f"  Length tertiles: short<{lo}, medium<{hi}, long>={hi}")

    rng = random.Random(args.seed)
    print(f"[prepare_2wiki_subset] Type-balanced train sampling (seed={args.seed})...")
    train_idx, train_ids, train_strata = _type_balanced_sample(train, args.train_size, lo, hi, rng)
    print(f"[prepare_2wiki_subset] Proportional val sampling...")
    val_idx, val_ids, val_strata = _proportional_sample(val, args.val_size, lo, hi, rng)

    print(f"\n[prepare_2wiki_subset] Train strata ({len(train_idx)}):")
    for k in sorted(train_strata): print(f"  {k:40s} {train_strata[k]:5d}")
    print(f"\n[prepare_2wiki_subset] Val strata ({len(val_idx)}):")
    for k in sorted(val_strata): print(f"  {k:40s} {val_strata[k]:5d}")

    manifest = {
        "meta": {
            "dataset": args.dataset,
            "revision": args.revision,
            "config": None,
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
            "train_strategy": "type_balanced",
            "val_strategy": "proportional_stratified",
        },
        "train": {"indices": train_idx, "ids": train_ids},
        "validation": {"indices": val_idx, "ids": val_ids},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[prepare_2wiki_subset] Wrote manifest -> {args.output}")


if __name__ == "__main__":
    main()
