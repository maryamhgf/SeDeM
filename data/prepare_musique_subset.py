import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
"""Build a stratified subset of MuSiQue.

Same manifest format as prepare_hotpot_subset.py / prepare_2wiki_subset.py
(consumable by training_utils.apply_subset_manifest). Stratification is by
hop-count × length bin.

  - train: HOP-BALANCED (equal allocation per hop count, length-stratified
    within hop). Ensures 4-hop questions aren't undertrained.
  - val: PROPORTIONAL stratified.
  - filter: only `answerable=True` examples (matches musique_qa_transform).

Hop count comes from `len(question_decomposition)` (= number of intermediate
sub-questions = number of supporting paragraphs).

Source: dgslibisey/MuSiQue.

Usage:
    python prepare_musique_subset.py \\
        --output data/musique_subset_6k1k.json \\
        --train-size 6000 \\
        --val-size 1000 \\
        --seed 42

Pass --train-size 0 to keep ALL train examples (the dataset is small at ~20k;
"full" is reasonable, and keeps the manifest format consistent for selection/
eval pipelines).
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from typing import Any, List, Tuple

from datasets import load_dataset


def _context_word_count(ex: dict[str, Any]) -> int:
    return sum(len(p["title"].split()) + len(p["paragraph_text"].split()) for p in ex["paragraphs"])


def _hop_count(ex: dict[str, Any]) -> int:
    return len(ex.get("question_decomposition", []))


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


def _stratum_key(ex, lo, hi):
    return f"hop{_hop_count(ex)}|{_bucket_length(_context_word_count(ex), lo, hi)}"


def _proportional_sample(
    indices_with_meta, target_size, lo, hi, rng,
):
    """indices_with_meta: list of (orig_idx, ex_dict). Returns (sel_orig_indices, ids, strata)."""
    strata: dict[str, list[int]] = defaultdict(list)
    for i, ex in indices_with_meta:
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
    return sel, dict(targets)


def _hop_balanced_sample(
    indices_with_meta, train_size, lo, hi, rng,
):
    by_hop: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for i, ex in indices_with_meta:
        h = _hop_count(ex)
        l = _bucket_length(_context_word_count(ex), lo, hi)
        by_hop[h][l].append(i)
    n_hops = len(by_hop)
    per_hop = train_size // n_hops
    print(f"[musique/hop-balanced] {n_hops} hops × {per_hop} each = {n_hops * per_hop} (target {train_size})")
    sel: list[int] = []
    targets: dict[str, int] = {}
    for h in sorted(by_hop.keys()):
        len_strata = by_hop[h]
        hop_total = sum(len(v) for v in len_strata.values())
        for lbucket in sorted(len_strata.keys()):
            pool = len_strata[lbucket]
            share = len(pool) / hop_total
            k = min(len(pool), int(round(share * per_hop)))
            if k > 0:
                sel.extend(rng.sample(pool, k))
                targets[f"hop{h}|{lbucket}"] = k
    sel.sort()
    return sel, targets


def _filter_answerable(dataset):
    """Return list of (orig_idx, example_dict) for answerable + non-empty answer examples."""
    out = []
    for i, ex in enumerate(dataset):
        if ex.get("answerable") and ex["answer"] and ex["answer"].strip():
            out.append((i, ex))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="dgslibisey/MuSiQue")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--train-size", type=int, default=6000,
                   help="Set 0 to keep all answerable train examples.")
    p.add_argument("--val-size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if os.path.exists(args.output) and not args.force:
        with open(args.output) as f:
            m = json.load(f)
        meta = m["meta"]
        print(f"[prepare_musique_subset] {args.output} exists; train={meta['actual_train']}, val={meta['actual_val']}")
        return

    print(f"[prepare_musique_subset] Loading {args.dataset}...")
    ds = load_dataset(args.dataset, cache_dir=args.cache_dir)
    train_full, val_full = ds["train"], ds["validation"]
    print(f"  train: {len(train_full)}, val: {len(val_full)}")

    print("[prepare_musique_subset] Filtering for answerable=True + non-empty answer...")
    train_filtered = _filter_answerable(train_full)
    val_filtered = _filter_answerable(val_full)
    print(f"  filtered train: {len(train_filtered)}, val: {len(val_filtered)}")
    print(f"  hop counts (train): {dict(Counter(_hop_count(ex) for _, ex in train_filtered))}")

    train_lengths = [_context_word_count(ex) for _, ex in train_filtered]
    lo, hi = _compute_length_bins(train_lengths)
    print(f"  Length tertiles (words): short<{lo}, medium<{hi}, long>={hi}")

    rng = random.Random(args.seed)
    if args.train_size == 0:
        # Keep all answerable
        train_idx = sorted(i for i, _ in train_filtered)
        train_strata = {}
        for i, ex in train_filtered:
            k = _stratum_key(ex, lo, hi)
            train_strata[k] = train_strata.get(k, 0) + 1
        print(f"[prepare_musique_subset] Keeping all {len(train_idx)} answerable train examples.")
    else:
        print(f"[prepare_musique_subset] Hop-balanced train sampling (seed={args.seed})...")
        train_idx, train_strata = _hop_balanced_sample(train_filtered, args.train_size, lo, hi, rng)
    print(f"[prepare_musique_subset] Proportional val sampling...")
    val_idx, val_strata = _proportional_sample(val_filtered, args.val_size, lo, hi, rng)

    print(f"\n[prepare_musique_subset] Train strata ({len(train_idx)}):")
    for k in sorted(train_strata): print(f"  {k:30s} {train_strata[k]:5d}")
    print(f"\n[prepare_musique_subset] Val strata ({len(val_idx)}):")
    for k in sorted(val_strata): print(f"  {k:30s} {val_strata[k]:5d}")

    train_ids = [str(train_full[i]["id"]) for i in train_idx]
    val_ids = [str(val_full[i]["id"]) for i in val_idx]

    manifest = {
        "meta": {
            "dataset": args.dataset,
            "config": None,
            "seed": args.seed,
            "target_train": args.train_size,
            "target_val": args.val_size,
            "actual_train": len(train_idx),
            "actual_val": len(val_idx),
            "length_bin_edges_words": [lo, hi],
            "strata_train": train_strata,
            "strata_val": val_strata,
            "source_train_size": len(train_full),
            "source_val_size": len(val_full),
            "answerable_filter": True,
            "train_strategy": "all_answerable" if args.train_size == 0 else "hop_balanced",
            "val_strategy": "proportional_stratified",
        },
        "train": {"indices": train_idx, "ids": train_ids},
        "validation": {"indices": val_idx, "ids": val_ids},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[prepare_musique_subset] Wrote manifest -> {args.output}")


if __name__ == "__main__":
    main()
