"""
Label-based block selection (evidence-label selection ceiling).

Given (context_text, evidence_strings) from a Qasper example, map each evidence
string to a set of memory blocks that "come from" the same segments as the
evidence tokens. All blocks within an evidence-containing segment are marked
positive (segment-level granularity; the compressor doesn't give us a cleaner
token→slot attribution, so this is the honest signal).

Blocks are groups of `block_size = compression_factor` consecutive memory
slots. With M=32 slots and c=4 per segment: 8 blocks per segment.

This module is used at eval time to:
  - Build per-example positive_block_mask
  - Produce selected_block_indices for `selection_mode ∈ {label, random}`
"""

import math
import random
import torch


def evidence_labels_from_qasper_spans(
    context_text,
    evidence_strings,
    tokenizer,
    model,
    max_length=4096,
    block_size=None,
):
    """Determine which memory blocks are evidence-positive for one Qasper example.

    Args:
        context_text: the paper text (what the compressor sees)
        evidence_strings: list[str] of evidence paragraphs from annotators
        tokenizer: the model's tokenizer (must be a fast tokenizer for offsets)
        model: SeDeMModel instance (uses .memory_slots, .compression_factor,
               .compute_num_segments)
        max_length: context truncation length (must match eval_qa.py setting)
        block_size: number of memory tokens per selection block. Default
                    `model.compression_factor` reproduces the Qasper
                    sub-segment block (8 blocks per segment at M=32, c=4).
                    Pass `block_size=model.memory_slots` to make 1 block = 1 full
                    segment's mem bundle (HotpotQA Option A: 1 paragraph =
                    1 block).

    Returns:
        positive_block_mask: (num_blocks,) bool tensor
        num_segments: int
        num_blocks: int
        stats: dict with per-example diagnostics
    """
    out = tokenizer(
        context_text,
        max_length=max_length,
        truncation=True,
        padding=False,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = out["input_ids"]
    offsets = out["offset_mapping"]  # list of (char_start, char_end) per token
    total_length = len(input_ids)

    num_segments = model.compute_num_segments(total_length)
    segment_length = math.ceil(total_length / num_segments)
    if block_size is None:
        block_size = model.compression_factor
    if model.memory_slots % block_size != 0:
        raise ValueError(
            f"memory_slots ({model.memory_slots}) must be divisible by block_size ({block_size})."
        )
    blocks_per_segment = model.memory_slots // block_size
    num_blocks = num_segments * blocks_per_segment

    positive_segments = set()
    n_matched = 0
    n_unmatched = 0
    n_beyond_truncation = 0

    for e in evidence_strings:
        pos = context_text.find(e)
        if pos < 0:
            n_unmatched += 1
            continue
        n_matched += 1
        e_start, e_end = pos, pos + len(e)

        first_tok = last_tok = None
        for i, (a, b) in enumerate(offsets):
            if a < e_end and b > e_start:
                if first_tok is None:
                    first_tok = i
                last_tok = i

        if first_tok is None:
            n_beyond_truncation += 1
            continue

        seg_start = first_tok // segment_length
        seg_end = last_tok // segment_length
        for s in range(seg_start, min(seg_end + 1, num_segments)):
            positive_segments.add(s)

    positive_block_mask = torch.zeros(num_blocks, dtype=torch.bool)
    for s in positive_segments:
        positive_block_mask[s * blocks_per_segment:(s + 1) * blocks_per_segment] = True

    stats = {
        "total_tokens": total_length,
        "num_segments": num_segments,
        "num_blocks": num_blocks,
        "blocks_per_segment": blocks_per_segment,
        "n_evidence": len(evidence_strings),
        "n_matched": n_matched,
        "n_unmatched": n_unmatched,
        "n_beyond_truncation": n_beyond_truncation,
        "num_positive_segments": len(positive_segments),
        "num_positive_blocks": int(positive_block_mask.sum().item()),
    }
    return positive_block_mask, num_segments, num_blocks, stats


def select_top_k_blocks(positive_block_mask, num_blocks, top_k, mode, rng):
    """Pick top-K block indices per `mode`.

    Modes:
        'label' : random K from positives; if fewer positives than K, fill with
                  random negatives (preserves positive selection priority).
        'random': K random blocks (ignores positive_block_mask). The floor.

    Args:
        positive_block_mask: (num_blocks,) bool tensor
        num_blocks: int, total blocks in this example's memory
        top_k: int, number of blocks to select (if >= num_blocks, returns all)
        mode: 'label' | 'random'
        rng: random.Random instance for reproducibility

    Returns:
        selected_indices: LongTensor(K_actual,) sorted ascending (to keep
                          reconstructed-token ordering deterministic)
        sel_stats: dict
    """
    if top_k >= num_blocks:
        return torch.arange(num_blocks, dtype=torch.long), {
            "k_requested": top_k, "k_selected": num_blocks,
            "n_from_positives": int(positive_block_mask.sum().item()),
            "n_from_negatives": int((~positive_block_mask).sum().item()),
        }

    all_indices = list(range(num_blocks))
    if mode == "random":
        selected = rng.sample(all_indices, top_k)
        n_pos = sum(1 for i in selected if positive_block_mask[i].item())
        sel_stats = {"k_requested": top_k, "k_selected": top_k,
                     "n_from_positives": n_pos, "n_from_negatives": top_k - n_pos}
    elif mode == "label":
        positives = [i for i in all_indices if positive_block_mask[i].item()]
        negatives = [i for i in all_indices if not positive_block_mask[i].item()]
        if len(positives) >= top_k:
            selected = rng.sample(positives, top_k)
            n_pos = top_k
        else:
            selected = list(positives)
            needed = top_k - len(selected)
            selected += rng.sample(negatives, min(needed, len(negatives)))
            n_pos = len(positives)
        sel_stats = {"k_requested": top_k, "k_selected": len(selected),
                     "n_from_positives": n_pos,
                     "n_from_negatives": len(selected) - n_pos}
    else:
        raise ValueError(f"Unknown mode='{mode}' (use 'label' or 'random')")

    selected_t = torch.tensor(sorted(selected), dtype=torch.long)
    return selected_t, sel_stats
