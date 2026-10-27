"""
Block-level positive-mask labeling for HotpotQA distractor (Option A:
1 paragraph = 1 block).

For each example, the context is the 10 distractor paragraphs (a list of
title + sentences). The selector's job is to pick the K paragraphs that
contain supporting facts.

Positive-mask construction:
  - HotpotQA `supporting_facts` is a dict {"title": list[str], "sent_id": list[int]}.
  - Mark a paragraph block as positive iff its title appears in the
    supporting_facts.title set.
  - Sentence-level granularity is NOT used — at paragraph-as-block granularity
    (Option A), if any supporting sentence is in this paragraph, the whole
    block is positive.

Returns a (num_blocks=10,) bool tensor and per-example diagnostics. The
diagnostics let us validate the title→paragraph mapping before training:
  - n_positive_blocks per example  (typically 2 for HotpotQA)
  - n_unmatched_titles  (supporting_fact titles not found in context.title)
  - n_paragraphs (should be 10 for the distractor split)
"""

from typing import Dict, List, Tuple

import torch


def evidence_labels_from_supporting_facts(
    context: Dict[str, List],
    supporting_facts: Dict[str, List],
    num_blocks: int = 10,
) -> Tuple[torch.Tensor, Dict]:
    """One HotpotQA distractor example → (positive_mask, stats).

    Args:
        context: dict with keys "title" (list[str]) and "sentences"
                 (list[list[str]]). For HotpotQA distractor this is a list
                 of `num_blocks` paragraphs in order.
        supporting_facts: dict with keys "title" and "sent_id" listing the
                 gold (paragraph_title, sentence_index) supporting facts.
        num_blocks: expected number of blocks per example (10 for distractor).

    Returns:
        positive_mask: (num_blocks,) bool tensor
        stats: dict with keys
            n_paragraphs, n_positive_blocks, n_supporting_titles,
            n_unmatched_titles, supporting_titles, paragraph_titles
    """
    paragraph_titles = list(context["title"])
    n_paragraphs = len(paragraph_titles)

    sf_titles = set(supporting_facts["title"])
    matched_indices = []
    matched_titles = set()
    for idx, t in enumerate(paragraph_titles):
        if t in sf_titles:
            matched_indices.append(idx)
            matched_titles.add(t)

    unmatched_titles = sf_titles - matched_titles

    # Pad / truncate to expected num_blocks. Real distractor always has 10
    # paragraphs, but be defensive against schema drift.
    mask = torch.zeros(num_blocks, dtype=torch.bool)
    for i in matched_indices:
        if i < num_blocks:
            mask[i] = True

    stats = {
        "n_paragraphs": n_paragraphs,
        "n_supporting_titles": len(sf_titles),
        "n_positive_blocks": int(mask.sum().item()),
        "n_unmatched_titles": len(unmatched_titles),
        "supporting_titles": sorted(sf_titles),
        "paragraph_titles": paragraph_titles,
        "matched_indices": matched_indices,
    }
    return mask, stats
