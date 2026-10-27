"""2WikiMultihopQA selection helpers (mirrors sedem.blocks_hotpot.py).

Same selection contract as HotpotQA: 1 paragraph = 1 block, num_blocks=10.

Source: xanhho/2WikiMultihopQA at revision='refs/convert/parquet'. The parquet
revision returns `context` and `supporting_facts` as JSON strings; we parse
them with `_twowiki_parse`.
"""

import json
from typing import Dict, List, Tuple

import torch


def _twowiki_parse(val):
    if isinstance(val, str):
        return json.loads(val)
    return val


def twowiki_eval_transform(examples: Dict[str, List]) -> Dict[str, List]:
    """Batched map for 2WikiMultihopQA → flat dict for selection pipelines."""
    inputs, all_paras, all_titles = [], [], []
    prompts, answers, all_ans, sup_titles = [], [], [], []

    for ctx, sf, q, a in zip(
        examples["context"], examples["supporting_facts"],
        examples["question"], examples["answer"],
    ):
        ctx = _twowiki_parse(ctx)
        sf = _twowiki_parse(sf)
        titles = [t for t, _ in ctx]
        paragraphs = [f"{t}: {''.join(s).strip()}" for t, s in ctx]
        inputs.append("\n\n".join(paragraphs))
        all_paras.append(paragraphs)
        all_titles.append(titles)
        prompts.append(q)
        answers.append(a)
        all_ans.append([a])
        sup_titles.append([t for t, _ in sf])

    return {
        "input": inputs, "paragraphs": all_paras, "paragraph_titles": all_titles,
        "prompt": prompts, "answer": answers, "all_answers": all_ans,
        "supporting_titles": sup_titles,
    }


@torch.no_grad()
def build_memory_blocks_twowiki(
    model, paragraphs: List[str], device,
    num_blocks: int = 10, chunk_size: int = 128,
) -> Tuple[torch.Tensor, List[str], List[List[int]], int, int]:
    """Same contract as build_memory_blocks_hotpot (10 paragraphs, 10 blocks)."""
    if chunk_size != model.memory_slots * model.compression_factor:
        raise ValueError(f"chunk_size {chunk_size} != M*c {model.memory_slots * model.compression_factor}")
    paragraphs = list(paragraphs)
    while len(paragraphs) < num_blocks:
        paragraphs.append("")
    paragraphs = paragraphs[:num_blocks]

    ids_per_block: List[List[int]] = []
    for p in paragraphs:
        out = model.tokenizer(
            p, max_length=chunk_size, truncation=True, padding="max_length",
            add_special_tokens=False, return_attention_mask=False,
        )
        ids_per_block.append(list(out["input_ids"]))

    flat = [i for block in ids_per_block for i in block]
    ctx_ids = torch.tensor([flat], dtype=torch.long, device=device)
    memory = model.compress_segments(ctx_ids)
    M_total, D = memory.size(1), memory.size(2)
    expected_M = num_blocks * model.memory_slots
    if M_total != expected_M:
        raise RuntimeError(f"compressor produced {M_total} mem tokens, expected {expected_M}")
    block_size = model.memory_slots
    blocks = memory.view(1, num_blocks, block_size, D)
    return blocks, paragraphs, ids_per_block, num_blocks, num_blocks


def evidence_labels_twowiki(
    paragraph_titles: List[str], supporting_titles: List[str], num_blocks: int = 10,
) -> Tuple[torch.Tensor, Dict]:
    """Mark blocks whose title is in supporting_facts.title as positive."""
    sf_set = set(supporting_titles)
    matched = [i for i, t in enumerate(paragraph_titles) if t in sf_set]
    mask = torch.zeros(num_blocks, dtype=torch.bool)
    for i in matched:
        if i < num_blocks:
            mask[i] = True
    return mask, {
        "n_paragraphs": len(paragraph_titles),
        "n_supporting_titles": len(sf_set),
        "n_positive_blocks": int(mask.sum().item()),
        "matched_indices": matched,
    }
