"""MuSiQue selection helpers (mirrors sedem.blocks_hotpot.py).

Different from HotpotQA: 20 paragraphs/example (not 10), and gold blocks are
identified by `is_supporting` flag inside each paragraph dict. Filter for
`answerable=True`.

Source: dgslibisey/MuSiQue.
"""

from typing import Dict, List, Tuple

import torch


def musique_eval_transform(examples: Dict[str, List]) -> Dict[str, List]:
    """Batched map for MuSiQue → flat dict for selection pipelines.

    Drops examples where answerable=False or answer is empty (matches the
    musique_qa_transform behaviour). `supporting_titles` is the list of titles
    where is_supporting=True; `is_supporting_mask` is a per-paragraph bool list.
    """
    inputs, all_paras, all_titles = [], [], []
    prompts, answers, all_ans, sup_titles, is_sup = [], [], [], [], []

    for paragraphs, q, a, aliases, answerable in zip(
        examples["paragraphs"], examples["question"], examples["answer"],
        examples["answer_aliases"], examples["answerable"],
    ):
        if not answerable or not a or not a.strip():
            continue
        titles = [p["title"] for p in paragraphs]
        paragraph_strs = [f"{p['title']}: {p['paragraph_text'].strip()}" for p in paragraphs]
        inputs.append("\n\n".join(paragraph_strs))
        all_paras.append(paragraph_strs)
        all_titles.append(titles)
        prompts.append(q)
        answers.append(a)
        all_ans.append([a] + [x for x in (aliases or []) if x and x.strip() and x != a])
        sup_titles.append([p["title"] for p in paragraphs if p["is_supporting"]])
        is_sup.append([bool(p["is_supporting"]) for p in paragraphs])

    return {
        "input": inputs, "paragraphs": all_paras, "paragraph_titles": all_titles,
        "prompt": prompts, "answer": answers, "all_answers": all_ans,
        "supporting_titles": sup_titles, "is_supporting": is_sup,
    }


@torch.no_grad()
def build_memory_blocks_musique(
    model, paragraphs: List[str], device,
    num_blocks: int = 20, chunk_size: int = 128,
) -> Tuple[torch.Tensor, List[str], List[List[int]], int, int]:
    """20-block version of build_memory_blocks_hotpot. Tokens are still
    chunk_size = M*c (e.g. 128) per paragraph; total memory = 20 * M slots."""
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


def evidence_labels_musique(
    is_supporting: List[bool], num_blocks: int = 20,
) -> Tuple[torch.Tensor, Dict]:
    """Direct mapping: each paragraph has an is_supporting flag → block label."""
    mask = torch.zeros(num_blocks, dtype=torch.bool)
    matched = []
    for i, s in enumerate(is_supporting):
        if i < num_blocks and s:
            mask[i] = True
            matched.append(i)
    return mask, {
        "n_paragraphs": len(is_supporting),
        "n_positive_blocks": int(mask.sum().item()),
        "matched_indices": matched,
    }
