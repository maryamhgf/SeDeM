"""
HotpotQA-distractor-specific helpers for the selection pipeline (Option A).

Bridges between the existing Qasper selection code (`train_selector.py`,
`train_stage2_joint.py`, `eval_qa.py`) and the HotpotQA distractor block
representation:

  * 1 example = 10 paragraphs
  * 1 paragraph = 1 selection block
  * Each paragraph padded/truncated to `chunk_size` (default 128) tokens
  * Compressor produces `memory_slots` (default 32) memory tokens per paragraph
  * Block size = memory_slots (paragraph-level granularity)
  * Total memory tokens per example = num_paragraphs × memory_slots = 10 × 32 = 320

Functions exposed:
  * build_memory_blocks_hotpot(model, paragraphs, device, ...)
        Returns blocks tensor in the same shape contract as
        `build_memory_blocks_for_document`: (1, num_blocks, block_size, D).
  * hotpot_eval_transform(examples)
        HF Datasets-compatible batched map producing
        {input, paragraphs, prompt, answer, all_answers, supporting_facts}.
        `paragraphs` is a list of strings, one per block, in order.
"""

from typing import Dict, List, Tuple

import torch


# -----------------------------------------------------------------------------
# Transform: raw HotpotQA distractor row → flat dict for selection pipelines.
# -----------------------------------------------------------------------------

def hotpot_eval_transform(examples: Dict[str, List]) -> Dict[str, List]:
    """Batched map for `hotpotqa/hotpot_qa` config=distractor.

    Output keys:
      input             : str    full 10-paragraph context (used by current Qasper
                                 dataloader interface; for HotpotQA we mostly
                                 work off `paragraphs` directly)
      paragraphs        : list[str]   per-block text, padded/truncated downstream
      paragraph_titles  : list[str]   titles aligned with paragraphs
      prompt            : str         the question
      answer            : str         the gold answer (single string)
      all_answers       : list[str]   for max-over-references EM/F1 (HotpotQA
                                      has one canonical answer; wrap as singleton)
      supporting_titles : list[str]   for positive-block label extraction
    """
    inputs, all_paras, all_titles = [], [], []
    prompts, answers, all_ans, sup_titles = [], [], [], []

    for ctx, q, a, sf in zip(
        examples["context"], examples["question"], examples["answer"], examples["supporting_facts"],
    ):
        titles = list(ctx["title"])
        sentences_lists = list(ctx["sentences"])
        # Build paragraph strings: "<title>: <sentence0><sentence1>..."
        paragraphs = []
        for t, sents in zip(titles, sentences_lists):
            joined = "".join(sents).strip()
            paragraphs.append(f"{t}: {joined}")
        full_text = "\n\n".join(paragraphs)

        inputs.append(full_text)
        all_paras.append(paragraphs)
        all_titles.append(titles)
        prompts.append(q)
        answers.append(a)
        all_ans.append([a])
        sup_titles.append(list(sf["title"]))

    return {
        "input": inputs,
        "paragraphs": all_paras,
        "paragraph_titles": all_titles,
        "prompt": prompts,
        "answer": answers,
        "all_answers": all_ans,
        "supporting_titles": sup_titles,
    }


# -----------------------------------------------------------------------------
# Block computation: paragraph-level segmentation, paragraph-level memory blocks.
# -----------------------------------------------------------------------------

@torch.no_grad()
def build_memory_blocks_hotpot(
    model,
    paragraphs: List[str],
    device,
    num_blocks: int = 10,
    chunk_size: int = 128,
) -> Tuple[torch.Tensor, List[str], List[List[int]], int, int]:
    """Tokenize each paragraph to a fixed `chunk_size`, concatenate to a single
    sequence, run the frozen compressor, and reshape per-paragraph memory into
    blocks.

    Pads/truncates each paragraph to `chunk_size` tokens. If `len(paragraphs)
    < num_blocks`, pads with empty paragraphs (compressor sees attn-mask zeros
    on padding; positive mask should already be False on those slots).

    Args:
        model:       SeDeMModel instance (frozen)
        paragraphs:  list[str], one per intended block (≤ num_blocks long).
        device:      torch device
        num_blocks:  total blocks to produce (default 10 for distractor split)
        chunk_size:  tokens per paragraph (= model.memory_slots * model.compression_factor)

    Returns:
        blocks       : (1, num_blocks, block_size, D) tensor
                       block_size = memory_slots (paragraph-level granularity)
        paragraphs   : list[str] (echo, may be padded)
        ids_per_block: list[list[int]] of token IDs per block, length num_blocks
        num_segments : int (= num_blocks, since 1 paragraph = 1 segment)
        num_blocks   : int (= num_blocks input)
    """
    pad_id = (
        model.tokenizer.pad_token_id
        if model.tokenizer.pad_token_id is not None
        else (model.tokenizer.eos_token_id or 0)
    )

    if chunk_size != model.memory_slots * model.compression_factor:
        raise ValueError(
            f"chunk_size ({chunk_size}) must equal memory_slots * compression_factor "
            f"({model.memory_slots} * {model.compression_factor} = "
            f"{model.memory_slots * model.compression_factor})."
        )

    # Pad paragraphs list to num_blocks by appending empty strings.
    paragraphs = list(paragraphs)
    while len(paragraphs) < num_blocks:
        paragraphs.append("")
    paragraphs = paragraphs[:num_blocks]

    # Per-paragraph tokenization with fixed chunk_size (pad/truncate).
    ids_per_block: List[List[int]] = []
    for p in paragraphs:
        out = model.tokenizer(
            p, max_length=chunk_size, truncation=True, padding="max_length",
            add_special_tokens=False, return_attention_mask=False,
        )
        ids_per_block.append(list(out["input_ids"]))

    # Flatten to single sequence; compressor segments by chunk_size internally.
    flat = [i for block in ids_per_block for i in block]
    ctx_ids = torch.tensor([flat], dtype=torch.long, device=device)

    memory = model.compress_segments(ctx_ids)            # (1, M_total, D), float32
    M_total, D = memory.size(1), memory.size(2)
    expected_M = num_blocks * model.memory_slots
    if M_total != expected_M:
        raise RuntimeError(
            f"Compressor produced {M_total} mem tokens, expected {expected_M} "
            f"(num_blocks={num_blocks} × memory_slots={model.memory_slots})."
        )

    # Block-size = memory_slots: 1 block = 1 paragraph's full mem bundle.
    block_size = model.memory_slots
    blocks = memory.view(1, num_blocks, block_size, D)
    return blocks, paragraphs, ids_per_block, num_blocks, num_blocks
