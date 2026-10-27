"""
Selector-only training on a frozen SeDeMModel checkpoint.

Freezes: backbone, compressor, decompressor, LoRA decoder, mem_norm/scale.
Trains: the pooled-query selector (PooledBlockEncoder + PooledQueryEncoder) or the late-interaction selector (LateInteractionScorer).

Loss: InfoNCE over evidence-based positive mask. Scores come from the chosen
selector. Per-paper batching amortizes the compression forward pass across
all questions about that paper.

Output: safetensors file with selector weights, loadable by eval_qa.py under
--selection_mode {r1_learned, r2_learned}.

Usage:
    python train_selector.py \
        --selector_type late_interaction \
        --base_checkpoint <output_dir>/default-config/model.safetensors \
        --output_dir <output_dir>/r1_retriever_run_N \
        --num_epochs 3 \
        --lr 1e-3 \
        --temperature 0.07 \
        ...
"""
import argparse
import json
import math
import os
import sys
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from safetensors.torch import load_file as safe_load, save_file as safe_save
from datasets import load_dataset

import transformers
import transformers.utils.import_utils
import transformers.trainer

from sedem.modeling import SeDeMModel, ModelArguments, TrainingArguments
from sedem.data import get_qa_eval_transform
from sedem.evidence_labels_qasper import evidence_labels_from_qasper_spans
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
from sedem.blocks_hotpot import (
    build_memory_blocks_hotpot,
    hotpot_eval_transform,
)
from sedem.selector import PooledBlockEncoder, PooledQueryEncoder, LateInteractionScorer


from sedem.selector import PooledQuerySelector, LateInteractionSelector, selection_infonce_loss, selection_margin_loss



# ---------------------------------------------------------------------------
# Frozen model loader
# ---------------------------------------------------------------------------

def load_frozen_codec(base_ckpt, model_args, training_args, device):
    """Load SeDeMModel with the default config flags pre-set, restore ckpt, freeze all."""
    model = SeDeMModel(model_args, training_args, lora_config=None)
    print(f"Loading frozen base checkpoint from {base_ckpt} ...")
    state = safe_load(base_ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # It's normal that backbone weights are "missing" — they come from the HF
    # auto-loaded base Llama, not from our stage-2 safetensors.
    print(f"  restored with {len(unexpected)} unexpected keys, "
          f"{sum(1 for k in missing if not k.startswith('backbone.'))} non-backbone missing.")
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def make_default_model_args(args=None):
    """Default SeDeM architecture arguments (the Stage-2 settings of the configs;
    override per config through `arch_args` in `scripts/common.sh`).
    `args` (parsed CLI) optionally overrides multi-layer mixing and inject layer
    so this trainer can also serve checkpoints trained with mix_extraction_layers=True.
    Also supports learnable_query (Perceiver) base ckpts by passing the relevant flags.
    """
    cv = getattr(args, "compressor_type", "pooled_linear") if args else "pooled_linear"
    ma = ModelArguments(
        model_name_or_path=getattr(args, "model_name_or_path", "meta-llama/Llama-3.2-3B") if args else "meta-llama/Llama-3.2-3B",
        encoder_name_or_path=getattr(args, "encoder_name_or_path", "") if args else "",
        decoder_name_or_path=getattr(args, "decoder_name_or_path", "") if args else "",
        compressor_type=cv,
        compressor_hidden_dim=getattr(args, "compressor_hidden_dim", 3072) if args else 3072,
        compressor_num_heads=8,
        extraction_layer=getattr(args, "extraction_layer", 14) if args else 14,
        # pooled_linear flags (kept for backward compat)
        compressor_no_global_context=getattr(args, "compressor_no_global_context", True) if args else True,
        compressor_slot_position_embedding=getattr(args, "compressor_slot_position_embedding", False) if args else False,
        compressor_linear_projection=getattr(args, "compressor_linear_projection", True) if args else True,
        # learnable_query (Perceiver) flags
        compressor_gated=getattr(args, "compressor_gated", True) if args else True,
        compressor_ignore_prev_mem=getattr(args, "compressor_ignore_prev_mem", False) if args else False,
        compressor_topics_init=getattr(args, "compressor_topics_init", "random") if args else "random",
        compressor_qk_norm=getattr(args, "compressor_qk_norm", False) if args else False,
        compressor_qk_norm_init_scale=getattr(args, "compressor_qk_norm_init_scale", 10.0) if args else 10.0,
        slot_competition=getattr(args, "slot_competition", False) if args else False,
        out_proj_mode=getattr(args, "out_proj_mode", "standard") if args else "standard",
        compressor_blend_mode=getattr(args, "compressor_blend_mode", "content_first") if args else "content_first",
        compressor_no_ffn=getattr(args, "compressor_no_ffn", False) if args else False,
        compressor_num_self_attn_layers=getattr(args, "compressor_num_self_attn_layers", 0) if args else 0,
        compressor_num_iter_blocks=getattr(args, "compressor_num_iter_blocks", 1) if args else 1,
        decompressor_type="mlp",
        inject_decompressed=True,
        injection_layer=getattr(args, "injection_layer", -999) if args else -999,
        lora_decoder=True,
        lora_r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        lora_target_modules="q_proj,k_proj,v_proj,o_proj",
        mix_extraction_layers=getattr(args, "mix_extraction_layers", False) if args else False,
        mixing_layers=getattr(args, "mixing_layers", "10,12,14,16") if args else "10,12,14,16",
        mixing_norm=getattr(args, "mixing_norm", "per_layer_ln") if args else "per_layer_ln",
        mixing_init=getattr(args, "mixing_init", "uniform") if args else "uniform",
        mixing_learnable=getattr(args, "mixing_learnable", False) if args else False,
    )
    return ma


def make_default_training_args(output_dir, args=None):
    ta = TrainingArguments(
        output_dir=output_dir,
        memory_slots_per_segment=getattr(args, "memory_slots_per_segment", 32) if args else 32,
        compression_factor=getattr(args, "compression_factor", 4) if args else 4,
        bf16=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        report_to="none",
    )
    return ta


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def iter_questions_by_document(dataset, rng):
    """Yield (paper_text, [(question, evidence), ...]) groups so the compressor
    forward is amortized across all questions about the same paper."""
    # Dataset rows are already flattened to (input, prompt, answer, all_answers, evidence).
    # Papers repeat as `input`; group by paper text.
    groups = {}
    for row in dataset:
        paper = row["input"]
        groups.setdefault(paper, []).append((row["prompt"], row.get("evidence", [])))
    # Shuffle papers (not questions within a paper — deterministic within group).
    paper_keys = list(groups.keys())
    rng.shuffle(paper_keys)
    for p in paper_keys:
        yield p, groups[p]


@torch.no_grad()
def build_memory_blocks_for_document(model, paper_text, device, max_length=4096, block_size=None):
    """Tokenize, compress, reshape memory into blocks. All under no_grad
    because compressor+backbone are frozen.

    block_size:
        - None (default): use `model.compression_factor` → Qasper sub-segment
          blocks (8 blocks per segment at M=32, c=4).
        - `model.memory_slots`: 1 block = 1 full segment's mem bundle. Used by the
          HotpotQA Option A path where 1 paragraph = 1 segment = 1 block.
    """
    ids = model.tokenizer(
        paper_text, max_length=max_length, truncation=True, padding=False,
        add_special_tokens=False, return_attention_mask=False,
    )["input_ids"]
    ctx_ids = torch.tensor([ids], dtype=torch.long, device=device)

    memory = model.compress_segments(ctx_ids)     # (1, M_total, D), float32
    M_total, D = memory.size(1), memory.size(2)
    num_segments = M_total // model.memory_slots
    if block_size is None:
        block_size = model.compression_factor
    if model.memory_slots % block_size != 0:
        raise ValueError(
            f"memory_slots ({model.memory_slots}) must be divisible by block_size ({block_size})."
        )
    blocks_per_segment = model.memory_slots // block_size
    num_blocks = num_segments * blocks_per_segment

    # (1, num_segments, memory_slots, D) → (1, num_blocks, block_size, D)
    blocks = memory.view(1, num_segments, model.memory_slots, D).view(
        1, num_blocks, block_size, D,
    )
    return blocks, paper_text, ids, num_segments, num_blocks


def encode_query(model, question_text, device):
    """Backbone forward over the question to get hidden states at extraction_layer
    (the layer the compressor writes from). Under no_grad: backbone is frozen.

    Cross-model: backbone is the encoder (enc_dim, e.g. 2048), but the selector
    operates in dec_dim (e.g. 3072) since blocks are slot tokens in dec_dim.
    Project question_hs through the compressor's input_proj so it lives in the
    same space as the blocks. In same-model mode input_proj is Identity (no-op).
    """
    q_ids = model.tokenizer(
        question_text, truncation=True, padding=False,
        return_attention_mask=False, add_special_tokens=False,
    )["input_ids"]
    if len(q_ids) == 0:
        q_ids = [model.tokenizer.eos_token_id or 0]
    q_tensor = torch.tensor([q_ids], dtype=torch.long, device=device).clamp(
        max=model.original_vocab_size - 1,
    )
    with torch.no_grad():
        out = model.backbone(q_tensor, output_hidden_states=True)
        hs = out.hidden_states[model.extraction_layer].float()  # (1, Q_tok, enc_dim)
    # Lift enc_dim → dec_dim if cross-model (Identity in same-model).
    if hasattr(model, "compressor") and hasattr(model.compressor, "input_proj"):
        with torch.no_grad():
            hs = model.compressor.input_proj(hs)  # (1, Q_tok, dec_dim)
    return hs


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ---- Model (frozen) -----------------------------------------------------
    model_args = make_default_model_args(args)
    training_args = make_default_training_args(args.output_dir, args)
    model = load_frozen_codec(args.base_checkpoint, model_args, training_args, device)

    # ---- Selector head (trainable) -----------------------------------------
    d_model = model.dim
    # Cross-model note: `encode_query` already projects q_hs from enc_dim
    # → dec_dim via compressor.input_proj. Both Q and K live in dec_dim, so the
    # selector head uses a single d_model.
    if args.selector_type == "pooled":
        head = PooledQuerySelector(d_model, d_key=args.d_key).to(device).float()
    elif args.selector_type == "late_interaction":
        head = LateInteractionSelector(d_model, num_heads=args.selector_heads).to(device).float()
    else:
        raise ValueError(f"selector_type must be 'pooled' or 'late_interaction', got {args.selector_type!r}")
    if args.warm_start is not None:
        from safetensors.torch import load_file as safe_load
        ws = safe_load(args.warm_start)
        missing, unexpected = head.load_state_dict(ws, strict=False)
        print(f"[warm-start] loaded {args.warm_start}  "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    head.train()
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[selector head] {args.selector_type}, params={n_params:,}, "
          f"lambda_margin={args.lambda_margin}, margin={args.margin}")

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.0)

    # ---- Data ---------------------------------------------------------------
    if args.dataset == "qasper":
        print("Loading Qasper train split...")
        ds = load_dataset("allenai/qasper", cache_dir=args.dataset_cache_dir,
                          revision="refs/convert/parquet")
        tr = get_qa_eval_transform("qasper")
        train_rows = ds["train"].map(
            tr, batched=True, remove_columns=ds["train"].column_names,
            load_from_cache_file=False,
        )
        if args.max_train_questions is not None:
            train_rows = train_rows.select(range(min(args.max_train_questions, len(train_rows))))
        print(f"Train rows (Qasper, after flatten): {len(train_rows)}")
        val_rows = None
        if args.eval_every > 0:
            vr = ds["validation"].map(
                tr, batched=True, remove_columns=ds["validation"].column_names,
                load_from_cache_file=False,
            )
            if args.max_val_questions is not None:
                vr = vr.select(range(min(args.max_val_questions, len(vr))))
            val_rows = vr
            print(f"Val rows: {len(val_rows)}")
    elif args.dataset == "hotpot":
        print("Loading HotpotQA distractor train split...")
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor",
                          cache_dir=args.dataset_cache_dir, trust_remote_code=True)
        transform_fn = hotpot_eval_transform
        train_rows = ds["train"].map(
            transform_fn, batched=True,
            remove_columns=ds["train"].column_names, load_from_cache_file=False,
        )
        if args.max_train_questions is not None:
            train_rows = train_rows.select(range(min(args.max_train_questions, len(train_rows))))
        print(f"Train rows (HotpotQA): {len(train_rows)}")
        val_rows = None
        if args.eval_every > 0:
            vr = ds["validation"].map(
                transform_fn, batched=True,
                remove_columns=ds["validation"].column_names, load_from_cache_file=False,
            )
            if args.max_val_questions is not None:
                vr = vr.select(range(min(args.max_val_questions, len(vr))))
            val_rows = vr
            print(f"Val rows: {len(val_rows)}")
    elif args.dataset == "twowiki":
        print("Loading 2WikiMultihopQA train split...")
        ds = load_dataset("xanhho/2WikiMultihopQA", revision="refs/convert/parquet",
                          cache_dir=args.dataset_cache_dir)
        transform_fn = twowiki_eval_transform
        train_rows = ds["train"].map(
            transform_fn, batched=True,
            remove_columns=ds["train"].column_names, load_from_cache_file=False,
        )
        if args.max_train_questions is not None:
            train_rows = train_rows.select(range(min(args.max_train_questions, len(train_rows))))
        print(f"Train rows (2Wiki): {len(train_rows)}")
        val_rows = None
        if args.eval_every > 0:
            vr = ds["validation"].map(
                transform_fn, batched=True,
                remove_columns=ds["validation"].column_names, load_from_cache_file=False,
            )
            if args.max_val_questions is not None:
                vr = vr.select(range(min(args.max_val_questions, len(vr))))
            val_rows = vr
            print(f"Val rows: {len(val_rows)}")
    elif args.dataset == "musique":
        print("Loading MuSiQue train split...")
        ds = load_dataset("dgslibisey/MuSiQue", cache_dir=args.dataset_cache_dir)
        transform_fn = musique_eval_transform
        train_rows = ds["train"].map(
            transform_fn, batched=True,
            remove_columns=ds["train"].column_names, load_from_cache_file=False,
        )
        if args.max_train_questions is not None:
            train_rows = train_rows.select(range(min(args.max_train_questions, len(train_rows))))
        print(f"Train rows (MuSiQue): {len(train_rows)}")
        val_rows = None
        if args.eval_every > 0:
            vr = ds["validation"].map(
                transform_fn, batched=True,
                remove_columns=ds["validation"].column_names, load_from_cache_file=False,
            )
            if args.max_val_questions is not None:
                vr = vr.select(range(min(args.max_val_questions, len(vr))))
            val_rows = vr
            print(f"Val rows: {len(val_rows)}")
    else:
        raise ValueError(
            f"Unknown --dataset {args.dataset!r}; use 'qasper', 'hotpot', 'twowiki' or 'musique'."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "train_log.jsonl")
    ckpt_path = os.path.join(args.output_dir, "selector.safetensors")
    cfg_path = os.path.join(args.output_dir, "selector_config.json")
    with open(cfg_path, "w") as f:
        json.dump({
            "selector_type": args.selector_type,
            "d_key": args.d_key if args.selector_type == "pooled" else None,
            "selector_heads": args.selector_heads if args.selector_type == "late_interaction" else None,
            "temperature": args.temperature,
            "lr": args.lr,
            "num_epochs": args.num_epochs,
            "lambda_margin": args.lambda_margin,
            "margin": args.margin,
            "warm_start": args.warm_start,
            "base_checkpoint": args.base_checkpoint,
            "seed": args.seed,
        }, f, indent=2)

    rng = random.Random(args.seed)
    global_step = 0
    best_val_recall = -1.0
    log_f = open(log_path, "w", buffering=1)

    # K list for periodic recall eval depends on dataset size:
    # Qasper ~230 blocks/paper -> sweep 8/32. Multi-hop datasets have:
    #   hotpot: 10 blocks (2 gold)   -> sweep 2/4/6/8
    #   twowiki: 10 blocks (2 gold)  -> sweep 2/4/6/8
    #   musique: 20 blocks (2-4 gold) -> sweep 2/4/6/8/10
    if args.dataset == "qasper":
        eval_k_list = (8, 32)
    elif args.dataset == "musique":
        eval_k_list = (2, 4, 6, 8, 10)
    else:  # hotpot, twowiki
        eval_k_list = (2, 4, 6, 8)

    # Dataset-specific dispatch: returns (blocks, num_blocks, pos_mask).
    def blocks_and_labels_for_example(row):
        if args.dataset == "qasper":
            blocks, _, _, _, num_blocks = build_memory_blocks_for_document(
                model, row["input"], device)
            pos_mask, _, _, _stats = evidence_labels_from_qasper_spans(
                row["input"], row.get("evidence") or [], model.tokenizer, model,
                max_length=4096,
            )
        elif args.dataset == "twowiki":
            blocks, _, _, _, num_blocks = build_memory_blocks_twowiki(
                model, row["paragraphs"], device,
                num_blocks=10, chunk_size=model.memory_slots * model.compression_factor,
            )
            pos_mask, _ = evidence_labels_twowiki(
                row["paragraph_titles"], row["supporting_titles"], num_blocks=10,
            )
        elif args.dataset == "musique":
            blocks, _, _, _, num_blocks = build_memory_blocks_musique(
                model, row["paragraphs"], device,
                num_blocks=20, chunk_size=model.memory_slots * model.compression_factor,
            )
            pos_mask, _ = evidence_labels_musique(
                row["is_supporting"], num_blocks=20,
            )
        else:  # hotpot
            blocks, _, _, _, num_blocks = build_memory_blocks_hotpot(
                model, row["paragraphs"], device,
                num_blocks=10, chunk_size=model.memory_slots * model.compression_factor,
            )
            ctx_dict = {"title": row["paragraph_titles"], "sentences": [[] for _ in row["paragraph_titles"]]}
            sf_dict = {"title": row["supporting_titles"], "sent_id": []}
            pos_mask, _ = evidence_labels_from_supporting_facts(
                ctx_dict, sf_dict, num_blocks=10,
            )
        return blocks, num_blocks, pos_mask

    def evaluate_selection_recall(head, rows, k_list=eval_k_list):
        head.eval()
        sums = {k: 0 for k in k_list}
        n = 0
        with torch.no_grad():
            for row in rows:
                blocks, num_blocks, pos_mask = blocks_and_labels_for_example(row)
                if pos_mask.sum() == 0:
                    continue
                q_hs = encode_query(model, row["prompt"], device)
                scores = head(q_hs, blocks).squeeze(0).float()
                pos_idx = set(torch.nonzero(pos_mask).flatten().tolist())
                for k in k_list:
                    topk = scores.topk(min(k, num_blocks))[1].tolist()
                    sums[k] += len(pos_idx.intersection(topk)) / max(len(pos_idx), 1)
                n += 1
        head.train()
        if n == 0:
            return {f"recall@{k}": 0.0 for k in k_list}
        return {f"recall@{k}": sums[k] / n for k in k_list}

    print(f"\nStarting training: {args.num_epochs} epochs, LR={args.lr}, τ={args.temperature}")
    for epoch in range(args.num_epochs):
        t_epoch = time.time()
        epoch_loss = 0.0
        epoch_skipped = 0
        epoch_steps = 0

        # Build dataset-specific iterator.
        # Qasper: yield (paper_text, [(q, ev), ...]) — compress once per paper.
        # Multi-hop (hotpot/twowiki/musique): yield (row, [(q,)]) — one question per row.
        if args.dataset == "qasper":
            iterator = iter_questions_by_document(train_rows, rng)
        else:
            shuffled = list(range(len(train_rows)))
            rng.shuffle(shuffled)
            def _row_iter():
                for i in shuffled:
                    r = train_rows[i]
                    yield r, [(r["prompt"],)]   # single-element entry; full row available via group_key
            iterator = _row_iter()

        for paper_idx, (group_key, q_list) in enumerate(iterator):
            # Compress once per group (paper for Qasper, row for multi-hop).
            if args.dataset == "qasper":
                paper_text = group_key
                blocks, _, _, _, num_blocks = build_memory_blocks_for_document(model, paper_text, device)
            elif args.dataset == "twowiki":
                blocks, _, _, _, num_blocks = build_memory_blocks_twowiki(
                    model, group_key["paragraphs"], device,
                    num_blocks=10, chunk_size=model.memory_slots * model.compression_factor,
                )
            elif args.dataset == "musique":
                blocks, _, _, _, num_blocks = build_memory_blocks_musique(
                    model, group_key["paragraphs"], device,
                    num_blocks=20, chunk_size=model.memory_slots * model.compression_factor,
                )
            else:  # hotpot
                blocks, _, _, _, num_blocks = build_memory_blocks_hotpot(
                    model, group_key["paragraphs"], device,
                    num_blocks=10, chunk_size=model.memory_slots * model.compression_factor,
                )

            # Per-question inner loop: accumulate losses, step once per group.
            losses = []
            inf_components = []
            mgn_components = []
            for entry in q_list:
                if args.dataset == "qasper":
                    q_text, evidence = entry
                    pos_mask, _, _, _stats = evidence_labels_from_qasper_spans(
                        group_key, evidence or [], model.tokenizer, model, max_length=4096,
                    )
                elif args.dataset == "twowiki":
                    (q_text,) = entry
                    pos_mask, _ = evidence_labels_twowiki(
                        group_key["paragraph_titles"], group_key["supporting_titles"], num_blocks=10,
                    )
                elif args.dataset == "musique":
                    (q_text,) = entry
                    pos_mask, _ = evidence_labels_musique(
                        group_key["is_supporting"], num_blocks=20,
                    )
                else:  # hotpot
                    (q_text,) = entry
                    ctx_dict = {"title": group_key["paragraph_titles"],
                                "sentences": [[] for _ in group_key["paragraph_titles"]]}
                    sf_dict = {"title": group_key["supporting_titles"], "sent_id": []}
                    pos_mask, _ = evidence_labels_from_supporting_facts(
                        ctx_dict, sf_dict, num_blocks=10,
                    )
                pos_mask = pos_mask.to(device)
                if not pos_mask.any():
                    epoch_skipped += 1
                    continue
                q_hs = encode_query(model, q_text, device)
                scores = head(q_hs, blocks).squeeze(0).float()  # (num_blocks,)
                L_inf = selection_infonce_loss(scores, pos_mask, args.temperature)
                if args.lambda_margin > 0:
                    L_mgn = selection_margin_loss(scores, pos_mask, margin=args.margin)
                    loss = L_inf + args.lambda_margin * L_mgn
                else:
                    L_mgn = torch.zeros((), device=device)
                    loss = L_inf
                losses.append(loss)
                inf_components.append(L_inf.detach().item())
                mgn_components.append(L_mgn.detach().item())

            if not losses:
                continue
            total = torch.stack(losses).mean()
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()

            epoch_loss += total.item() * len(losses)
            epoch_steps += len(losses)
            recent_inf = sum(inf_components) / max(len(inf_components), 1)
            recent_mgn = sum(mgn_components) / max(len(mgn_components), 1)
            global_step += 1

            if global_step % args.log_every == 0:
                avg = epoch_loss / max(epoch_steps, 1)
                log_f.write(json.dumps({
                    "step": global_step, "epoch": epoch, "paper_idx": paper_idx,
                    "avg_loss": round(avg, 4),
                    "L_inf": round(recent_inf, 4),
                    "L_mgn": round(recent_mgn, 4),
                    "n_papers": paper_idx + 1,
                    "n_questions": epoch_steps,
                }) + "\n")
                print(f"  [step {global_step:4d}] epoch={epoch} paper={paper_idx} "
                      f"avg_loss={avg:.4f} L_inf={recent_inf:.4f} L_mgn={recent_mgn:.4f}")

            if val_rows is not None and global_step % args.eval_every == 0 and global_step > 0:
                recalls = evaluate_selection_recall(head, val_rows)
                # Best ckpt selection metric: largest K in the eval list.
                # Qasper: recall@32. HotpotQA: recall@8.
                primary_key = f"recall@{max(eval_k_list)}"
                primary_val = recalls[primary_key]
                print(f"  [eval step={global_step}] " +
                      ", ".join(f"{k}={v:.4f}" for k, v in recalls.items()))
                log_f.write(json.dumps({
                    "step": global_step, "eval": True, **recalls,
                }) + "\n")
                if primary_val > best_val_recall:
                    best_val_recall = primary_val
                    safe_save({k: v.contiguous().cpu() for k, v in head.state_dict().items()},
                              ckpt_path)
                    print(f"    ↳ saved best ckpt ({primary_key}={primary_val:.4f}) → {ckpt_path}")

        dt = time.time() - t_epoch
        avg = epoch_loss / max(epoch_steps, 1)
        print(f"[epoch {epoch}] avg_loss={avg:.4f} steps={epoch_steps} "
              f"skipped_no_positives={epoch_skipped} time={dt/60:.1f}min")

    # Final save (always, regardless of val eval)
    safe_save({k: v.contiguous().cpu() for k, v in head.state_dict().items()},
              ckpt_path if best_val_recall < 0 else ckpt_path.replace(".safetensors", "_final.safetensors"))
    print(f"[done] selector ckpt saved. best val recall@{max(eval_k_list)} = {best_val_recall:.4f}")
    log_f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selector_type", choices=["pooled", "late_interaction"], required=True)
    ap.add_argument("--dataset", choices=["qasper", "hotpot", "twowiki", "musique"],
                    default="qasper",
                    help="Which dataset / block-segmentation regime to use.")
    ap.add_argument("--base_checkpoint", required=True,
                    help="Stage-2 (Qasper) or Stage-1 (HotpotQA) safetensors (frozen)")
    ap.add_argument("--output_dir", required=True)
    # Cross-model support: separate encoder + decoder names. Empty falls back to
    # model_name_or_path. For cross-model 1B/3B set encoder=Llama-3.2-1B, decoder=Llama-3.2-3B.
    ap.add_argument("--model_name_or_path", default="meta-llama/Llama-3.2-3B",
                    help="Default model (used as fallback for encoder/decoder if those are empty).")
    ap.add_argument("--encoder_name_or_path", default="",
                    help="Cross-model encoder name. Empty = use model_name_or_path.")
    ap.add_argument("--decoder_name_or_path", default="",
                    help="Cross-model decoder name. Empty = use model_name_or_path.")
    ap.add_argument("--extraction_layer", type=int, default=14,
                    help="Encoder layer to extract HS from (must match the trained ckpt).")
    ap.add_argument("--memory_slots_per_segment", type=int, default=32,
                    help="N_M: memory slots per context segment (segment length T = N_M * C).")
    ap.add_argument("--compression_factor", type=int, default=4,
                    help="C: tokens summarised by each memory slot.")
    ap.add_argument("--dataset_cache_dir", default=None)
    ap.add_argument("--num_epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--lambda_margin", type=float, default=0.0,
                    help="Hinge margin loss weight. 0 = InfoNCE only (default).")
    ap.add_argument("--margin", type=float, default=2.0,
                    help="Hinge margin: enforce min(pos) - max(neg) >= margin.")
    ap.add_argument("--warm_start", default=None,
                    help="Optional path to a selector .safetensors to initialize from.")
    # Optional compressor-variant overrides (defaults match SeDeM / default config)
    ap.add_argument("--compressor_type", type=str, default="pooled_linear",
                    choices=["pooled_linear", "learnable_query"])
    ap.add_argument("--compressor_no_global_context", type=lambda s: s.lower() == "true", default=True)
    ap.add_argument("--compressor_slot_position_embedding", type=lambda s: s.lower() == "true", default=False)
    ap.add_argument("--compressor_linear_projection", type=lambda s: s.lower() == "true", default=True)
    # learnable_query (Perceiver) flags
    ap.add_argument("--compressor_gated", type=lambda s: s.lower() == "true", default=True)
    ap.add_argument("--compressor_ignore_prev_mem", type=lambda s: s.lower() == "true", default=False)
    ap.add_argument("--compressor_topics_init", type=str, default="random",
                    choices=["random", "sinusoidal", "orthogonal"])
    ap.add_argument("--compressor_qk_norm", type=lambda s: s.lower() == "true", default=False)
    ap.add_argument("--compressor_qk_norm_init_scale", type=float, default=10.0)
    ap.add_argument("--slot_competition", type=lambda s: s.lower() == "true", default=False)
    ap.add_argument("--out_proj_mode", type=str, default="standard",
                    choices=["standard", "identity", "block_diagonal"])
    ap.add_argument("--compressor_blend_mode", type=str, default="content_first")
    ap.add_argument("--compressor_no_ffn", type=lambda s: s.lower() == "true", default=False)
    ap.add_argument("--compressor_num_self_attn_layers", type=int, default=0)
    ap.add_argument("--compressor_num_iter_blocks", type=int, default=1)
    # Codec bottleneck = decoder hidden size (3072 for 3B, 2048 for 1B same-model).
    # Default 3072 preserves prior 3B-decoder callers; must match the base ckpt.
    ap.add_argument("--compressor_hidden_dim", type=int, default=3072)
    # Optional model-arch overrides for ckpts trained with multi-layer mixing
    ap.add_argument("--mix_extraction_layers", type=lambda s: s.lower() == "true", default=False)
    ap.add_argument("--mixing_layers", type=str, default="10,12,14,16")
    ap.add_argument("--mixing_norm", type=str, default="per_layer_ln")
    ap.add_argument("--mixing_init", type=str, default="uniform")
    ap.add_argument("--mixing_learnable", type=lambda s: s.lower() == "true", default=False)
    ap.add_argument("--injection_layer", type=int, default=-999)
    ap.add_argument("--d_key", type=int, default=256, help="pooled-query selector only")
    ap.add_argument("--selector_heads", type=int, default=4, help="late-interaction selector only")
    ap.add_argument("--max_train_questions", type=int, default=None,
                    help="Cap train set (debug). None = use all.")
    ap.add_argument("--max_val_questions", type=int, default=50,
                    help="Cap val set for periodic recall eval.")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=500,
                    help="Steps between recall@K evals. 0 = never.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
