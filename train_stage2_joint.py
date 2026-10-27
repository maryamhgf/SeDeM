"""
Joint selection-aware Stage 2 training.

Base: Stage-1 checkpoint (compressor + decompressor + mem_norm/scale, no LoRA).
      Adds a **fresh LoRA decoder** and loads a trained late-interaction selector as warm start.

All modules trainable from step 0 (no phase-based freezing). Separate LRs per
module group:
    selector:    lr_selector    (default 1e-3)
    compressor:   lr_compressor   (default 5e-5)
    decompressor: lr_decompressor (default 5e-5)
    LoRA + rest:  lr_lora         (default 1e-4)

Selection in the decoder path from step 0:
    compress → select top-K=24 (label curriculum) → decompress all → gather K blocks
    → inject at layer ℓ via two-phase forward → LoRA decoder → L_QA on answer

Loss:  L = L_QA + λ_ret · L_ret + λ_decomp · L_decomp   (λ_ret=0.1, λ_decomp=0.01)

Label curriculum (uses Qasper's evidence field as supervised selection signal
during warmup; the technique is identical to prior "evidence-label" curriculum —
renamed to match the `label_k*` eval config naming convention):
    p_label = 1.0                for steps [0, warmup]
    p_label: 1.0 → 0.0 linearly  for steps [warmup, warmup + anneal]
    p_label = 0.0                for steps thereafter

K handling: fixed K=24 at training. Eval sweep at K ∈ {16, 24, 32}.

Batching: per-paper (compressor amortizes across questions on the same paper).
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
import transformers.utils.import_utils
import transformers.trainer

from safetensors.torch import load_file as safe_load, save_file as safe_save
from datasets import load_dataset

from sedem.modeling import (
    SeDeMModel,
    ModelArguments,
    TrainingArguments,
)
from sedem.data import get_qa_eval_transform
from sedem.evidence_labels_qasper import evidence_labels_from_qasper_spans
from sedem.evidence_labels_hotpot import evidence_labels_from_supporting_facts
from sedem.blocks_hotpot import (
    build_memory_blocks_hotpot,
    hotpot_eval_transform,
)
from sedem.selector import PooledQuerySelector, LateInteractionSelector, selection_infonce_loss, selection_margin_loss


from sedem.pipeline import (
    evidence_label_probability, make_default_model_args, make_default_training_args, load_sedem_model,
    load_selector, set_trainable_for_joint_training, compress_context, decompress_all_segments, encode_query_from_prompt,
    select_blocks_with_label_curriculum, gather_selected_blocks, decode_with_injected_memory, normalize_memory, embed_prompt_text,
    prefill_with_injected_memory_cached, decode_step_with_injected_memory_cached, answer_ce_loss,
)


# ---------------------------------------------------------------------------
# Per-question training step
# ---------------------------------------------------------------------------

def train_step(model, selector, question_text, answer_text, paper_text, evidence,
               memory, num_segments, num_blocks, blocks, block_size, tokens_per_block,
               total_mem_len_of_memory, inject_layer,
               global_step, K, args, rng, device,
               dataset="qasper", supporting_titles=None, paragraph_titles=None):
    """Process one (paper, question, answer) with pre-computed memory/blocks.

    Returns dict with loss components or None if skipped (no positives, no answer, etc.).
    """
    tok = model.tokenizer
    # Build prompt_answer_ids + labels in the format existing code expects
    ctx_max = args.context_max_length

    # Chat template question (matches stage2_tokenize_function)
    has_chat = (hasattr(tok, "chat_template") and tok.chat_template is not None)
    if has_chat:
        from sedem.data import QA_SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": QA_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question_text}"},
        ]
        try:
            chat = tok.apply_chat_template(messages, enable_thinking=False,
                                           tokenize=False, add_generation_prompt=True)
        except TypeError:
            chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        chat_ids = tok(chat, add_special_tokens=False, truncation=False)["input_ids"]
    else:
        chat_ids = tok(question_text, add_special_tokens=False, truncation=False)["input_ids"]

    answer_ids = tok(answer_text, add_special_tokens=False, truncation=False)["input_ids"] + [model.eos_id]
    # We use the REAL K=24 memory in the prompt, but the decoder won't see the placeholder
    # tokens — those are just a bookkeeping convention the existing code uses to size labels.
    # We use total_mem_len_of_memory (= num_segments * memory_slots) as placeholders here because
    # decode_with_injected_memory will strip the same count.
    total_mem_len = total_mem_len_of_memory
    mem_placeholder_id = model.mem_id_start
    prompt_ids = [mem_placeholder_id] * total_mem_len + [model.ft_token_id] + chat_ids
    full_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids

    prompt_answer_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    labels_t = torch.tensor([labels], dtype=torch.long, device=device)

    # Positive mask
    if dataset == "hotpot":
        ctx_dict = {"title": paragraph_titles, "sentences": [[] for _ in paragraph_titles]}
        sf_dict = {"title": supporting_titles, "sent_id": []}
        pos_mask, _ = evidence_labels_from_supporting_facts(
            ctx_dict, sf_dict, num_blocks=num_blocks,
        )
    else:
        pos_mask, _, num_blocks_ev, _ = evidence_labels_from_qasper_spans(
            paper_text, evidence or [], tok, model, max_length=ctx_max,
        )
        assert num_blocks_ev == num_blocks, (num_blocks_ev, num_blocks)
    pos_mask = pos_mask.to(device)

    # Question hidden states for selector input (at layer ℓ)
    label_start = (labels_t != -100).float().argmax(dim=1).item()
    question_hs = encode_query_from_prompt(model, prompt_answer_ids, total_mem_len, label_start)

    # Selector scores + InfoNCE (always trainable). With the late-interaction selector,
    # scores are bounded in [-Q_tok, Q_tok] instead of [-1, 1]. τ must be rescaled
    # so logits/τ stay in a comparable range. The default --temperature in the
    # script is bumped from 0.07 → 1.0 to compensate (≈ 0.07 · Q_tok at typical
    # Q_tok≈14 for HotpotQA short questions).
    scores = selector(question_hs, blocks).squeeze(0)  # (num_blocks,)
    if pos_mask.any():
        L_ret = selection_infonce_loss(scores, pos_mask, args.temperature)
        L_margin = selection_margin_loss(scores, pos_mask, margin=args.margin) if args.lambda_margin > 0 else torch.zeros((), device=device)
    else:
        L_ret = torch.zeros((), device=device)  # no signal this question
        L_margin = torch.zeros((), device=device)

    # Top-K selection with label curriculum
    p_l = evidence_label_probability(global_step, args.label_warmup, args.label_anneal)
    sel_idx, from_label = select_blocks_with_label_curriculum(scores, pos_mask, K, p_l, rng, device)

    # Decompress all segments (needed for gather + optional L_decomp preservation).
    # The decompressor is trainable from step 0 — always enable grad.
    decompressed_hs, _ = decompress_all_segments(model, memory, decompressor_trainable=True)

    # Gather K blocks' reconstructed tokens
    selected_hs = gather_selected_blocks(decompressed_hs, sel_idx, tokens_per_block)

    # Two-phase decoder forward on gathered Ĥ + text
    text_logits = decode_with_injected_memory(model, selected_hs, prompt_answer_ids, total_mem_len)
    vocab_size = min(text_logits.size(-1), model.original_vocab_size)
    L_qa = answer_ce_loss(text_logits, labels_t, total_mem_len, vocab_size)

    # L_decomp preservation (optional; small weight). Skip if decompressor frozen and
    # compressor frozen — value is constant.
    L_decomp = torch.zeros((), device=device)  # keep placeholder; add below once wired

    # Monitor: recall@K
    picked = set(sel_idx.tolist())
    num_pos = int(pos_mask.sum().item())
    recall_at_k = (
        sum(1 for i in picked if pos_mask[i].item()) / max(num_pos, 1)
        if num_pos > 0 else 0.0
    )

    return {
        "L_qa": L_qa,
        "L_ret": L_ret,
        "L_margin": L_margin,
        "L_decomp": L_decomp,
        "p_label": p_l,
        "from_label": from_label,
        "recall_at_k": recall_at_k,
        "num_positives": num_pos,
        "selected_k": len(picked),
    }


# ---------------------------------------------------------------------------
# Data helpers (per-paper grouping)
# ---------------------------------------------------------------------------

def iter_questions_by_document(rows, rng):
    groups = {}
    for row in rows:
        groups.setdefault(row["input"], []).append(
            (row["prompt"], row["answer"], row.get("evidence", []))
        )
    keys = list(groups.keys())
    rng.shuffle(keys)
    for k in keys:
        yield k, groups[k]


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load base + selector ----------------------------------------------
    model = load_sedem_model(args.base_checkpoint, args.output_dir, device, args=args)
    selector = load_selector(args.selector_type, args.selector_checkpoint, model.dim, device)

    # ---- Data ---------------------------------------------------------------
    if args.dataset == "qasper":
        tr = get_qa_eval_transform("qasper")
        ds = load_dataset("allenai/qasper", cache_dir=args.dataset_cache_dir,
                          revision="refs/convert/parquet")
        train_rows = ds["train"].map(tr, batched=True,
                                      remove_columns=ds["train"].column_names,
                                      load_from_cache_file=True)
        val_rows = ds["validation"].map(tr, batched=True,
                                         remove_columns=ds["validation"].column_names,
                                         load_from_cache_file=True)
    elif args.dataset == "hotpot":
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor",
                          cache_dir=args.dataset_cache_dir, trust_remote_code=True)
        train_rows = ds["train"].map(hotpot_eval_transform, batched=True,
                                      remove_columns=ds["train"].column_names,
                                      load_from_cache_file=True)
        val_rows = ds["validation"].map(hotpot_eval_transform, batched=True,
                                         remove_columns=ds["validation"].column_names,
                                         load_from_cache_file=True)
    else:
        raise ValueError(f"Unknown --dataset {args.dataset!r}")
    if args.max_train_questions:
        train_rows = train_rows.select(range(min(args.max_train_questions, len(train_rows))))
    print(f"train rows: {len(train_rows)}, val rows: {len(val_rows)}  (dataset={args.dataset})")

    # ---- Unfreeze all modules (no phase-based freezing) -----------
    set_trainable_for_joint_training(model, selector)

    # ---- Three-group optimizer (separate LRs per module) -------------------
    selector_params = [p for p in selector.parameters() if p.requires_grad]
    compressor_params = [p for n, p in model.named_parameters()
                         if p.requires_grad and n.startswith("compressor.")]
    decompressor_params = [p for n, p in model.named_parameters()
                           if p.requires_grad and n.startswith("decompressor.")]
    lora_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "lora_" in n]
    other_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and not n.startswith("compressor.")
                    and not n.startswith("decompressor.") and "lora_" not in n]

    def collect_trainable():
        return (selector_params + compressor_params + decompressor_params
                + lora_params + other_params)

    opt = torch.optim.AdamW([
        {"params": selector_params,     "lr": args.lr_selector},
        {"params": compressor_params,    "lr": args.lr_compressor},
        {"params": decompressor_params,  "lr": args.lr_decompressor},
        {"params": lora_params,          "lr": args.lr_lora},
        {"params": other_params,         "lr": args.lr_lora},
    ], weight_decay=0.0)
    print(f"[optimizer] selector@{args.lr_selector}, compressor@{args.lr_compressor}, "
          f"decompressor@{args.lr_decompressor}, lora@{args.lr_lora}, other@{args.lr_lora}")

    rng = random.Random(args.seed)
    log_f = open(os.path.join(args.output_dir, "train_log.jsonl"), "w", buffering=1)

    # ---- Training loop ------------------------------------------------------
    global_step = 0
    # Block size: Qasper uses sub-segment blocks (= compression_factor);
    # HotpotQA uses paragraph-level blocks (= memory_slots, 1 block per paragraph).
    if args.dataset == "qasper":
        block_size = model.compression_factor
    else:
        block_size = model.memory_slots
    tokens_per_block = block_size * model.compression_factor
    blocks_per_segment = model.memory_slots // block_size
    while global_step < args.total_steps:
        # Build dataset-specific iterator. Both yield (group_key, q_list).
        if args.dataset == "qasper":
            iterator = iter_questions_by_document(train_rows, rng)
        else:
            shuffled = list(range(len(train_rows)))
            rng.shuffle(shuffled)
            def _hotpot_iter():
                for i in shuffled:
                    r = train_rows[i]
                    yield r, [(r["prompt"], r["answer"], r["supporting_titles"], r["paragraph_titles"], r["paragraphs"])]
            iterator = _hotpot_iter()

        for group_key, q_list in iterator:
            if global_step >= args.total_steps:
                break

            # Tokenize + compress once per group (paper for Qasper, example for HotpotQA)
            if args.dataset == "qasper":
                paper_text = group_key
                ctx_ids = model.tokenizer(
                    paper_text, max_length=args.context_max_length, truncation=True,
                    padding=False, return_attention_mask=False, add_special_tokens=False,
                )["input_ids"]
                ctx_t = torch.tensor([ctx_ids], dtype=torch.long, device=device)
                memory = compress_context(model, ctx_t, compressor_trainable=True)
                num_segments = memory.size(1) // model.memory_slots
                total_mem_len_of_memory = num_segments * model.memory_slots
                D = memory.size(-1)
                num_blocks = num_segments * blocks_per_segment
                blocks = memory.view(1, num_segments, model.memory_slots, D).view(
                    1, num_blocks, block_size, D,
                ).float()
            else:
                # HotpotQA: paragraph-level blocks. build_memory_blocks_hotpot
                # is @no_grad — we need a trainable variant for end-to-end. Inline
                # the same logic with grad enabled.
                paragraphs_list = group_key["paragraphs"]
                chunk_size = model.memory_slots * model.compression_factor  # 128
                num_blocks = 10  # distractor split
                # Pad paragraphs to num_blocks
                paragraphs_padded = list(paragraphs_list)
                while len(paragraphs_padded) < num_blocks:
                    paragraphs_padded.append("")
                paragraphs_padded = paragraphs_padded[:num_blocks]
                # Tokenize each para to chunk_size
                ids_per_block = []
                for p in paragraphs_padded:
                    out = model.tokenizer(
                        p, max_length=chunk_size, truncation=True, padding="max_length",
                        add_special_tokens=False, return_attention_mask=False,
                    )
                    ids_per_block.append(list(out["input_ids"]))
                flat = [i for blk in ids_per_block for i in blk]
                ctx_t = torch.tensor([flat], dtype=torch.long, device=device)
                paper_text = group_key.get("input", "")  # for any incidental logging
                memory = compress_context(model, ctx_t, compressor_trainable=True)
                num_segments = num_blocks
                total_mem_len_of_memory = num_segments * model.memory_slots
                D = memory.size(-1)
                blocks = memory.view(1, num_blocks, block_size, D).float()

            # Loop over this group's questions (1 for HotpotQA, ≥1 for Qasper)
            losses = []
            meta = []
            for entry in q_list:
                if args.dataset == "qasper":
                    q_text, ans_text, evidence = entry
                    sup_titles, para_titles = None, None
                else:
                    q_text, ans_text, sup_titles, para_titles, _paragraphs = entry
                    evidence = None
                if not ans_text:
                    continue
                try:
                    out = train_step(
                        model, selector, q_text, ans_text, paper_text, evidence,
                        memory, num_segments, num_blocks, blocks,
                        block_size, tokens_per_block, total_mem_len_of_memory,
                        model._injection_layer,
                        global_step, args.train_top_k, args, rng, device,
                        dataset=args.dataset,
                        supporting_titles=sup_titles,
                        paragraph_titles=para_titles,
                    )
                except Exception as e:
                    print(f"  [skip] question failed with {type(e).__name__}: {e}")
                    continue
                if out["num_positives"] == 0:
                    continue
                L = (
                    out["L_qa"]
                    + args.lambda_sel * out["L_ret"]
                    + args.lambda_margin * out["L_margin"]
                    + args.lambda_decomp * out["L_decomp"]
                )
                losses.append(L)
                meta.append(out)

            if not losses:
                continue
            total = torch.stack(losses).mean()
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(collect_trainable(), 1.0)
            opt.step()

            global_step += 1

            if global_step % args.log_every == 0:
                avg_qa = sum(m["L_qa"].item() for m in meta) / len(meta)
                avg_ret = sum(m["L_ret"].item() for m in meta) / len(meta)
                avg_margin = sum(m["L_margin"].item() for m in meta) / len(meta)
                avg_rec = sum(m["recall_at_k"] for m in meta) / len(meta)
                avg_pl = meta[0]["p_label"]
                print(f"  [step {global_step:4d}] "
                      f"L_qa={avg_qa:.4f} L_ret={avg_ret:.4f} L_mgn={avg_margin:.4f} "
                      f"recall@K={avg_rec:.3f} p_label={avg_pl:.2f} "
                      f"(n_q={len(meta)})")
                log_f.write(json.dumps({
                    "step": global_step,
                    "L_qa": round(avg_qa, 4), "L_ret": round(avg_ret, 4),
                    "L_margin": round(avg_margin, 4),
                    "recall_at_k": round(avg_rec, 4),
                    "p_label": round(avg_pl, 4),
                    "n_questions": len(meta),
                }) + "\n")

            if global_step % args.save_every == 0:
                ckpt = os.path.join(args.output_dir, f"model_step{global_step}.safetensors")
                # Save compressor + decompressor + LoRA + selector + mem_norm/scale
                state = {}
                for n, p in model.state_dict().items():
                    if any(k in n for k in ["compressor.", "decompressor.", "lora_",
                                             "mem_norm", "mem_scale", "special_token_embed"]):
                        state[n] = p.contiguous().cpu()
                retr_state = {f"selector.{n}": p.contiguous().cpu()
                              for n, p in selector.state_dict().items()}
                safe_save({**state, **retr_state}, ckpt)
                print(f"    ↳ saved {ckpt}")

            if global_step >= args.total_steps:
                break

    # Final save
    ckpt_final = os.path.join(args.output_dir, "model_final.safetensors")
    state = {}
    for n, p in model.state_dict().items():
        if any(k in n for k in ["compressor.", "decompressor.", "lora_",
                                 "mem_norm", "mem_scale", "special_token_embed"]):
            state[n] = p.contiguous().cpu()
    retr_state = {f"selector.{n}": p.contiguous().cpu()
                  for n, p in selector.state_dict().items()}
    safe_save({**state, **retr_state}, ckpt_final)

    # Also save selector alone for eval_qa.py compatibility
    retr_only_path = os.path.join(args.output_dir, "selector_final.safetensors")
    safe_save({n: p.contiguous().cpu() for n, p in selector.state_dict().items()},
              retr_only_path)
    retr_cfg_path = os.path.join(args.output_dir, "selector_config.json")
    with open(retr_cfg_path, "w") as f:
        json.dump({
            "selector_type": args.selector_type,
            "d_key": 256 if args.selector_type == "pooled" else None,
            "selector_heads": 4 if args.selector_type == "late_interaction" else None,
            "temperature": args.temperature,
            "base_checkpoint": args.base_checkpoint,
            "selector_warm_start": args.selector_checkpoint,
            "total_steps": args.total_steps,
        }, f, indent=2)

    log_f.close()
    print(f"[done] final ckpt: {ckpt_final}")
    print(f"[done] selector only: {retr_only_path}")


def _add_dataset_arg(ap):
    ap.add_argument("--dataset", choices=["qasper", "hotpot"], default="qasper",
                    help="Which dataset / block-segmentation regime.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_checkpoint", required=True,
                   help="Stage-1 default config safetensors (compressor + decompressor, no LoRA).")
    p.add_argument("--selector_checkpoint", required=True,
                   help="trained selector.safetensors (for warm start).")
    p.add_argument("--selector_type", choices=["pooled", "late_interaction"], default="late_interaction")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--dataset_cache_dir", default=None)
    # Cross-model support: separate encoder + decoder names. Empty falls back to model_name_or_path.
    p.add_argument("--model_name_or_path", default="meta-llama/Llama-3.2-3B")
    p.add_argument("--encoder_name_or_path", default="",
                   help="Cross-model encoder. Empty = use model_name_or_path.")
    p.add_argument("--decoder_name_or_path", default="",
                   help="Cross-model decoder. Empty = use model_name_or_path.")
    p.add_argument("--extraction_layer", type=int, default=14)
    _add_dataset_arg(p)

    # Training schedule
    p.add_argument("--memory_slots_per_segment", type=int, default=32,
                    help="N_M: memory slots per context segment (segment length T = N_M * C).")
    p.add_argument("--compression_factor", type=int, default=4,
                    help="C: tokens summarised by each memory slot.")
    p.add_argument("--total_steps", type=int, default=6000,
                   help="Total optimizer steps. All modules trainable from step 0.")
    p.add_argument("--train_top_k", type=int, default=24)
    p.add_argument("--context_max_length", type=int, default=4096)
    p.add_argument("--max_train_questions", type=int, default=None)

    # Label curriculum (supervised selection signal during warmup)
    p.add_argument("--label_warmup", type=int, default=200)
    p.add_argument("--label_anneal", type=int, default=400)

    # Loss weights
    p.add_argument("--lambda_sel", type=float, default=0.1)
    p.add_argument("--lambda_decomp", type=float, default=0.01)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--lambda_margin", type=float, default=0.0,
                   help="A6 hinge margin weight on selector scores. 0 disables; "
                        "0.5 is a reasonable starting value alongside InfoNCE.")
    p.add_argument("--margin", type=float, default=4.0,
                   help="Required gap min(pos)−max(neg) ≥ margin. Sum-aggregator-scaled "
                        "(score ∈ [-Q_tok, Q_tok]). For HotpotQA Q_tok≈14, margin=4 ≈ "
                        "cosine 0.29 — above bridge gap (0.18), below comparison (0.34).")

    # Optimizer — three groups with separate LRs
    p.add_argument("--lr_selector", type=float, default=1e-3)
    p.add_argument("--lr_compressor", type=float, default=5e-5)
    p.add_argument("--lr_decompressor", type=float, default=5e-5)
    p.add_argument("--lr_lora", type=float, default=1e-4)

    # Logging
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)

    # Mixed extraction layers (App. A.12). When enabled,
    # `extraction_layer` is IGNORED for source extraction (mixer takes layers
    # from --mixing_layers); decomp loss target still uses extraction_layer.
    _bool = lambda x: str(x).strip().lower() in ("true", "1", "yes")
    p.add_argument("--mix_extraction_layers", type=_bool, default=False)
    p.add_argument("--mixing_layers", default="10,12,14,16")
    p.add_argument("--mixing_norm", default="per_layer_ln",
                   choices=["per_layer_ln", "shared_ln", "none"])
    p.add_argument("--mixing_init", default="uniform",
                   choices=["uniform", "peaked_last"])
    p.add_argument("--mixing_learnable", type=_bool, default=False)
    p.add_argument("--injection_layer", type=int, default=-999,
                   help="Layer to inject decompressed Ĥ at. -999 = match HS_LAYER.")

    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
