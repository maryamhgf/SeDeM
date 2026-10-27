"""Inference / joint-training helpers: load model + selector, compress, decompress,
select top-K blocks, inject at the decoder layer, generate.
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


# ---------------------------------------------------------------------------
# Label-curriculum schedule (formerly "evidence-label curriculum")
# ---------------------------------------------------------------------------

def evidence_label_probability(step: int, warmup: int = 200, anneal: int = 400) -> float:
    """p_label curve: 1 for [0, warmup], linear 1→0 over [warmup, warmup+anneal], 0 after."""
    if step < warmup:
        return 1.0
    if step < warmup + anneal:
        return 1.0 - (step - warmup) / anneal
    return 0.0


# ---------------------------------------------------------------------------
# Model/selector builders
# ---------------------------------------------------------------------------

def make_default_model_args(base_ckpt: str, stage1: bool, args=None):
    """Model args matching Qasper default config SeDeM. `stage1=True` means we're
    loading the Stage-1 ckpt (no LoRA weights); lora_decoder flag still True so the
    model attaches fresh LoRA. `args` (parsed CLI) optionally overrides defaults
    for multi-layer source mixing and inject layer."""
    def _g(key, default):
        return getattr(args, key, default) if args else default

    return ModelArguments(
        model_name_or_path=_g("model_name_or_path", "meta-llama/Llama-3.2-3B"),
        encoder_name_or_path=_g("encoder_name_or_path", ""),
        decoder_name_or_path=_g("decoder_name_or_path", ""),
        compressor_type=_g("compressor_type", "pooled_linear"),
        compressor_hidden_dim=_g("compressor_hidden_dim", 3072),
        compressor_num_heads=8,
        extraction_layer=_g("extraction_layer", 14),
        compressor_no_global_context=True,
        compressor_slot_position_embedding=False,
        compressor_linear_projection=_g("compressor_linear_projection", True),
        decompressor_type=_g("decompressor_type", "mlp"),
        inject_decompressed=True,
        injection_layer=_g("injection_layer", -999),
        lora_decoder=True,
        lora_r=_g("lora_r", 64),
        lora_alpha=_g("lora_alpha", 128),
        lora_dropout=0.05,
        lora_target_modules="q_proj,k_proj,v_proj,o_proj",
        train=True,
        mix_extraction_layers=_g("mix_extraction_layers", False),
        mixing_layers=_g("mixing_layers", "10,12,14,16"),
        mixing_norm=_g("mixing_norm", "per_layer_ln"),
        mixing_init=_g("mixing_init", "uniform"),
        mixing_learnable=_g("mixing_learnable", False),
        # R7 / Perceiver-specific flags (no-ops for VB)
        slot_competition=_g("slot_competition", False),
        out_proj_mode=_g("out_proj_mode", "standard"),
        compressor_gated=_g("compressor_gated", False),
        compressor_ignore_prev_mem=_g("compressor_ignore_prev_mem", False),
        compressor_topics_init=_g("compressor_topics_init", "random"),
        compressor_qk_norm=_g("compressor_qk_norm", False),
    )


def make_default_training_args(output_dir: str, args=None):
    """Training args for inference-time model construction. Memory geometry (N_M slots per
    segment, compression factor C) can be overridden from the CLI (`args`)."""
    return TrainingArguments(
        output_dir=output_dir,
        memory_slots_per_segment=getattr(args, "memory_slots_per_segment", 32) if args else 32,
        compression_factor=getattr(args, "compression_factor", 4) if args else 4,
        bf16=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        report_to="none",
    )


def load_sedem_model(base_ckpt: str, output_dir: str, device, args=None):
    """Create SeDeMModel with fresh LoRA + Stage-1 compressor/decompressor.

    The decompressor uses slot-only input (m_global permanently removed). The
    base checkpoint must have been trained with this architecture. `args`
    (parsed CLI) is optional and forwards mixing/inject overrides into model_args.
    """
    from peft import LoraConfig
    model_args = make_default_model_args(base_ckpt, stage1=True, args=args)
    training_args = make_default_training_args(output_dir, args)

    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=[m.strip() for m in model_args.lora_target_modules.split(",")],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = SeDeMModel(model_args, training_args, lora_config=lora_config)

    print(f"Loading Stage-1 weights from {base_ckpt}")
    state = safe_load(base_ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    non_backbone_missing = [k for k in missing if not k.startswith("backbone.") and "lora_" not in k]
    print(f"  restored: {len(state)} tensors; non-backbone/lora missing: {len(non_backbone_missing)}; "
          f"unexpected: {len(unexpected)}")
    if unexpected:
        print(f"    unexpected keys (first 5): {unexpected[:5]}")

    model.to(device)
    return model


def load_selector(kind: str, selector_ckpt_path: str, d_model: int, device):
    """Construct a selector head ('pooled' or 'late_interaction') and warm-start it from a trained checkpoint."""
    from sedem.selector import load_selector_config, build_selector_from_config, resolve_selector_checkpoint, LEGACY_SELECTOR_TYPES
    selector_ckpt_path = resolve_selector_checkpoint(selector_ckpt_path)
    ret_cfg = load_selector_config(selector_ckpt_path)
    kind = LEGACY_SELECTOR_TYPES.get(kind, kind)
    if ret_cfg.get("selector_type") != kind:
        raise ValueError(f"selector_type mismatch: requested {kind!r}, checkpoint config says {ret_cfg.get('selector_type')!r}")
    print(f"Selector config: {ret_cfg}")
    head = build_selector_from_config(ret_cfg, d_model)
    w = safe_load(selector_ckpt_path)
    missing, unexpected = head.load_state_dict(w, strict=True)
    head.to(device).float()
    print(f"  selector warm-start: loaded {len(w)} tensors "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")
    return head


# ---------------------------------------------------------------------------
# Parameter unfreezing (all modules trainable from step 0)
# ---------------------------------------------------------------------------

def set_trainable_for_joint_training(model, selector):
    """All modules trainable from step 0. No phase-based staging."""
    for p in model.parameters():
        p.requires_grad_(False)
    for p in selector.parameters():
        p.requires_grad_(True)

    lora_params = 0
    for n, p in model.decoder.named_parameters():
        if "lora_" in n:
            p.requires_grad_(True)
            lora_params += p.numel()

    compressor_params = 0
    for p in model.compressor.parameters():
        p.requires_grad_(True)
        compressor_params += p.numel()

    decompressor_params = 0
    for p in model.decompressor.parameters():
        p.requires_grad_(True)
        decompressor_params += p.numel()

    # mem_norm / mem_scale / special_token_embed — lightweight but trainable
    other_params = 0
    for n, p in model.named_parameters():
        if any(k in n for k in ["mem_norm", "mem_scale", "special_token_embed"]):
            p.requires_grad_(True)
            other_params += p.numel()

    retr_params = sum(p.numel() for p in selector.parameters())
    print(f"[set_trainable_for_joint_training] trainable params: selector={retr_params:,}, lora={lora_params:,}, "
          f"compressor={compressor_params:,}, decompressor={decompressor_params:,}, other={other_params:,}")


# ---------------------------------------------------------------------------
# Forward helpers
# ---------------------------------------------------------------------------

def compress_context(model, context_ids, compressor_trainable: bool):
    """Run compressor; honor train/no-grad based on phase."""
    ctx = torch.enable_grad() if compressor_trainable else torch.no_grad()
    with ctx:
        memory = model.compress_segments(context_ids)  # (1, M_total, D) float32
    return memory


def decompress_all_segments(model, memory, decompressor_trainable: bool):
    """Apply decompressor segment-wise. Always in grad context (compressor grad may flow in)."""
    ctx = torch.enable_grad() if decompressor_trainable else torch.no_grad()
    all_recon = []
    num_segments = memory.size(1) // model.memory_slots
    with ctx:
        for s in range(num_segments):
            seg_mem = memory[:, s * model.memory_slots : (s + 1) * model.memory_slots, :].float()
            all_recon.append(model.decompressor(seg_mem))
    return torch.cat(all_recon, dim=1), num_segments  # (1, T_recon_all, D)


def encode_query_from_prompt(model, prompt_answer_ids, total_mem_len: int, label_start: int):
    """Extract question tokens from prompt_answer_ids — between mem-placeholders + ft and answer start.

    Cross-model: backbone is the encoder (enc_dim), but selector expects dec_dim
    (matches the slot/block space). Project hs through compressor.input_proj to lift
    enc_dim → dec_dim. In same-model mode input_proj is Identity (no-op).
    """
    # Layout: [mem_placeholder * total_mem_len] + [ft_token] + question_tokens + answer_tokens
    q_start = total_mem_len + 1  # skip mem placeholders + ft_token
    q_end = label_start
    if q_end <= q_start:
        q_end = min(q_start + 16, prompt_answer_ids.size(1))
    q_ids = prompt_answer_ids[:, q_start:q_end].clamp(max=model.original_vocab_size - 1)
    with torch.no_grad():
        out = model.backbone(q_ids, output_hidden_states=True)
        hs = out.hidden_states[model.extraction_layer].float()
    if hasattr(model, "compressor") and hasattr(model.compressor, "input_proj"):
        with torch.no_grad():
            hs = model.compressor.input_proj(hs)
    return hs


def select_blocks_with_label_curriculum(scores, positive_mask, K: int, p_label: float, rng: random.Random, device):
    """Top-K block indices under label curriculum. Falls back to learned if no positives."""
    N = scores.size(0)
    if rng.random() < p_label and positive_mask.any():
        pos = positive_mask.nonzero(as_tuple=True)[0].tolist()
        if len(pos) >= K:
            chosen = rng.sample(pos, K)
        else:
            negs = [i for i in range(N) if i not in pos]
            chosen = list(pos) + rng.sample(negs, K - len(pos))
        return torch.tensor(sorted(chosen), dtype=torch.long, device=device), True
    _, top = scores.topk(min(K, N))
    return top.sort().values.to(device), False


def gather_selected_blocks(decompressed_hs, selected_indices, tokens_per_block: int):
    """Gather reconstructed tokens for the K selected blocks, preserving order."""
    parts = [decompressed_hs[:, k * tokens_per_block : (k + 1) * tokens_per_block, :]
             for k in selected_indices.tolist()]
    return torch.cat(parts, dim=1)


def decode_with_injected_memory(model, selected_hs, prompt_answer_ids, total_mem_len: int,
                   inference_only: bool = False):
    """Normalize Ĥ and run the two-phase decoder forward. Returns text logits + answer label slice.

    When inference_only=True, the final norm + lm_head are computed only on the last text
    position (bit-exact equivalent for greedy decoding, since RMSNorm + Linear are per-token).
    Returns text_logits of shape (B, 1, vocab) instead of (B, S_text, vocab). Callers that
    only use logits[:, -1, :] (the eval scripts) work either way."""
    # Normalize decompressed states (matches generate path)
    hs = selected_hs
    if hasattr(model, "mem_norm") and hasattr(model, "mem_scale") and model._normalize_mem:
        max_scale = 25.0 / math.sqrt(model.dim)
        eff_scale = torch.clamp(model.mem_scale, min=0.01, max=max_scale)
        hs = model.mem_norm(hs) * eff_scale
    hs = hs.to(model._backbone_dtype)

    # Strip memory-placeholder region from prompt_answer_ids → text_ids
    text_ids = prompt_answer_ids[:, total_mem_len:]
    safe_text_ids = text_ids.clamp(max=model.original_vocab_size - 1)
    # Cross-model: must use decoder embed_tokens (dec_dim) not backbone (enc_dim).
    # In same-model mode this is equivalent (decoder aliases backbone).
    text_embs = model._decoder_embed_tokens(safe_text_ids).detach()

    # Replace FT special token with its learned embedding
    ft_mask = text_ids == model.ft_token_id
    if ft_mask.any():
        ft_emb = model.special_token_embed(
            torch.ones(1, dtype=torch.long, device=text_embs.device)
        ).to(text_embs.dtype).expand_as(text_embs)
        ft_mask_3d = ft_mask.unsqueeze(-1).expand_as(text_embs)
        text_embs = torch.where(ft_mask_3d, ft_emb, text_embs)

    # Two-phase forward
    inject_layer = model._injection_layer
    text_logits = model.forward_with_injection(text_embs, hs, inject_layer, _verbose=False,
                                            inference_only=inference_only)
    return text_logits


def normalize_memory(model, selected_hs):
    """Apply mem_norm + mem_scale + dtype cast. Output is constant across decode steps,
    so call this ONCE before the decode loop and reuse for every decode_with_injected_memory_cached call."""
    hs = selected_hs
    if hasattr(model, "mem_norm") and hasattr(model, "mem_scale") and model._normalize_mem:
        max_scale = 25.0 / math.sqrt(model.dim)
        eff_scale = torch.clamp(model.mem_scale, min=0.01, max=max_scale)
        hs = model.mem_norm(hs) * eff_scale
    return hs.to(model._backbone_dtype)


def embed_prompt_text(model, prompt_answer_ids, total_mem_len: int):
    """Embed the text portion of prompt_answer_ids (post mem-placeholder region).
    Replace FT special token positions with the learned ft embedding. Returns the
    full text_embs tensor (same as decode_with_injected_memory's middle stage)."""
    text_ids = prompt_answer_ids[:, total_mem_len:]
    safe_text_ids = text_ids.clamp(max=model.original_vocab_size - 1)
    text_embs = model._decoder_embed_tokens(safe_text_ids).detach()
    ft_mask = text_ids == model.ft_token_id
    if ft_mask.any():
        ft_emb = model.special_token_embed(
            torch.ones(1, dtype=torch.long, device=text_embs.device)
        ).to(text_embs.dtype).expand_as(text_embs)
        ft_mask_3d = ft_mask.unsqueeze(-1).expand_as(text_embs)
        text_embs = torch.where(ft_mask_3d, ft_emb, text_embs)
    return text_embs


def prefill_with_injected_memory_cached(model, selected_hs_normalized, prompt_answer_ids, total_mem_len: int,
                                   max_new_tokens: int = 0):
    """Cached-path prefill. selected_hs_normalized is the output of normalize_memory.
    Returns (last_logits, past_kv_phase1, past_kv_phase2, text_pos_past) where text_pos_past
    is the cached text length to pass into subsequent decode_step_with_injected_memory calls.

    When model._shape_stable_kv=True, max_new_tokens must be > 0 so the StaticCache
    can be sized for prefill + full decode horizon. Default 0 keeps the DynamicCache
    path (existing behavior bit-for-bit)."""
    text_embs = embed_prompt_text(model, prompt_answer_ids, total_mem_len)
    inject_layer = model._injection_layer
    _static_max_text_len = (text_embs.size(1) + int(max_new_tokens)) if getattr(model, '_shape_stable_kv', False) else None
    last_logits, pkv1, pkv2 = model.forward_with_injection_cached(
        text_embs, selected_hs_normalized, inject_layer,
        past_kv_phase1=None, past_kv_phase2=None, text_pos_past=0,
        static_max_text_len=_static_max_text_len,
    )
    return last_logits, pkv1, pkv2, text_embs.size(1)


def decode_step_with_injected_memory_cached(model, selected_hs_normalized, new_token_id,
                                       pkv1, pkv2, text_pos_past):
    """One decode step with KV cache. new_token_id is the previously-predicted token (int or 0-d tensor).
    Returns (last_logits, pkv1, pkv2, new_text_pos_past)."""
    device = selected_hs_normalized.device
    if not torch.is_tensor(new_token_id):
        new_token_id = torch.tensor([[new_token_id]], dtype=torch.long, device=device)
    elif new_token_id.dim() == 0:
        new_token_id = new_token_id.view(1, 1)
    # Embed the single new token; check for FT replacement (rare on decode but defensive)
    safe_tok = new_token_id.clamp(max=model.original_vocab_size - 1)
    new_emb = model._decoder_embed_tokens(safe_tok).detach()
    if (new_token_id == model.ft_token_id).any():
        ft_emb = model.special_token_embed(
            torch.ones(1, dtype=torch.long, device=device)
        ).to(new_emb.dtype).expand_as(new_emb)
        new_emb = torch.where((new_token_id == model.ft_token_id).unsqueeze(-1).expand_as(new_emb), ft_emb, new_emb)
    inject_layer = model._injection_layer
    # Decode reuses the already-allocated caches, but static_max_text_len must
    # still be passed: it is what makes use_static_kv=True inside the cached
    # forward so the explicit StaticCache mask is built. Without it, decode falls
    # to mask=None over a StaticCache and attends to unwritten zero slots
    # (degenerate output). Recover it from the Phase-1 StaticCache buffer size
    # (== static_max_text_len by construction); None for the DynamicCache path.
    from transformers.cache_utils import StaticCache as _StaticCache
    _smt = pkv1.max_cache_len if isinstance(pkv1, _StaticCache) else None
    last_logits, pkv1, pkv2 = model.forward_with_injection_cached(
        new_emb, selected_hs_normalized, inject_layer,
        past_kv_phase1=pkv1, past_kv_phase2=pkv2, text_pos_past=text_pos_past,
        static_max_text_len=_smt,
    )
    return last_logits, pkv1, pkv2, text_pos_past + 1


def answer_ce_loss(text_logits, labels, total_mem_len: int, vocab_size: int):
    """Compute CE on answer tokens. Labels were built at prompt_answer_ids length; strip mem region."""
    text_labels = labels[:, total_mem_len:]
    # shift: predict token t+1 from logits at t
    shift_logits = text_logits[:, :-1, :vocab_size].contiguous()
    shift_labels = text_labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,
    )



