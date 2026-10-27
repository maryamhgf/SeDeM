# SeDeM that supports multi span concat

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
import torch
import torch.nn as nn
import random
from dataclasses import dataclass, field
from typing import Optional
from peft import (
    get_peft_model,
    LoraConfig,
)
import torch.nn.functional as F
import math
from safetensors.torch import load_file
from sedem.compressor import build_compressor
def effective_rank(Z: torch.Tensor) -> float:
    """Effective rank of a 2D matrix via Shannon entropy of normalized singular values.

    erank(Z) = exp(-sum(p_i * log(p_i)))  where p_i = sigma_i / sum(sigma_j)

    References:
      - "When Less is More: The LLM Scaling Paradox in Context Compression" (arXiv:2602.09789)
      - Roy & Vetterli, "The effective rank of a matrix" (2007)
    """
    if Z.dim() != 2 or Z.size(0) < 2:
        return float('nan')
    sigma = torch.linalg.svdvals(Z.float())
    sigma = sigma[sigma > 1e-10]
    if sigma.numel() == 0:
        return 0.0
    p = sigma / sigma.sum()
    entropy = -(p * p.log()).sum()
    return entropy.exp().item()


def memory_diversity_metrics(mem_per_segment: list, verbose: bool = False) -> dict:
    """Compute diversity metrics for memory tokens across segments.

    Returns dict with:
      - inter_seg_cosine: mean pairwise cosine between consecutive segment centroids
      - intra_slot_cosine: mean pairwise cosine between slots within segments
      - erank_per_seg: effective rank of the M×D matrix per segment (averaged)
      - erank_global: effective rank of the full (num_seg*M)×D memory matrix
    """
    output = {}
    num_segments = len(mem_per_segment)
    device = mem_per_segment[0].device

    # 1) Inter-segment cosine (consecutive centroids)
    if num_segments > 1:
        inter_cos_vals = []
        for si in range(1, num_segments):
            pool_prev = mem_per_segment[si - 1].float().mean(dim=1)
            pool_curr = mem_per_segment[si].float().mean(dim=1)
            inter_cos_vals.append(
                F.cosine_similarity(pool_prev, pool_curr, dim=-1).mean().item()
            )
        output["inter_seg_cosine"] = sum(inter_cos_vals) / len(inter_cos_vals)

    # 2) Intra-slot cosine (pairwise within segment)
    intra_cos_vals = []
    for seg_m in mem_per_segment:
        slots = F.normalize(seg_m[0].float(), dim=-1)
        sim_matrix = slots @ slots.T
        M = slots.size(0)
        mask_diag = ~torch.eye(M, dtype=torch.bool, device=device)
        intra_cos_vals.append(sim_matrix[mask_diag].mean().item())
    output["intra_slot_cosine"] = sum(intra_cos_vals) / len(intra_cos_vals)

    # 3) Effective rank per segment (averaged)
    erank_per_seg = []
    for seg_m in mem_per_segment:
        er = effective_rank(seg_m[0])  # M×D
        if not math.isnan(er):
            erank_per_seg.append(er)
    if erank_per_seg:
        output["erank_per_seg"] = sum(erank_per_seg) / len(erank_per_seg)

    # 4) Effective rank of global memory matrix (all segments concatenated)
    all_mem = torch.cat([seg_m[0] for seg_m in mem_per_segment], dim=0)  # (num_seg*M, D)
    output["erank_global"] = effective_rank(all_mem)

    if verbose:
        parts = []
        for k in ["inter_seg_cosine", "intra_slot_cosine", "erank_per_seg", "erank_global"]:
            if k in output:
                parts.append(f"{k}={output[k]:.4f}")
        print(f"  diversity: {', '.join(parts)}")

    return output



class _BackboneEarlyExit(Exception):
    """Sentinel raised by the backbone-early-exit forward hook to bail out
    of the remaining-layers loop in HF's backbone forward. Caught by
    SeDeMModel.extract_hidden_states_early_exit. Not a real error."""
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="mistralai/Mistral-7B-v0.1")
    encoder_name_or_path: str = field(
        default="",
        metadata={"help": "Encoder (compress-phase) model. Empty = fall back to model_name_or_path. "
                  "Set to a smaller model (e.g. Llama-3.2-1B) to run a cross-model variant where "
                  "the encoder produces hidden states the decoder consumes via the decompressor."}
    )
    decoder_name_or_path: str = field(
        default="",
        metadata={"help": "Decoder (answer-generation) model. Empty = fall back to model_name_or_path. "
                  "Hidden dim must match the compressor's output_dim; dim crossing happens at the "
                  "compressor input_proj when encoder hidden_size != decoder hidden_size."}
    )
    lora_r: int = field(
        default=128,
        metadata={"help": "lora rank"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "lora dropout"}
    )
    train: bool = field(
        default=True,
        metadata={"help": "if true, the model ckpt will be initialized for training; else, it's for inference"}
    )
    # --- hidden-state compressor options ---
    compressor_type: str = field(
        default="pooled_linear",
        metadata={"help": "learnable_query | pooled_linear  (only used when compressor_type=hidden_state)"}
    )
    compressor_hidden_dim: int = field(
        default=512,
        metadata={"help": "bottleneck dim inside the hidden-state compressor"}
    )
    compressor_num_heads: int = field(
        default=8,
        metadata={"help": "attention heads in the compressor"}
    )
    compressor_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout in compressor"}
    )
    extraction_layer: int = field(
        default=-1,
        metadata={"help": "which backbone layer to extract hidden states from (-1 = last)"}
    )
    injection_layer: int = field(
        default=-999,
        metadata={"help": "which decoder layer to inject decompressed hidden states at. "
                  "Default -999 means 'same as extraction_layer' (backward compat). "
                  "Set to 0 to inject at the embedding level (before any transformer layers); "
                  "the decompressed sequence then passes through all L decoder layers. "
                  "Set to -1 for the last layer, or any explicit index in [0, L-1]."}
    )
    gate_init: float = field(
        default=0.0,
        metadata={"help": "initial value for raw_gate parameters in cross-attention and FFN. "
                  "0.0 = classic zero-init (Flamingo). 0.1 gives effective gate ~0.05, "
                  "avoiding the dead-branch effect where gates never open."}
    )
    gate_max: float = field(
        default=0.5,
        metadata={"help": "maximum effective gate value. gate = gate_max * tanh(raw_gate). "
                  "0.5 = context can contribute at most 50%% of the residual. "
                  "1.0 = no cap, context can fully override queries."}
    )
    compressor_blend_mode: str = field(
        default="content_first",
        metadata={"help": "How the compressor blends queries with cross-attention output. "
                  "'content_first': g*Q + (1-g)*attn (default, memory is mostly content). "
                  "'perturbation': Q + g*attn (old Flamingo-style, memory is mostly query). "
                  "'no_residual': attn only (no query component in output)."}
    )
    slot_competition: bool = field(
        default=False,
        metadata={"help": "Enable Slot Attention-style competition in the compressor. "
                  "After softmax-over-keys, normalizes attention weights over the "
                  "slot/query dimension so slots compete for each key position. "
                  "No new parameters (uses same MHA weights)."}
    )
    out_proj_mode: str = field(
        default="standard",
        metadata={"help": "Output projection mode for compressor cross-attention. "
                  "'standard': shared (D,D) out_proj (default). "
                  "'identity': no output projection (preserve per-head geometry). "
                  "'block_diagonal': per-head (d_head,d_head) projections (no cross-head mixing)."}
    )
    compressor_no_ffn: bool = field(
        default=False,
        metadata={"help": "Skip the FFN after cross-attention in the compressor. "
                  "Memory = cross-attention output directly (Perceiver/IC-Former pattern)."}
    )
    compressor_num_self_attn_layers: int = field(
        default=0,
        metadata={"help": "Number of Perceiver-style self-attention layers on the latents/topics "
                  "after cross-attention (0 = current behavior, no self-attn). "
                  "Lets the M slots coordinate with each other; addresses diversity collapse "
                  "from purely-passive query embeddings. Perceiver paper uses 6."}
    )
    compressor_num_iter_blocks: int = field(
        default=1,
        metadata={"help": "Number of iterative Perceiver blocks with shared weights "
                  "(1 = current behavior, single pass). Each iteration re-attends to the "
                  "input and refines the latents. Perceiver paper uses 8."}
    )
    compressor_gated: bool = field(
        default=True,
        metadata={"help": "Enable gated cross-attention residual in learnable_query compressor "
                  "(Flamingo-style: tanh gate + content-first/perturbation blend + "
                  "norm-matching). Default True for backward compat. Set False for "
                  "pure Perceiver-IO residual (queries + attn_out, no gate, no blend, "
                  "no norm-match)."}
    )
    compressor_ignore_prev_mem: bool = field(
        default=False,
        metadata={"help": "Make the learnable_query compressor stateless across segments — "
                  "drop the prev_mem KV concatenation. Mirrors the pooled_linear "
                  "'no recurrence' behavior. Addresses inter-segment memory convergence "
                  "(temporal accumulation pathology) at the cost of cross-segment context."}
    )
    compressor_topics_init: str = field(
        default="random",
        metadata={"help": "Initialization scheme for learnable_query topic queries. "
                  "'random' (default, randn*0.02 — canonical Perceiver-IO). "
                  "'sinusoidal' (deterministic per-slot positional embedding — breaks "
                  "cold-start query symmetry). "
                  "'orthogonal' (orthogonal random rows)."}
    )
    compressor_qk_norm: bool = field(
        default=False,
        metadata={"help": "L2-normalize Q^h and K^h per head before dot-product, "
                  "scale by learnable per-head temperature (Henry et al. 2020 / used "
                  "in CLIP). Directly attacks low score variance — the diagnosed "
                  "cold-start collapse cause (score_var=0.24 << log(T)=4.85)."}
    )
    compressor_qk_norm_init_scale: float = field(
        default=10.0,
        metadata={"help": "Initial value for the QK-norm temperature (per head). "
                  "10.0 gives scores in [-10, +10] which is well above log(T=128)≈4.85 "
                  "needed for sharp softmax."}
    )
    lora_decoder: bool = field(
        default=False,
        metadata={"help": "apply LoRA to the decoder in SeDeMModel"}
    )
    lora_alpha: int = field(
        default=128,
        metadata={"help": "LoRA alpha (scaling = lora_alpha / lora_r)"}
    )
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj",
        metadata={"help": "comma-separated list of decoder modules to apply LoRA to"}
    )
    cache_hidden_states: bool = field(
        default=False,
        metadata={"help": "Cache backbone hidden states to disk. Eliminates repeated backbone "
                  "forward passes across epochs (backbone is frozen). Set --hidden_states_cache_dir too."}
    )
    hidden_states_cache_dir: str = field(
        default=None,
        metadata={"help": "Directory to store cached hidden states. Required when cache_hidden_states=True."}
    )

    # --- SeDeM: decompress-inject mode ---
    inject_decompressed: bool = field(
        default=False,
        metadata={"help": "SeDeM mode: compress → decompress → inject at extraction layer. "
                  "Decompressed hidden states enter the decoder at the extraction layer via "
                  "two-phase forward (layers 0..ℓ-1 text-only, layers ℓ..L-1 with "
                  "concatenated [decompressed | text]). Requires compressor_type=chunked_gated."}
    )
    compressor_no_global_context: bool = field(
        default=False,
        metadata={"help": "Drop z_global from SegmentCompressor input. "
                  "Input changes from [prev_mem; z_local; z_global] to [prev_mem; z_local]."}
    )
    compressor_slot_position_embedding: bool = field(
        default=False,
        metadata={"help": "Add learned per-slot embedding to z_local in SegmentCompressor."}
    )
    compressor_linear_projection: bool = field(
        default=False,
        metadata={"help": "Replace candidate MLP with single Linear in SegmentCompressor."}
    )
    slot_mixer_type: str = field(
        default="linear",
        metadata={"help": "How M memory slots talk to each other after the per-slot writer. "
                  "'linear' (default): LN + Linear(M,M) residual, init-as-identity (current). "
                  "'attn': LN + MultiheadAttention over slots with tanh-gated residual; gate "
                  "starts at 0 so module is a strict pass-through at step 0 (Tier 1.2 test). "
                  "'none': no mixing (slots stay independent — useful for ablation)."}
    )
    slot_mixer_attn_heads: int = field(
        default=4,
        metadata={"help": "Number of attention heads when slot_mixer_type='attn'. "
                  "Default 4 (D=3072 → head_dim=768). Ignored otherwise."}
    )
    decompressor_type: str = field(
        default="mlp",
        metadata={"help": "Decompressor variant. Only 'mlp' (Decompressor: "
                  "slot-only per-slot MLP) is supported."}
    )
    # --- multi-layer source mixing ---------------------------------------------
    mix_extraction_layers: bool = field(
        default=False,
        metadata={"help": "If True, extract hidden states from multiple backbone layers "
                  "(listed in mixing_layers) and combine them via per-layer LN + "
                  "learned softmax-weighted sum. Overrides extraction_layer for source "
                  "extraction; injection layer (injection_layer) is unaffected."}
    )
    mixing_layers: str = field(
        default="10,12,14,16",
        metadata={"help": "Comma-separated backbone layer indices to mix when "
                  "mix_extraction_layers=True. Ignored otherwise. Supports 0-indexed layer IDs "
                  "(0 = embedding output, 1..L = transformer layers)."}
    )
    mixing_norm: str = field(
        default="per_layer_ln",
        metadata={"help": "Normalization applied to each layer before mixing. "
                  "'per_layer_ln' (default): each layer gets its own learnable LayerNorm. "
                  "'shared_ln': single LayerNorm shared across all layers. "
                  "'none': raw hidden states."}
    )
    mixing_init: str = field(
        default="uniform",
        metadata={"help": "Initialization for raw mixing logits w (softmax gives alpha). "
                  "'uniform' (default): w=0 → alpha = 1/L. "
                  "'peaked_last': w[-1]=2, others=0 (emphasizes last layer in mixing_layers)."}
    )
    mixing_learnable: bool = field(
        default=True,
        metadata={"help": "If True (default), w is a trainable parameter. "
                  "If False, w is frozen at its init value — useful for M1u uniform-mix "
                  "ablation (alpha = 1/L, no learned weighting)."}
    )

    inference_mode: str = field(
        default="qa",
        metadata={"help": "Top-level inference loop. "
                  "'qa' (default) = single-shot compress-then-answer. "
                  "'trajectory' = rolling per-turn compress + decode loop "
                  "(maintains compressed bank + recent_raw window across env turns)."}
    )

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data (local JSONL). Ignored when dataset_name is set."})
    eval_data_path: str = field(default=None, metadata={"help": "Path to the eval data. Defaults to data_path with 'train' replaced by 'val'."})
    debug_data: bool = field(default=False, metadata={"help": "Enable debug dataset to quickly verify the training process"})
    dataset_name: str = field(default=None, metadata={"help": "HuggingFace dataset name, e.g. 'hotpotqa/hotpot_qa'. When set, data_path is ignored."})
    dataset_config: str = field(default=None, metadata={"help": "HuggingFace dataset config, e.g. 'distractor'."})
    dataset_cache_dir: str = field(default=None, metadata={"help": "Cache directory for HuggingFace datasets."})
    dataset_transform: str = field(default=None, metadata={"help": "Transform name, e.g. 'hotpotqa' or 'qasper'. Auto-detected from dataset_name if not set."})
    subset_manifest: str = field(default=None, metadata={"help": "Path to a JSON subset manifest (e.g. from prepare_hotpot_subset.py). When set, train/validation splits are filtered to the indices listed in the manifest. Enables reproducible mini-runs without re-sampling."})
    dataset_streaming: bool = field(default=False, metadata={"help": "Stream dataset from HF Hub without caching Arrow files. Required for very large datasets (e.g. SlimPajama) where local disk is limited. Incompatible with subset_manifest and load_best_model_at_end."})
    max_train_eval_samples: int = field(default=None, metadata={"help": "If set, deterministically slice the training-time eval_dataset to first N examples (after transform/map). Speeds up Trainer's periodic eval used for load_best_model_at_end + U-turn detection. None = use full eval split. Has no effect on post-training eval_qa.py (which has its own --max_eval_samples)."})
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=28000,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    memory_slots_per_segment: int = field(
        default=128,
        metadata={"help": "Enalbing the fixed mem size."},
    )
    compression_factor: int = field(
        default=4,
        metadata={"help": "Mean compression rate; default=4"},
    )
    min_tokens_for_lm: int = field(
        default=64,
        metadata={"help": "Minimum tokens for lm objective learning"},
    )
    leave_tokens_for_lm: int = field(
        default=8,
        metadata={"help": "Leave some tokens without loss for lm objective"},
    )
    lm_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio for LM training."},
    )
    add_special_token_for_lm: bool = field(
        default=False,
        metadata={"help": "Add a special token for the prompt of language modeling; default: False"},
    )
    restore_from: str = field(
        default="",
        metadata={"help": "The checkpoint that should be restored from for fine-tuning"}
    )
    freeze_compressor: bool = field(
        default=False,
        metadata={"help": "Freeze compressor weights after loading checkpoint. "
                  "Used for decompressor-only training with a fixed compressor ."}
    )
    distill_loss_weight: float = field(
        default=0.0,
        metadata={"help": "weight for KL-distillation loss (0 = disabled)"}
    )
    distill_temperature: float = field(
        default=2.0,
        metadata={"help": "temperature for KL-distillation softmax"}
    )
    qa_weight: float = field(
        default=0.0,
        metadata={"help": "weight for QA answer cross-entropy loss (0 = disabled)"}
    )
    rec_loss_weight: float = field(
        default=0.0,
        metadata={"help": "weight for hidden-state reconstruction (decompressor) loss (0 = disabled)"}
    )
    ctx_token_loss_weight: float = field(
        default=0.0,
        metadata={"help": "weight for token reconstruction CE loss on decompressed positions (0 = disabled). "
                  "L_tok = CE(decoder(decompressed_hs), original_context_tokens)."}
    )
    pretrain_window_size: int = field(
        default=128,
        metadata={"help": "Window size for windowed reconstruction pretraining. "
                  "Decoder sees only this many tokens (random window from full context). "
                  "First token predicted from memory alone, forcing memory usage."}
    )
    memonly_fraction: float = field(
        default=0.0,
        metadata={"help": "Fraction of training samples that use a very short window (4-16 tokens) "
                  "instead of pretrain_window_size. Forces the decoder to predict from memory alone, "
                  "creating gradient pressure for document-specific content encoding. "
                  "0.0 = disabled (all samples use pretrain_window_size). "
                  "0.2-0.3 = recommended starting range."}
    )
def print_trainable_parameters(model):
    trainable_parameters = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()
    print(f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: {100 * trainable_parameters / all_param}")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.shape)


def freeze_model(model):
    for _, param in model.named_parameters():
        param.requires_grad = False


# SeDeM with Hidden-State Compressor
# =====================================================================

class ExtractionLayerMixer(torch.nn.Module):
    """
    Per-layer LN + learned softmax-weighted sum over multiple backbone layers.

    z[B, T, D] = sum_i  softmax(w)_i * LN_i( H^(mix_layers[i]) )

    - mix_layers: list of backbone layer indices to extract from (0 = embedding output).
    - normalization ∈ {"per_layer_ln", "shared_ln", "none"}.
    - weight_init ∈ {"uniform", "peaked_last"}.
    - learnable: if False, w is frozen → fixed uniform mix (M1u ablation).

    Trainable params: (num_layers * 2*D) for per-layer LN + num_layers scalar logits.
    Feeds into the existing compressor unchanged.
    """
    def __init__(self, dim: int, mix_layers, normalization: str = "per_layer_ln",
                 weight_init: str = "uniform", learnable: bool = True):
        super().__init__()
        self.dim = dim
        self.mix_layers = list(mix_layers)
        self.num_layers = len(self.mix_layers)
        if self.num_layers < 2:
            raise ValueError(
                f"ExtractionLayerMixer requires ≥2 layers; got mix_layers={self.mix_layers}"
            )
        self.normalization = normalization

        if normalization == "per_layer_ln":
            self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(dim) for _ in range(self.num_layers)])
        elif normalization == "shared_ln":
            shared = torch.nn.LayerNorm(dim)
            self.norms = torch.nn.ModuleList([shared] * self.num_layers)
        elif normalization == "none":
            self.norms = None
        else:
            raise ValueError(f"Unknown mix normalization: {normalization!r}")

        if weight_init == "uniform":
            w_init = torch.zeros(self.num_layers)
        elif weight_init == "peaked_last":
            w_init = torch.zeros(self.num_layers); w_init[-1] = 2.0
        else:
            raise ValueError(f"Unknown mix weight_init: {weight_init!r}")

        self.w = torch.nn.Parameter(w_init, requires_grad=bool(learnable))

    @property
    def alpha(self) -> torch.Tensor:
        return torch.softmax(self.w, dim=0)

    def forward(self, hidden_by_layer):
        """hidden_by_layer: list of [B, T, D] tensors (one per mix_layers), fp32 or bf16."""
        if len(hidden_by_layer) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} layer tensors, got {len(hidden_by_layer)}"
            )
        alpha = self.alpha
        if self.norms is not None:
            normed = [norm(h) for h, norm in zip(hidden_by_layer, self.norms)]
        else:
            normed = hidden_by_layer
        z = alpha[0] * normed[0]
        for i in range(1, self.num_layers):
            z = z + alpha[i] * normed[i]
        return z


class SeDeMModel(torch.nn.Module):
    """
    Hidden-state compression variant of SeDeM.

    Flow
    ----
    1. Run the frozen backbone on the CONTEXT to obtain hidden states.
    2. Feed hidden states into a small compressor network →  M memory tokens.
    3. Decoder sees  [memory tokens] + [question/instruction tokens]  and
       generates the target (reconstructed context or QA answer).

    The question / instruction is NEVER compressed.

    Losses
    ------
    - Reconstruction CE:  reconstruct context from memory tokens alone.
    - KL distillation:    match logits of (full-context decoder) vs
                          (memory-only decoder).  Controlled by
                          distill_loss_weight / distill_temperature.
    - QA answer CE:       standard next-token CE on answer span when
                          qa_weight > 0.
    """

    _keys_to_ignore_on_save = None

    def __init__(self, model_args, training_args, lora_config=None):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.model_name = model_args.model_name_or_path

        # ---- encoder + decoder names (cross-model support) -------------
        # Empty fields fall back to model_name_or_path → same-model behaviour.
        # Different fields → cross-model variant (e.g. 1B encoder + 3B decoder).
        encoder_name = getattr(model_args, 'encoder_name_or_path', '') or self.model_name
        decoder_name = getattr(model_args, 'decoder_name_or_path', '') or self.model_name
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        self._cross_model = (encoder_name != decoder_name)

        # ---- backbone (frozen, encoder-role: produces context hidden states) -
        dtype = torch.bfloat16 if training_args.bf16 else torch.float16
        self.backbone = AutoModelForCausalLM.from_pretrained(
            encoder_name,
            dtype=dtype,
            attn_implementation="sdpa",  # 5-10x faster than eager for long T; output_attentions=True paths auto-fall back to eager
        )
        freeze_model(self.backbone)
        self.backbone.eval()

        # ---- decoder (frozen base; LoRA-trained at Stage-2) ------------
        # In cross-model or training mode, load a separate decoder model.
        # In same-model inference, alias decoder→backbone (avoids duplicating
        # GPU memory; the `self.decoder is not self.backbone` check elsewhere
        # still distinguishes "have a separate decoder" from "no separate decoder").
        self._is_training = model_args.train
        if self._is_training or self._cross_model:
            self.decoder = AutoModelForCausalLM.from_pretrained(
                decoder_name,
                dtype=dtype,
                attn_implementation="sdpa",
            )
            freeze_model(self.decoder)
            self.decoder.eval()
        else:
            self.decoder = self.backbone

        # ---- tokenizer --------------------------------------------------
        # Decoder's tokenizer is canonical (it tokenizes prompt+answer text).
        # For Llama-3.2-1B/3B these share the same tokenizer, so this is a no-op
        # in practice; the choice matters only for genuinely different families.
        self.tokenizer = AutoTokenizer.from_pretrained(
            decoder_name, use_fast=True
        )
        # Qwen3 / many modern tokenizers already have pad; fall back safely
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_id = self.tokenizer.eos_token_id

        # ---- dims -------------------------------------------------------
        # enc_dim: hidden size of the compress-phase encoder (backbone).
        # dec_dim: hidden size of the answer-generation decoder.
        # In same-model mode they coincide; the rest of the model uses
        # self.dim = self.dec_dim because slots, decompressor output, and
        # mem_norm/special_token_embed all live in decoder space.
        # The dim-crossing happens once at the compressor's input_proj.
        self.enc_dim = self.backbone.config.hidden_size
        _decoder_for_dim = getattr(self, 'decoder', self.backbone)
        self.dec_dim = _decoder_for_dim.config.hidden_size
        self.dim = self.dec_dim
        self._frozen_slot_dim = 0
        self.slot_dim = self.dec_dim
        self.memory_slots = training_args.memory_slots_per_segment
        self.compression_factor = training_args.compression_factor
        if self._cross_model:
            print(f"[SeDeM cross-model] encoder={self.encoder_name} (dim={self.enc_dim}, "
                  f"layers={self.backbone.config.num_hidden_layers}) | "
                  f"decoder={self.decoder_name} (dim={self.dec_dim}, "
                  f"layers={_decoder_for_dim.config.num_hidden_layers})")

        # Original vocab (no resizing needed -- we never embed memory token
        # IDs through the backbone; the compressor produces them directly).
        # Use decoder vocab since prompt/answer tokens are tokenized with decoder's tokenizer.
        self.original_vocab_size = _decoder_for_dim.config.vocab_size

        # We still use integer IDs > vocab_size as *placeholders* in the
        # prompt_answer_ids tensor so the data-collator can build batches.
        # These are replaced by compressor outputs before the decoder runs.
        self.mem_id_start = self.original_vocab_size + 1  # +1 for PAD
        self.ae_token_id = self.mem_id_start + self.memory_slots
        self.ft_token_id = self.mem_id_start + self.memory_slots + 1

        # Learnable embedding only for the special AE / FT tokens (2 tokens).
        # Memory-token embeddings come from the compressor, NOT from a table.
        # Init from decoder's real token embeddings so the norm matches what
        # the decoder expects (~1.0). Cross-model: decoder space (dec_dim).
        self.special_token_embed = nn.Embedding(2, self.dim)
        with torch.no_grad():
            vocab_embeds = _decoder_for_dim.model.embed_tokens.weight
            indices = torch.randint(0, vocab_embeds.size(0), (2,))
            self.special_token_embed.weight.copy_(vocab_embeds[indices])

        # ---- memory token normalization ----------------------------------
        # Compressor outputs have norms ~55 while text embeddings have norms ~1.
        # LayerNorm brings memory tokens to unit scale; the learned scalar lets
        # the model adjust the magnitude during training.
        self.mem_norm = nn.LayerNorm(self.dim)
        # After LayerNorm, ||mem|| ≈ sqrt(D) ≈ 50.6 for D=2560.
        # For prepend mode: text embeddings have ||x|| ≈ 1.0 → init scale = 1/sqrt(D).
        # For inject_decompressed: text hidden states at layer ℓ have ||x|| ≈ 15-20
        #   → init scale = 15/sqrt(D) so decompressed states start at text norm.
        if getattr(model_args, 'inject_decompressed', False):
            _init_scale = 15.0 / math.sqrt(self.dim)
        else:
            _init_scale = 1.0 / math.sqrt(self.dim)
        self.mem_scale = nn.Parameter(torch.tensor(_init_scale))
        self._normalize_mem = True  # disabled when loading old checkpoints without mem_norm

        compressor_num_tokens = self.memory_slots

        # ---- compressor -------------------------------------------------
        blend_mode = getattr(model_args, 'compressor_blend_mode', 'content_first')
        slot_competition = getattr(model_args, 'slot_competition', False)
        out_proj_mode = getattr(model_args, 'out_proj_mode', 'standard')
        no_ffn = getattr(model_args, 'compressor_no_ffn', False)
        no_z_global = getattr(model_args, 'compressor_no_global_context', False)
        slot_embedding = getattr(model_args, 'compressor_slot_position_embedding', False)
        linear_writer = getattr(model_args, 'compressor_linear_projection', False)
        slot_mixer_type = getattr(model_args, 'slot_mixer_type', 'linear')
        slot_mixer_attn_heads = getattr(model_args, 'slot_mixer_attn_heads', 4)
        num_self_attn_layers = int(getattr(model_args, 'compressor_num_self_attn_layers', 0))
        num_iter_blocks = int(getattr(model_args, 'compressor_num_iter_blocks', 1))
        compressor_gated = bool(getattr(model_args, 'compressor_gated', True))
        compressor_ignore_prev_mem = bool(getattr(model_args, 'compressor_ignore_prev_mem', False))
        compressor_topics_init = getattr(model_args, 'compressor_topics_init', 'random')
        compressor_qk_norm = bool(getattr(model_args, 'compressor_qk_norm', False))
        compressor_qk_norm_init_scale = float(getattr(model_args, 'compressor_qk_norm_init_scale', 10.0))
        # Compressor: input in encoder-dim (backbone hidden states), output in
        # decoder-dim (slots feed decompressor → injected at decoder layer).
        # When enc_dim != dec_dim, SegmentCompressor's input_proj does
        # the dim crossing once at the input.
        self.compressor = build_compressor(
            compressor_type=model_args.compressor_type,
            input_dim=self.enc_dim,
            output_dim=self.slot_dim,
            num_memory_tokens=compressor_num_tokens,
            bottleneck_dim=model_args.compressor_hidden_dim,
            num_heads=model_args.compressor_num_heads,
            dropout=model_args.compressor_dropout,
            gate_init=model_args.gate_init,
            gate_max=getattr(model_args, 'gate_max', 0.5),
            blend_mode=blend_mode,
            slot_competition=slot_competition,
            out_proj_mode=out_proj_mode,
            no_ffn=no_ffn,
            no_z_global=no_z_global,
            slot_embedding=slot_embedding,
            linear_writer=linear_writer,
            slot_mixer_type=slot_mixer_type,
            slot_mixer_attn_heads=slot_mixer_attn_heads,
            num_self_attn_layers=num_self_attn_layers,
            num_iter_blocks=num_iter_blocks,
            gated=compressor_gated,
            ignore_prev_mem=compressor_ignore_prev_mem,
            topics_init=compressor_topics_init,
            qk_norm=compressor_qk_norm,
            qk_norm_init_scale=compressor_qk_norm_init_scale,
        )
        print(f"Compressor blend_mode: {blend_mode}, slot_competition: {slot_competition}, out_proj_mode: {out_proj_mode}")

        # which backbone layer to tap
        self.extraction_layer = model_args.extraction_layer

        # ---- multi-layer source mixing -------------------------------------
        self._mix_extraction_layers = bool(getattr(model_args, 'mix_extraction_layers', False))
        self.mixer = None
        self._mix_layers = None
        if self._mix_extraction_layers:
            mix_str = getattr(model_args, 'mixing_layers', '') or ''
            mix_layers = [int(x) for x in mix_str.split(',') if x.strip()]
            num_backbone_layers = self.backbone.config.num_hidden_layers
            for ℓ in mix_layers:
                if ℓ < 0 or ℓ > num_backbone_layers:
                    raise ValueError(
                        f"mixing_layers contains out-of-range layer {ℓ} "
                        f"(valid: 0..{num_backbone_layers}). Got mix_layers={mix_layers}"
                    )
            self._mix_layers = mix_layers
            # Mixer operates on encoder hidden states → must be in encoder dim.
            self.mixer = ExtractionLayerMixer(
                dim=self.enc_dim,
                mix_layers=mix_layers,
                normalization=getattr(model_args, 'mixing_norm', 'per_layer_ln'),
                weight_init=getattr(model_args, 'mixing_init', 'uniform'),
                learnable=bool(getattr(model_args, 'mixing_learnable', True)),
            )
            print(
                f"[SeDeM] mix_extraction_layers=True, mix_layers={mix_layers}, "
                f"normalization={model_args.mixing_norm}, "
                f"weight_init={model_args.mixing_init}, "
                f"learnable={model_args.mixing_learnable}, "
                f"extraction_layer={self.extraction_layer} (IGNORED for source extraction)"
            )

        # ---- decompress-inject mode (SeDeM) ---------------------------------
        self._inject_decompressed = getattr(model_args, 'inject_decompressed', False)
        if self._inject_decompressed:
            assert model_args.compressor_type in ('pooled_linear', 'learnable_query'), (
                f"inject_decompressed requires compressor_type in "
                f"('pooled_linear', 'learnable_query'), got '{model_args.compressor_type}'"
            )
            # Inject layer is decoder-side; source/HS layer is encoder-side.
            # Validate each against the model that hosts it.
            num_dec_layers = (self.decoder.config.num_hidden_layers
                              if hasattr(self, 'decoder')
                              else self.backbone.config.num_hidden_layers)
            num_enc_layers = self.backbone.config.num_hidden_layers
            _inject_raw = getattr(model_args, 'injection_layer', -999)
            # -999 sentinel: fall back to extraction_layer (old same-model behaviour).
            inject_layer = self.extraction_layer if _inject_raw == -999 else _inject_raw
            if inject_layer < 0:
                inject_layer = num_dec_layers + inject_layer
            # Allow 0 (embedding-level injection) up through last decoder layer.
            inject_layer = max(0, min(inject_layer, num_dec_layers - 1))
            self._injection_layer = inject_layer
            print(f"[SeDeM] inject_decompressed: source_layer={self.extraction_layer}/{num_enc_layers} (encoder), "
                  f"inject_layer={inject_layer}/{num_dec_layers} (decoder)")

        # ---- decompressor (auxiliary: reconstruction from memory → hidden states)
        # Always created for eval logging; only contributes to training loss if weight > 0
        # For inject_decompressed mode, also needed at inference.
        self.rec_loss_weight = training_args.rec_loss_weight
        self.ctx_token_loss_weight = getattr(training_args, 'ctx_token_loss_weight', 0.0)
        _need_decompressor = self._is_training or self._inject_decompressed
        if _need_decompressor:
            if model_args.compressor_type not in ('pooled_linear', 'learnable_query'):
                raise ValueError(
                    f"Decompressor requires compressor_type in "
                    f"('pooled_linear', 'learnable_query'), got '{model_args.compressor_type}'. "
                    f"Decompressor takes (B, M, D) memory tokens — both compressor "
                    f"variants produce that shape."
                )
            decompressor_type = getattr(model_args, 'decompressor_type', 'mlp')
            if decompressor_type != 'mlp':
                raise ValueError(
                    f"Only decompressor_type='mlp' is supported, got '{decompressor_type}'."
                )
            from sedem.compressor import Decompressor
            compression_factor = getattr(training_args, 'compression_factor', 4)
            self.decompressor = Decompressor(
                d_model=self.dim,
                chunk_size=compression_factor,
                input_dim=self.slot_dim,
            )
            # Cross-model L_decomp: decompressor outputs dec_dim, but the
            # supervised target for reconstruction loss is encoder hidden states
            # (enc_dim). Project rec → enc_dim before the cosine/MSE comparison.
            # In same-model mode this is identity (enc_dim == dec_dim).
            if self._cross_model:
                self._decomp_loss_proj = nn.Linear(self.dec_dim, self.enc_dim, bias=False)
                # Initialize as a left-pseudo-inverse-ish small scale to keep
                # gradients well-conditioned at step 0.
                nn.init.xavier_uniform_(self._decomp_loss_proj.weight, gain=0.5)
            else:
                self._decomp_loss_proj = None

        # ---- loss -------------------------------------------------------
        self.loss_ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.distill_loss_weight = training_args.distill_loss_weight
        self.distill_temperature = training_args.distill_temperature
        self.qa_weight = training_args.qa_weight

        # ---- LoRA on decoder (optional) ---------------------------------
        if lora_config is not None and self._is_training:
            self.decoder = get_peft_model(self.decoder, lora_config)
            print("LoRA applied to decoder:")
            self.decoder.print_trainable_parameters()

        # ---- trainable modules stay in fp32 for optimizer precision ------
        # bf16 has ULP ~0.001 at typical param values, which silently kills
        # AdamW updates of ~5e-5.  Keeping trainable params in fp32 lets
        # the optimizer accumulate small updates.  Forward-pass dtype casts
        # are handled explicitly (compressor output → bf16 before decoder).
        self._trainable_dtype = torch.float32
        self._backbone_dtype = dtype  # bf16
        self.compressor.to(torch.float32)
        self.special_token_embed.to(torch.float32)
        self.mem_norm.to(torch.float32)
        self.mem_scale.data = self.mem_scale.data.to(torch.float32)
        if hasattr(self, 'decompressor'):
            self.decompressor.to(torch.float32)

        if self._is_training:
            self._init_training()

    # ------------------------------------------------------------------
    def _init_training(self):
        print_trainable_parameters(self)
        if self.training_args.restore_from and self.training_args.restore_from != "":
            print(f"Loading from checkpoint: {self.training_args.restore_from} ...")
            state_dict = load_file(self.training_args.restore_from)
            # Detect old checkpoints that don't have mem_norm/mem_scale.
            if "mem_norm.weight" not in state_dict:
                print("  (old checkpoint without mem_norm — disabling memory normalization)")
                self._normalize_mem = False
            # Filter out keys with shape mismatches (e.g., old decompressor weights
            # when switching to a different decompressor variant)
            model_state = self.state_dict()
            filtered_keys = []
            for k, v in list(state_dict.items()):
                if k in model_state and model_state[k].shape != v.shape:
                    filtered_keys.append(k)
                    del state_dict[k]
            if filtered_keys:
                print(f"  Filtered {len(filtered_keys)} shape-mismatched keys: {filtered_keys[:10]}"
                      f"{'...' if len(filtered_keys) > 10 else ''}")
            result = self.load_state_dict(state_dict, strict=False)
            if result.missing_keys:
                print(f"  Missing keys (will use fresh init): {result.missing_keys[:10]}"
                      f"{'...' if len(result.missing_keys) > 10 else ''}")
            if result.unexpected_keys:
                print(f"  Unexpected keys (ignored): {result.unexpected_keys[:10]}"
                      f"{'...' if len(result.unexpected_keys) > 10 else ''}")
            print("Checkpoint loaded.")

            # Freeze compressor if requested (decompressor-only training)
            if getattr(self.training_args, 'freeze_compressor', False):
                for p in self.compressor.parameters():
                    p.requires_grad = False
                print("[freeze_compressor] Freezing compressor + mem_scale + mem_norm + special_token_embed...")
                self.mem_scale.requires_grad = False
                for p in self.mem_norm.parameters():
                    p.requires_grad = False
                self.special_token_embed.requires_grad = False
                trainable_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
                print(f"  Trainable after freeze: {trainable_count:,} params")

        if self._inject_decompressed:
            # Two-phase manual layer forward is incompatible with gradient checkpointing
            print("[inject_decompressed] Gradient checkpointing DISABLED (manual layer forward)")
        else:
            self.decoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # Verify fp32 fix survived checkpoint loading
        print("DTYPE AUDIT after init:")
        for name, p in self.compressor.named_parameters():
            if p.requires_grad:
                print(f"  compressor.{name}: dtype={p.dtype}, shape={p.shape}")
        print(f"  mem_scale: dtype={self.mem_scale.dtype}")
        print(f"  mem_norm.weight: dtype={self.mem_norm.weight.dtype}")
        print(f"  special_token_embed: dtype={self.special_token_embed.weight.dtype}")

        # ---- hidden state caching setup ----
        self._use_hs_cache = getattr(self.model_args, 'cache_hidden_states', False)
        self._hs_cache_dir = getattr(self.model_args, 'hidden_states_cache_dir', None)
        if self._use_hs_cache:
            assert self._hs_cache_dir is not None, \
                "hidden_states_cache_dir must be set when cache_hidden_states=True"
            import os
            # Include encoder name in cache path to prevent cross-model contamination
            # (hidden states differ in dim and content between encoder models).
            model_tag = self.encoder_name.replace("/", "_")
            self._hs_cache_dir = os.path.join(self._hs_cache_dir, model_tag)
            os.makedirs(self._hs_cache_dir, exist_ok=True)
            self._hs_cache_hits = 0
            self._hs_cache_misses = 0
            print(f"[HS Cache] Enabled, dir={self._hs_cache_dir}")

    # ------------------------------------------------------------------
    def compute_num_segments(self, total_length):
        assert total_length > 0
        return math.ceil(total_length / (self.memory_slots * self.compression_factor))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def extract_hidden_states(self, input_ids, attention_mask=None):
        """Run frozen backbone, return hidden states from the chosen layer.

        When mix_extraction_layers=True, extracts hidden states from each layer in
        self._mix_layers, applies the ExtractionLayerMixer (per-layer LN + learned α),
        and returns the mixed [B, T, D] tensor. Backbone params remain frozen;
        gradients flow only through the mixer.

        attention_mask is optional. Default None preserves the original
        unmasked call (used by sequential compress_segments, where every segment is
        passed as-is with no padding). The parallel-segment compress_segments path
        (SeDeM / no-recurrence only) passes a 2D mask to ignore right-
        padding on the last segment.

        Optional fast path (env/attribute gated): when
        ``model._backbone_early_exit`` is True AND mix_extraction_layers is False
        AND extraction_layer < num_hidden_layers, skip computing layers
        ``extraction_layer..L-1`` (the output at layer ``extraction_layer``
        is a pure function of layers 0..ℓ-1; subsequent layers' outputs are
        never read). For Llama-3.2-1B with extraction_layer=8 this saves
        ~50% of backbone compute on the compress hotpath. Bit-exact.
        """
        # Fast path: early-exit at layer ℓ (skip ℓ..L-1). Bit-exact.
        if (
            getattr(self, '_backbone_early_exit', False)
            and not self._mix_extraction_layers
        ):
            fast_out = self.extract_hidden_states_early_exit(
                input_ids, attention_mask=attention_mask
            )
            if fast_out is not None:
                # Optional bit-exact verifier: A/B vs full forward on first N calls.
                n_verify = int(getattr(self, '_verify_backbone_early_exit_remaining', 0))
                if n_verify > 0:
                    with torch.no_grad():
                        if attention_mask is None:
                            ref_out = self.backbone(
                                input_ids=input_ids, output_hidden_states=True
                            )
                        else:
                            ref_out = self.backbone(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                output_hidden_states=True,
                            )
                        ref_hs = ref_out.hidden_states[self.extraction_layer]
                    max_abs_diff = (fast_out.float() - ref_hs.float()).abs().max().item()
                    _tol = 0.0  # require strict bit-exact
                    print(f"  [backbone_early_exit verify] max_abs_diff={max_abs_diff:.3e} "
                          f"(tol={_tol:.0e}, remaining={n_verify - 1})")
                    if max_abs_diff > _tol:
                        raise RuntimeError(
                            f"backbone_early_exit not bit-exact: max_abs_diff={max_abs_diff:.3e} > tol={_tol:.0e}"
                        )
                    self._verify_backbone_early_exit_remaining = n_verify - 1
                return fast_out

        with torch.no_grad():
            if attention_mask is None:
                outputs = self.backbone(
                    input_ids=input_ids, output_hidden_states=True
                )
            else:
                outputs = self.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
        # outputs.hidden_states is a tuple of (num_layers+1) tensors
        if self._mix_extraction_layers:
            # gather + upcast outside no_grad so mixer params get gradients
            layer_hidden = [outputs.hidden_states[ℓ].float() for ℓ in self._mix_layers]
            return self.mixer(layer_hidden)  # (B, T, D)
        layer_idx = self.extraction_layer
        return outputs.hidden_states[layer_idx]  # (B, T, D)

    def extract_hidden_states_early_exit(self, input_ids, attention_mask=None):
        """Fast variant of extract_hidden_states: stops the backbone
        forward at layer ``self.extraction_layer``. Returns the same tensor
        ``outputs.hidden_states[self.extraction_layer]`` would, but without
        computing the unused tail layers.

        Returns None and lets the caller fall back to the full forward when:
          - extraction_layer == 0 (would want raw embeddings — uncommon path)
          - extraction_layer >= num_hidden_layers (no skip possible)
          - the hook never fires (unexpected — silent failure protection)

        Bit-exact: yes. Uses HF's own forward up to layer ℓ-1, then captures the
        output via a forward hook on that layer. The hook raises a sentinel
        exception to bail out of the remaining layers in HF's loop.
        """
        inner = self.backbone.model.model if hasattr(self.backbone, 'model') and hasattr(self.backbone.model, 'model') else self.backbone.model
        num_layers = len(inner.layers)
        exit_layer = int(self.extraction_layer)

        if exit_layer <= 0 or exit_layer > num_layers:
            return None  # let caller fall back

        target_layer_0idx = exit_layer - 1  # layer whose OUTPUT == hidden_states[exit_layer]

        captured = {}
        def _hook(module, args, output):
            captured['out'] = output[0] if isinstance(output, tuple) else output
            raise _BackboneEarlyExit()

        handle = inner.layers[target_layer_0idx].register_forward_hook(_hook)
        try:
            with torch.no_grad():
                kwargs = {'input_ids': input_ids, 'output_hidden_states': False}
                if attention_mask is not None:
                    kwargs['attention_mask'] = attention_mask
                self.backbone(**kwargs)
        except _BackboneEarlyExit:
            pass
        finally:
            handle.remove()

        return captured.get('out', None)

    def extract_or_load_cached_hidden_states(self, input_ids, num_segments, segment_length):
        """Extract hidden states with optional disk caching.

        Two cache formats depending on layer-mixing:
          - `mix_extraction_layers=False`: store the post-extraction tensor of shape
            (B, T, D). Stable across runs (frozen encoder, deterministic per
            input). Loaded as-is.
          - `mix_extraction_layers=True`: store the PER-LAYER tensor of shape
            (L_mix, B, T, D) where L_mix is the number of mix-source layers.
            The mixer's learnable α weights are applied at *load time* so the
            cached per-layer tensors stay valid even as α evolves during
            training. Per-layer H is deterministic from the frozen encoder.

        Cache files use distinct suffixes (.perlayer.pt vs .pt) so the two
        formats can coexist in the same directory.

        Caveat: this assumes the encoder backbone is truly frozen during the
        run. In SAME-MODEL setups with LoRA on the decoder, LoRA wraps the
        backbone modules in-place (because `self.decoder = self.backbone`),
        which mutates the encoder forward output. Disable cache in that case.
        """
        use_cache = getattr(self, '_use_hs_cache', False)

        if use_cache:
            import hashlib, os
            ids_flat = input_ids[0]  # (T,) — batch size is always 1
            T = ids_flat.size(0)
            key_tokens = torch.cat([ids_flat[:64], ids_flat[-64:], torch.tensor([T], device=ids_flat.device)])
            ids_bytes = key_tokens.cpu().numpy().tobytes()
            cache_key = hashlib.sha256(ids_bytes).hexdigest()[:16]

            if self._mix_extraction_layers:
                cache_path = os.path.join(self._hs_cache_dir, f"{cache_key}.perlayer.pt")
            else:
                cache_path = os.path.join(self._hs_cache_dir, f"{cache_key}.pt")

            if os.path.exists(cache_path):
                # Cache hit — guard against truncated files written during disk-full events
                try:
                    cached = torch.load(cache_path, map_location=input_ids.device,
                                        weights_only=True)
                except Exception as _cache_err:
                    # Quarantine corrupt file and fall through to recompute
                    _quarantine_dir = os.path.join(os.path.dirname(self._hs_cache_dir),
                                                   "_quarantine_corrupt")
                    os.makedirs(_quarantine_dir, exist_ok=True)
                    _dest = os.path.join(_quarantine_dir, os.path.basename(cache_path))
                    os.rename(cache_path, _dest)
                    print(f"  [HS Cache] CORRUPT file quarantined: {cache_path} → {_dest} ({_cache_err})")
                else:
                    if self._mix_extraction_layers:
                        # cached: (L_mix, B, T, D). Apply mixer at load time so α
                        # gradient still flows through the mixer parameters.
                        layer_list = [cached[i].float() for i in range(cached.size(0))]
                        hidden_states = self.mixer(layer_list)               # (B, T, D)
                    else:
                        hidden_states = cached
                    self._hs_cache_hits += 1
                    total = self._hs_cache_hits + self._hs_cache_misses
                    if total <= 5 or total % 1000 == 0:
                        fmt = "perlayer" if self._mix_extraction_layers else "mixed"
                        print(f"  [HS Cache] HIT ({fmt}) (hits={self._hs_cache_hits}, "
                              f"misses={self._hs_cache_misses}, "
                              f"rate={self._hs_cache_hits/total:.1%})")
                    return hidden_states

        # Cache miss or caching disabled — compute hidden states via backbone
        total_length = input_ids.size(1)
        all_hidden = []
        # When caching per-layer, collect per-segment per-layer tensors too.
        per_layer_segments = [] if (use_cache and self._mix_extraction_layers) else None

        for seg_idx in range(num_segments):
            start = seg_idx * segment_length
            end = min((seg_idx + 1) * segment_length, total_length)
            seg_ids = input_ids[:, start:end]

            if per_layer_segments is not None:
                # Inline the encoder forward so we can grab per-layer H for caching.
                with torch.no_grad():
                    outputs = self.backbone(
                        input_ids=seg_ids, output_hidden_states=True
                    )
                layer_hidden = [outputs.hidden_states[ℓ].float() for ℓ in self._mix_layers]
                # Stash a stacked copy for the cache: (L_mix, B, T_seg, D)
                per_layer_segments.append(torch.stack(layer_hidden, dim=0))
                # Compute the mixed output for this segment (applies current α)
                seg_hidden = self.mixer(layer_hidden)
            else:
                seg_hidden = self.extract_hidden_states(seg_ids)

            all_hidden.append(seg_hidden)

        hidden_states = torch.cat(all_hidden, dim=1)                    # (B, T_ctx, D)

        if use_cache:
            if self._mix_extraction_layers:
                # Concat per-layer segments over T → (L_mix, B, T_ctx, D)
                to_save = torch.cat(per_layer_segments, dim=2).to(torch.bfloat16)
                torch.save(to_save, cache_path)
            else:
                torch.save(hidden_states.to(torch.bfloat16), cache_path)
            self._hs_cache_misses += 1
            total = self._hs_cache_hits + self._hs_cache_misses
            if total <= 5 or total % 500 == 0:
                fmt = "perlayer" if self._mix_extraction_layers else "mixed"
                print(f"  [HS Cache] MISS ({fmt}), saved to {cache_key}.pt "
                      f"(hits={self._hs_cache_hits}, misses={self._hs_cache_misses})")

        return hidden_states

    def embed_decoder_input_with_prefix(self, prompt_answer_ids, memory_tokens):
        """
        Build decoder input embeddings by replacing placeholder IDs with
        compressor outputs and special-token embeddings.

        prompt_answer_ids : (B, S)  -- contains normal token IDs, memory
                            placeholder IDs, and special-token IDs
        memory_tokens     : (B, M_total, D)  from compressor

        IMPORTANT: Uses out-of-place ops (torch.where) instead of in-place
        indexed assignment to ensure gradients flow back through memory_tokens
        to the compressor parameters. In-place ops on a non-grad base tensor
        (from frozen backbone) sever the autograd chain.
        """
        B, S = prompt_answer_ids.shape

        # Start from backbone embeddings (clamps IDs to valid range first)
        safe_ids = prompt_answer_ids.clamp(max=self.original_vocab_size - 1)
        embs = self._decoder_embed_tokens(safe_ids).detach()  # (B, S, D), no grad

        # Identify memory-token placeholders
        mem_mask = (prompt_answer_ids >= self.mem_id_start) & (
            prompt_answer_ids < self.mem_id_start + self.memory_slots
        )
        num_mem_slots = mem_mask.sum().item()
        num_mem_provided = memory_tokens.reshape(-1, self.dim).size(0)
        if num_mem_slots != num_mem_provided:
            raise ValueError(
                f"Memory slot count mismatch! "
                f"prompt_answer_ids has {num_mem_slots} mem placeholders, "
                f"but compressor produced {num_mem_provided} memory tokens. "
                f"prompt_answer_ids shape={prompt_answer_ids.shape}, "
                f"memory_tokens shape={memory_tokens.shape}"
            )

        # Normalize memory tokens to match text embedding scale
        if hasattr(self, 'mem_norm') and hasattr(self, 'mem_scale') and self._normalize_mem:
            # Cap at 1.5/sqrt(D) so memory norm stays ≤ 1.5x text norm.
            # Without cap, mem_scale drifts upward (0.020→0.044 in prior run),
            # giving memory 2.2x text norm and starving text attention to ~4%.
            max_scale = 1.5 / math.sqrt(self.dim)
            effective_scale = torch.clamp(self.mem_scale, min=0.01, max=max_scale)
            normed_mem = self.mem_norm(memory_tokens) * effective_scale  # (B, M, D), requires grad
        else:
            normed_mem = memory_tokens  # (B, M, D), requires grad

        # Build full-size memory tensor at correct positions (out-of-place).
        # Memory placeholders are always the first M positions in prompt_answer_ids.
        M = normed_mem.size(1)  # total_mem_length
        # Pad normed_mem to full sequence length (M → S) with zeros on the right.
        # F.pad is out-of-place and preserves the autograd chain.
        mem_full = F.pad(normed_mem.to(embs.dtype), (0, 0, 0, S - M), value=0.0)  # (B, S, D)

        # Build special-token embeddings at correct positions (out-of-place)
        ae_mask = prompt_answer_ids == self.ae_token_id
        ft_mask = prompt_answer_ids == self.ft_token_id

        # For special tokens, broadcast a single embedding to all positions
        # then use torch.where to select. This avoids in-place indexed assignment.
        special_full = embs  # fallback (will be masked out anyway)
        has_special = ae_mask.any() or ft_mask.any()
        if has_special:
            # Each special token type gets one embedding, broadcast to (B, S, D)
            ae_emb = self.special_token_embed(
                torch.zeros(1, dtype=torch.long, device=embs.device)
            ).to(embs.dtype).expand(B, S, -1)  # (B, S, D)
            ft_emb = self.special_token_embed(
                torch.ones(1, dtype=torch.long, device=embs.device)
            ).to(embs.dtype).expand(B, S, -1)  # (B, S, D)
            # Select: ae positions get ae_emb, ft positions get ft_emb, rest unchanged
            ae_mask_3d = ae_mask.unsqueeze(-1).expand_as(embs)
            ft_mask_3d = ft_mask.unsqueeze(-1).expand_as(embs)
            special_full = torch.where(ae_mask_3d, ae_emb, embs)
            special_full = torch.where(ft_mask_3d, ft_emb, special_full)

        # Combine using torch.where (out-of-place, preserves autograd chain)
        # Priority: memory tokens > special tokens > base text embeddings
        mem_mask_3d = mem_mask.unsqueeze(-1).expand_as(embs)  # (B, S, D)

        # Start with base text + special tokens, then overlay memory tokens
        result = special_full if has_special else embs
        result = torch.where(mem_mask_3d, mem_full, result)

        return result

    # ------------------------------------------------------------------
    # Forward  (training)
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        prompt_answer_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        teacher_input_ids: Optional[torch.LongTensor] = None,
        teacher_labels: Optional[torch.LongTensor] = None,
        teacher_offset: Optional[torch.LongTensor] = None,
        evidence_mask: Optional[torch.LongTensor] = None,
    ):
        """
        Parameters
        ----------
        input_ids          : (B, T_ctx)   -- context token IDs
        prompt_answer_ids  : (B, S)       -- decoder input with mem placeholders
        labels             : (B, S)       -- target IDs (-100 for masked positions)
        teacher_input_ids  : (B, T_teach) -- full-context input for distillation
                             (optional, only when distill_loss_weight > 0)
        teacher_labels     : (B, S_teach) -- labels aligned with teacher_input_ids
                             (optional)
        """
        B = input_ids.size(0)
        total_length = input_ids.size(1)

        # ---- step counter for periodic diagnostics (every 50 steps) ----
        if not hasattr(self, "_fwd_step"):
            self._fwd_step = 0
            self._train_fwd_step = 0
        self._fwd_step += 1
        if self.training and torch.is_grad_enabled():
            self._train_fwd_step += 1
        _is_training = self.training and torch.is_grad_enabled()
        _verbose = _is_training and ((self._train_fwd_step <= 3) or (self._train_fwd_step % 200 == 0))

        if _verbose:
            t_off = teacher_offset[0].item() if teacher_offset is not None else 0
            total_mem = self.compute_num_segments(total_length) * self.memory_slots
            window_len = prompt_answer_ids.size(1) - total_mem - 1  # subtract mem + AE
            eff_scale = torch.clamp(self.mem_scale, min=0.01).item()
            raw_scale = self.mem_scale.item()
            _mode = "TRAIN" if (self.training and torch.is_grad_enabled()) else "EVAL"
            print(f"\n[step={self._fwd_step} {_mode} train_fwd={self._train_fwd_step}] "
                  f"ctx={total_length}, window={window_len}(start={t_off}), "
                  f"segs={self.compute_num_segments(total_length)}, "
                  f"mem_scale={eff_scale:.4f} (raw={raw_scale:.4f})")

            dtype_parts = []
            for name, p in self.compressor.named_parameters():
                if 'weight' in name and p.requires_grad:
                    dtype_parts.append(f"{name}: dtype={p.dtype}")
                    break
            dtype_parts.append(f"mem_scale: dtype={self.mem_scale.dtype}")
            print(f"  DTYPE CHECK: {', '.join(dtype_parts)}")
            if hasattr(self, '_debug_mem_grad_norm'):
                mem_g = self._debug_mem_grad_norm
                text_g = self._debug_text_grad_norm
                ratio = mem_g / max(text_g, 1e-10)
                print(f"  grads(prev): mem={mem_g:.6f}, text={text_g:.6f}, ratio={ratio:.4f}")

        # 1) Extract hidden states from backbone (frozen), with optional caching
        num_segments = self.compute_num_segments(total_length)
        segment_length = math.ceil(total_length / num_segments)

        hidden_states = self.extract_or_load_cached_hidden_states(input_ids, num_segments, segment_length)

        # Check backbone output for NaN/Inf
        if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
            nan_count = torch.isnan(hidden_states).sum().item()
            inf_count = torch.isinf(hidden_states).sum().item()
            hs_norm = hidden_states.float().norm(dim=-1)
            raise ValueError(
                f"[step={self._fwd_step}] NaN/Inf in backbone hidden states! "
                f"nan={nan_count}, inf={inf_count}, "
                f"shape={list(hidden_states.shape)}, "
                f"norm min={hs_norm.min().item():.2f}, max={hs_norm.max().item():.2f}, "
                f"input_ids range=[{input_ids.min().item()}, {input_ids.max().item()}]"
            )

        # 2) Compress hidden states → M memory tokens per segment
        #    Each segment's compressor receives the previous segment's memory
        #    tokens (detached) as additional KV context for cross-attention.
        mem_per_segment = []
        prev_mem = None
        attn_past_mass_sum = 0.0
        attn_past_mass_count = 0
        _alloc_accum = {}  # allocation diagnostics (keyed by stat name)
        _alloc_count = 0
        for seg_idx in range(num_segments):
            start = seg_idx * segment_length
            end = min((seg_idx + 1) * segment_length, total_length)
            seg_hidden = hidden_states[:, start:end, :]
            # Upcast to fp32 for compressor (trainable params are fp32 for optimizer precision)
            seg_hidden_fp32 = seg_hidden.float()
            # verbose for first & last segment only (avoid flooding)
            _comp_verbose = _verbose and (seg_idx == 0 or seg_idx == num_segments - 1)
            seg_mem, stats = self.compressor(
                seg_hidden_fp32, prev_mem=prev_mem, return_attn_stats=True,
                verbose=_comp_verbose, seg_idx=seg_idx,
            )
            # Per-segment NaN check to isolate which segment causes issues
            if torch.isnan(seg_mem).any() or torch.isinf(seg_mem).any():
                nan_count = torch.isnan(seg_mem).sum().item()
                inf_count = torch.isinf(seg_mem).sum().item()
                seg_hs_nan = torch.isnan(seg_hidden).any().item()
                seg_hs_norm = seg_hidden.float().norm(dim=-1)
                mem_norm = seg_mem.float().norm(dim=-1)
                raise ValueError(
                    f"[step={self._fwd_step}] NaN/Inf in seg_mem at seg {seg_idx}! "
                    f"nan={nan_count}, inf={inf_count}, "
                    f"seg_hidden NaN={seg_hs_nan}, "
                    f"seg_hidden norm=[{seg_hs_norm.min().item():.2f}, {seg_hs_norm.max().item():.2f}], "
                    f"seg_mem norm=[{mem_norm[~torch.isnan(mem_norm)].min().item():.2f}, {mem_norm[~torch.isnan(mem_norm)].max().item():.2f}], "
                    f"prev_mem={'None' if prev_mem is None else f'norm={prev_mem.float().norm(dim=-1).mean().item():.2f}'}"
                )
            mem_per_segment.append(seg_mem)

            if _verbose and (seg_idx == 0 or seg_idx == num_segments - 1):
                parts = [f"seg {seg_idx}"]
                pm = stats.get("attn_past_mass", 0)
                pm_val = pm.item() if torch.is_tensor(pm) else pm
                if pm_val > 0:
                    parts.append(f"past_mass={pm_val:.4f}")
                parts.append(f"pre_norm={stats.get('pre_mlp_norm', 0):.1f}")
                parts.append(f"post_norm={stats.get('post_mlp_norm', 0):.1f}")
                parts.append(f"ffn_cos={stats.get('ffn_cosine', 0):.4f}")
                if "ffn_contrib_ratio" in stats:
                    parts.append(f"ffn_contrib={stats['ffn_contrib_ratio']:.4f}")
                if "sa_gate" in stats:
                    parts.append(f"sa_gate={stats['sa_gate']:.6f}")
                if "xattn_gate" in stats:
                    parts.append(f"xattn={stats['xattn_gate']:.6f}")
                if "xattn_update_ratio" in stats:
                    parts.append(f"update_ratio={stats['xattn_update_ratio']:.4f}")
                if "attn_entropy_normalized" in stats:
                    parts.append(f"attn_ent={stats['attn_entropy_normalized']:.3f}")
                if "attn_alloc_cosine" in stats:
                    parts.append(f"alloc_cos={stats['attn_alloc_cosine']:.3f}")
                    # Per-head range (compact)
                    if "attn_alloc_cos_per_head_min" in stats:
                        parts.append(f"alloc_cos/hd=[{stats['attn_alloc_cos_per_head_min']:.3f},{stats['attn_alloc_cos_per_head_max']:.3f}]")
                if "attn_effective_rank" in stats:
                    parts.append(f"eff_rank={stats['attn_effective_rank']:.1f}")
                    if "attn_eff_rank_per_head_min" in stats:
                        parts.append(f"eff_rank/hd=[{stats['attn_eff_rank_per_head_min']:.1f},{stats['attn_eff_rank_per_head_max']:.1f}]")
                if "attn_pre_alloc_cosine" in stats:
                    parts.append(f"pre_alloc_cos={stats['attn_pre_alloc_cosine']:.3f}")
                    parts.append(f"pre_eff_rank={stats.get('attn_pre_effective_rank', 0):.1f}")
                if "slot_mass_mean" in stats:
                    parts.append(f"slot_mass={stats['slot_mass_mean']:.3f}±{stats['slot_mass_std']:.3f}")
                    parts.append(f"dead={stats['slot_near_dead_frac']:.2f}")
                print(f"  compressor: {', '.join(parts)}")
                # Pipeline diversity (compact, first + last segment only)
                if "pipe_V_cos" in stats or "pipe_raw_hs_cos" in stats:
                    stages = ["raw_hs", "V", "AV", "out_proj", "norm_match", "blend", "memory"]
                    pipe_parts = []
                    for s in stages:
                        cos_k = f"pipe_{s}_cos"
                        rank_k = f"pipe_{s}_eff_rank"
                        if cos_k in stats:
                            pipe_parts.append(f"{s}=[cos:{stats[cos_k]:.3f},rk:{stats.get(rank_k, 0):.1f}]")
                    print(f"  pipeline: {', '.join(pipe_parts)}")
                # Query specialization & coverage diagnostics
                if "cov_max_mean" in stats:
                    spec_parts = []
                    spec_parts.append(f"cov_max={stats['cov_max_mean']:.4f}(min={stats['cov_max_min']:.4f})")
                    spec_parts.append(f"dead={stats['cov_dead_frac']:.3f}")
                    spec_parts.append(f"gini={stats['cov_gini']:.3f}")
                    spec_parts.append(f"com_std_n={stats['topic_com_std_norm']:.3f}")
                    spec_parts.append(f"com_range={stats['topic_com_range']:.1f}")
                    spec_parts.append(f"width_n={stats['topic_support_width_mean_norm']:.3f}")
                    spec_parts.append(f"ent=[{stats['topic_entropy_min']:.3f},{stats['topic_entropy_max']:.3f}]μ={stats['topic_entropy_mean']:.3f}")
                    if "topic_top1_mass_mean" in stats:
                        spec_parts.append(f"top1={stats['topic_top1_mass_mean']:.4f}")
                    if "topic_top4_mass_mean" in stats:
                        spec_parts.append(f"top4={stats['topic_top4_mass_mean']:.4f}")
                    if "topic_top8_mass_mean" in stats:
                        spec_parts.append(f"top8={stats['topic_top8_mass_mean']:.4f}")
                    print(f"  specialization: {', '.join(spec_parts)}")
            # Accumulate allocation diagnostics (all attn_*, slot_* keys)
            _alloc_keys = [k for k in stats if k.startswith(("attn_alloc", "attn_eff",
                           "attn_pre_alloc", "attn_pre_eff", "slot_", "pipe_",
                           "cov_", "topic_"))]
            if _alloc_keys:
                _alloc_count += 1
                for k in _alloc_keys:
                    _alloc_accum[k] = _alloc_accum.get(k, 0.0) + stats[k]
            prev_mem = seg_mem
            if seg_idx > 0:
                pm = stats.get("attn_past_mass", 0)
                attn_past_mass_sum += (pm.item() if torch.is_tensor(pm) else pm)
                attn_past_mass_count += 1
        memory_tokens = torch.cat(mem_per_segment, dim=1)  # (B, M_total, D)
        if attn_past_mass_count > 0:
            self._last_attn_past_mass = attn_past_mass_sum / attn_past_mass_count
        else:
            self._last_attn_past_mass = 0.0

        # Cross-slot mixer gate diagnostic (Tier 1.2): tanh(raw_gate) ∈ (-1, 1).
        # 0 at init means strict pass-through; non-zero means model is using the
        # attn-based slot-axis mixer. Only meaningful when slot_mixer_type='attn'.
        _gate_raw = getattr(self.compressor, 'cross_slot_attn_gate', None)
        if _gate_raw is not None:
            self._last_cross_slot_mixer_gate = float(torch.tanh(_gate_raw.detach()).item())
        else:
            self._last_cross_slot_mixer_gate = 0.0
        # Attn-shape diagnostics (entropy, normalized entropy, diagonal mass).
        # Populated on verbose steps inside the compressor; otherwise the prior
        # value is preserved. Surface to output dict for wandb logging.
        for _key in ("cross_slot_attn_entropy",
                     "cross_slot_attn_entropy_norm",
                     "cross_slot_attn_diag_mass"):
            if hasattr(self.compressor, f"_last_attn_{_key.split('_', 3)[-1]}"):
                # The compressor stores under _last_attn_entropy / _last_attn_entropy_norm /
                # _last_attn_diag_mass — read directly.
                pass  # handled by reading _LAST_ATTN_* below
        for _src, _dst in (
            ("_last_attn_entropy", "_last_cross_slot_attn_entropy"),
            ("_last_attn_entropy_norm", "_last_cross_slot_attn_entropy_norm"),
            ("_last_attn_diag_mass", "_last_cross_slot_attn_diag_mass"),
        ):
            if hasattr(self.compressor, _src):
                setattr(self, _dst, float(getattr(self.compressor, _src)))

        # ---- safety checks ----
        if torch.isnan(memory_tokens).any() or torch.isinf(memory_tokens).any():
            raise ValueError(f"[step={self._fwd_step}] NaN/Inf in memory_tokens!")

        # ---- memory diversity diagnostics (compact) ----
        if _verbose:
            with torch.no_grad():
                # Intra-slot diversity (avg pairwise cosine within each segment)
                intra_cos_parts = []
                for si, seg_m in enumerate(mem_per_segment):
                    slots = F.normalize(seg_m[0].float(), dim=-1)
                    sim = slots @ slots.T
                    M = slots.size(0)
                    mask = ~torch.eye(M, dtype=torch.bool, device=sim.device)
                    intra_cos_parts.append(f"{sim[mask].mean().item():.4f}")
                # Inter-segment cosine (if >1 segment)
                inter_cos_parts = []
                for si in range(1, num_segments):
                    p = mem_per_segment[si-1].float().mean(dim=1)
                    c = mem_per_segment[si].float().mean(dim=1)
                    inter_cos_parts.append(f"{F.cosine_similarity(p, c, dim=-1).mean().item():.4f}")
                print(f"  memory: norm={memory_tokens.float().norm(dim=-1).mean().item():.2f}, "
                      f"intra_cos=[{','.join(intra_cos_parts)}]"
                      + (f", inter_cos=[{','.join(inter_cos_parts)}]" if inter_cos_parts else ""))

        decoder_memory = memory_tokens
        decoder_num_segments = num_segments

        # ==================================================================
        # SeDeM: decompress-inject path
        # ==================================================================
        if self._inject_decompressed:
            # 2c) Decompress memory → reconstructed hidden states at layer ℓ.
            all_reconstructed = []
            for seg_idx in range(decoder_num_segments):
                seg_mem = decoder_memory[
                    :, seg_idx * self.memory_slots : (seg_idx + 1) * self.memory_slots, :
                ]
                seg_recon = self.decompressor(seg_mem.float())  # (B, M*chunk_size, D)
                all_reconstructed.append(seg_recon)
            decompressed_hs = torch.cat(all_reconstructed, dim=1)  # (B, T_recon_total, D)

            # Normalize to match hidden-state scale at inject layer
            if hasattr(self, 'mem_norm') and hasattr(self, 'mem_scale') and self._normalize_mem:
                max_scale = 25.0 / math.sqrt(self.dim)  # allow norms up to ~25 (layer-14 text ≈ 17)
                effective_scale = torch.clamp(self.mem_scale, min=0.01, max=max_scale)
                decompressed_hs = self.mem_norm(decompressed_hs) * effective_scale

            # Build text-only decoder input (strip memory placeholders)
            total_mem_in_prompt = num_segments * self.memory_slots
            text_ids = prompt_answer_ids[:, total_mem_in_prompt:]
            safe_text_ids = text_ids.clamp(max=self.original_vocab_size - 1)
            if self._is_training:
                text_embs = self._decoder_embed_tokens(safe_text_ids).detach()
            else:
                text_embs = self._decoder_embed_tokens(safe_text_ids)

            # Replace special tokens (AE, FT) with learned embeddings
            ae_mask = text_ids == self.ae_token_id
            ft_mask = text_ids == self.ft_token_id
            if ae_mask.any() or ft_mask.any():
                ae_emb = self.special_token_embed(
                    torch.zeros(1, dtype=torch.long, device=text_embs.device)
                ).to(text_embs.dtype).expand_as(text_embs)
                ft_emb = self.special_token_embed(
                    torch.ones(1, dtype=torch.long, device=text_embs.device)
                ).to(text_embs.dtype).expand_as(text_embs)
                ae_mask_3d = ae_mask.unsqueeze(-1).expand_as(text_embs)
                ft_mask_3d = ft_mask.unsqueeze(-1).expand_as(text_embs)
                text_embs = torch.where(ae_mask_3d, ae_emb, text_embs)
                text_embs = torch.where(ft_mask_3d, ft_emb, text_embs)

            # Adjust labels: strip memory placeholder region
            if labels is not None:
                labels = labels[:, total_mem_in_prompt:]
            total_mem_length = 0  # no memory in decoder input sequence

            # -------- replace path (current/legacy): concat [Ĥ|text] at inject_layer --------
            # Two-phase forward: layers 0..ℓ-1 text-only, ℓ..L-1 [decompressed | text]
            _need_decomp_logits = self.ctx_token_loss_weight > 0
            # Also request attention for attn_past_mass measurement on verbose steps
            _fwd_result = self.forward_with_injection(
                text_embs, decompressed_hs,
                inject_layer=self._injection_layer,
                _verbose=_verbose,
                return_decomp_logits=_need_decomp_logits,
                measure_mem_attention=_verbose,
            )
            if _need_decomp_logits:
                logits, decomp_logits = _fwd_result[:2]
                if len(_fwd_result) > 2:
                    self._last_attn_past_mass = _fwd_result[2]
            else:
                if isinstance(_fwd_result, tuple):
                    logits = _fwd_result[0]
                    if len(_fwd_result) > 1:
                        self._last_attn_past_mass = _fwd_result[1]
                else:
                    logits = _fwd_result
                decomp_logits = None
            vocab_size = logits.size(-1)

            if _verbose:
                with torch.no_grad():
                    d_norm = decompressed_hs.float().norm(dim=-1).mean().item()
                    t_norm = text_embs.float().norm(dim=-1).mean().item()
                    print(f"  decoder_in(DECOMPRESS_INJECT/replace): "
                          f"decompressed={decompressed_hs.size(1)} tokens (norm={d_norm:.2f}), "
                          f"text={text_embs.size(1)} tokens (norm={t_norm:.2f}), "
                          f"inject_layer={self._injection_layer}, "
                          f"attn_past_mass={self._last_attn_past_mass:.4f}")

        # Skip to loss computation (steps 5+)
        # We need: logits, labels, vocab_size, total_mem_length — all set above.

            # 5) Reconstruction / QA cross-entropy loss
            shift_logits = logits[:, :-1, :].reshape(-1, vocab_size)
            shift_labels = labels[:, 1:].reshape(-1)
            loss_next_token = self.loss_ce(shift_logits, shift_labels)

            num_valid_labels = (shift_labels != -100).sum().item()
            if num_valid_labels == 0:
                raise ValueError(f"[step={self._fwd_step}] No valid labels! All labels are -100. "
                                 f"labels shape={labels.shape}")
            if torch.isnan(loss_next_token) or torch.isinf(loss_next_token):
                raise ValueError(f"[step={self._fwd_step}] NaN/Inf in loss_next_token={loss_next_token.item()}")

            total_loss = loss_next_token
            output = {"loss_next_token": loss_next_token.detach()}

            # Window size
            total_mem = num_segments * self.memory_slots
            window_len = prompt_answer_ids.size(1) - total_mem - 1
            output["window_size"] = window_len

            # Multi-layer mixing: log alpha per layer (HF Trainer scalarizes 0-d tensors)
            if self._mix_extraction_layers and self.mixer is not None:
                alpha_vec = self.mixer.alpha.detach()
                for i, ℓ in enumerate(self.mixer.mix_layers):
                    output[f"mix_alpha_L{ℓ}"] = alpha_vec[i]

            # 6) Optional KL distillation loss
            if self.distill_loss_weight > 0 and teacher_input_ids is not None:
                loss_distill = self.distillation_loss(
                    student_logits=logits,
                    teacher_input_ids=teacher_input_ids,
                    labels=labels,
                    teacher_offset=teacher_offset,
                    _verbose=_verbose,
                )
                if torch.isnan(loss_distill) or torch.isinf(loss_distill):
                    raise ValueError(f"[step={self._fwd_step}] NaN/Inf in loss_distill")
                total_loss = total_loss + loss_distill
                output["loss_distill"] = loss_distill.detach()

            # 7) Decompressor loss (reconstruction fidelity)
            if hasattr(self, "decompressor"):
                loss_decomp = self.hidden_state_reconstruction_loss(
                    memory_tokens=memory_tokens,
                    hidden_states=hidden_states,
                    num_segments=num_segments,
                    segment_length=segment_length,
                    total_length=total_length,
                    _verbose=_verbose,
                )
                if torch.isnan(loss_decomp) or torch.isinf(loss_decomp):
                    raise ValueError(f"[step={self._fwd_step}] NaN/Inf in loss_decomp")
                if self.rec_loss_weight > 0:
                    total_loss = total_loss + loss_decomp
                output["loss_decomp"] = loss_decomp.detach()

            # 7b) Token reconstruction CE loss (L_tok)
            # Passes decompressed hidden states through layers ℓ..L → logits,
            # then computes CE against original context token IDs.
            if self.ctx_token_loss_weight > 0 and decomp_logits is not None:
                # decomp_logits: (B, T_recon, V) — logits at decompressed positions
                # input_ids: (B, T_ctx) — original context tokens
                T_recon = decomp_logits.size(1)
                T_ctx = input_ids.size(1)
                # Autoregressive shift: predict token t+1 from position t
                # Use min(T_recon, T_ctx) to handle edge cases
                T_align = min(T_recon, T_ctx)
                tok_logits = decomp_logits[:, :T_align - 1, :].reshape(-1, vocab_size)
                tok_targets = input_ids[:, 1:T_align].reshape(-1)
                loss_tok = self.ctx_token_loss_weight * self.loss_ce(tok_logits, tok_targets)
                if _verbose:
                    with torch.no_grad():
                        tok_preds = tok_logits.argmax(dim=-1)
                        tok_acc = (tok_preds == tok_targets).float().mean().item()
                        print(f"  L_tok: loss={loss_tok.item():.4f}, "
                              f"acc={tok_acc:.4f}, T_recon={T_recon}, T_ctx={T_ctx}")
                if not (torch.isnan(loss_tok) or torch.isinf(loss_tok)):
                    total_loss = total_loss + loss_tok
                output["loss_tok"] = loss_tok.detach()

            # ---- total loss ----
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                raise ValueError(f"[step={self._fwd_step}] NaN/Inf in total_loss!")
            if _verbose:
                parts = [f"next_token={loss_next_token.item():.4f}"]
                for k in ["loss_distill", "loss_decomp"]:
                    if k in output:
                        parts.append(f"{k.split('_',1)[1]}={output[k].item():.4f}")
                print(f"  loss: total={total_loss.item():.4f} ({', '.join(parts)})")

            output["loss"] = total_loss
            output["logits"] = logits
            output["attn_past_mass"] = self._last_attn_past_mass
            output["cross_slot_mixer_gate"] = getattr(self, '_last_cross_slot_mixer_gate', 0.0)
            for _k in ("cross_slot_attn_entropy",
                       "cross_slot_attn_entropy_norm",
                       "cross_slot_attn_diag_mass"):
                _v = getattr(self, f"_last_{_k}", None)
                if _v is not None:
                    output[_k] = _v

            # Diversity diagnostics
            div_metrics = memory_diversity_metrics(mem_per_segment, verbose=_verbose)
            output.update(div_metrics)

            return output
        # ==================================================================
        # Prefix path (no decompression): memory slots are prepended to the
        # decoder input as a soft prefix (Table 4 'no decompression' ablation).
        # ==================================================================
        # Standard mode: prepend memory tokens to decoder input
        decoder_embs = self.embed_decoder_input_with_prefix(prompt_answer_ids, memory_tokens)
        total_mem_length = num_segments * self.memory_slots

        if _verbose:
            with torch.no_grad():
                _is_proj = False  # FFN is always standard residual now
                mem_norm = decoder_embs[:, :total_mem_length, :].float().norm(dim=-1).mean().item()
                text_norm = decoder_embs[:, total_mem_length:, :].float().norm(dim=-1).mean().item()
                sample_text = decoder_embs[:, total_mem_length:total_mem_length+32, :].float()
                sample_mem = decoder_embs[:, :32, :].float()
                if sample_text.size(1) > 0 and sample_mem.size(1) > 0:
                    mem_centroid = F.normalize(sample_mem.mean(dim=1), dim=-1)
                    text_centroid = F.normalize(sample_text.mean(dim=1), dim=-1)
                    space_cos = F.cosine_similarity(mem_centroid, text_centroid, dim=-1).item()
                else:
                    space_cos = 0.0
                mode_str = "PROJ" if _is_proj else "GATE"
                print(f"  decoder_in({mode_str}): mem_norm={mem_norm:.2f}, text_norm={text_norm:.2f}, "
                      f"ratio={mem_norm / max(text_norm, 1e-8):.2f}, "
                      f"mem↔text_cos={space_cos:.4f}")

    # Gradient hook: measure gradient flow to memory vs text positions
        if decoder_embs.requires_grad:
            _total_mem = total_mem_length
            _self = self
            def _grad_hook(grad):
                mem_grad = grad[:, :_total_mem, :].float()
                text_grad = grad[:, _total_mem:, :].float()
                _self._debug_mem_grad_norm = mem_grad.norm().item()
                _self._debug_text_grad_norm = text_grad.norm().item()
                return grad
            decoder_embs.register_hook(_grad_hook)

        # 4) Decoder forward
        # Request attention weights sparingly (expensive with gradient checkpointing)
        _output_attentions = (self._fwd_step <= 3) or (self._fwd_step % 200 == 0)
        # Use self.decoder for both training and inference. In same-model inference
        # it aliases self.backbone; in cross-model it's the 3B answer-generation model.
        decoder_out = self.decoder(
            inputs_embeds=decoder_embs,
            output_hidden_states=False,
            output_attentions=_output_attentions,
        )

        # NOTE: Do NOT clear memory here — gradient checkpointing reruns the
        # forward pass during backward, and needs the hooks to produce the same
        # outputs. Memory is overwritten at the start of the next forward pass.

        logits = decoder_out.logits
        vocab_size = logits.size(-1)

        # Decoder attention diagnostic
        if _verbose:
            if (hasattr(decoder_out, 'attentions') and decoder_out.attentions is not None
                    and len(decoder_out.attentions) > 0 and total_mem_length > 0):
                # Self-attention mode: memory is prepended to decoder input.
                # Guard against SDPA returning an empty attentions tuple even when
                # output_attentions=True (Transformers can silently skip the eager
                # fallback in some versions, leaving attentions=()).
                with torch.no_grad():
                    last_attn = decoder_out.attentions[-1].float()  # (B, H, S, S)
                    mem_mass = last_attn[:, :, :, :total_mem_length].sum(dim=-1).mean().item()
                    text_mass = last_attn[:, :, :, total_mem_length:].sum(dim=-1).mean().item()
                    total = mem_mass + text_mass
                    print(f"  dec_attn: mem_frac={mem_mass/max(total,1e-8):.4f}, "
                          f"text_frac={text_mass/max(total,1e-8):.4f}")
                    del last_attn

        # 5) Reconstruction / QA cross-entropy loss
        shift_logits = logits[:, :-1, :].reshape(-1, vocab_size)
        shift_labels = labels[:, 1:].reshape(-1)
        loss_next_token = self.loss_ce(shift_logits, shift_labels)

        num_valid_labels = (shift_labels != -100).sum().item()
        if num_valid_labels == 0:
            raise ValueError(f"[step={self._fwd_step}] No valid labels! All labels are -100. "
                             f"labels shape={labels.shape}")
        if torch.isnan(loss_next_token) or torch.isinf(loss_next_token):
            raise ValueError(f"[step={self._fwd_step}] NaN/Inf in loss_next_token={loss_next_token.item()}")

        total_loss = loss_next_token
        output = {"loss_next_token": loss_next_token.detach()}

        # Log window size (useful for tracking memonly fraction)
        total_mem = num_segments * self.memory_slots
        window_len = prompt_answer_ids.size(1) - total_mem - 1  # subtract mem + AE
        output["window_size"] = window_len

        # Multi-layer mixing: log alpha per layer
        if self._mix_extraction_layers and self.mixer is not None:
            alpha_vec = self.mixer.alpha.detach()
            for i, ℓ in enumerate(self.mixer.mix_layers):
                output[f"mix_alpha_L{ℓ}"] = alpha_vec[i]

        # 6) Optional KL distillation loss
        if self.distill_loss_weight > 0 and teacher_input_ids is not None:
            loss_distill = self.distillation_loss(
                student_logits=logits,
                teacher_input_ids=teacher_input_ids,
                labels=labels,
                teacher_offset=teacher_offset,
                _verbose=_verbose,
            )
            if torch.isnan(loss_distill) or torch.isinf(loss_distill):
                raise ValueError(f"[step={self._fwd_step}] NaN/Inf in loss_distill={loss_distill.item()}")
            total_loss = total_loss + loss_distill
            output["loss_distill"] = loss_distill.detach()

        # 7) Hidden-state decompressor loss (reconstruction from memory → hidden states)
        # Always compute for logging (eval plots); only add to training loss if weight > 0
        if hasattr(self, "decompressor"):
            loss_decomp = self.hidden_state_reconstruction_loss(
                memory_tokens=memory_tokens,
                hidden_states=hidden_states,
                num_segments=num_segments,
                segment_length=segment_length,
                total_length=total_length,
                _verbose=_verbose,
            )
            if torch.isnan(loss_decomp) or torch.isinf(loss_decomp):
                raise ValueError(f"[step={self._fwd_step}] NaN/Inf in loss_decomp={loss_decomp.item()}")
            if self.rec_loss_weight > 0:
                total_loss = total_loss + loss_decomp
            output["loss_decomp"] = loss_decomp.detach()

        # ---- total loss ----
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            raise ValueError(f"[step={self._fwd_step}] NaN/Inf in total_loss!")
        if _verbose:
            parts = [f"next_token={loss_next_token.item():.4f}"]
            for k in ["loss_distill", "loss_decomp"]:
                if k in output:
                    parts.append(f"{k.split('_',1)[1]}={output[k].item():.4f}")
            print(f"  loss: total={total_loss.item():.4f} ({', '.join(parts)})")

        output["loss"] = total_loss
        output["logits"] = logits

        # Cross-segment attention diagnostic
        output["attn_past_mass"] = self._last_attn_past_mass
        # Cross-slot mixer gate (Tier 1.2): tanh(raw) ∈ (-1, 1); 0 = pass-through
        output["cross_slot_mixer_gate"] = getattr(self, '_last_cross_slot_mixer_gate', 0.0)
        for _k in ("cross_slot_attn_entropy",
                   "cross_slot_attn_entropy_norm",
                   "cross_slot_attn_diag_mass"):
            _v = getattr(self, f"_last_{_k}", None)
            if _v is not None:
                output[_k] = _v

        # Diversity diagnostics (inter-segment, intra-slot, effective rank)
        div_metrics = memory_diversity_metrics(mem_per_segment, verbose=_verbose)
        output.update(div_metrics)

        # Allocation diagnostics: pairwise attention overlap & effective rank
        if _alloc_count > 0:
            for k, v in _alloc_accum.items():
                output[k] = v / _alloc_count

        return output

    # ------------------------------------------------------------------
    def distillation_loss(self, student_logits, teacher_input_ids, labels,
                           teacher_offset=None, _verbose=False):
        """
        KL(teacher || student) on positions where labels != -100.
        Teacher = backbone with full context; student = decoder with memory.

        For completion-style pretraining, the student predicts a continuation
        starting at position `split` in the original context, while the teacher
        sees the full context.  teacher_offset tells us where in the teacher
        sequence to start alignment:

          student logit at (prompt_len + k)  predicts  a[split + k + 1]
          teacher logit at (split + k)       predicts  a[split + k + 1]

        For backward compatibility (full reconstruction), teacher_offset=0
        recovers the original alignment.
        """
        T = self.distill_temperature
        t_off = 0
        if teacher_offset is not None:
            t_off = teacher_offset[0].item()  # batch_size=1 assumed

        with torch.no_grad():
            # Teacher must share the student's vocab + arch to be a valid
            # uncompressed reference — that's the decoder, not the encoder.
            # Same-model: self.decoder aliases self.backbone (no-op).
            # Cross-model: teacher is the 3B base (LoRA disabled by no_grad).
            teacher_out = self.decoder(
                input_ids=teacher_input_ids, output_hidden_states=False
            )
        teacher_logits = teacher_out.logits

        # prompt_len = number of leading -100 positions in labels
        # (memory placeholders + ae/ft token).
        non_masked = (labels[0] != -100).nonzero(as_tuple=True)[0]
        if len(non_masked) == 0:
            if _verbose:
                print("  [distill] ALL labels are -100 → returning 0")
            return torch.tensor(0.0, device=student_logits.device)
        prompt_len = non_masked[0].item()

        # Number of comparable (shifted) positions after alignment
        student_answer_len = student_logits.size(1) - prompt_len - 1
        teacher_answer_len = teacher_logits.size(1) - t_off - 1
        align_len = min(student_answer_len, teacher_answer_len)

        if _verbose:
            print(f"  distill: prompt_len={prompt_len}, teacher_offset={t_off}, align_len={align_len}")

        if align_len <= 0:
            if _verbose:
                print(f"  [distill] align_len={align_len} <= 0 → returning 0")
            return torch.tensor(0.0, device=student_logits.device)

        # Aligned slices: both predict the same next tokens
        s_logits = student_logits[:, prompt_len : prompt_len + align_len, :]
        t_logits = teacher_logits[:, t_off : t_off + align_len, :]

        # Mask (should be all-True for standard AE/completion, but check for safety)
        mask = labels[:, prompt_len + 1 : prompt_len + 1 + align_len] != -100

        if mask.sum() == 0:
            if _verbose:
                print("  [distill] mask is all-False → returning 0")
            return torch.tensor(0.0, device=student_logits.device)

        # Alignment sanity check (step 1 only)
        if self._fwd_step == 1:
            s_targets = labels[0, prompt_len + 1 : prompt_len + 1 + min(5, align_len)]
            t_targets = teacher_input_ids[0, t_off + 1 : t_off + 1 + min(5, align_len)]
            match = "OK" if torch.equal(s_targets, t_targets) else "MISMATCH"
            print(f"  distill alignment: {match} (student={s_targets.tolist()}, teacher={t_targets.tolist()})")

        s_log_probs = F.log_softmax(s_logits / T, dim=-1)
        t_probs = F.softmax(t_logits / T, dim=-1)

        # Check for NaN in softmax outputs
        if torch.isnan(s_log_probs).any():
            raise ValueError(f"[step={self._fwd_step}] NaN in student log_softmax! "
                             f"s_logits range: [{s_logits.min():.4f}, {s_logits.max():.4f}]")
        if torch.isnan(t_probs).any():
            raise ValueError(f"[step={self._fwd_step}] NaN in teacher softmax! "
                             f"t_logits range: [{t_logits.min():.4f}, {t_logits.max():.4f}]")

        kl = F.kl_div(s_log_probs, t_probs, reduction="none").sum(-1)  # (B, L)
        kl_per_pos = kl * mask
        kl_mean = kl_per_pos.sum() / mask.sum()
        final_loss = self.distill_loss_weight * (T ** 2) * kl_mean

        if _verbose:
            print(f"  distill: kl={kl_mean.item():.6f}, loss={final_loss.item():.6f}")

        return final_loss

    # ------------------------------------------------------------------
    def hidden_state_reconstruction_loss(
        self, memory_tokens, hidden_states, num_segments, segment_length, total_length,
        _verbose=False,
    ):
        """
        Reconstruct backbone hidden states from compressed memory tokens.

        Hybrid loss (scale-invariant):
            L_cos   = 1 - cosine_similarity(predicted, target)   — direction
            L_mse   = MSE after LayerNorm                        — coordinates (kept as diagnostic only, weight=0)
            L_chunk = 1 - cosine(pooled_pred_chunk, pooled_target_chunk) — chunk summary

        Total: alpha * L_cos + beta * L_mse + gamma * L_chunk
        """
        _alpha, _beta, _gamma = 1.0, 0.0, 0.5

        seg_losses = []
        _diag_cos, _diag_mse, _diag_chunk = 0.0, 0.0, 0.0

        for seg_idx in range(num_segments):
            start = seg_idx * segment_length
            end = min((seg_idx + 1) * segment_length, total_length)
            seg_len = end - start

            seg_mem = memory_tokens[
                :, seg_idx * self.memory_slots : (seg_idx + 1) * self.memory_slots, :
            ]
            target = hidden_states[:, start:end, :].detach()

            reconstructed = self.decompressor(seg_mem, target_len=seg_len)
            reconstructed = reconstructed[:, :seg_len, :]  # trim if needed

            rec_f = reconstructed.float()
            # Cross-model: rec is dec_dim, target is enc_dim → project rec → enc_dim.
            if self._decomp_loss_proj is not None:
                rec_f = self._decomp_loss_proj(rec_f)
            tgt_f = target.float()

            # 1) Cosine loss: per-token direction alignment
            cos_sim = F.cosine_similarity(
                rec_f.reshape(-1, rec_f.size(-1)),
                tgt_f.reshape(-1, tgt_f.size(-1)),
                dim=-1,
            )
            loss_cos = (1.0 - cos_sim).mean()

            # 2) Normalized MSE: coordinate alignment after unit normalization
            rec_norm = F.normalize(rec_f, dim=-1)
            tgt_norm = F.normalize(tgt_f, dim=-1)
            loss_mse = F.mse_loss(rec_norm, tgt_norm)

            # 3) Chunk-summary loss: cosine between pooled chunks
            M = self.memory_slots
            chunk_size = seg_len // M if M > 0 else seg_len
            if chunk_size > 0 and seg_len >= M:
                rec_chunks = rec_f[:, :M * chunk_size].view(-1, M, chunk_size, rec_f.size(-1)).mean(dim=2)
                tgt_chunks = tgt_f[:, :M * chunk_size].view(-1, M, chunk_size, tgt_f.size(-1)).mean(dim=2)
                chunk_cos = F.cosine_similarity(rec_chunks, tgt_chunks, dim=-1)
                loss_chunk = (1.0 - chunk_cos).mean()
            else:
                loss_chunk = loss_cos.new_zeros(1).squeeze()

            seg_loss = _alpha * loss_cos + _beta * loss_mse + _gamma * loss_chunk

            if torch.isnan(seg_loss) or torch.isinf(seg_loss):
                raise ValueError(
                    f"[step={self._fwd_step}] NaN/Inf in decomp seg {seg_idx}: "
                    f"cos={loss_cos.item():.4f}, mse={loss_mse.item():.4f}, "
                    f"chunk={loss_chunk.item():.4f}")

            seg_losses.append(seg_loss)
            _diag_cos += (1 - loss_cos.item())
            _diag_mse += loss_mse.item()
            _diag_chunk += (1 - loss_chunk.item())

        avg_loss = sum(seg_losses) / len(seg_losses)
        final_loss = self.rec_loss_weight * avg_loss

        if _verbose:
            n = len(seg_losses)
            print(f"  decomp: cos={_diag_cos/n:.4f}, norm_mse={_diag_mse/n:.6f}, "
                  f"chunk_cos={_diag_chunk/n:.4f}, loss={final_loss.item():.6f}")

            # Detailed diagnostics: is the decompressor output actually collapsed?
            with torch.no_grad():
                # Pick first and last segment for comparison
                for _si in [0, num_segments - 1]:
                    _start = _si * segment_length
                    _end = min((_si + 1) * segment_length, total_length)
                    _seg_mem = memory_tokens[:, _si * self.memory_slots : (_si + 1) * self.memory_slots, :]
                    _rec = self.decompressor(_seg_mem.float(), target_len=_end - _start)
                    _rec = _rec[:, :_end - _start, :]
                    if self._decomp_loss_proj is not None:
                        _rec = self._decomp_loss_proj(_rec)
                    _tgt = hidden_states[:, _start:_end, :].float()

                    # 1) Decompressor output diversity: pairwise cosine among output tokens
                    _rec_n = F.normalize(_rec[0], dim=-1)
                    _rec_cos = _rec_n @ _rec_n.T
                    _T = _rec_cos.size(0)
                    _mask = ~torch.eye(_T, dtype=torch.bool, device=_rec_cos.device)
                    _rec_pairwise = _rec_cos[_mask].mean().item()

                    # 2) Target diversity for comparison
                    _tgt_n = F.normalize(_tgt[0], dim=-1)
                    _tgt_cos = _tgt_n @ _tgt_n.T
                    _tgt_pairwise = _tgt_cos[_mask].mean().item()

                    # 3) Per-token cosine: distribution (min, mean, max, std)
                    _per_tok_cos = F.cosine_similarity(_rec[0], _tgt[0], dim=-1)

                    # 4) Memory slot diversity (input to decompressor)
                    _mem_n = F.normalize(_seg_mem[0], dim=-1)
                    _mem_cos = _mem_n @ _mem_n.T
                    _M = _mem_cos.size(0)
                    _mmask = ~torch.eye(_M, dtype=torch.bool, device=_mem_cos.device)
                    _mem_pairwise = _mem_cos[_mmask].mean().item()

                    # 5) Norms
                    _rec_norm = _rec[0].norm(dim=-1)
                    _tgt_norm = _tgt[0].norm(dim=-1)

                    print(f"  decomp_diag seg={_si}: "
                          f"rec_pairwise_cos={_rec_pairwise:.4f} (target={_tgt_pairwise:.4f}), "
                          f"mem_slot_cos={_mem_pairwise:.4f}, "
                          f"per_tok_cos=[{_per_tok_cos.min().item():.4f}, "
                          f"{_per_tok_cos.mean().item():.4f}, "
                          f"{_per_tok_cos.max().item():.4f}, "
                          f"std={_per_tok_cos.std().item():.4f}], "
                          f"rec_norm={_rec_norm.mean().item():.2f}, "
                          f"tgt_norm={_tgt_norm.mean().item():.2f}")

        return final_loss

    # ------------------------------------------------------------------
    # Decompress-inject helpers (SeDeM)
    # ------------------------------------------------------------------
    def _get_decoder_internals(self):
        """Return (layers, final_norm, lm_head, rotary_emb) from the decoder, handling LoRA wrapping."""
        if hasattr(self.decoder, 'model') and hasattr(self.decoder.model, 'model'):
            # LoRA-wrapped: PeftModel.model.model
            inner = self.decoder.model.model
            lm_head = self.decoder.model.lm_head
        else:
            inner = self.decoder.model
            lm_head = self.decoder.lm_head
        rotary_emb = getattr(inner, 'rotary_emb', None)
        return inner.layers, inner.norm, lm_head, rotary_emb

    def _get_backbone_internals(self):
        """Return (layers, final_norm, lm_head, rotary_emb) from the backbone."""
        inner = self.backbone.model
        rotary_emb = getattr(inner, 'rotary_emb', None)
        return inner.layers, inner.norm, self.backbone.lm_head, rotary_emb

    def _decoder_embed_tokens(self, ids):
        """Embed token IDs in decoder-dim space, robust to LoRA wrapping.

        Use this anywhere the resulting embeddings feed self.decoder (i.e. text
        prompt/answer embeddings, growing_embs in generation, etc.). In same-model
        mode self.decoder may alias self.backbone, so this is a no-op rename.
        In cross-model mode the decoder lives at dec_dim (e.g. 3072), distinct
        from the encoder's enc_dim (e.g. 2048).
        """
        dec = self.decoder
        # PeftModel wraps the base model: dec.model is the LlamaForCausalLM,
        # dec.model.model is the inner LlamaModel that owns embed_tokens.
        if hasattr(dec, 'model') and hasattr(dec.model, 'model'):
            return dec.model.model.embed_tokens(ids)
        return dec.model.embed_tokens(ids)

    def forward_with_injection(self, text_embs, decompressed_hs, inject_layer,
                           _verbose=False, return_decomp_logits=False,
                           measure_mem_attention=False, inference_only=False):
        """Two-phase decoder forward for decompress-inject mode.

        Phase 1: Run text through decoder layers 0..inject_layer-1 (text only).
        Phase 2: Concatenate [decompressed | text_at_ℓ] and run through
                 layers inject_layer..L-1, then final norm + LM head.

        Returns logits for the TEXT portion only (decompressed prefix stripped).
        If return_decomp_logits=True, also returns logits for decompressed positions
        (for L_tok token reconstruction loss).
        If measure_mem_attention=True, also returns the mean attention mass from
        text-query positions to decompressed-key positions, averaged across the
        last-layer attention heads. This is the FIXED attn_past_mass diagnostic
        (previously was reading compressor internal stats — always 0 for the pooled-linear compressor).

        Position encoding: text gets positions [T_recon, T_recon+1, ...] at ALL
        layers so RoPE is consistent.  Decompressed gets positions [0..T_recon-1]
        at layers ≥ inject_layer.
        """
        # Always use decoder internals: in same-model inference self.decoder
        # aliases self.backbone (so internals are the same weights); in
        # cross-model the decoder is the 3B with dec_dim. The backbone path
        # would inject decoder-dim Ĥ into the encoder's smaller residual stream.
        layers, final_norm, lm_head, rotary_emb = self._get_decoder_internals()

        B = text_embs.size(0)
        S_text = text_embs.size(1)
        T_recon = decompressed_hs.size(1)
        device = text_embs.device
        dtype = text_embs.dtype

        # Position IDs — text offset by T_recon at all layers for consistent RoPE
        text_pos = torch.arange(T_recon, T_recon + S_text, device=device).unsqueeze(0).expand(B, -1)
        min_val = torch.finfo(dtype).min

        # ---- Phase 1: layers 0..inject_layer-1 on text only ----
        hidden = text_embs
        if inject_layer > 0:
            # Causal mask for text-only phase
            text_mask = torch.full((S_text, S_text), min_val, device=device, dtype=dtype)
            text_mask = torch.triu(text_mask, diagonal=1)
            text_mask = text_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S_text, S_text)
            # Compute RoPE position embeddings for text
            text_pos_emb = rotary_emb(hidden, text_pos) if rotary_emb is not None else None
            for layer_idx in range(inject_layer):
                layer_output = layers[layer_idx](
                    hidden,
                    attention_mask=text_mask,
                    position_ids=text_pos,
                    position_embeddings=text_pos_emb,
                )
                hidden = layer_output[0] if isinstance(layer_output, tuple) else layer_output if isinstance(layer_output, tuple) else layer_output
        text_at_inject = hidden  # (B, S_text, D)

        # ---- Phase 2: [decompressed | text] through layers inject_layer..L-1 ----
        decompressed_bf16 = decompressed_hs.to(dtype)
        combined = torch.cat([decompressed_bf16, text_at_inject], dim=1)  # (B, T_recon+S_text, D)

        # Combined position IDs
        decomp_pos = torch.arange(T_recon, device=device).unsqueeze(0).expand(B, -1)
        combined_pos = torch.cat([decomp_pos, text_pos], dim=1)  # (B, T_recon+S_text)

        # Compute RoPE position embeddings for combined sequence
        combined_pos_emb = rotary_emb(combined, combined_pos) if rotary_emb is not None else None

        # 4D causal mask for the combined sequence
        total_len = T_recon + S_text
        causal_mask = torch.full((total_len, total_len), min_val, device=device, dtype=dtype)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, total_len, total_len)

        # Optionally measure text→memory attention mass from the last layer.
        # We do this by running the last layer separately with output_attentions=True.
        # To keep cost low, only measure when asked (e.g. verbose steps).
        mem_attn_mass = None

        hidden = combined
        num_layers = len(layers)
        for layer_idx in range(inject_layer, num_layers):
            # For the very last layer, optionally request attention weights to measure
            # text→memory mass. This is expensive and only done when verbose.
            _is_last = (layer_idx == num_layers - 1)
            if measure_mem_attention and _is_last:
                layer_output = layers[layer_idx](
                    hidden,
                    attention_mask=causal_mask,
                    position_ids=combined_pos,
                    position_embeddings=combined_pos_emb,
                    output_attentions=True,
                )
                if isinstance(layer_output, tuple) and len(layer_output) >= 2 and layer_output[1] is not None:
                    # attention_weights: (B, num_heads, S, S)
                    attn_w = layer_output[1].float()
                    # text queries: positions [T_recon..total_len)
                    # memory keys: positions [0..T_recon)
                    with torch.no_grad():
                        text_to_mem = attn_w[:, :, T_recon:, :T_recon]  # (B, H, S_text, T_recon)
                        # Sum over memory keys, mean over text queries + heads + batch
                        mem_attn_mass = text_to_mem.sum(dim=-1).mean().item()
                hidden = layer_output[0] if isinstance(layer_output, tuple) else layer_output
            else:
                layer_output = layers[layer_idx](
                    hidden,
                    attention_mask=causal_mask,
                    position_ids=combined_pos,
                    position_embeddings=combined_pos_emb,
                )
                hidden = layer_output[0] if isinstance(layer_output, tuple) else layer_output

        # Final norm + LM head
        # P2 fast path: when inference_only and no auxiliary logits/attention are needed,
        # compute the LM head only on the last text position (the only one used for greedy
        # next-token prediction). RMSNorm + Linear are per-token, so this is bit-exact
        # equivalent to slicing the full output.
        if inference_only and not return_decomp_logits and not measure_mem_attention:
            last_hidden = final_norm(hidden[:, -1:, :])
            text_logits = lm_head(last_hidden)  # (B, 1, V)
            decomp_logits = None
        else:
            hidden = final_norm(hidden)
            logits = lm_head(hidden)
            # Split logits: decompressed prefix | text suffix
            decomp_logits = logits[:, :T_recon, :]
            text_logits = logits[:, T_recon:, :]

        if _verbose:
            with torch.no_grad():
                recon_norm = decompressed_bf16.float().norm(dim=-1).mean().item()
                text_norm = text_at_inject.float().norm(dim=-1).mean().item()
                msg = (f"  two_phase_fwd: inject_layer={inject_layer}, "
                       f"T_recon={T_recon}, S_text={S_text}, "
                       f"recon_norm={recon_norm:.2f}, text_norm_at_ℓ={text_norm:.2f}")
                if mem_attn_mass is not None:
                    msg += f", last-layer text→mem attention mass={mem_attn_mass:.4f}"
                print(msg)

        # Build return tuple based on flags
        if return_decomp_logits and measure_mem_attention:
            return text_logits, decomp_logits, (mem_attn_mass if mem_attn_mass is not None else 0.0)
        if return_decomp_logits:
            return text_logits, decomp_logits
        if measure_mem_attention:
            return text_logits, (mem_attn_mass if mem_attn_mass is not None else 0.0)
        return text_logits

    def forward_with_injection_cached(self, text_embs_new, decompressed_hs, inject_layer,
                                   past_kv_phase1=None, past_kv_phase2=None,
                                   text_pos_past=0, static_max_text_len=None):
        """Cached two-phase decoder forward. Output-equivalent to
        `forward_with_injection` modulo numerical noise from skipping recomputation.

        On the first call (past_kv_*=None) this runs a full prefill over the
        complete text sequence and the [decomp | text] combined sequence,
        populating both caches. Returns the last-position logits + both caches.

        On subsequent calls, pass only the new text token(s) in `text_embs_new`
        (typically S_text_new=1 for greedy decode) along with `text_pos_past`
        equal to the number of text positions already cached. The function
        processes only the new positions, attending to the cached K,V for the
        prior text (Phase 1) and the cached decomp+text K,V (Phase 2).

        Position-ID convention (RoPE-consistent with `forward_with_injection`):
            decomp positions: [0..T_recon-1]
            text positions:   [T_recon..T_recon + S_text_total - 1]

        Cache slots:
            Phase 1: text K,V only; slot i == text position i (0-indexed)
            Phase 2: [decomp | text]; slots 0..T_recon-1 = decomp, T_recon..= text

        Shape-stable KV (opt-in via model._shape_stable_kv=True + static_max_text_len):
            On prefill, allocates StaticCache(max_cache_len) instead of DynamicCache.
            All decode steps see the same cache-tensor shape, so dynamo guards on
            past_key_values.layers stop firing per step (eliminates compile thrash
            in all-opt regime). Bit-exact vs DynamicCache to bf16 reduction-order.

        Returns:
            last_logits: (B, 1, vocab)  — logits at the last text position only
            past_kv_phase1, past_kv_phase2: Cache objects, updated in place
        """
        from transformers.cache_utils import DynamicCache, StaticCache

        layers, final_norm, lm_head, rotary_emb = self._get_decoder_internals()
        B = text_embs_new.size(0)
        S_text_new = text_embs_new.size(1)
        T_recon = decompressed_hs.size(1)
        device = text_embs_new.device
        dtype = text_embs_new.dtype
        min_val = torch.finfo(dtype).min

        # Resolve shape-stable KV mode once. Requires both the flag and a known
        # max text length so we can size the static cache up-front.
        use_static_kv = bool(getattr(self, '_shape_stable_kv', False)) and (static_max_text_len is not None)

        is_prefill = past_kv_phase1 is None

        # ---- Phase 1: text-only layers 0..inject_layer-1 ----
        if past_kv_phase1 is None:
            if use_static_kv:
                _decoder_for_cfg = getattr(self, 'decoder', self.backbone)
                past_kv_phase1 = StaticCache(
                    config=_decoder_for_cfg.config,
                    max_batch_size=B,
                    max_cache_len=int(static_max_text_len),
                    device=device,
                    dtype=dtype,
                )
            else:
                past_kv_phase1 = DynamicCache()

        # Position IDs for the NEW text positions (RoPE: text is offset by T_recon)
        text_pos_start = T_recon + text_pos_past
        text_pos_end = text_pos_start + S_text_new
        text_pos_ids = torch.arange(text_pos_start, text_pos_end, device=device).unsqueeze(0).expand(B, -1)
        cache_pos_p1 = torch.arange(text_pos_past, text_pos_past + S_text_new, device=device)

        # Phase 1 mask: only build for multi-query prefill (S_text_new > 1) when
        # using DynamicCache. StaticCache requires an explicit mask covering the
        # full max_cache_len (positions beyond the active region must be masked).
        # Mask is built vectorized: row q_off allows keys [0, text_pos_past+q_off].
        if use_static_kv:
            kv_len_p1 = int(static_max_text_len)
            q_ids = torch.arange(S_text_new, device=device).view(-1, 1)
            k_ids = torch.arange(kv_len_p1, device=device).view(1, -1)
            # allowed: k <= text_pos_past + q_off
            allowed_p1 = k_ids <= (text_pos_past + q_ids)
            p1_mask = torch.where(
                allowed_p1,
                torch.zeros((), dtype=dtype, device=device),
                torch.full((), min_val, dtype=dtype, device=device),
            ).unsqueeze(0).unsqueeze(0)  # (1, 1, S_text_new, kv_len_p1)
        elif S_text_new > 1:
            kv_len_p1 = text_pos_past + S_text_new
            q_ids = torch.arange(S_text_new, device=device).view(-1, 1)
            k_ids = torch.arange(kv_len_p1, device=device).view(1, -1)
            allowed_p1 = k_ids <= (text_pos_past + q_ids)
            p1_mask = torch.where(
                allowed_p1,
                torch.zeros((), dtype=dtype, device=device),
                torch.full((), min_val, dtype=dtype, device=device),
            ).unsqueeze(0).unsqueeze(0)  # (1, 1, S_text_new, kv_len_p1)
        else:
            p1_mask = None

        text_pos_emb = rotary_emb(text_embs_new, text_pos_ids) if rotary_emb is not None else None

        hidden = text_embs_new
        for li in range(inject_layer):
            out = layers[li](
                hidden,
                attention_mask=p1_mask,
                position_ids=text_pos_ids,
                past_key_values=past_kv_phase1,
                use_cache=True,
                cache_position=cache_pos_p1,
                position_embeddings=text_pos_emb,
            )
            hidden = out[0] if isinstance(out, tuple) else out

        text_at_inject_new = hidden  # (B, S_text_new, D)

        # ---- Phase 2: layers inject_layer..L-1 ----
        # Phase 2 cache allocation. Static path needs full size from the start.
        if past_kv_phase2 is None and use_static_kv:
            _decoder_for_cfg = getattr(self, 'decoder', self.backbone)
            past_kv_phase2 = StaticCache(
                config=_decoder_for_cfg.config,
                max_batch_size=B,
                max_cache_len=int(T_recon + static_max_text_len),
                device=device,
                dtype=dtype,
            )
        if is_prefill:
            # Prefill: process [decomp | text_at_inject_new] together
            decompressed_bf16 = decompressed_hs.to(dtype)
            combined = torch.cat([decompressed_bf16, text_at_inject_new], dim=1)
            decomp_pos_ids = torch.arange(T_recon, device=device)
            combined_pos_ids = torch.cat([decomp_pos_ids, text_pos_ids[0]]).unsqueeze(0).expand(B, -1)
            cache_pos_p2 = torch.arange(T_recon + S_text_new, device=device)
            combined_pos_emb = rotary_emb(combined, combined_pos_ids) if rotary_emb is not None else None
            if past_kv_phase2 is None:
                past_kv_phase2 = DynamicCache()
            # Phase 2 prefill: q_len = T_recon + S_text_new, k_len = q_len for
            # DynamicCache, or full static cache len for StaticCache. Build mask
            # vectorized via broadcasting (replaces the prior triu+full loop).
            q_len_p2 = T_recon + S_text_new
            if use_static_kv:
                k_len_p2 = int(T_recon + static_max_text_len)
            else:
                k_len_p2 = T_recon + S_text_new
            q_ids = torch.arange(q_len_p2, device=device).view(-1, 1)
            k_ids = torch.arange(k_len_p2, device=device).view(1, -1)
            allowed_p2 = k_ids <= q_ids
            p2_mask = torch.where(
                allowed_p2,
                torch.zeros((), dtype=dtype, device=device),
                torch.full((), min_val, dtype=dtype, device=device),
            ).unsqueeze(0).unsqueeze(0)
        else:
            # Decode step: only the new text token; decomp K,V are already cached
            combined = text_at_inject_new
            combined_pos_ids = text_pos_ids                                   # same position IDs as Phase 1
            cache_pos_p2 = torch.arange(T_recon + text_pos_past,
                                       T_recon + text_pos_past + S_text_new,
                                       device=device)
            combined_pos_emb = rotary_emb(combined, combined_pos_ids) if rotary_emb is not None else None
            # Phase 2 decode: build mask vectorized when needed. For StaticCache
            # we always need to mask out unwritten future slots; for DynamicCache,
            # single-query (S_text_new=1) can use mask=None (SDPA fast path).
            if use_static_kv:
                q_len_p2 = S_text_new
                k_len_p2 = int(T_recon + static_max_text_len)
                # Active key indices: [0, T_recon + text_pos_past + q_off]
                q_ids = torch.arange(S_text_new, device=device).view(-1, 1)
                k_ids = torch.arange(k_len_p2, device=device).view(1, -1)
                allowed_p2 = k_ids <= (T_recon + text_pos_past + q_ids)
                p2_mask = torch.where(
                    allowed_p2,
                    torch.zeros((), dtype=dtype, device=device),
                    torch.full((), min_val, dtype=dtype, device=device),
                ).unsqueeze(0).unsqueeze(0)
            elif S_text_new > 1:
                q_len_p2 = S_text_new
                k_len_p2 = T_recon + text_pos_past + S_text_new
                q_ids = torch.arange(S_text_new, device=device).view(-1, 1)
                k_ids = torch.arange(k_len_p2, device=device).view(1, -1)
                allowed_p2 = k_ids <= (T_recon + text_pos_past + q_ids)
                p2_mask = torch.where(
                    allowed_p2,
                    torch.zeros((), dtype=dtype, device=device),
                    torch.full((), min_val, dtype=dtype, device=device),
                ).unsqueeze(0).unsqueeze(0)
            else:
                p2_mask = None

        hidden = combined
        for li in range(inject_layer, len(layers)):
            out = layers[li](
                hidden,
                attention_mask=p2_mask,
                position_ids=combined_pos_ids,
                past_key_values=past_kv_phase2,
                use_cache=True,
                cache_position=cache_pos_p2,
                position_embeddings=combined_pos_emb,
            )
            hidden = out[0] if isinstance(out, tuple) else out

        # LM head on last text position only (skip decomp prefix, skip earlier text)
        last_hidden = final_norm(hidden[:, -1:, :])
        last_logits = lm_head(last_hidden)

        return last_logits, past_kv_phase1, past_kv_phase2

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def compressor_is_stateless(self):
        """Return True if compressor has no cross-segment dependency
        (prev_mem is ignored). Required for the parallel-segment compress_segments
        path to produce output identical to sequential.

        SegmentCompressor: always stateless (SeDeM / Run-9 design).
        LearnableQueryCompressor: stateless iff ignore_prev_mem=True (Run-6 style).
        Anything else: assumed stateful for safety; falls back to sequential.

        If self.compressor is wrapped in torch._dynamo's OptimizedModule
        (i.e. --compile_compressor is on), unwrap via _orig_mod before the
        class-name check so the parallel fast path is still taken.
        """
        comp = self.compressor
        comp = getattr(comp, '_orig_mod', comp)  # unwrap torch.compile wrapper
        cls_name = type(comp).__name__
        if cls_name == 'SegmentCompressor':
            return True
        if cls_name == 'LearnableQueryCompressor':
            return bool(getattr(comp, 'ignore_prev_mem', False))
        return False

    @torch.no_grad()
    def compress_segments(self, input_ids):
        """Compress context token IDs into memory-token embeddings."""
        total_length = input_ids.size(1)
        num_segments = self.compute_num_segments(total_length)
        segment_length = math.ceil(total_length / num_segments)

        # Optional parallel-segment fast path (SeDeM / no-recurrence only).
        # Set model._parallel_segments=True (e.g. via eval_qa.py --parallel) to
        # enable. Default off → existing sequential behavior preserved exactly.
        if getattr(self, '_parallel_segments', False) and num_segments > 1:
            if self.compressor_is_stateless():
                return self.compress_segments_parallel(input_ids, num_segments, segment_length)
            else:
                if not getattr(self, '_warned_parallel_fallback', False):
                    print(f"[parallel] WARNING: compressor {type(self.compressor).__name__} "
                          f"is not stateless; falling back to sequential compress_segments.")
                    self._warned_parallel_fallback = True

        mem_per_segment = []
        prev_mem = None
        for seg_idx in range(num_segments):
            start = seg_idx * segment_length
            end = min((seg_idx + 1) * segment_length, total_length)
            seg_ids = input_ids[:, start:end]
            seg_hidden = self.extract_hidden_states(seg_ids).float()
            seg_mem = self.compressor(seg_hidden, prev_mem=prev_mem,
                                     verbose=False,
                                     seg_idx=seg_idx)
            mem_per_segment.append(seg_mem)
            prev_mem = seg_mem

        return torch.cat(mem_per_segment, dim=1)  # (B, M_total, D)

    @torch.no_grad()
    def compress_segments_parallel(self, input_ids, num_segments, segment_length):
        """Parallel-segment variant of compress_segments (Option A — batch along B).

        Reshapes [1, T_total] → [num_segments, segment_length] with right-
        padding on the last (possibly short) segment, runs ONE backbone
        forward over the batched tensor with an attention_mask, then calls
        the compressor per-segment on the trimmed (real-length) hidden
        states. Output is identical to sequential compress_segments up to float
        reduction-order noise (~1e-6).

        Preconditions:
          - batch_size == 1 (matches sequential compress_segments at inference)
          - compressor is stateless (compressor_is_stateless() == True)
          - num_segments > 1 (single-segment case has no parallelism to win)

        Trade-off: peak activation memory rises by ~num_segments× over
        sequential (all segments held in one forward), but TTFT drops in
        proportion to num_segments since the backbone forward is amortized.

        Optional `self._batched_compressor_fast` (default False) collapses the
        per-segment compressor for-loop into ONE batched compressor call when
        every segment is full-length. Bit-exact: SegmentCompressor ops
        are all batch-independent (input_proj is per-position Linear;
        chunks.mean(dim=2) is per-(B,M) slice; LayerNorm normalizes feature
        dim per-sample; cross_slot_mixer Linear(M,M) is per-batch). For
        mixed lengths (last segment shorter), full segments are batched and
        the last is run separately — also bit-exact since the sequential
        code pads internally with zeros inside the compressor.
        """
        assert input_ids.size(0) == 1, "compress_segments_parallel assumes batch_size=1"
        total_length = input_ids.size(1)
        device = input_ids.device

        # Win A: real_lens precomputed on CPU (no per-seg .item() sync).
        real_lens = [
            min((i + 1) * segment_length, total_length) - i * segment_length
            for i in range(num_segments)
        ]

        pad_id = int(self.pad_token_id) if hasattr(self, 'pad_token_id') else 0
        batched_ids = torch.full(
            (num_segments, segment_length), pad_id,
            dtype=input_ids.dtype, device=device,
        )
        batched_mask = torch.zeros(
            (num_segments, segment_length), dtype=torch.long, device=device,
        )
        for seg_idx in range(num_segments):
            start = seg_idx * segment_length
            end = min((seg_idx + 1) * segment_length, total_length)
            batched_ids[seg_idx, :real_lens[seg_idx]] = input_ids[0, start:end]
            batched_mask[seg_idx, :real_lens[seg_idx]] = 1

        hidden_states = self.extract_hidden_states(
            batched_ids, attention_mask=batched_mask
        )  # (num_segments, segment_length, D)

        def _seq_path():
            mem_per_segment = []
            for seg_idx in range(num_segments):
                seg_hidden = hidden_states[seg_idx:seg_idx+1, :real_lens[seg_idx], :].float()
                seg_mem = self.compressor(
                    seg_hidden, prev_mem=None,
                    verbose=False,
                    seg_idx=seg_idx,
                )
                mem_per_segment.append(seg_mem)
            return torch.cat(mem_per_segment, dim=1)  # (1, M_total, D)

        def _batched_path():
            all_full = all(rl == segment_length for rl in real_lens)
            if all_full:
                # All segments full-length → ONE batched compressor call.
                all_mem = self.compressor(
                    hidden_states.float(), prev_mem=None,
                    verbose=False, seg_idx=None,
                )  # (num_segments, M, D)
            elif num_segments > 1:
                # Mixed: batch all full segments, run last (partial) separately.
                full_hs = hidden_states[:-1].float()
                mem_full = self.compressor(
                    full_hs, prev_mem=None, verbose=False, seg_idx=None,
                )
                last_hs = hidden_states[-1:, :real_lens[-1], :].float()
                mem_last = self.compressor(
                    last_hs, prev_mem=None,
                    verbose=False, seg_idx=num_segments - 1,
                )
                all_mem = torch.cat([mem_full, mem_last], dim=0)
            else:
                # num_segments == 1 — no batching to do.
                seg_hidden = hidden_states[0:1, :real_lens[0], :].float()
                all_mem = self.compressor(
                    seg_hidden, prev_mem=None, verbose=False, seg_idx=0,
                )
            return all_mem.reshape(1, num_segments * all_mem.size(1), all_mem.size(2))

        if getattr(self, '_batched_compressor_fast', False):
            # Optional verifier: on first N examples, run both paths and assert equality.
            n_verify = int(getattr(self, '_verify_batched_remaining', 0))
            if n_verify > 0:
                mem_seq = _seq_path()
                mem_bat = _batched_path()
                max_abs_diff = (mem_seq - mem_bat).abs().max().item()
                _tol = float(getattr(self, '_batched_bitexact_tol', 1e-3))
                print(f"  [verify_batched_bitexact] num_seg={num_segments} "
                      f"max_abs_diff={max_abs_diff:.3e}  tol={_tol:.0e}")
                assert max_abs_diff <= _tol, (
                    f"compress_segments_parallel batched-vs-seq drift: max_abs_diff={max_abs_diff:.3e} > tol={_tol:.0e}; "
                    f"batched compressor diverged beyond cuDNN/cuBLAS dispatch fp noise. "
                    f"Set self._batched_bitexact_tol = <value> to override."
                )
                self._verify_batched_remaining = n_verify - 1
                return mem_bat
            return _batched_path()

        return _seq_path()

    @torch.no_grad()
    def generate(self, input_ids, prompt_ids=None, max_new_tokens=512):
        """
        End-to-end: compress context, optionally prepend a prompt, and
        autoregressively generate.

        input_ids  : (1, T_ctx)  -- context
        prompt_ids : (1, T_p)    -- question / instruction  (uncompressed)
        """
        memory = self.compress_segments(input_ids)  # (1, M_total, D)

        # ---- Decompress-inject mode: pass RAW memory (decompressor handles normalization) ----
        if self._inject_decompressed:
            return self.generate_with_injection(memory, prompt_ids, max_new_tokens)

        # Normalize memory tokens to match text embedding scale
        # (same normalization as embed_decoder_input_with_prefix)
        if hasattr(self, 'mem_norm') and hasattr(self, 'mem_scale') and self._normalize_mem:
            max_scale = 1.5 / math.sqrt(self.dim)
            effective_scale = torch.clamp(self.mem_scale, min=0.01, max=max_scale)
            memory = self.mem_norm(memory) * effective_scale

        memory = memory.to(self._backbone_dtype)  # fp32 → bf16 for decoder

        # ---- Standard mode: prepend memory and generate ----
        # Build initial decoder input
        parts = [memory]
        if prompt_ids is not None:
            prompt_embs = self._decoder_embed_tokens(prompt_ids)
            parts.append(prompt_embs)
        decoder_input = torch.cat(parts, dim=1)

        generated = []
        past_key_values = None
        cur_input = decoder_input

        for _ in range(max_new_tokens):
            # Use self.decoder for generation (same-model: aliases self.backbone;
            # cross-model: routes through the 3B answer-generation model).
            out = self.decoder(
                inputs_embeds=cur_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logit = out.logits[:, -1, : self.original_vocab_size]
            past_key_values = out.past_key_values
            next_id = logit.argmax(dim=-1)
            if next_id.item() == self.eos_id:
                break
            generated.append(next_id.item())
            cur_input = self._decoder_embed_tokens(next_id).unsqueeze(1)

        return self.tokenizer.decode(generated, skip_special_tokens=True)

    @torch.no_grad()
    def generate_with_injection(self, memory_raw, prompt_ids, max_new_tokens,
                                    override_decompressed_hs=None,
                                    override_skip_mem_norm=False,
                                    selected_block_indices=None,
                                    return_timing=False):
        """Generate with decompress-inject: two-phase prefill then autoregressive decode.

        Args:
            memory_raw: (1, M_total, D) — raw compressor output (NOT normalized).
                        Decompressor runs on raw memory, then output is normalized.
                        This matches the training forward() path.
            override_decompressed_hs: (1, T, D) or None. If provided, bypass the
                        decompressor and use this tensor directly as the injected
                        hidden states. Used for evidence-label-H eval (real layer-ℓ H from
                        the frozen backbone in place of decompressor output).
                        Still goes through mem_norm * mem_scale so the decoder sees
                        its trained input distribution (unless override_skip_mem_norm).
            override_skip_mem_norm: if True and override_decompressed_hs is provided,
                        skip the mem_norm * mem_scale step. Used to test whether
                        mem_norm (trained on Ĥ's distribution) is distorting real H.
            selected_block_indices: optional LongTensor(K,) of global block indices.
                        When provided, decompress every segment as usual but then
                        gather only the reconstructed tokens corresponding to the
                        selected blocks before injection. Each block contributes
                        block_size*chunk_size (= compression_factor**2) tokens.
                        Used by the selection eval paths.

        Phase 1 (prefill): Run prompt through layers 0..ℓ-1, then
        [decompressed | prompt_at_ℓ] through layers ℓ..L-1.
        Phase 2 (decode): Autoregressive token-by-token, re-running the
        two-phase forward each step (O(n²) — no KV cache optimization).

        """
        if return_timing:
            import time as _time
            torch.cuda.synchronize()
            _t_prefill_start = _time.perf_counter()
        # Tracks whether the efficient-decompress path already produced a
        # block-subsetted reconstruction, so the generic subset step is skipped.
        _did_efficient_subset = False
        if override_decompressed_hs is not None:
            # Oracle-H path: skip decompressor, use supplied hidden states directly.
            decompressed_hs = override_decompressed_hs.float()
        else:
            # Decompress raw memory tokens (matches forward() path), per segment.
            if getattr(self, '_efficient_decompress', False) and selected_block_indices is not None:
                # Efficient decompress (flag-gated, default off): gather ONLY the
                # chosen blocks' raw memory slots and run a SINGLE decompressor call,
                # instead of decompressing all M_total slots into the full
                # reconstruction and discarding all but the chosen blocks.
                # Bit-exact for a per-slot decompressor (Decompressor: one
                # shared MLP applied independently per slot, no cross-slot signal,
                # no position embeddings) — each slot's output depends only on that
                # slot, so gather-then-decompress == decompress-then-gather.
                # Block k occupies raw mem slots [k*block_mem, k*block_mem+block_mem),
                # where block_mem = tokens_per_block / chunk_size mem slots per block.
                B, M_total, D = memory_raw.shape
                c = self.compression_factor
                tokens_per_block = c * c
                chunk = getattr(self.decompressor, 'chunk_size', c)
                block_mem = tokens_per_block // chunk
                sel = []
                for k in selected_block_indices.tolist():
                    start = k * block_mem
                    end = start + block_mem
                    if start < 0 or end > M_total:
                        raise IndexError(
                            f"block index {k} → mem slots [{start},{end}) out of "
                            f"range [0,{M_total}); check num_blocks computation"
                        )
                    sel.append(memory_raw[:, start:end, :])
                sel_mem = torch.cat(sel, dim=1)                       # (1, K*block_mem, D)
                decompressed_hs = self.decompressor(sel_mem.float())  # (1, K*c*c, D)
                _did_efficient_subset = True
                # Optional strict bit-exact A/B vs decompress-all-then-subset.
                _n_verify = int(getattr(self, '_verify_efficient_decom_remaining', 0))
                if _n_verify > 0:
                    _all_rec = []
                    _ns = M_total // self.memory_slots
                    for _s in range(_ns):
                        _sm = memory_raw[:, _s * self.memory_slots:(_s + 1) * self.memory_slots, :]
                        _all_rec.append(self.decompressor(_sm.float()))
                    _full = torch.cat(_all_rec, dim=1)
                    _ref = torch.cat(
                        [_full[:, k * tokens_per_block:(k + 1) * tokens_per_block, :]
                         for k in selected_block_indices.tolist()], dim=1)
                    _mad = (decompressed_hs - _ref).abs().max().item()
                    # On GPU, fp32 GEMM picks a different tiling for the small (K-block)
                    # vs full (M_total) decompressor call -> reduction-order noise (~1e-6),
                    # not exact 0. The decompressed states are cast to the backbone dtype
                    # (bf16) next, which washes this out, so injection is F1-preserving.
                    _mad_bf16 = (decompressed_hs.to(self._backbone_dtype)
                                 - _ref.to(self._backbone_dtype)).abs().max().item()
                    _tol = 1e-3
                    print(f"[efficient_decompress verify] fp32 max_abs_diff={_mad:.3e}, "
                          f"bf16-cast max_abs_diff={_mad_bf16:.3e} "
                          f"(K={len(selected_block_indices)} blocks; "
                          f"{decompressed_hs.size(1)} recon toks vs full {_full.size(1)}; "
                          f"tol={_tol:.0e})")
                    if _mad > _tol:
                        raise RuntimeError(
                            f"efficient_decompress mismatch: fp32 max_abs_diff={_mad:.3e} > tol={_tol:.0e} "
                            "(decompressor may not be per-slot independent)")
                    self._verify_efficient_decom_remaining = _n_verify - 1
            else:
                all_reconstructed = []
                B, M_total, D = memory_raw.shape
                num_segments = M_total // self.memory_slots
                for seg_idx in range(num_segments):
                    seg_mem = memory_raw[:, seg_idx * self.memory_slots : (seg_idx + 1) * self.memory_slots, :]
                    seg_recon = self.decompressor(seg_mem.float())
                    all_reconstructed.append(seg_recon)
                decompressed_hs = torch.cat(all_reconstructed, dim=1)  # (1, T_recon, D)

        # Block-level subset selection for selection eval paths.
        # Each block spans block_size*chunk_size = compression_factor**2 reconstructed tokens;
        # keep only the indices in selected_block_indices, preserving their order.
        # Skipped when the efficient-decompress path already produced the subset.
        if selected_block_indices is not None and not _did_efficient_subset:
            c = self.compression_factor
            tokens_per_block = c * c
            T_all = decompressed_hs.size(1)
            gathered = []
            for k in selected_block_indices.tolist():
                start = k * tokens_per_block
                end = start + tokens_per_block
                if start < 0 or end > T_all:
                    raise IndexError(
                        f"block index {k} → tokens [{start},{end}) out of range "
                        f"[0,{T_all}); check num_blocks computation"
                    )
                gathered.append(decompressed_hs[:, start:end, :])
            decompressed_hs = torch.cat(gathered, dim=1)  # (1, K*c*c, D)

        # Normalize decompressed output (same as forward() path; also applied to
        # evidence-label-H override so the decoder's input distribution matches training).
        # Skip only if caller explicitly asked (raw evidence-label-H diagnostic mode).
        _skip_norm = (override_decompressed_hs is not None) and override_skip_mem_norm
        if (not _skip_norm) and hasattr(self, 'mem_norm') and hasattr(self, 'mem_scale') and self._normalize_mem:
            max_scale = 25.0 / math.sqrt(self.dim)  # match layer-ℓ hidden state norms
            effective_scale = torch.clamp(self.mem_scale, min=0.01, max=max_scale)
            decompressed_hs = self.mem_norm(decompressed_hs) * effective_scale

        decompressed_hs = decompressed_hs.to(self._backbone_dtype)

        # Build prompt embeddings with ft_token prefix (matches training format)
        # Clamp to original_vocab_size: chat-template special tokens (e.g. <|im_start|>=151644)
        # exceed config.vocab_size=151643; training does the same clamp at every embed call.
        if prompt_ids is not None:
            prompt_ids = prompt_ids.clamp(max=self.original_vocab_size - 1)
            prompt_embs = self._decoder_embed_tokens(prompt_ids)
        else:
            # Need at least one token to start generation
            bos = torch.tensor([[self.tokenizer.bos_token_id or self.eos_id]],
                               device=memory_raw.device)
            prompt_embs = self._decoder_embed_tokens(bos)
        # Prepend ft_token learned embedding (training uses [mem_placeholders] + [ft] + question)
        ft_emb = self.special_token_embed(
            torch.ones(1, dtype=torch.long, device=prompt_embs.device)
        ).unsqueeze(0).to(prompt_embs.dtype)  # (1, 1, D)
        prompt_embs = torch.cat([ft_emb, prompt_embs], dim=1)

        # Two-phase prefill to get logits for the prompt
        # Use decoder (with LoRA) if available, else backbone
        inject_layer = self._injection_layer
        if hasattr(self, 'decoder') and self.decoder is not self.backbone:
            layers, final_norm, lm_head, rotary_emb = self._get_decoder_internals()
        else:
            layers, final_norm, lm_head, rotary_emb = self._get_backbone_internals()
        T_recon = decompressed_hs.size(1)
        S_prompt = prompt_embs.size(1)
        device = memory_raw.device
        dtype = prompt_embs.dtype
        min_val = torch.finfo(dtype).min

        # ============================================================
        # KV-cached two-phase decode path (env-gated; default OFF).
        # Set SEDEM_KV_CACHE=1 to enable. Default off preserves the uncached
        # path bit-for-bit. The cached path is mathematically equivalent up to
        # fp16/bf16 reduction-order noise (decode kernel dispatch).
        # ============================================================
        if os.environ.get("SEDEM_KV_CACHE", "0") == "1":
            # When shape-stable KV is on, size the cache for the full possible
            # decode horizon (S_prompt + max_new_tokens) so dynamo sees one shape
            # across prefill + all decode steps.
            _static_max_text_len = S_prompt + int(max_new_tokens) if getattr(self, '_shape_stable_kv', False) else None
            last_logit, past_kv1, past_kv2 = self.forward_with_injection_cached(
                prompt_embs.to(dtype), decompressed_hs.to(dtype), inject_layer,
                past_kv_phase1=None, past_kv_phase2=None, text_pos_past=0,
                static_max_text_len=_static_max_text_len,
            )
            logit = last_logit[:, 0, :self.original_vocab_size]

            generated = []
            next_id = logit.argmax(dim=-1)
            if not hasattr(self, '_gen_debug_count'):
                self._gen_debug_count = 0
            self._gen_debug_count += 1
            if self._gen_debug_count <= 3:
                top5_vals, top5_ids = logit.topk(5, dim=-1)
                top5_toks = [self.tokenizer.decode([t.item()]) for t in top5_ids[0]]
                print(f"  [GEN DEBUG SEDEM_KV_CACHE] first token: id={next_id.item()}, "
                      f"text={repr(self.tokenizer.decode([next_id.item()]))}, "
                      f"eos_id={self.eos_id}, "
                      f"top5={list(zip(top5_toks, top5_vals[0].tolist()))}")

            if next_id.item() == self.eos_id:
                if return_timing:
                    torch.cuda.synchronize()
                    _t_first = _time.perf_counter()
                    return {
                        "text": "",
                        "prefill_ms": (_t_first - _t_prefill_start) * 1000.0,
                        "decode_ms": 0.0,
                        "num_tokens_generated": 0,
                    }
                return ""
            generated.append(next_id.item())

            if return_timing:
                torch.cuda.synchronize()
                _t_first = _time.perf_counter()

            text_pos_past = S_prompt
            for _ in range(max_new_tokens - 1):
                next_id_clamped = next_id.clamp(max=self.original_vocab_size - 1)
                next_emb = self._decoder_embed_tokens(next_id_clamped.unsqueeze(0))
                last_logit, past_kv1, past_kv2 = self.forward_with_injection_cached(
                    next_emb.to(dtype), decompressed_hs.to(dtype), inject_layer,
                    past_kv_phase1=past_kv1, past_kv_phase2=past_kv2,
                    text_pos_past=text_pos_past,
                    static_max_text_len=_static_max_text_len,
                )
                text_pos_past += 1
                logit = last_logit[:, 0, :self.original_vocab_size]
                next_id = logit.argmax(dim=-1)
                if next_id.item() == self.eos_id:
                    break
                generated.append(next_id.item())

            if return_timing:
                torch.cuda.synchronize()
                _t_end = _time.perf_counter()
                return {
                    "text": self.tokenizer.decode(generated, skip_special_tokens=True),
                    "prefill_ms": (_t_first - _t_prefill_start) * 1000.0,
                    "decode_ms": (_t_end - _t_first) * 1000.0,
                    "num_tokens_generated": len(generated),
                }
            return self.tokenizer.decode(generated, skip_special_tokens=True)

        # ============================================================
        # Default uncached path: O(n²) two-phase decode
        # ============================================================
        # Phase 1: layers 0..ℓ-1 on prompt only
        text_pos = torch.arange(T_recon, T_recon + S_prompt, device=device).unsqueeze(0)
        hidden = prompt_embs
        if inject_layer > 0:
            text_pos_emb = rotary_emb(hidden, text_pos) if rotary_emb is not None else None
            text_mask = torch.full((S_prompt, S_prompt), min_val, device=device, dtype=dtype)
            text_mask = torch.triu(text_mask, diagonal=1).unsqueeze(0).unsqueeze(0)
            for layer_idx in range(inject_layer):
                lo = layers[layer_idx](
                    hidden, attention_mask=text_mask, position_ids=text_pos,
                    position_embeddings=text_pos_emb,
                )
                hidden = lo[0] if isinstance(lo, tuple) else lo
        prompt_at_inject = hidden

        # Phase 2: [decompressed | prompt_at_ℓ] through layers ℓ..L-1
        combined = torch.cat([decompressed_hs, prompt_at_inject], dim=1)
        decomp_pos = torch.arange(T_recon, device=device).unsqueeze(0)
        combined_pos = torch.cat([decomp_pos, text_pos], dim=1)
        combined_pos_emb = rotary_emb(combined, combined_pos) if rotary_emb is not None else None

        total_len = T_recon + S_prompt
        causal_mask = torch.full((total_len, total_len), min_val, device=device, dtype=dtype)
        causal_mask = torch.triu(causal_mask, diagonal=1).unsqueeze(0).unsqueeze(0)

        hidden = combined
        for layer_idx in range(inject_layer, len(layers)):
            lo = layers[layer_idx](
                hidden, attention_mask=causal_mask, position_ids=combined_pos,
                position_embeddings=combined_pos_emb,
            )
            hidden = lo[0] if isinstance(lo, tuple) else lo

        hidden = final_norm(hidden)
        logit = lm_head(hidden[:, -1:, :])[:, 0, :self.original_vocab_size]

        # Autoregressive decode: for simplicity, re-run full two-phase
        # forward with growing prompt. This is O(n²) but correct.
        # For long generations, a KV-cache version would be needed.
        generated = []
        next_id = logit.argmax(dim=-1)
        # Debug: print first token prediction
        if not hasattr(self, '_gen_debug_count'):
            self._gen_debug_count = 0
        self._gen_debug_count += 1
        if self._gen_debug_count <= 3:
            top5_vals, top5_ids = logit.topk(5, dim=-1)
            top5_toks = [self.tokenizer.decode([t.item()]) for t in top5_ids[0]]
            print(f"  [GEN DEBUG] first token: id={next_id.item()}, "
                  f"text={repr(self.tokenizer.decode([next_id.item()]))}, "
                  f"eos_id={self.eos_id}, "
                  f"top5={list(zip(top5_toks, top5_vals[0].tolist()))}")
        if next_id.item() == self.eos_id:
            if return_timing:
                torch.cuda.synchronize()
                _t_first = _time.perf_counter()
                return {
                    "text": "",
                    "prefill_ms": (_t_first - _t_prefill_start) * 1000.0,
                    "decode_ms": 0.0,
                    "num_tokens_generated": 0,
                }
            return ""
        generated.append(next_id.item())

        # Mark the first-token boundary for TTFT vs decode split.
        if return_timing:
            torch.cuda.synchronize()
            _t_first = _time.perf_counter()

        # Grow the prompt for subsequent tokens
        growing_ids = prompt_ids.clone() if prompt_ids is not None else bos.clone()
        growing_ids = torch.cat([growing_ids, next_id.unsqueeze(0)], dim=1)

        for _ in range(max_new_tokens - 1):
            # Re-embed growing prompt
            growing_embs = self._decoder_embed_tokens(growing_ids)
            S_cur = growing_embs.size(1)
            text_pos_cur = torch.arange(T_recon, T_recon + S_cur, device=device).unsqueeze(0)

            # Phase 1
            hidden = growing_embs
            if inject_layer > 0:
                text_pos_emb_cur = rotary_emb(hidden, text_pos_cur) if rotary_emb is not None else None
                text_mask_cur = torch.full((S_cur, S_cur), min_val, device=device, dtype=dtype)
                text_mask_cur = torch.triu(text_mask_cur, diagonal=1).unsqueeze(0).unsqueeze(0)
                for layer_idx in range(inject_layer):
                    lo = layers[layer_idx](
                        hidden, attention_mask=text_mask_cur, position_ids=text_pos_cur,
                        position_embeddings=text_pos_emb_cur,
                    )
                    hidden = lo[0] if isinstance(lo, tuple) else lo

            # Phase 2
            combined = torch.cat([decompressed_hs, hidden], dim=1)
            combined_pos_cur = torch.cat([decomp_pos, text_pos_cur], dim=1)
            combined_pos_emb_cur = rotary_emb(combined, combined_pos_cur) if rotary_emb is not None else None
            total_len_cur = T_recon + S_cur
            causal_mask_cur = torch.full(
                (total_len_cur, total_len_cur), min_val, device=device, dtype=dtype
            )
            causal_mask_cur = torch.triu(causal_mask_cur, diagonal=1).unsqueeze(0).unsqueeze(0)

            hidden = combined
            for layer_idx in range(inject_layer, len(layers)):
                lo = layers[layer_idx](
                    hidden, attention_mask=causal_mask_cur, position_ids=combined_pos_cur,
                    position_embeddings=combined_pos_emb_cur,
                )
                hidden = lo[0] if isinstance(lo, tuple) else lo

            hidden = final_norm(hidden)
            logit = lm_head(hidden[:, -1:, :])[:, 0, :self.original_vocab_size]
            next_id = logit.argmax(dim=-1)
            if next_id.item() == self.eos_id:
                break
            generated.append(next_id.item())
            growing_ids = torch.cat([growing_ids, next_id.unsqueeze(0)], dim=1)

        if return_timing:
            torch.cuda.synchronize()
            _t_end = _time.perf_counter()
            return {
                "text": self.tokenizer.decode(generated, skip_special_tokens=True),
                "prefill_ms": (_t_first - _t_prefill_start) * 1000.0,
                "decode_ms": (_t_end - _t_first) * 1000.0,
                "num_tokens_generated": len(generated),
            }
        return self.tokenizer.decode(generated, skip_special_tokens=True)
