"""Top-K selector over memory blocks (paper Sec. 3.2, App. A.3).

LateInteraction scoring: per head r, q = LN(H_Q) W_Q^(r), k = LN(B_n) W_K^(r),
alpha = cos(q, k); MaxSim over slots in a block, summed over query tokens,
averaged over heads. Trained with InfoNCE (+ optional margin) on evidence labels.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PooledBlockEncoder(nn.Module):
    """Encodes a memory block into a selection key.

    block (G slots, D each) → mean-pool → LayerNorm → linear → L2-normalize
    Output: unit-norm key vector in R^{D_key}
    """

    def __init__(self, d_model: int, d_key: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_key, bias=False)

    def forward(self, blocks: torch.Tensor) -> torch.Tensor:
        """
        blocks: (B, num_blocks, G, D)
        returns: (B, num_blocks, D_key) — unit-norm keys
        """
        pooled = blocks.mean(dim=2)                  # (B, num_blocks, D)
        return F.normalize(self.proj(self.norm(pooled)), dim=-1)


class PooledQueryEncoder(nn.Module):
    """Encodes question hidden states into a query vector for selection.

    question_hs (B, Q, D) → mean-pool → LayerNorm → linear → L2-normalize
    Output: unit-norm query vector in R^{D_key}
    """

    def __init__(self, d_model: int, d_key: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_key, bias=False)

    def forward(self, question_hs: torch.Tensor) -> torch.Tensor:
        """
        question_hs: (B, Q, D)
        returns: (B, D_key) — unit-norm query
        """
        pooled = question_hs.mean(dim=1)             # (B, D)
        return F.normalize(self.proj(self.norm(pooled)), dim=-1)



class LateInteractionScorer(nn.Module):
    """Per-block score via late-interaction (ColBERT-style MaxSim) on raw logits.

    **Design note:** an earlier version applied `softmax` across
    all N·G slots before aggregating per block. That produced probability-valued
    scores summing to 1 → double-normalization collapse when fed into an outer
    InfoNCE softmax. Rankings became insensitive to weight updates even as loss
    slowly drifted. Fixed by removing the inner softmax and using raw logit
    aggregation (this version).

    Formulation:
        Q = q_proj(LN(H_q))                     # (B, Q_tok, H*Dh)
        K = k_proj(LN(blocks.flatten_slots))    # (B, N*G, H*Dh)
        ℓ_{h,t,i,g} = (Q · Kᵀ)                  # (B, H, Q_tok, N, G)  cosine in [-1, 1]
        z_{h,t,i}   = max_g ℓ_{h,t,i,g}          # MaxSim per (head, token, block)
        s_i         = sum_t mean_h z_{h,t,i}     # (B, N) per-block scalar

    Each query token picks its best slot in each block (MaxSim); per-block
    scores are then SUMMED over query tokens (ColBERT convention) and meaned
    over heads. Sum-over-tokens (vs prior mean) preserves the per-token gradient
    signal that mean-aggregation was diluting by 1/Q_tok.

    Parameters ≈ 2 · d_model² + 2 · d_model (LayerNorms).
    """

    def __init__(self, d_model: int, num_heads: int = 4, d_head: int = None,
                 q_dim: int = None):
        """`q_dim` (defaults to `d_model`) is the dim of the question hidden
        states. In cross-model setups (encoder dim ≠ decoder dim) the question
        comes from the encoder while blocks come from the compressor's decoder
        space. Norm + proj on the Q-side use q_dim; K-side uses d_model.
        """
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_head or (d_model // num_heads)
        q_dim = q_dim or d_model
        self.q_dim = q_dim
        self.norm_q = nn.LayerNorm(q_dim)
        self.norm_k = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(q_dim, num_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, num_heads * self.d_head, bias=False)

    def forward(self, question_hs: torch.Tensor, blocks: torch.Tensor) -> torch.Tensor:
        """
        question_hs: (B, Q_tok, D)
        blocks:      (B, N, G, D)
        returns:     (B, N) per-block scalar scores (higher = more relevant)

        Uses unit-normalized Q and K (ColBERT convention) so every logit is a
        cosine in [-1, 1]. Without this, raw dot products explode the outer
        InfoNCE loss at random init (observed: loss ≈ 200-240 on real data).
        Per-block scores remain bounded in [-1, 1] after MaxSim + mean.
        """
        B, N, G, D = blocks.shape
        Q_tok = question_hs.size(1)
        H, Dh = self.num_heads, self.d_head

        Q = self.q_proj(self.norm_q(question_hs))          # (B, Q_tok, H*Dh)
        Q = Q.view(B, Q_tok, H, Dh).transpose(1, 2)        # (B, H, Q_tok, Dh)
        Q = F.normalize(Q, dim=-1)                         # cosine basis

        slots = blocks.reshape(B, N * G, D)
        K = self.k_proj(self.norm_k(slots))                # (B, N*G, H*Dh)
        K = K.view(B, N * G, H, Dh).transpose(1, 2)        # (B, H, N*G, Dh)
        K = F.normalize(K, dim=-1)                         # cosine basis

        # Cosine similarity per (head, query_token, slot) — logits bounded in [-1, 1].
        # Note: no √Dh scaling since vectors are unit-norm and products are cosines.
        logits = torch.matmul(Q, K.transpose(-2, -1))       # (B, H, Q_tok, N*G)
        logits = logits.reshape(B, H, Q_tok, N, G)

        # MaxSim: for each (head, query_token, block), pick best-matching slot.
        # Aggregate per-block: sum over Q_tok (ColBERT), mean over heads. The sum
        # preserves per-token gradient magnitude; mean-over-tokens (prior) shrank
        # gradients by 1/Q_tok and made the effective τ too soft.
        z = logits.max(dim=-1).values                       # (B, H, Q_tok, N)
        scores = z.sum(dim=2).mean(dim=1)                   # (B, N) — bounded in [-Q_tok, Q_tok]
        return scores




# ---------------------------------------------------------------------------
# Selector head wrappers
# ---------------------------------------------------------------------------

class PooledQuerySelector(nn.Module):
    """Dot-product selector: scalar per block via q · k."""

    def __init__(self, d_model, d_key=256):
        super().__init__()
        self.block_enc = PooledBlockEncoder(d_model, d_key)
        self.query_enc = PooledQueryEncoder(d_model, d_key)

    def forward(self, question_hs, blocks):
        """
        question_hs: (B, Q_tok, D)
        blocks:      (B, N, G, D)
        returns:     (B, N) per-block scores
        """
        keys = self.block_enc(blocks)                          # (B, N, D_key) unit-norm
        q_vec = self.query_enc(question_hs)                    # (B, D_key) unit-norm
        scores = torch.einsum("bd,bnd->bn", q_vec, keys)       # cosine sim
        return scores


class LateInteractionSelector(nn.Module):
    """Cross-attention scorer: per-block score from full question sequence.

    `q_dim` (defaults to d_model) lets cross-model setups pass the encoder's
    hidden dim for the question side while blocks live in decoder dim.
    """

    def __init__(self, d_model, num_heads=4, q_dim=None):
        super().__init__()
        self.scorer = LateInteractionScorer(d_model, num_heads=num_heads, q_dim=q_dim)

    def forward(self, question_hs, blocks):
        return self.scorer(question_hs, blocks)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def selection_infonce_loss(scores, positive_mask, temperature):
    """Sum-of-log-probs InfoNCE for a single example (or batch of one).

    scores:        (N,)
    positive_mask: (N,) bool — at least one True; otherwise caller should skip
    temperature:   float > 0

    L = logsumexp(all/tau) - logsumexp(positives/tau)
    Equivalent to   −log( Σ_{i∈pos} exp(s_i/τ) / Σ_all exp(s_j/τ) )
    """
    s = scores / temperature
    log_z_all = torch.logsumexp(s, dim=0)
    log_z_pos = torch.logsumexp(s[positive_mask], dim=0)
    return log_z_all - log_z_pos


def selection_margin_loss(scores, positive_mask, margin: float = 0.4):
    """Hinge margin: enforce min(positives) − max(negatives) ≥ margin.

    Motivated by the observation that on HotpotQA bridge questions the gold-distractor cosine
    gap is 0.178 while comparison is 0.342. InfoNCE alone cannot enlarge this
    gap aggressively because LSE(all)/LSE(pos) saturates once any positive scores
    above all negatives. A direct hinge on the *worst* positive vs *best*
    negative keeps gradient flow even after InfoNCE plateaus.

    scores:        (N,) per-block scores (any scale; choose `margin` accordingly)
    positive_mask: (N,) bool
    margin:        scalar; for cosine-style scores in [-1, 1], 0.3-0.5 is reasonable

    Returns 0 when no positives or no negatives present.
    """
    if not positive_mask.any() or positive_mask.all():
        return torch.zeros((), device=scores.device, dtype=scores.dtype)
    pos = scores[positive_mask]
    neg = scores[~positive_mask]
    return F.relu(margin - (pos.min() - neg.max()))

# ---------------------------------------------------------------------------
# Checkpoint / config helpers (accept the pre-release file and key names)
# ---------------------------------------------------------------------------
import json as _json
import os as _os

LEGACY_SELECTOR_TYPES = {"r1": "pooled", "r2": "late_interaction"}  # pre-release names
SELECTOR_WEIGHTS_FILENAME = "selector.safetensors"
SELECTOR_CONFIG_FILENAME = "selector_config.json"
_LEGACY_WEIGHTS = ("retriever.safetensors",)  # pre-release filename
_LEGACY_CONFIGS = ("retriever_config.json",)  # pre-release filename


def resolve_selector_checkpoint(path):
    """Accept a directory or a file; fall back to the pre-release filename."""
    if _os.path.isdir(path):
        for name in (SELECTOR_WEIGHTS_FILENAME,) + _LEGACY_WEIGHTS:
            cand = _os.path.join(path, name)
            if _os.path.exists(cand):
                return cand
        raise FileNotFoundError(f"no selector weights found in {path}")
    if not _os.path.exists(path):
        for name in _LEGACY_WEIGHTS:
            cand = _os.path.join(_os.path.dirname(path), name)
            if _os.path.exists(cand):
                return cand
    return path


def load_selector_config(ckpt_path, config_path=None):
    """Read selector_config.json next to a checkpoint (or the pre-release selector_config.json)
    and normalise key/value names to the release vocabulary."""
    if config_path is None:
        d = _os.path.dirname(_os.path.abspath(ckpt_path))
        for name in (SELECTOR_CONFIG_FILENAME,) + _LEGACY_CONFIGS:
            cand = _os.path.join(d, name)
            if _os.path.exists(cand):
                config_path = cand
                break
    if config_path is None:
        raise FileNotFoundError(f"no selector config found next to {ckpt_path}")
    with open(config_path) as f:
        cfg = _json.load(f)
    if "selector_type" not in cfg and "retriever_type" in cfg:  # pre-release key
        cfg["selector_type"] = cfg.pop("retriever_type")
    cfg["selector_type"] = LEGACY_SELECTOR_TYPES.get(cfg.get("selector_type"), cfg.get("selector_type"))
    if "selector_heads" not in cfg and "r2_heads" in cfg:  # pre-release key
        cfg["selector_heads"] = cfg.pop("r2_heads")
    return cfg


def build_selector_from_config(cfg, d_model):
    """Instantiate the selector head described by a (normalised) config dict."""
    kind = LEGACY_SELECTOR_TYPES.get(cfg.get("selector_type"), cfg.get("selector_type"))
    if kind == "pooled":
        return PooledQuerySelector(d_model, d_key=cfg.get("d_key") or 256)
    if kind == "late_interaction":
        return LateInteractionSelector(d_model, num_heads=cfg.get("selector_heads") or 4)
    raise ValueError(f"selector_type must be 'pooled' or 'late_interaction', got {kind!r}")
