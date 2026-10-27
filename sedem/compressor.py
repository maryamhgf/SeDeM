"""
Hidden-state compressor modules for SeDeM v2.

Compressor variants that take backbone LLM hidden states and produce
a fixed set of M memory tokens:

1. LearnableQueryCompressor   -- cross-attention (or mean) pooling + MLP
2. SegmentCompressor -- local, recurrent, gated memory writer
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class CrossAttentionBlock(nn.Module):
    """Multi-head cross-attention: queries attend to key/value context.

    With gating enabled, the merge uses content-first blending:
        result = gate * queries + (1 - gate) * norm_match(attn_out, queries)
    where gate = g_max * tanh(raw_gate).

    With gate_init=0.1 → gate≈0.05, the output is ~95% cross-attention
    content and ~5% query structure. This ensures memory tokens carry
    document-specific information from the start, while the query serves
    primarily as an addressing mechanism for the attention computation.

    Norm-matching rescales the attention output to match the query norm
    per-token, keeping both branches at comparable scale regardless of
    MHA weight magnitudes (BranchNorm-style, arXiv:2305.02790).
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.0,
                 gated: bool = False, gate_max: float = 0.5, gate_init: float = 0.0,
                 blend_mode: str = "content_first",
                 slot_competition: bool = False,
                 out_proj_mode: str = "standard",
                 qk_norm: bool = False,
                 qk_norm_init_scale: float = 10.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.out_proj_mode = out_proj_mode  # "standard", "identity", "block_diagonal"
        # QK-normalization (Henry et al. 2020): L2-normalize Q^h, K^h before dot
        # product, scale by learnable per-head temperature. Forces score variance
        # to a controlled range regardless of W_Q/W_K weight magnitudes — directly
        # attacks the cold-start uniform-attention pathology where Var(QK) is too
        # small to peak the softmax over T keys.
        self.qk_norm = bool(qk_norm)
        if self.qk_norm:
            # init log_temp so softmax is moderately peaked at start.
            # softmax(temp * cos(Q,K)) with temp=10, cos∈[-1,1] -> scores∈[-10,10]
            # which is well above log(T=128)=4.85 needed for sharp softmax.
            import math as _math
            self.log_qk_temp = nn.Parameter(
                torch.full((num_heads,), _math.log(float(qk_norm_init_scale)))
            )
            print(f"[CrossAttn] qk_norm=True, init_temp={qk_norm_init_scale}")
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        self.gated = gated
        self.blend_mode = blend_mode  # "content_first", "perturbation", "no_residual"
        self.slot_competition = slot_competition

        # Block-diagonal out_proj: H independent (d_head, d_head) projections.
        # Preserves per-head geometry without cross-head mixing.
        if out_proj_mode == "block_diagonal":
            d_head = d_model // num_heads
            self.block_diag_proj = nn.Parameter(
                torch.eye(d_head).unsqueeze(0).expand(num_heads, -1, -1).clone()
            )
            self.block_diag_bias = nn.Parameter(torch.zeros(d_model))
            print(f"[CrossAttn] out_proj_mode=block_diagonal: {num_heads} heads × ({d_head},{d_head})")
        elif out_proj_mode == "identity":
            print(f"[CrossAttn] out_proj_mode=identity: no output projection")

        if gated:
            self.gate_max = gate_max
            self.raw_gate = nn.Parameter(torch.tensor(float(gate_init)))

    @property
    def gate(self):
        """Effective gate value (bounded). Only valid when gated=True."""
        return self.gate_max * torch.tanh(self.raw_gate)

    def _apply_out_proj(self, out, B, M, D):
        """Apply output projection based on out_proj_mode."""
        if self.out_proj_mode == "standard":
            return F.linear(out, self.attn.out_proj.weight, self.attn.out_proj.bias)
        elif self.out_proj_mode == "identity":
            return out
        elif self.out_proj_mode == "block_diagonal":
            H = self.num_heads
            d_head = D // H
            x = out.view(B, M, H, d_head)
            x = torch.einsum('bmhi,hij->bmhj', x, self.block_diag_proj)
            return x.reshape(B, M, D) + self.block_diag_bias

    def _manual_forward(self, q, kv_normed, need_weights):
        """Manual MHA forward with configurable out_proj and optional slot competition.

        Used instead of nn.MultiheadAttention.forward() when out_proj_mode != "standard"
        or slot_competition is enabled. Reuses the same in_proj weights from self.attn.

        Returns:
            out: (B, M, D)
            weights: (B, H, M, T) attention weights (or None)
            weights_pre: (B, H, M, T) pre-competition weights (or None, slot_competition only)
        """
        B, M, D = q.shape
        T = kv_normed.size(1)
        H = self.num_heads
        d_head = D // H

        # Unpack QKV projections from MHA's packed in_proj_weight
        W_q, W_k, W_v = self.attn.in_proj_weight.chunk(3, dim=0)
        if self.attn.in_proj_bias is not None:
            b_q, b_k, b_v = self.attn.in_proj_bias.chunk(3, dim=0)
        else:
            b_q = b_k = b_v = None

        Q = F.linear(q, W_q, b_q).view(B, M, H, d_head).transpose(1, 2)
        K = F.linear(kv_normed, W_k, b_k).view(B, T, H, d_head).transpose(1, 2)
        V = F.linear(kv_normed, W_v, b_v).view(B, T, H, d_head).transpose(1, 2)

        if self.qk_norm:
            # L2-normalize Q^h, K^h per head, then scale by learnable per-head temp.
            # scores_ij = temp_h * (Q^h_i / ||Q^h_i||) . (K^h_j / ||K^h_j||)
            #          = temp_h * cos(Q^h_i, K^h_j)
            Q_n = F.normalize(Q, dim=-1)
            K_n = F.normalize(K, dim=-1)
            temp = self.log_qk_temp.exp().view(1, -1, 1, 1)  # (1, H, 1, 1)
            scores = temp * torch.matmul(Q_n, K_n.transpose(-2, -1))
        else:
            # Standard attention scores + softmax over keys
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_head)
        attn_pre = F.softmax(scores, dim=-1)  # (B, H, M, T)

        # Optional slot competition: normalize over queries
        weights_pre = None
        if self.slot_competition:
            attn = attn_pre / (attn_pre.sum(dim=-2, keepdim=True) + 1e-8)
            weights_pre = attn_pre
        else:
            attn = attn_pre

        if self.training and self.attn.dropout > 0:
            attn = F.dropout(attn, p=self.attn.dropout, training=True)

        # Weighted sum over values
        out = torch.matmul(attn, V)  # (B, H, M, d_head)
        out = out.transpose(1, 2).contiguous().view(B, M, D)

        # Output projection (mode-dependent)
        out = self._apply_out_proj(out, B, M, D)

        if need_weights:
            return out, attn, weights_pre
        return out, None, None

    def forward(self, queries, kv, need_weights=False):
        """
        queries : (B, M, D)
        kv      : (B, T, D)
        returns : (B, M, D)  or  ((B, M, D), (B, H, M, T)) when need_weights
        """
        q = self.norm_q(queries)
        k = v = self.norm_kv(kv)
        weights_pre = None  # pre-competition weights (only with slot_competition)
        if self.slot_competition or self.out_proj_mode != "standard" or self.qk_norm:
            # Manual forward: needed for slot competition, non-standard out_proj,
            # or QK-normalization (since nn.MultiheadAttention has no qk-norm).
            out, weights, weights_pre = self._manual_forward(
                q, k, need_weights=need_weights,
            )
        else:
            out, weights = self.attn(
                q, k, v,
                need_weights=need_weights,
                average_attn_weights=False,
            )
        if self.gated:
            # Norm-match: rescale attn output to match query norm per-token.
            # This keeps both branches at comparable scale so the gate
            # controls the blend ratio directly without norm inflation.
            q_norm = queries.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            out_norm = out.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            # Clamp scale to prevent bf16 overflow (max 65504) when out_norm is tiny
            scale = (q_norm / out_norm).clamp(max=1e4)
            out = out * scale
            g = self.gate
            if self.blend_mode == "content_first":
                # Content-first: low gate → output is mostly cross-attention content.
                result = g * queries + (1.0 - g) * out
            elif self.blend_mode == "perturbation":
                # Old Flamingo-style: output is query + small content perturbation.
                result = queries + g * out
            elif self.blend_mode == "no_residual":
                # Pure cross-attention output, no query component.
                result = out
            else:
                raise ValueError(f"Unknown blend_mode: {self.blend_mode}")
        else:
            result = queries + out  # ungated (backward compatible)
        if need_weights:
            return result, weights, weights_pre  # weights: (B, H, M, T)
        return result


class FFN(nn.Module):
    """Standard pre-norm feed-forward block: output = x + FFN(LayerNorm(x)).

    Standard transformer FFN with residual connection. No gating, no
    norm-matching — the FFN freely transforms the representation.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0,
                 **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        # Keep raw_gate as a frozen dummy for checkpoint backward compatibility
        self.raw_gate = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def forward(self, x):
        h = self.norm(x)
        h = self.w2(self.dropout(F.gelu(self.w1(h))))
        return x + h


class SelfAttentionBlock(nn.Module):
    """Pre-norm self-attention + FFN with standard residuals (Perceiver-style).

    Applied on the latent/topic array so the M slots can coordinate
    with each other ("I'll cover X, you cover Y"). Without this, slots
    only see the input via cross-attention and have no mechanism to
    differentiate, which empirically drives diversity collapse.
    """

    def __init__(self, d_model: int, num_heads: int = 8,
                 d_ff: int = None, dropout: float = 0.0):
        super().__init__()
        if d_ff is None:
            d_ff = 2 * d_model
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm_attn(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        h = self.norm_ffn(x)
        h = self.w2(self.dropout(F.gelu(self.w1(h))))
        return x + h


# ---------------------------------------------------------------------------
# Topic Router: selects top-k topics from a larger pool per segment
# ---------------------------------------------------------------------------

class LearnableQueryCompressor(nn.Module):
    """
    Produces M memory tokens from variable-length hidden states.

    M learnable topics cross-attend to the hidden-state sequence,
    then an MLP projects the result.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_memory_tokens: int = 128,
        bottleneck_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.0,
        gate_init: float = 0.0,
        gate_max: float = 0.5,
        blend_mode: str = "content_first",
        slot_competition: bool = False,
        out_proj_mode: str = "standard",
        qk_norm: bool = False,
        qk_norm_init_scale: float = 10.0,
        no_ffn: bool = False,
        num_self_attn_layers: int = 0,
        num_iter_blocks: int = 1,
        gated: bool = True,
        ignore_prev_mem: bool = False,
        topics_init: str = "random",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_memory_tokens = num_memory_tokens
        self.enable_routing = False
        self.routing_top_k = num_memory_tokens
        self.no_ffn = no_ffn
        self.num_self_attn_layers = int(num_self_attn_layers)
        self.num_iter_blocks = int(num_iter_blocks)

        # Topics initialization. 'random' = canonical Perceiver-IO randn*0.02.
        # 'sinusoidal' = deterministic per-slot positional embedding -> breaks
        # cold-start query symmetry. Motivation:
        # at random init, all M topics have similar distribution -> W_Q@LN(q_i)
        # produces near-identical Q^h_i across slots -> uniform attention
        # allocation -> AV is slot-shared -> slots collapse. Sinusoidal init
        # gives slot-distinct anchors that survive W_Q regardless of W_Q noise.
        if topics_init == "sinusoidal":
            pe = torch.zeros(num_memory_tokens, input_dim)
            position = torch.arange(0, num_memory_tokens, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, input_dim, 2, dtype=torch.float)
                * (-math.log(10000.0) / input_dim)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            # Match the post-LN scale a learnable random init would reach.
            # ||sinusoidal||_2 per row ≈ sqrt(input_dim/2). After LN it becomes
            # ≈ sqrt(input_dim) regardless. Scale here is for residual blend
            # (gated=False: result = topics + attn_out, so ||topics|| matters).
            self.topics = nn.Parameter(pe.unsqueeze(0) * 0.5)
            print(f"[MLPPool] topics_init=sinusoidal, ||topics_i||~={pe.norm(dim=-1).mean().item() * 0.5:.3f}")
        elif topics_init == "orthogonal":
            t = torch.empty(num_memory_tokens, input_dim)
            nn.init.orthogonal_(t)
            self.topics = nn.Parameter((t * 0.02).unsqueeze(0))
            print(f"[MLPPool] topics_init=orthogonal")
        elif topics_init == "random":
            self.topics = nn.Parameter(
                torch.randn(1, num_memory_tokens, input_dim) * 0.02
            )
        else:
            raise ValueError(
                f"Unknown topics_init: '{topics_init}'. "
                "Choose 'random', 'sinusoidal', or 'orthogonal'."
            )
        # gated=False -> pure Perceiver-IO residual: result = queries + attn_out
        # (drops the Flamingo-derived tanh gate, content-first blend, and
        #  norm-matching). gated=True keeps the existing behavior.
        self.gated = bool(gated)
        # ignore_prev_mem=True -> stateless Perceiver. Drops the prev_mem KV
        # concatenation that re-introduces cross-segment recurrence (the
        # mechanism deliberately removed; the "prev_mem -> writer feedback loop"
        # was the dominant collapse cause in the chunked compressor's recurrent
        # variants and equally applies here when prev_mem is fed as KV).
        self.ignore_prev_mem = bool(ignore_prev_mem)
        self.cross_attn = CrossAttentionBlock(
            d_model=input_dim, num_heads=num_heads, dropout=dropout,
            gated=self.gated, gate_max=gate_max, gate_init=gate_init,
            blend_mode=blend_mode, slot_competition=slot_competition,
            out_proj_mode=out_proj_mode,
            qk_norm=qk_norm, qk_norm_init_scale=qk_norm_init_scale,
        )

        if self.num_iter_blocks < 1:
            raise ValueError(f"num_iter_blocks must be >= 1, got {self.num_iter_blocks}")
        if (self.num_self_attn_layers > 0 or self.num_iter_blocks > 1) and input_dim != output_dim:
            raise ValueError(
                "Perceiver-style self-attn / iteration requires input_dim == output_dim "
                f"(got {input_dim} vs {output_dim})"
            )
        if self.num_self_attn_layers > 0:
            self.self_attn_stack = nn.ModuleList([
                SelfAttentionBlock(
                    d_model=input_dim, num_heads=num_heads,
                    d_ff=bottleneck_dim, dropout=dropout,
                )
                for _ in range(self.num_self_attn_layers)
            ])
            print(f"[Compressor] Perceiver self-attn stack: "
                  f"{self.num_self_attn_layers} layers × {num_heads} heads, "
                  f"d_ff={bottleneck_dim}")
        else:
            self.self_attn_stack = None
        if self.num_iter_blocks > 1:
            print(f"[Compressor] Iterative refinement: {self.num_iter_blocks} blocks "
                  f"with shared weights (Perceiver RNN-over-depth)")

        self.router = None
        if no_ffn:
            self.mlp = None
            print(f"[Compressor] FFN disabled (no_ffn=True): memory = cross-attention output directly")
        else:
            assert input_dim == output_dim, (
                f"FFN requires input_dim == output_dim, "
                f"got {input_dim} vs {output_dim}"
            )
            self.mlp = FFN(input_dim, bottleneck_dim, dropout)

    @property
    def output_memory_tokens(self):
        """Number of memory tokens output per segment (k if routing, else N)."""
        if self.enable_routing:
            return self.routing_top_k
        return self.num_memory_tokens

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_mem: torch.Tensor = None,
        return_attn_stats: bool = False,
        router_kv: torch.Tensor = None,
        verbose: bool = False,
        seg_idx: int = None,
        **kwargs,
    ):
        """
        hidden_states    : (B, T, D_in)
        prev_mem         : (B, K, D_in) or None — previous segment's memory
                           tokens, detached and prepended to KV context.
        return_attn_stats: if True, also return a dict with attention diagnostics.
        router_kv        : (B, T', D_in) or None — if provided, the router uses
                           this as input instead of the full [prev_mem; hidden_states].
                           Cross-attention still uses the full KV. This allows ablating
        verbose, seg_idx : accepted for call-site signature compatibility with
                           SegmentCompressor; not used here.
                           whether prev_mem in the router input helps or hurts.
        returns          : (B, M, D_out)  or  ((B, M, D_out), dict)
        """
        B, T, D = hidden_states.shape
        attn_weights = None
        attn_weights_pre = None  # pre-competition weights (slot_competition only)
        prev_len = 0
        router_indices = None
        router_logits = None
        router_weights = None
        topics_selected = None

        topics = self.topics.expand(B, -1, -1)  # (B, N, D)

        # Stateless mode: drop prev_mem entirely ("no recurrence").
        if getattr(self, 'ignore_prev_mem', False):
            prev_mem = None

        # Build KV: optionally prepend previous memory tokens
        if prev_mem is not None:
            prev_mem = prev_mem.detach()
            prev_len = prev_mem.size(1)
            kv = torch.cat([prev_mem, hidden_states], dim=1)  # (B, K+T, D)
        else:
            kv = hidden_states

        # === Routing (runs once before iteration): select which topics participate ===
        _routing_dbg = False
        pooled_before_gate = None
        num_clamped = 0
        sel_logits = None
        if self.enable_routing:
            if not hasattr(self, '_routing_dbg_count'):
                self._routing_dbg_count = 0
            _routing_dbg = self._routing_dbg_count < 1

            # Select-Then-Attend: router picks k topics, then cross-attn
            # runs only on selected topics (saves N/k compute).
            router_input = router_kv if router_kv is not None else kv
            router_indices, router_logits = self.router(router_input)

            # Canonical ordering: sort so decoder sees consistent positions.
            router_indices_unsorted = router_indices
            router_indices, _ = router_indices.sort(dim=-1)

            if _routing_dbg:
                with torch.no_grad():
                    print(f"[Routing dbg #{self._routing_dbg_count}] "
                          f"topics_pool={list(topics.shape)}, kv={list(kv.shape)}")
                    print(f"  indices before sort: {router_indices_unsorted[0].tolist()}")
                    print(f"  indices after sort:  {router_indices[0].tolist()}")

            expanded_idx = router_indices.unsqueeze(-1).expand(-1, -1, D)
            topics_selected = torch.gather(topics, 1, expanded_idx)  # (B, k, D)

            if _routing_dbg:
                with torch.no_grad():
                    print(f"  topics_selected: shape={list(topics_selected.shape)}, "
                          f"norm={topics_selected.float().norm(dim=-1).mean().item():.4f}")

            # Hook to capture total gradient on router_logits (from ALL paths)
            if router_logits.requires_grad:
                def _router_logits_total_grad_hook(grad, _self=self):
                    with torch.no_grad():
                        g = grad.float()
                        _self._debug_router_logits_total_grad_norm = g.norm().item()
                        _self._debug_router_logits_total_grad_mean = g.abs().mean().item()
                    return grad
                router_logits.register_hook(_router_logits_total_grad_hook)

            # Sigmoid-based independent gating; scale by 2 so mean weight ≈ 1.0.
            sel_logits = torch.gather(router_logits, 1, router_indices)  # (B, k)
            router_weights_raw = 2.0 * torch.sigmoid(sel_logits)
            num_clamped = (router_weights_raw > 2.0).sum().item()
            router_weights = router_weights_raw.clamp(max=2.0)

            latents = topics_selected
        else:
            latents = topics

        # === Iterative Perceiver block (cross-attn + FFN + self-attn stack) ===
        # When num_iter_blocks=1 and num_self_attn_layers=0, this matches the
        # original single-pass behavior exactly. Higher values give true
        # Perceiver semantics with weight sharing across iterations.
        for iter_idx in range(self.num_iter_blocks):
            is_last_iter = iter_idx == (self.num_iter_blocks - 1)

            if return_attn_stats and is_last_iter:
                pooled, attn_weights, attn_weights_pre = self.cross_attn(
                    latents, kv, need_weights=True
                )
            else:
                pooled = self.cross_attn(latents, kv)

            if _routing_dbg and iter_idx == 0:
                with torch.no_grad():
                    print(f"  after xattn: pooled shape={list(pooled.shape)}, "
                          f"norm={pooled.float().norm(dim=-1).mean().item():.4f}")

            # MoE router-weight coupling: applied only on the first iteration so
            # scaling does not compound. Identical to single-pass behavior at R=1.
            if self.enable_routing and iter_idx == 0:
                pooled_before_gate = pooled
                pooled = router_weights.unsqueeze(-1) * pooled

                if _routing_dbg:
                    with torch.no_grad():
                        print(f"  MoE gate (sigmoid): sel_logits min={sel_logits.min().item():.4f}, "
                              f"max={sel_logits.max().item():.4f}, "
                              f"mean={sel_logits.mean().item():.4f}")
                        print(f"  router_weights: min={router_weights.min().item():.4f}, "
                              f"max={router_weights.max().item():.4f}, "
                              f"mean={router_weights.mean().item():.4f}, "
                              f"std={router_weights.std().item():.4f}, "
                              f"clamped={num_clamped}")
                        q_norm = topics_selected.float().norm(dim=-1).mean().item()
                        content_norm = pooled_before_gate.float().norm(dim=-1).mean().item()
                        g_xattn = self.cross_attn.gate.item() if self.cross_attn.gated else 1.0
                        print(f"  content_norm={content_norm:.4f}, "
                              f"Q_sel norm={q_norm:.4f}, "
                              f"g_xattn={g_xattn:.6f}")
                        content_frac = 1.0 - abs(g_xattn)
                        print(f"  content_frac={content_frac:.4f} (topic_frac={abs(g_xattn):.4f})")
                        pooled_norm = pooled.float().norm(dim=-1).mean().item()
                        print(f"  final pooled: shape={list(pooled.shape)}, "
                              f"norm={pooled_norm:.4f}")
                    self._routing_dbg_count += 1

            # FFN with residual
            if self.no_ffn:
                latents = pooled
            else:
                latents = self.mlp(pooled)  # residual FFN: pooled + FFN(pooled)

            # Self-attention stack: latents coordinate with each other
            if self.self_attn_stack is not None:
                for layer in self.self_attn_stack:
                    latents = layer(latents)

        mem = latents

        if not return_attn_stats:
            return mem

        # ---- collect diagnostic stats ----
        stats = {}

        # Attention mass on previous segment's memory (cross-segment attention)
        if prev_len > 0 and attn_weights is not None:
            past_mass = attn_weights[:, :, :, :prev_len].sum(dim=-1).mean()
            stats["attn_past_mass"] = past_mass.detach()
        else:
            stats["attn_past_mass"] = hidden_states.new_tensor(0.0)

        # Cross-attention gate (only for attention pooling with gating)
        if self.cross_attn.gated:
            with torch.no_grad():
                stats["xattn_gate"] = self.cross_attn.gate.item()
                stats["xattn_raw_gate"] = self.cross_attn.raw_gate.item()
                stats["xattn_gate_max"] = self.cross_attn.gate_max

                # Content fraction: how much of the output diverged from the query.
                # With content-first blending, this should be high (~0.95 at gate=0.05).
                if self.enable_routing and topics_selected is not None:
                    ref_topics = topics_selected[0:1]
                else:
                    ref_topics = self.topics
                q_norms = ref_topics.float().norm(dim=-1).mean().item()
                delta = (pooled[0] - ref_topics[0]).float().norm(dim=-1).mean().item()
                stats["xattn_update_ratio"] = delta / max(q_norms, 1e-6)

        # Pre-MLP vs post-MLP: verify residual FFN preserves diversity
        with torch.no_grad():
            stats["pre_mlp_pooled"] = pooled.detach()
            pre_f = pooled[0].float()
            post_f = mem[0].float().detach()
            stats["pre_mlp_norm"] = pre_f.norm(dim=-1).mean().item()
            stats["post_mlp_norm"] = post_f.norm(dim=-1).mean().item()
            if not self.no_ffn:
                # Cosine between pre and post FFN — how much the FFN transforms the representation
                stats["ffn_cosine"] = F.cosine_similarity(pre_f, post_f, dim=-1).mean().item()
                # FFN contribution: ||FFN(x)|| / ||x + FFN(x)||
                ffn_contrib = (post_f - pre_f).norm(dim=-1)
                stats["ffn_contrib_norm"] = ffn_contrib.mean().item()
                stats["ffn_contrib_ratio"] = (ffn_contrib / post_f.norm(dim=-1).clamp(min=1e-6)).mean().item()

        # Attention entropy (only for attention pooling)
        if attn_weights is not None:
            with torch.no_grad():
                aw = attn_weights.float().clamp(min=1e-8)
                entropy = -(aw * aw.log()).sum(dim=-1)
                max_entropy = math.log(aw.size(-1))
                stats["attn_entropy_normalized"] = (
                    entropy.mean() / max(max_entropy, 1e-8)
                ).item()

                # --- Allocation diagnostics ---
                M_q = aw.size(2)
                diag_mask = ~torch.eye(M_q, dtype=torch.bool, device=aw.device)

                def _alloc_metrics(w, prefix):
                    """Compute alloc_cosine and effective_rank for weight tensor (B,H,M,T)."""
                    # Head-averaged metrics
                    w_avg = w.mean(dim=1)  # (B, M, T)
                    w_normed = F.normalize(w_avg[0], dim=-1)  # (M, T)
                    sim = w_normed @ w_normed.T
                    stats[f"{prefix}_alloc_cosine"] = sim[diag_mask].mean().item()
                    sv = torch.linalg.svdvals(w_avg[0])
                    sv_n = sv / (sv.sum() + 1e-8)
                    sv_n = sv_n.clamp(min=1e-8)
                    stats[f"{prefix}effective_rank"] = torch.exp(
                        -(sv_n * sv_n.log()).sum()
                    ).item()

                    # Per-head metrics
                    H = w.size(1)
                    per_head_cos = []
                    per_head_rank = []
                    for h in range(H):
                        wh = w[0, h]  # (M, T)
                        wh_n = F.normalize(wh, dim=-1)
                        sim_h = wh_n @ wh_n.T
                        per_head_cos.append(sim_h[diag_mask].mean().item())
                        sv_h = torch.linalg.svdvals(wh)
                        sv_h = sv_h / (sv_h.sum() + 1e-8)
                        sv_h = sv_h.clamp(min=1e-8)
                        per_head_rank.append(torch.exp(-(sv_h * sv_h.log()).sum()).item())
                    cos_t = torch.tensor(per_head_cos)
                    rank_t = torch.tensor(per_head_rank)
                    stats[f"{prefix}_alloc_cos_per_head_min"] = cos_t.min().item()
                    stats[f"{prefix}_alloc_cos_per_head_max"] = cos_t.max().item()
                    stats[f"{prefix}_alloc_cos_per_head_std"] = cos_t.std().item()
                    stats[f"{prefix}_eff_rank_per_head_min"] = rank_t.min().item()
                    stats[f"{prefix}_eff_rank_per_head_max"] = rank_t.max().item()

                # Post-competition (or standard) attention metrics
                _alloc_metrics(aw, "attn")

                # Pre-competition metrics (only with slot_competition)
                if attn_weights_pre is not None:
                    _alloc_metrics(attn_weights_pre.float().clamp(min=1e-8), "attn_pre")

                # Per-slot mass: total attention mass assigned to each slot.
                # After slot competition, rows no longer sum to 1.
                # Low mass = near-dead slot.
                slot_mass = aw.sum(dim=-1).mean(dim=1)[0]  # (M,) avg over heads
                stats["slot_mass_mean"] = slot_mass.mean().item()
                stats["slot_mass_std"] = slot_mass.std().item()
                stats["slot_mass_min"] = slot_mass.min().item()
                stats["slot_mass_max"] = slot_mass.max().item()
                dead_thresh = 0.1 * slot_mass.mean().item()
                stats["slot_near_dead_frac"] = (
                    (slot_mass < dead_thresh).float().mean().item()
                )

                # --- Query specialization & text coverage diagnostics ---
                # Focus on text positions only (strip prev-memory prefix).
                # aw shape: (B, H, M, T_total) where T_total = prev_len + T_text
                aw_text = aw[:, :, :, prev_len:]  # (B, H, M, T_text)
                w_ha = aw_text.mean(dim=1)[0]     # (M, T_text) head-averaged
                T_text = w_ha.size(-1)

                if T_text > 0:
                    positions = torch.arange(
                        T_text, dtype=w_ha.dtype, device=w_ha.device
                    )

                    # -- 1. Coverage: per-position attention across topics --
                    cov_max, _ = w_ha.max(dim=0)   # (T_text,) best-query per pos
                    cov_mean = w_ha.mean(dim=0)     # (T_text,) avg query mass per pos
                    stats["cov_max_mean"] = cov_max.mean().item()
                    stats["cov_max_min"] = cov_max.min().item()

                    # Dead-zone fraction: positions where even the best query
                    # assigns < 10% of the uniform baseline (1/T_text).
                    uniform_level = 1.0 / T_text
                    stats["cov_dead_frac"] = (
                        (cov_max < 0.1 * uniform_level).float().mean().item()
                    )

                    # Gini coefficient for balanced coverage (0=uniform, 1=all on one pos)
                    sorted_cov, _ = cov_mean.sort()
                    n = sorted_cov.size(0)
                    idx = torch.arange(
                        1, n + 1, dtype=sorted_cov.dtype, device=sorted_cov.device
                    )
                    cov_sum = sorted_cov.sum().clamp(min=1e-8)
                    stats["cov_gini"] = (
                        (2 * (idx * sorted_cov).sum() / (n * cov_sum))
                        - (n + 1) / n
                    ).item()

                    # -- 2. Per-query specialization: center of mass & support width --
                    # Center of mass: weighted-average position per query
                    com = (w_ha * positions.unsqueeze(0)).sum(dim=-1)  # (M,)
                    stats["topic_com_std"] = com.std().item()
                    stats["topic_com_range"] = (com.max() - com.min()).item()
                    # Normalize by T so metrics are comparable across lengths
                    stats["topic_com_std_norm"] = (com.std() / max(T_text, 1)).item()

                    # Support width: std of attention distribution per query
                    var = (w_ha * (positions.unsqueeze(0) - com.unsqueeze(-1)) ** 2).sum(dim=-1)
                    support_w = var.clamp(min=0).sqrt()  # (M,)
                    stats["topic_support_width_mean"] = support_w.mean().item()
                    stats["topic_support_width_std"] = support_w.std().item()
                    stats["topic_support_width_mean_norm"] = (
                        support_w.mean() / max(T_text, 1)
                    ).item()

                    # -- 3. Per-query sharpness: entropy & top-k concentration --
                    per_q_ent = -(w_ha * w_ha.clamp(min=1e-8).log()).sum(dim=-1)  # (M,)
                    max_ent = math.log(T_text) if T_text > 1 else 1.0
                    stats["topic_entropy_mean"] = (per_q_ent.mean() / max_ent).item()
                    stats["topic_entropy_std"] = (per_q_ent.std() / max_ent).item()
                    stats["topic_entropy_min"] = (per_q_ent.min() / max_ent).item()
                    stats["topic_entropy_max"] = (per_q_ent.max() / max_ent).item()

                    # Top-k mass: fraction of each query's total text mass in top-k positions
                    row_mass = w_ha.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # (M, 1)
                    w_ha_normed = w_ha / row_mass  # normalize rows to sum=1
                    for topk in (1, 4, 8):
                        if topk <= T_text:
                            topk_vals, _ = w_ha_normed.topk(topk, dim=-1)
                            topk_mass = topk_vals.sum(dim=-1)  # (M,)
                            stats[f"topic_top{topk}_mass_mean"] = topk_mass.mean().item()
                            stats[f"topic_top{topk}_mass_std"] = topk_mass.std().item()

        # Router diagnostics (only when routing is enabled)
        if self.enable_routing and router_logits is not None:
            stats["router_logits"] = router_logits  # (B, N) — kept for load balancing loss
            with torch.no_grad():
                # Router entropy (on softmax of full logits): high = diverse selection
                probs = F.softmax(router_logits.float(), dim=-1)
                ent = -(probs * probs.clamp(min=1e-8).log()).sum(dim=-1)
                max_ent = math.log(self.num_memory_tokens)
                stats["router_entropy_normalized"] = (
                    ent.mean() / max(max_ent, 1e-8)
                ).item()
                # Unique topics selected in this batch
                stats["router_unique_topics"] = router_indices.unique().numel()
                # MoE-style gate weights for selected topics
                if router_weights is not None:
                    stats["router_sel_weight_mean"] = router_weights.mean().item()
                    stats["router_sel_weight_std"] = router_weights.std().item()
                # Selected indices (first example in batch, for debugging)
                stats["router_selected_indices"] = router_indices[0].tolist()

        # --- Pipeline diversity diagnostics ---
        # Track where representation diversity dies along the compressor pipeline.
        # Stages: V → AV → out_proj → norm_match → blend → memory
        # Each stage reports effective rank and mean pairwise cosine.
        if attn_weights is not None:
            with torch.no_grad():
                def _diversity(X, prefix):
                    """X: (N, D) — compute eff_rank and mean pairwise cosine."""
                    X_f = X.float()
                    N_x = X_f.size(0)
                    if N_x < 2:
                        return
                    # Mean pairwise cosine
                    X_n = F.normalize(X_f, dim=-1)
                    sim = X_n @ X_n.T
                    mask = ~torch.eye(N_x, dtype=torch.bool, device=X.device)
                    stats[f"pipe_{prefix}_cos"] = sim[mask].mean().item()
                    # Effective rank (SVD-based Shannon entropy of singular values)
                    sv = torch.linalg.svdvals(X_f)
                    sv_n = sv / (sv.sum() + 1e-8)
                    sv_n = sv_n.clamp(min=1e-8)
                    stats[f"pipe_{prefix}_eff_rank"] = torch.exp(
                        -(sv_n * sv_n.log()).sum()
                    ).item()

                # Queries used for this segment
                if self.enable_routing and topics_selected is not None:
                    q_used = topics_selected[0:1]  # (1, M, D)
                else:
                    q_used = self.topics  # (1, M, D)

                M_q = q_used.size(1)
                xattn = self.cross_attn

                # Stage: raw_hs (backbone hidden states, before any learned transform)
                # This is the only stage independent of compressor weights,
                # so it gives an unconfounded measure of layer quality.
                _diversity(kv[0], "raw_hs")

                kv_normed = xattn.norm_kv(kv[0:1])  # (1, T, D)

                # Extract V projection weights from packed MHA
                W_q, W_k, W_v = xattn.attn.in_proj_weight.chunk(3, dim=0)
                b_v = None
                if xattn.attn.in_proj_bias is not None:
                    _, _, b_v = xattn.attn.in_proj_bias.chunk(3, dim=0)

                H = xattn.num_heads
                d_head = xattn.d_model // H

                # Stage: topics (queries before any projection — input to cross-attn)
                _diversity(q_used[0], "topics")
                # Topic norms (per-slot ||q||) — exposes init scale collapse / explosion
                stats["topics_norm_mean"] = q_used[0].float().norm(dim=-1).mean().item()
                stats["topics_norm_std"]  = q_used[0].float().norm(dim=-1).std().item()

                # Stage: post-LN topics (input to W_Q after norm_q)
                q_normed = xattn.norm_q(q_used)  # (1, M, D)
                _diversity(q_normed[0], "Qln")

                # Stage: post-W_Q queries (per-head, what enters QK^T)
                if 'W_q' in dir() or W_q is not None:
                    Q_proj = F.linear(q_normed, W_q,
                                      None if xattn.attn.in_proj_bias is None
                                      else xattn.attn.in_proj_bias.chunk(3, dim=0)[0])  # (1, M, D)
                    _diversity(Q_proj[0], "Q")
                    stats["Q_norm_mean"] = Q_proj[0].float().norm(dim=-1).mean().item()
                    stats["Q_norm_std"]  = Q_proj[0].float().norm(dim=-1).std().item()

                    # Score variance — predicts softmax peakiness directly.
                    # Var(QK^T/sqrt(d)) ~ uniformity of attention.
                    Q_h = Q_proj.view(1, M_q, H, d_head).transpose(1, 2)  # (1,H,M,d_h)
                    K_proj = F.linear(kv_normed, W_k,
                                      None if xattn.attn.in_proj_bias is None
                                      else xattn.attn.in_proj_bias.chunk(3, dim=0)[1])  # (1,T,D)
                    K_h = K_proj.view(1, -1, H, d_head).transpose(1, 2)  # (1,H,T,d_h)
                    raw_scores = torch.matmul(Q_h.float(), K_h.float().transpose(-2, -1)) / math.sqrt(d_head)
                    stats["score_var"] = raw_scores.var().item()
                    stats["score_std"] = raw_scores.std().item()
                    stats["score_range"] = (raw_scores.max() - raw_scores.min()).item()

                # Stage: V (value vectors, T×D, before attention weighting)
                V = F.linear(kv_normed, W_v, b_v)  # (1, T, D)
                _diversity(V[0], "V")
                stats["V_norm_mean"] = V[0].float().norm(dim=-1).mean().item()

                # Stage: AV (attention-weighted values, M×D, before out_proj)
                V_h = V.view(1, -1, H, d_head).transpose(1, 2)  # (1, H, T, d_head)
                aw = attn_weights[0:1].float()  # (1, H, M, T)
                AV_h = torch.matmul(aw, V_h)  # (1, H, M, d_head)
                AV = AV_h.transpose(1, 2).contiguous().view(1, M_q, xattn.d_model)
                _diversity(AV[0], "AV")

                # Stage: after out_proj (mode-dependent)
                out_proj_AV = xattn._apply_out_proj(AV, 1, M_q, xattn.d_model)
                _diversity(out_proj_AV[0], "out_proj")

                # Stage: after norm-matching (rescale to query norm)
                q_norm = q_used.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                out_norm = out_proj_AV.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                scale = (q_norm / out_norm).clamp(max=1e4)
                norm_matched = out_proj_AV * scale
                _diversity(norm_matched[0], "norm_match")

                # Stage: after blend (content_first: g*topics + (1-g)*norm_matched)
                if xattn.gated:
                    g = xattn.gate
                    if xattn.blend_mode == "content_first":
                        blended = g * q_used + (1.0 - g) * norm_matched
                    elif xattn.blend_mode == "perturbation":
                        blended = q_used + g * norm_matched
                    elif xattn.blend_mode == "no_residual":
                        blended = norm_matched
                    else:
                        blended = norm_matched
                    _diversity(blended[0], "blend")

                # Stage: final memory (after router weights + FFN)
                _diversity(mem[0].detach(), "memory")

        return mem, stats


# ---------------------------------------------------------------------------
# Decompressor: reconstruct hidden states from memory tokens
# ---------------------------------------------------------------------------

class Decompressor(nn.Module):
    """Chunked decompressor.

    One shared MLP applied independently per slot, slot-only input (no cross-slot
    signal). No position embeddings.

        H_hat_i = reshape(MLP(m_i), C, D)
        MLP: LN(D) → Linear(D→D) → GELU → Linear(D→C*D)

    Discarded at inference — zero runtime cost.
    """

    def __init__(self, d_model: int, chunk_size: int = 4, input_dim: int = None, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size
        # Decoupled-memory mode: the (frozen) compressor may emit slots at a
        # different dim than this decoder's d_model. A bias-free in_proj lifts
        # slot_dim -> d_model; the rest of the decompressor runs in decoder space.
        # When input_dim == d_model (default) this is identity (no extra module).
        self.input_dim = input_dim if input_dim is not None else d_model
        self.in_proj = (nn.Linear(self.input_dim, d_model, bias=False)
                        if self.input_dim != d_model else None)

        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, chunk_size * d_model),
        )

        total_params = sum(p.numel() for p in self.parameters())
        print(f"[Decompressor] d_model={d_model}, input_dim={self.input_dim}, "
              f"chunk_size={chunk_size}, in_proj={'yes' if self.in_proj is not None else 'no'}, "
              f"total params: {total_params:,}")

    def forward(self, memory_tokens: torch.Tensor, target_len: int = None) -> torch.Tensor:
        """
        memory_tokens: (B, M, input_dim)
        target_len: ignored (kept for interface compatibility)
        returns: (B, M*chunk_size, d_model)
        """
        B, M, _ = memory_tokens.shape
        if self.in_proj is not None:
            memory_tokens = self.in_proj(memory_tokens)   # (B, M, d_model)
        out = self.mlp(memory_tokens)                     # (B, M, C*d_model)
        out = out.view(B, M * self.chunk_size, self.d_model)
        return out



# =====================================================================
# Chunked Gated Compressor
# =====================================================================

class SegmentCompressor(nn.Module):
    """Local chunked memory writer (no recurrence).

    For each segment:
        1. Split T tokens into M chunks of (T/M) tokens
        2. Pool each chunk → local summary z_i
        3. (optional) Mean-pool full segment → global summary z_global
        4. For each memory slot i:
             u_i = LN([z_i; z_global])   (z_global dropped if no_z_global=True)
             c_i = MLP_c(u_i)             — candidate content (or Linear if linear_writer)
             m_i = LN(c_i)                — stateless, no prev_mem, no gate

    This is the SeDeM "no recurrence" architecture: the writer never reads
    prev_mem, avoiding the feedback-loop collapse found in the 12-run
    compressor ablation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_memory_tokens: int = 32,
        bottleneck_dim: int = 3072,
        cross_slot_mixer: bool = True,
        slot_mixer_type: str = "linear",
        slot_mixer_attn_heads: int = 4,
        no_z_global: bool = False,
        slot_embedding: bool = False,
        linear_writer: bool = False,
        raw_chunks: bool = False,
        raw_chunk_size: int = 4,
        weighted_pool: bool = False,
        weighted_pool_d_attn: int = 256,
        attn_pool: bool = False,
        # optional soft-keep gate (ablation, off by default)
        soft_keep_gate: bool = False,
        **kwargs,
    ):
        super().__init__()
        D = output_dim
        M = num_memory_tokens
        self.input_dim = input_dim
        self.output_dim = D
        self.num_memory_tokens = M
        self.no_z_global = no_z_global
        self.slot_embedding_enabled = slot_embedding
        self.linear_writer = linear_writer
        self.raw_chunks = raw_chunks
        self.raw_chunk_size = raw_chunk_size
        self.weighted_pool = weighted_pool
        self.attn_pool = attn_pool
        self.slot_mixer_type = slot_mixer_type if cross_slot_mixer else "none"
        assert self.slot_mixer_type in ("none", "linear", "attn"), (
            f"slot_mixer_type must be 'none' | 'linear' | 'attn', "
            f"got '{self.slot_mixer_type}'"
        )
        assert not (weighted_pool and attn_pool), \
            "weighted_pool and attn_pool are mutually exclusive"

        # Input projection if dims differ
        if input_dim != D:
            self.input_proj = nn.Linear(input_dim, D, bias=False)
        else:
            self.input_proj = nn.Identity()

        # Slot-conditioned weighted pooling over chunk tokens
        if weighted_pool:
            D_a = weighted_pool_d_attn
            self.wp_W_x = nn.Linear(D, D_a, bias=False)
            self.wp_W_e = nn.Linear(D, D_a, bias=False)
            self.wp_b = nn.Parameter(torch.zeros(D_a))
            self.wp_v = nn.Linear(D_a, 1, bias=False)

        # Shared-query attention pooling over chunk tokens (slot-agnostic)
        # One learned query q in R^D, used for all slots.
        # z_i = sum_j softmax(q^T h_{i,j} / sqrt(D)) * h_{i,j}
        if attn_pool:
            self.ap_query = nn.Parameter(torch.randn(D) * 0.02)

        # Local input dimension: raw_chunk_size * D if raw_chunks, else D
        local_dim = raw_chunk_size * D if raw_chunks else D

        # Writer input: [z_local (+ z_global?)], no prev_mem ever.
        cat_dim = local_dim if no_z_global else local_dim + D

        # Input norm on the concatenation
        self.input_norm = nn.LayerNorm(cat_dim)

        # Candidate content: MLP or single Linear
        if linear_writer:
            self.candidate_linear = nn.Linear(cat_dim, D)
        else:
            self.candidate_mlp = nn.Sequential(
                nn.Linear(cat_dim, bottleneck_dim),
                nn.GELU(),
                nn.Linear(bottleneck_dim, D),
            )

        # Optional per-slot learned embedding for differentiation
        if slot_embedding:
            self.slot_emb = nn.Parameter(torch.randn(M, D) * 0.02)

        # Output norm
        self.out_norm = nn.LayerNorm(D)

        # Cross-slot mixer: how the M slots talk to each other after the writer.
        #   'none'   -> no mixing (slots independent)
        #   'linear' -> LN + Linear(M,M) residual, init-as-identity (legacy default,
        #               preserves current production behavior bit-for-bit)
        #   'attn'   -> LN + MultiheadAttention over slots, gated residual; gate
        #               starts at tanh(0)=0 so module is a pass-through at step 0
        self.cross_slot_mixer = None
        self.cross_slot_attn = None
        self.cross_slot_attn_norm = None
        self.cross_slot_attn_gate = None
        if self.slot_mixer_type == "linear":
            self.cross_slot_mixer = nn.Sequential(
                nn.LayerNorm(D),
                nn.Linear(M, M, bias=False),
            )
            # Init near-identity so mixer starts as pass-through
            nn.init.eye_(self.cross_slot_mixer[1].weight)
        # optional soft-keep gate per slot. Replaces m_i with
        #   m'_i = ρ_i · m_i + (1 - ρ_i) · e_null
        # ρ_i = sigmoid(w · LN(m_i)). At init: w=0 (so ρ ≈ 0.5) plus
        # LN(m_i) → bias toward open gate during early training; e_null=0.
        # The β-annealed budget loss + entropy loss + delayed L_null are applied
        # at the model level, not here.
        self.soft_keep_gate_enabled = bool(soft_keep_gate)
        self.soft_keep_norm = None
        self.soft_keep_w = None
        self.e_null = None
        if self.soft_keep_gate_enabled:
            self.soft_keep_norm = nn.LayerNorm(D)
            # Zero init for w: ρ_i = sigmoid(0) = 0.5 at step 0, neither
            # open nor closed. Combined with β_start=1.0 (no budget pressure
            # initially), the gate has a chance to learn before β closes it.
            self.soft_keep_w = nn.Parameter(torch.zeros(D))
            # e_null = 0 at init (a safe pass-through value through LN).
            # L_null regularizer keeps it small later in training.
            self.e_null = nn.Parameter(torch.zeros(D))

        if self.slot_mixer_type == "attn":
            self.cross_slot_attn_norm = nn.LayerNorm(D)
            self.cross_slot_attn = nn.MultiheadAttention(
                embed_dim=D, num_heads=slot_mixer_attn_heads,
                batch_first=True, dropout=0.0,
            )
            # Belt-and-suspenders safety: even after the gate γ moves off zero,
            # the attention output O_s is multiplied by W_O. Initializing W_O
            # small (xavier with gain=0.01) keeps O_s norm small in the first
            # few hundred steps, so the warm-up is gradual on both axes.
            nn.init.xavier_uniform_(self.cross_slot_attn.out_proj.weight, gain=0.01)
            nn.init.zeros_(self.cross_slot_attn.out_proj.bias)
            # Tanh-gated residual, gate=0 at init -> exact pass-through.
            # Lets attn warm up gradually without destabilizing SeDeM at step 0.
            self.cross_slot_attn_gate = nn.Parameter(torch.zeros(1))

        total_params = sum(p.numel() for p in self.parameters())
        print(f"[SegmentCompressor] D={D}, M={M}, "
              f"bottleneck={bottleneck_dim}, "
              f"cross_slot_mixer={self.slot_mixer_type}"
              + (f"(heads={slot_mixer_attn_heads})" if self.slot_mixer_type == "attn" else "") +
              f", no_z_global={no_z_global}, slot_embedding={slot_embedding}, "
              f"linear_writer={linear_writer}, "
              f"raw_chunks={raw_chunks}" + (f"(size={raw_chunk_size})" if raw_chunks else "") +
              f", weighted_pool={weighted_pool}" + (f"(d_a={weighted_pool_d_attn})" if weighted_pool else "") +
              f", attn_pool={attn_pool}"
              f", soft_keep_gate={self.soft_keep_gate_enabled}"
              f", total params: {total_params:,}")

    def forward(self, segment_hidden_states, prev_mem=None,
                return_attn_stats=False, verbose=False, seg_idx=None, **kwargs):
        """
        segment_hidden_states: (B, T, input_dim)
        prev_mem: accepted for caller-signature compatibility; ignored (stateless).
        returns: (B, M, D)
        """
        B, T, _ = segment_hidden_states.shape
        M = self.num_memory_tokens
        D = self.output_dim

        # Project input
        seg = self.input_proj(segment_hidden_states)  # (B, T, D)

        # Step 1: chunk segment into M groups
        chunk_size = T // M
        if T % M != 0:
            # Pad to make evenly divisible
            pad_len = M - (T % M)
            seg = F.pad(seg, (0, 0, 0, pad_len))  # pad seq dim
            chunk_size = seg.size(1) // M

        chunks = seg.view(B, M, chunk_size, D)  # (B, M, chunk_size, D)

        # Step 2: local summaries
        if self.weighted_pool and self.slot_embedding_enabled:
            # Slot-conditioned weighted pooling: s_{i,j} = v^T tanh(W_x x_{i,j} + W_e e_i + b)
            # chunks: (B, M, K, D), slot_emb: (M, D)
            x_proj = self.wp_W_x(chunks)              # (B, M, K, D_a)
            e_proj = self.wp_W_e(self.slot_emb)       # (M, D_a)
            scores = self.wp_v(torch.tanh(x_proj + e_proj[None, :, None, :] + self.wp_b))  # (B, M, K, 1)
            attn_weights = torch.softmax(scores, dim=2)  # (B, M, K, 1)
            z_local = (attn_weights * chunks).sum(dim=2)  # (B, M, D)
            # slot_emb already incorporated via attention; add it to z_local too for the writer
            z_local = z_local + self.slot_emb         # (M, D) broadcasts over B
        elif self.attn_pool:
            # Shared-query attention pooling (slot-agnostic):
            #   s_{i,j} = q^T h_{i,j} / sqrt(D)
            #   alpha_{i,j} = softmax_j(s_{i,j})
            #   z_i = sum_j alpha_{i,j} * h_{i,j}
            # chunks: (B, M, K, D), ap_query: (D,) cast to chunks dtype for bf16 safety.
            scale = D ** -0.5
            q = self.ap_query.to(chunks.dtype)
            scores = torch.einsum("bmkd,d->bmk", chunks, q) * scale  # (B, M, K)
            attn_weights = torch.softmax(scores, dim=-1)             # (B, M, K)
            z_local = torch.einsum("bmk,bmkd->bmd", attn_weights, chunks)  # (B, M, D)
            if self.slot_embedding_enabled:
                z_local = z_local + self.slot_emb    # (M, D) broadcasts over B
        elif self.raw_chunks:
            # Raw chunks: pad/truncate to fixed size, then flatten
            C = self.raw_chunk_size
            if chunk_size >= C:
                z_local = chunks[:, :, :C, :].reshape(B, M, C * D)  # (B, M, C*D)
            else:
                # Pad chunks shorter than raw_chunk_size
                pad = torch.zeros(B, M, C - chunk_size, D, device=seg.device, dtype=seg.dtype)
                z_local = torch.cat([chunks, pad], dim=2).reshape(B, M, C * D)
            # Optional: per-slot embedding
            if self.slot_embedding_enabled:
                z_local = z_local + self.slot_emb.repeat(1, self.raw_chunk_size)  # (M, C*D)
        else:
            z_local = chunks.mean(dim=2)             # (B, M, D)
            # Optional: per-slot embedding for differentiation
            if self.slot_embedding_enabled:
                z_local = z_local + self.slot_emb    # (M, D) broadcasts over B

        # Step 3: build writer input (no prev_mem — stateless) and compute candidate
        parts = [z_local]
        if not self.no_z_global:
            z_global = seg.mean(dim=1, keepdim=True).expand(-1, M, -1)  # (B, M, D)
            parts.append(z_global)
        u = torch.cat(parts, dim=-1)
        u = self.input_norm(u)

        # Candidate content
        if self.linear_writer:
            c = self.candidate_linear(u)          # (B, M, D)
        else:
            c = self.candidate_mlp(u)             # (B, M, D)

        # State: m_i = LN(c_i) — no recurrence, no gate, no prev_mem
        pre_norm = c
        new_mem = self.out_norm(pre_norm)         # (B, M, D)

        # Step 4: optional cross-slot mixer
        pre_mixer = new_mem
        if self.slot_mixer_type == "linear" and self.cross_slot_mixer is not None:
            # Apply norm, then linear across slot dim: (B, D, M) → (B, D, M)
            normed = self.cross_slot_mixer[0](new_mem)          # (B, M, D)
            mixed = self.cross_slot_mixer[1](
                normed.transpose(1, 2)                          # (B, D, M)
            ).transpose(1, 2)                                   # (B, M, D)
            new_mem = new_mem + mixed  # residual
        if self.slot_mixer_type == "attn" and self.cross_slot_attn is not None:
            # Self-attention over the M slot axis, content-conditional cross-slot
            # communication. Residual gate starts at tanh(0)=0 so this is a strict
            # pass-through at step 0; gradients warm up gradually.
            normed = self.cross_slot_attn_norm(new_mem)         # (B, M, D)
            # On verbose steps, request attention weights so we can log entropy
            # and diagonal mass. Costs ~5% extra time on the verbose step only.
            _need_w = bool(verbose)
            attn_out, attn_weights = self.cross_slot_attn(
                normed, normed, normed,
                need_weights=_need_w, average_attn_weights=True,
            )                                                   # (B, M, D), (B, M, M) | None
            gate_val = torch.tanh(self.cross_slot_attn_gate)    # scalar in (-1, 1)
            new_mem = new_mem + gate_val * attn_out
            # Diagnostics: surface attention shape only on verbose steps.
            if attn_weights is not None:
                with torch.no_grad():
                    aw = attn_weights.float()                   # (B, M, M)
                    # Per-row entropy averaged over rows + batch
                    ent = -(aw * (aw + 1e-10).log()).sum(dim=-1).mean().item()
                    max_ent = float(torch.tensor(aw.size(-1)).log().item())
                    # Diagonal mass = 1/M * sum_i A[i,i]
                    diag_mass = aw.diagonal(dim1=-2, dim2=-1).mean().item()
                    self._last_attn_entropy = ent
                    self._last_attn_entropy_norm = ent / (max_ent + 1e-8)
                    self._last_attn_diag_mass = diag_mass

        # ---- optional soft-keep gate ----
        # m'_i = ρ_i · m_i + (1 - ρ_i) · e_null
        # ρ_i = sigmoid(w · LN(m_i)). Operates on the M-axis only; segments
        # remain stateless. ρ flows into stats so the model-level forward can
        # compute L_budget / L_entropy / evidence-AUC diagnostics.
        rho = None
        if self.soft_keep_gate_enabled:
            normed_m = self.soft_keep_norm(new_mem.to(self.soft_keep_norm.weight.dtype))
            rho_logits = (normed_m * self.soft_keep_w).sum(dim=-1)   # (B, M)
            rho = torch.sigmoid(rho_logits)                          # (B, M) ∈ [0, 1]
            # Broadcast ρ over D and interpolate against e_null
            mixed = (
                rho.unsqueeze(-1) * new_mem
                + (1.0 - rho.unsqueeze(-1)) * self.e_null.to(new_mem.dtype)
            )
            new_mem = mixed

        # ---- Per-step diagnostics ----
        if verbose:
            with torch.no_grad():
                _tag = f"seg={seg_idx}" if seg_idx is not None else ""

                def _pw_cos(x):
                    """Pairwise cosine of (M, D) matrix, returns mean off-diagonal."""
                    n = F.normalize(x, dim=-1)
                    cm = n @ n.T
                    mask = ~torch.eye(cm.size(0), dtype=torch.bool, device=cm.device)
                    return cm[mask].mean().item()

                def _erank(x):
                    """Effective rank of (M, D) matrix."""
                    sigma = torch.linalg.svdvals(x.float())
                    sigma = sigma[sigma > 1e-10]
                    if sigma.numel() == 0:
                        return 0.0
                    p = sigma / sigma.sum()
                    return (-(p * p.log()).sum()).exp().item()

                zl = z_local[0]        # (M, D) or (M, C*D)
                pn = pre_norm[0]       # (M, D) — before LayerNorm
                nm = pre_mixer[0]      # (M, D) — after LayerNorm, before mixer
                fm = new_mem[0]        # (M, D) — final output

                zl_pw = _pw_cos(zl)
                if not self.no_z_global:
                    zl_zg_cos = F.cosine_similarity(zl, z_global[0], dim=-1).mean().item()
                else:
                    zl_zg_cos = float('nan')

                c_pw = _pw_cos(c[0])
                pn_pw = _pw_cos(pn)
                nm_pw = _pw_cos(nm)
                fm_pw = _pw_cos(fm)

                zl_er = _erank(zl)
                pn_er = _erank(pn)
                nm_er = _erank(nm)
                fm_er = _erank(fm)

                _flags = []
                if self.no_z_global: _flags.append("no_global_context")
                if self.slot_embedding_enabled: _flags.append("slot_position_embedding")
                if self.linear_writer: _flags.append("linear_projection")
                if self.raw_chunks: _flags.append(f"raw_chunks({self.raw_chunk_size})")
                _flag_str = f" [{','.join(_flags)}]" if _flags else ""
                print(f"    compressor_diag {_tag}{_flag_str}:")
                _global_part = "" if self.no_z_global else f", z_local↔z_global cos={zl_zg_cos}"
                print(f"      inputs: z_local(pw_cos={zl_pw:.4f}, erank={zl_er:.2f}){_global_part}")
                print(f"      candidate pw_cos={c_pw:.4f}")
                print(f"      diversity: pre_norm pw_cos={pn_pw:.4f}→post_LN pw_cos={nm_pw:.4f}"
                      f"→post_mixer pw_cos={fm_pw:.4f}")
                print(f"      erank: pre_norm={pn_er:.2f}→post_LN={nm_er:.2f}→post_mixer={fm_er:.2f}")

        # Stats for logging
        stats = {}
        if self.slot_mixer_type == "attn" and self.cross_slot_attn_gate is not None:
            # Effective gate = tanh(raw); 0 at init, drifts in (-1, 1) during training.
            # If this stays near 0, the model declined to use the attn mixer.
            stats["cross_slot_mixer_gate"] = torch.tanh(
                self.cross_slot_attn_gate.detach()
            ).item()
            # Attention-shape diagnostics (only populated on verbose steps;
            # otherwise fall back to last seen value).
            if hasattr(self, '_last_attn_entropy'):
                stats["cross_slot_attn_entropy"] = self._last_attn_entropy
                stats["cross_slot_attn_entropy_norm"] = self._last_attn_entropy_norm
                stats["cross_slot_attn_diag_mass"] = self._last_attn_diag_mass
        # soft-keep: pass ρ tensor through stats so the model-level forward can
        # compute L_budget, L_entropy, mean ρ, evidence AUC, spatial coherence.
        if rho is not None:
            stats["soft_keep_rho"] = rho                     # (B, M) tensor, kept differentiable
        if return_attn_stats:
            return new_mem, stats
        return new_mem


# =====================================================================
# KV Injection: MemoryKVProjector (DEPRECATED — removed)
# =====================================================================
# The shared-projection approach (single W_k/W_v + per-layer bias) has been
# replaced by running memory tokens through the decoder's own per-layer
# K/V projections, matching the standard approach from 500xCompressor
# (Li et al., ACL 2025) and Activation Beacon (Zhang et al., ICLR 2025).
# See sedem.modeling.py, memory_injection_mode='kv_inject'.
# =====================================================================


def build_compressor(
    compressor_type: str,
    input_dim: int,
    output_dim: int,
    num_memory_tokens: int = 128,
    bottleneck_dim: int = 512,
    num_heads: int = 8,
    dropout: float = 0.0,
    gate_init: float = 0.0,
    gate_max: float = 0.5,
    blend_mode: str = "content_first",
    slot_competition: bool = False,
    out_proj_mode: str = "standard",
    qk_norm: bool = False,
    qk_norm_init_scale: float = 10.0,
    no_ffn: bool = False,
    no_z_global: bool = False,
    slot_embedding: bool = False,
    linear_writer: bool = False,
    raw_chunks: bool = False,
    raw_chunk_size: int = 4,
    weighted_pool: bool = False,
    weighted_pool_d_attn: int = 256,
    attn_pool: bool = False,
    slot_mixer_type: str = "linear",
    slot_mixer_attn_heads: int = 4,
    soft_keep_gate: bool = False,
    num_self_attn_layers: int = 0,
    num_iter_blocks: int = 1,
    gated: bool = True,
    ignore_prev_mem: bool = False,
    topics_init: str = "random",
    **kwargs,
) -> nn.Module:
    """
    Factory that returns the requested compressor.

    compressor_type:
        'learnable_query'   -> LearnableQueryCompressor
        'pooled_linear' -> SegmentCompressor

    When enable_routing=True, the LearnableQueryCompressor will use a QueryRouter
    to select top-k topics from the pool per segment (content-adaptive).
    """
    if compressor_type == "learnable_query":
        return LearnableQueryCompressor(
            input_dim=input_dim,
            output_dim=output_dim,
            num_memory_tokens=num_memory_tokens,
            bottleneck_dim=bottleneck_dim,
            num_heads=num_heads,
            dropout=dropout,
            gate_init=gate_init,
            gate_max=gate_max,
            blend_mode=blend_mode,
            slot_competition=slot_competition,
            out_proj_mode=out_proj_mode,
            qk_norm=qk_norm,
            qk_norm_init_scale=qk_norm_init_scale,
            no_ffn=no_ffn,
            num_self_attn_layers=num_self_attn_layers,
            num_iter_blocks=num_iter_blocks,
            gated=gated,
            ignore_prev_mem=ignore_prev_mem,
            topics_init=topics_init,
        )
    elif compressor_type == "pooled_linear":
        return SegmentCompressor(
            input_dim=input_dim,
            output_dim=output_dim,
            num_memory_tokens=num_memory_tokens,
            bottleneck_dim=bottleneck_dim,
            no_z_global=no_z_global,
            slot_embedding=slot_embedding,
            linear_writer=linear_writer,
            raw_chunks=raw_chunks,
            raw_chunk_size=raw_chunk_size,
            weighted_pool=weighted_pool,
            weighted_pool_d_attn=weighted_pool_d_attn,
            attn_pool=attn_pool,
            slot_mixer_type=slot_mixer_type,
            slot_mixer_attn_heads=slot_mixer_attn_heads,
            soft_keep_gate=soft_keep_gate,
        )
    else:
        raise ValueError(
            f"Unknown compressor_type='{compressor_type}'. "
            "Choose 'learnable_query' or 'pooled_linear'."
        )