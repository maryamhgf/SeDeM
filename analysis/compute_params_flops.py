#!/usr/bin/env python3
"""
Added parameters and inference FLOPs per query of SeDeM (App. A.9) on
Llama-3.2-1B and Llama-3.2-3B (same-model encoder = decoder).

Both quantities are HARDWARE-INDEPENDENT:
  - Added params  : pure architecture (does not depend on the GPU).
  - FLOPs/query   : theoretical compute (depends only on model + seq-len + batch).
FLOP/s (throughput) and latency ARE GPU-dependent and are NOT computed here.

Numbers are auditable: param totals below reproduce the published 1.24B / 3.21B
backbone sizes, which validates the configs.

Run:  python analysis/compute_params_flops.py
"""

# ---------------------------------------------------------------- backbone cfg
LLAMA = {
    "1B": dict(D=2048, L=16, n_heads=32, head_dim=64, kv_heads=8, I=8192, vocab=128256),
    "3B": dict(D=3072, L=28, n_heads=24, head_dim=128, kv_heads=8, I=8192, vocab=128256),
}

# ------------------------------------------------------------ method hyperparams
SEDEM = dict(lora_r=64, lora_targets=("q", "k", "v", "o"), memory_slots=32, rate=4,
            hs_layer=14, selector="late_interaction", d_key=256, selector_heads=4, mixer_M=32,
            chunk_size=4)

# ----------------------------------------------- FLOPs scenario (HotpotQA-like)
SCN = dict(N=1536,        # context tokens
           Q=30,          # question tokens
           A=16,          # generated answer tokens
           top_k=4)       # selected blocks (SeDeM); each block = memory_slots tokens


# ----------------------------------------------------------------- param helpers
def proj_dims(cfg):
    """(in,out) for each attention projection given GQA."""
    D = cfg["D"]
    qo = cfg["n_heads"] * cfg["head_dim"]      # = D for Llama-3.2
    kvo = cfg["kv_heads"] * cfg["head_dim"]
    return {"q": (D, qo), "k": (D, kvo), "v": (D, kvo), "o": (qo, D)}


def backbone_params(cfg):
    D, L, I, V = cfg["D"], cfg["L"], cfg["I"], cfg["vocab"]
    pd = proj_dims(cfg)
    attn = sum(i * o for i, o in pd.values())          # per layer
    mlp = 3 * D * I                                     # SwiGLU gate/up/down
    body = L * (attn + mlp)
    embed = V * D                                       # tied (counted once)
    return body, embed


def lora_params(cfg, r, targets):
    pd = proj_dims(cfg)
    per_layer = sum(r * (pd[t][0] + pd[t][1]) for t in targets)
    return cfg["L"] * per_layer


def ln(D):           # LayerNorm weight+bias
    return 2 * D


def sedem_added(cfg):
    D = cfg["D"]
    # compressor (linear_writer, no_z_global, same-model => input_proj=Identity)
    comp = ln(D) + (D * D + D) + ln(D) + ln(D) + SEDEM["mixer_M"] ** 2
    # selector
    if SEDEM["selector"] == "pooled":
        retr = 2 * (ln(D) + SEDEM["d_key"] * D)         # block + query encoders
        retr_name = "pooled-query selector (d_key=256)"
    else:
        retr = 2 * ln(D) + 2 * (D * D)                 # norm_q,norm_k + q_proj,k_proj
        retr_name = "late-interaction selector (4 heads)"
    lora = lora_params(cfg, SEDEM["lora_r"], SEDEM["lora_targets"])
    # decompressor: ACTIVE at inference in the use_selection path.
    #   decompress_all_segments(...) expands the top-K selected
    #   blocks into latent decoder tokens.  Decompressor.mlp =
    #   LN(D) -> Linear(D,D) -> GELU -> Linear(D, C*D)  => 5 D^2 + 7 D.
    decomp = ln(D) + (D * D + D) + (D * SEDEM["chunk_size"] * D + SEDEM["chunk_size"] * D)
    return {"compressor (linear)": comp,
            "decompressor": decomp,
            retr_name: retr,
            "LoRA q,k,v,o (r=64)": lora}


# ------------------------------------------------------------------- FLOPs model
# Matmul-dominant convention: a forward over `t` tokens through `frac` of the
# layers costs ~2 * t * (frac * body_params).  Attention score/value matmuls add
# ~4 * (frac*L) * t^2 * D and are listed separately.  lm_head adds 2*g*V*D for g
# generated tokens.  (Backward not included: inference only.)
def fwd_flops(cfg, tokens, frac_layers=1.0):
    body, _ = backbone_params(cfg)
    mm = 2 * tokens * body * frac_layers
    attn = 4 * (cfg["L"] * frac_layers) * (tokens ** 2) * cfg["D"]
    return mm + attn


def lmhead_flops(cfg, gen_tokens):
    _, embed = backbone_params(cfg)
    return 2 * gen_tokens * embed


def sedem_flops(cfg):
    D = cfg["D"]
    N, Q, A = SCN["N"], SCN["Q"], SCN["A"]
    m, c, k = SEDEM["memory_slots"], SEDEM["rate"], SCN["top_k"]
    frac = SEDEM["hs_layer"] / cfg["L"]              # encode stops at layer 14
    selected_mem = k * m                           # only top-k blocks prepended
    # encode: partial forward to hs_layer over context (no mem tokens appended)
    enc = fwd_flops(cfg, N, frac_layers=frac)
    # compressor: pooled chunks (N/c slots) through one D->D linear  (tiny)
    enc += 2 * (N // c) * (D * D)
    # select + decompress: score over Q+(N/c) tokens, then the decompressor
    #   expands ALL N/c slots (LN+Linear(D,D)+Linear(D,4D) ~ 5 D^2 MAC/slot).
    retr = 2 * (Q + N // c) * (D * D)             # late-interaction scoring
    retr += (N // c) * 2 * (5 * D * D)            # decompressor over all slots
    prefill = fwd_flops(cfg, selected_mem + Q, frac_layers=1.0)
    gen = A * fwd_flops(cfg, selected_mem + Q + A, frac_layers=1.0) / (selected_mem + Q + A)
    gen += lmhead_flops(cfg, A)
    return dict(encode=enc, select=retr, decode=prefill + gen,
                total=enc + retr + prefill + gen, prefix_tokens=selected_mem)


# ----------------------------------------------------------------------- report
def fmt_p(n):
    return f"{n/1e6:8.2f} M"


def fmt_f(n):
    return f"{n/1e9:9.2f} GFLOP"


def row(label, v):
    print(f"      {label:<26} | {fmt_p(v)}")


def main():
    for size in ("1B", "3B"):
        cfg = LLAMA[size]
        body, embed = backbone_params(cfg)
        bb = body + embed
        print("=" * 72)
        print(f"Llama-3.2-{size}: backbone = {bb/1e9:.3f} B "
              f"(body {body/1e9:.3f} B + embed {embed/1e9:.3f} B)")
        print("=" * 72)

        added = sedem_added(cfg)
        lora = added["LoRA q,k,v,o (r=64)"]
        comp = added["compressor (linear)"]
        decomp = added["decompressor"]
        sel = next(v for k, v in added.items() if "selector" in k)
        nonlora = comp + decomp + sel
        total = lora + nonlora

        print("\n  ADDED PARAMETERS  (all rows active at inference)")
        row("LoRA q,k,v,o (r=64)", lora)
        row("compressor", comp)
        row("decompressor", decomp)
        row("selector", sel)
        row("non-LoRA subtotal", nonlora)
        print(f"      {'-' * 40}")
        row("TOTAL (inference)", total)
        print(f"      {'':<26} | {total/bb*100:5.2f}% of backbone")

        print(f"\n  INFERENCE FLOPs / query  (N={SCN['N']}, Q={SCN['Q']}, "
              f"A={SCN['A']}, top_k={SCN['top_k']})")
        f = sedem_flops(cfg)
        print(f"  encode {fmt_f(f['encode'])} | select {fmt_f(f['select'])}"
              f" | decode {fmt_f(f['decode'])} | prefix {f['prefix_tokens']} tok")
        print(f"  TOTAL  {fmt_f(f['total'])}")
        print()


if __name__ == "__main__":
    main()