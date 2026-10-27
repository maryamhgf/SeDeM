import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
"""
Causal activation patching on frozen Llama backbone — HotpotQA-Distractor.

Scientific question
-------------------
Are the answer-relevant hidden states at layers {10,12,14,16} causally
necessary for answer prediction, and is this signal distributed across
multiple layers rather than concentrated in one?

Protocol (per example)
----------------------
1. CLEAN         — normal forward; save H^(Lk) at supporting-fact positions
2. CORRUPTED     — add Gaussian noise to embeddings at support positions
3. PATCHED-{S}   — corrupted forward, but restore clean H^(Lk) at layer k
                   for every k in set S

Recovery(S) = (NLL_corrupt − NLL_S) / (NLL_corrupt − NLL_clean)
  0 = no help, 1 = full recovery, >1 = improved beyond clean, <0 = hurt

Hook targets (HF Llama indexing)
---------------------------------
  hidden_states[0]  = embed_tokens output  → hook model.model.embed_tokens
  hidden_states[k]  = output of model.model.layers[k-1]
  → to intercept H^(L10), hook model.model.layers[9]; L12→layers[11]; etc.

Noise scale
-----------
  sigma = noise_std_multiplier × std(embed_activations at support positions)
  multiplier default 3.0 — large enough to shift representations well outside
  clean manifold while keeping token positions aligned.

Usage
-----
  CUDA_VISIBLE_DEVICES=0 python analysis/layer_causal_patching.py \\
      --model_name_or_path meta-llama/Llama-3.2-3B \\
      --max_samples 300 --noise_std_multiplier 3.0 \\
      --output_json outputs/layer_causal_patching_hotpot.json

  # smoke test (3 examples):
  CUDA_VISIBLE_DEVICES=0 python analysis/layer_causal_patching.py \\
      --max_samples 3 --output_json outputs/smoke_patching.json
"""

import argparse, json, math, os, sys, time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

DATASET_CACHE = "$HF_HOME"
RECOVERY_EPS  = 1e-3   # min |NLL_corrupt - NLL_clean| to include in recovery calc
EFF_CORRUPT_THRESH = 0.10  # min gap for "effective corruption" subset

# Patch sets run for each example.
# Keys used in output JSON correspond to these sets.
PATCH_SETS: Dict[str, frozenset] = {
    "L10":            frozenset({10}),
    "L12":            frozenset({12}),
    "L14":            frozenset({14}),
    "L16":            frozenset({16}),
    "cumul_10_12":    frozenset({10, 12}),
    "cumul_10_12_14": frozenset({10, 12, 14}),
    "multi":          frozenset({10, 12, 14, 16}),  # also = cumul_10_12_14_16
}


# ─────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────

def load_model(model_name_or_path: str, device: torch.device):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        cache_dir=DATASET_CACHE,
        device_map={"": device},
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=DATASET_CACHE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


# ─────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────

def build_prompt(ex: dict) -> Tuple[str, Dict[Tuple[str, int], Tuple[int, int]], int]:
    """
    Build full prompt string with character-level sentence span tracking.

    Returns
    -------
    prompt_str       : complete string  "Context:\\n[T]...\\nQuestion: q\\nAnswer: a"
    sent_spans       : {(title, sent_id): (char_start, char_end)} in prompt_str
    answer_char_start: char index of first character AFTER "Answer:" (incl. space)
    """
    context = ex["context"]
    titles, sents_list = context["title"], context["sentences"]
    question, answer = ex["question"], ex["answer"]

    sent_spans: Dict[Tuple[str, int], Tuple[int, int]] = {}
    buf = "Context:\n"

    for title, sents in zip(titles, sents_list):
        buf += f"[{title}]"
        for i, sent in enumerate(sents):
            # Ensure exactly one space before each sentence
            sep = "" if sent.startswith(" ") else " "
            sent_start = len(buf)
            buf += sep + sent
            sent_end = len(buf)
            sent_spans[(title, i)] = (sent_start, sent_end)
        buf += "\n"

    buf += f"\nQuestion: {question}\nAnswer:"
    answer_char_start = len(buf)   # position of the space (or first answer char)
    buf += f" {answer}"

    return buf, sent_spans, answer_char_start


# ─────────────────────────────────────────────────────────────
# Token alignment
# ─────────────────────────────────────────────────────────────

def make_masks(
    prompt_str: str,
    sent_spans: Dict[Tuple[str, int], Tuple[int, int]],
    sf_set: set,            # {(title, sent_id), ...}
    offset_mapping: list,   # list of (char_start, char_end) per token
    answer_char_start: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns three boolean masks over token positions:
      support_mask  — supporting-fact tokens (context, in sf_set)
      bg_mask       — context tokens NOT in support
      answer_mask   — gold-answer tokens (used for NLL)
    """
    T = len(offset_mapping)
    support_mask = np.zeros(T, dtype=bool)
    bg_mask      = np.zeros(T, dtype=bool)
    answer_mask  = np.zeros(T, dtype=bool)

    # Character span of the context block (before "\nQuestion:")
    q_pos = prompt_str.rfind("\nQuestion:")
    context_end_char = q_pos if q_pos >= 0 else answer_char_start

    # Support sentence char spans
    support_spans: List[Tuple[int, int]] = []
    for key in sf_set:
        if key in sent_spans:
            support_spans.append(sent_spans[key])

    for t, (cs, ce) in enumerate(offset_mapping):
        if cs == ce:   # special / padding token
            continue
        in_context = ce <= context_end_char
        in_answer  = cs >= answer_char_start

        if in_answer:
            answer_mask[t] = True
        elif in_context:
            in_sup = any(cs < sup_e and ce > sup_s for sup_s, sup_e in support_spans)
            if in_sup:
                support_mask[t] = True
            else:
                bg_mask[t] = True

    return support_mask, bg_mask, answer_mask


# ─────────────────────────────────────────────────────────────
# NLL helper
# ─────────────────────────────────────────────────────────────

def answer_nll(logits: torch.Tensor, input_ids: torch.Tensor,
               answer_mask: np.ndarray) -> float:
    """
    Cross-entropy NLL per token at answer positions.

    logits   : (1, T, V)
    input_ids: (1, T)
    answer_mask: bool (T,) — positions whose token is part of the answer

    For each answer position t, the prediction comes from logits[t-1].
    Equivalently: targets = input_ids[1:], logits = logits[:-1], restricted
    to positions where the SHIFTED answer_mask (answer_mask[1:]) is True.
    """
    T = input_ids.shape[1]
    if answer_mask.sum() == 0:
        return float("nan")

    shift_logits = logits[0, :-1, :].float()   # (T-1, V)
    shift_labels = input_ids[0, 1:]            # (T-1,)
    # answer_mask[t] is True for the answer token at position t;
    # its prediction is at logits[t-1], i.e. shifted by -1 in the shift view
    shifted_mask = torch.tensor(answer_mask[1:], dtype=torch.bool,
                                device=shift_labels.device)   # (T-1,)

    if shifted_mask.sum() == 0:
        return float("nan")

    loss = F.cross_entropy(shift_logits[shifted_mask],
                           shift_labels[shifted_mask],
                           reduction="mean")
    return loss.item()


# ─────────────────────────────────────────────────────────────
# Forward passes with hooks
# ─────────────────────────────────────────────────────────────

def _layer_hook_idx(layer: int) -> int:
    """hidden_states[k] = output of model.model.layers[k-1]"""
    return layer - 1


@torch.no_grad()
def run_clean(
    model,
    input_ids: torch.Tensor,
    support_idx: np.ndarray,     # integer indices of support token positions
    layers: List[int],
    answer_mask: np.ndarray,
) -> Tuple[float, Dict[int, torch.Tensor]]:
    """
    Normal forward. Captures H^(Lk)[support_idx] for each k in layers.

    Returns (nll_clean, clean_hs) where clean_hs[k] is float32 CPU tensor (N_sup, D).
    """
    clean_hs: Dict[int, torch.Tensor] = {}
    hooks = []

    for k in layers:
        def _make_save(lk):
            def _h(module, inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                clean_hs[lk] = hs[0, support_idx, :].float().detach().cpu()
            return _h
        hooks.append(
            model.model.layers[_layer_hook_idx(k)].register_forward_hook(_make_save(k))
        )

    out = model(input_ids=input_ids, use_cache=False)
    nll = answer_nll(out.logits, input_ids, answer_mask)
    for h in hooks:
        h.remove()
    return nll, clean_hs


@torch.no_grad()
def _compute_noise_std(model, input_ids: torch.Tensor,
                       support_idx: np.ndarray, multiplier: float) -> float:
    """
    sigma = multiplier × std(embed_activations at support positions)

    We run embed_tokens once to measure the feature-level std at the
    support positions, then scale accordingly. This anchors noise magnitude
    to the actual activation scale of each example.
    """
    embed = model.model.embed_tokens(input_ids)  # (1, T, D)
    support_embed = embed[0, support_idx, :].float()
    std = support_embed.std().item()
    if std < 1e-6:
        std = 1.0
    return multiplier * std


@torch.no_grad()
def run_corrupted(
    model,
    input_ids: torch.Tensor,
    support_idx: np.ndarray,
    noise_std: float,
    answer_mask: np.ndarray,
    seed: int = 0,
) -> float:
    """
    Add Gaussian noise (N(0, sigma²)) to embed_tokens output at support
    positions, then run full forward. Returns answer NLL.
    """
    torch.manual_seed(seed)
    N_sup = len(support_idx)
    device = input_ids.device

    def _corrupt(module, inp, out):
        noisy = out.clone()
        D = out.shape[-1]
        noise = torch.randn(N_sup, D, dtype=out.dtype, device=device) * noise_std
        noisy[0, support_idx, :] += noise
        return noisy

    h = model.model.embed_tokens.register_forward_hook(_corrupt)
    out = model(input_ids=input_ids, use_cache=False)
    nll = answer_nll(out.logits, input_ids, answer_mask)
    h.remove()
    return nll


@torch.no_grad()
def run_patched(
    model,
    input_ids: torch.Tensor,
    support_idx: np.ndarray,
    noise_std: float,
    layers_to_patch: frozenset,
    clean_hs: Dict[int, torch.Tensor],
    answer_mask: np.ndarray,
    seed: int = 0,
) -> float:
    """
    Corrupted forward with H^(Lk)[support_idx] restored to clean values
    for every k in layers_to_patch.

    Intervention order: embedding corruption is applied first; then at each
    patched layer k, the support positions are overwritten with clean states
    before they continue to layer k+1. Patching at multiple layers is
    additive and consistent — each restores the state to what the clean
    forward would have produced at that layer.
    """
    torch.manual_seed(seed)
    N_sup = len(support_idx)
    device = input_ids.device
    hooks = []

    # 1. Embedding corruption
    def _corrupt(module, inp, out):
        noisy = out.clone()
        D = out.shape[-1]
        noise = torch.randn(N_sup, D, dtype=out.dtype, device=device) * noise_std
        noisy[0, support_idx, :] += noise
        return noisy
    hooks.append(model.model.embed_tokens.register_forward_hook(_corrupt))

    # 2. Layer patches
    for k in layers_to_patch:
        saved = clean_hs[k].to(device)
        def _make_patch(saved_H):
            def _h(module, inp, out):
                if isinstance(out, tuple):
                    hs = out[0].clone()
                    hs[0, support_idx, :] = saved_H.to(hs.dtype)
                    return (hs,) + out[1:]
                else:
                    hs = out.clone()
                    hs[0, support_idx, :] = saved_H.to(hs.dtype)
                    return hs
            return _h
        hooks.append(
            model.model.layers[_layer_hook_idx(k)].register_forward_hook(
                _make_patch(saved)
            )
        )

    out = model(input_ids=input_ids, use_cache=False)
    nll = answer_nll(out.logits, input_ids, answer_mask)
    for h in hooks:
        h.remove()
    return nll


# ─────────────────────────────────────────────────────────────
# Per-example processing
# ─────────────────────────────────────────────────────────────

def process_example(
    ex: dict,
    ex_idx: int,
    model,
    tok,
    layers: List[int],
    noise_std_multiplier: float,
    max_length: int,
    run_background_control: bool,
    bg_rng: np.random.Generator,
) -> dict:
    """
    Returns a result dict with all NLL values and recovery fractions.
    On any skip condition, returns {"idx": ex_idx, "skip_reason": "..."}.
    """
    answer = ex.get("answer", "")
    if not answer or not answer.strip():
        return {"idx": ex_idx, "skip_reason": "empty_answer"}

    # ── Build prompt ──────────────────────────────────────────
    prompt_str, sent_spans, answer_char_start = build_prompt(ex)

    # ── Tokenize ──────────────────────────────────────────────
    enc = tok(
        prompt_str,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    input_ids = enc["input_ids"].to(next(model.parameters()).device)
    T = input_ids.shape[1]

    # Offset mapping is on CPU
    offset_mapping = enc["offset_mapping"][0].tolist()

    # ── Align masks ───────────────────────────────────────────
    sf = ex["supporting_facts"]
    sf_set = set(zip(sf["title"], sf["sent_id"]))

    support_mask, bg_mask, answer_mask = make_masks(
        prompt_str, sent_spans, sf_set, offset_mapping, answer_char_start
    )

    n_support = int(support_mask.sum())
    n_bg      = int(bg_mask.sum())
    n_answer  = int(answer_mask.sum())

    if n_support == 0:
        return {"idx": ex_idx, "skip_reason": "no_support_tokens_aligned"}
    if n_answer == 0:
        return {"idx": ex_idx, "skip_reason": "no_answer_tokens"}

    support_idx = np.where(support_mask)[0]

    # ── Noise scale ───────────────────────────────────────────
    noise_std = _compute_noise_std(model, input_ids, support_idx, noise_std_multiplier)

    # ── Clean run ─────────────────────────────────────────────
    nll_clean, clean_hs = run_clean(model, input_ids, support_idx, layers, answer_mask)
    if math.isnan(nll_clean):
        return {"idx": ex_idx, "skip_reason": "nll_clean_nan"}

    # ── Corrupted run ─────────────────────────────────────────
    nll_corrupt = run_corrupted(model, input_ids, support_idx, noise_std,
                                answer_mask, seed=ex_idx)

    gap = nll_corrupt - nll_clean

    def recovery(nll_patch):
        denom = gap
        if abs(denom) < RECOVERY_EPS:
            return None  # denominator too small; skip
        return (nll_corrupt - nll_patch) / denom

    result: dict = {
        "idx": ex_idx,
        "answer": answer,
        "n_support_tokens": n_support,
        "n_bg_tokens": n_bg,
        "n_answer_tokens": n_answer,
        "noise_std": round(float(noise_std), 4),
        "nll_clean": round(nll_clean, 5),
        "nll_corrupt": round(nll_corrupt, 5),
        "gap": round(gap, 5),
    }

    # ── Patched runs ──────────────────────────────────────────
    for cond_name, patch_set in PATCH_SETS.items():
        nll_p = run_patched(model, input_ids, support_idx, noise_std,
                            patch_set, clean_hs, answer_mask, seed=ex_idx)
        rec   = recovery(nll_p)
        result[f"nll_{cond_name}"]      = round(nll_p, 5)
        result[f"recovery_{cond_name}"] = round(rec, 5) if rec is not None else None

    # ── Background control ────────────────────────────────────
    if run_background_control and n_bg >= n_support:
        bg_idx = np.where(bg_mask)[0]
        bg_sample = bg_rng.choice(bg_idx, size=n_support, replace=False)

        bg_noise_std = _compute_noise_std(model, input_ids, bg_sample, noise_std_multiplier)

        nll_bg_corrupt = run_corrupted(model, input_ids, bg_sample, bg_noise_std,
                                       answer_mask, seed=ex_idx + 100000)
        bg_gap = nll_bg_corrupt - nll_clean
        result["nll_bg_corrupt"] = round(nll_bg_corrupt, 5)
        result["bg_gap"]         = round(bg_gap, 5)

        # For background control we need clean_hs at bg positions too
        _, clean_hs_bg = run_clean(model, input_ids, bg_sample, layers, answer_mask)

        def bg_recovery(nll_p):
            denom = bg_gap
            if abs(denom) < RECOVERY_EPS:
                return None
            return (nll_bg_corrupt - nll_p) / denom

        for cond_name, patch_set in PATCH_SETS.items():
            nll_p = run_patched(model, input_ids, bg_sample, bg_noise_std,
                                patch_set, clean_hs_bg, answer_mask, seed=ex_idx + 200000)
            rec   = bg_recovery(nll_p)
            result[f"bg_nll_{cond_name}"]      = round(nll_p, 5)
            result[f"bg_recovery_{cond_name}"] = round(rec, 5) if rec is not None else None

    return result


# ─────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────

def aggregate(examples: List[dict], subset: str = "all") -> dict:
    """Compute mean ± std / n for every NLL and recovery field."""
    valid = [e for e in examples if "skip_reason" not in e]
    if subset == "effective":
        valid = [e for e in valid if e.get("gap", 0) >= EFF_CORRUPT_THRESH]

    if not valid:
        return {"n": 0}

    def _agg(key):
        vals = [e[key] for e in valid if e.get(key) is not None]
        if not vals:
            return None
        a = np.array(vals, dtype=np.float64)
        return {"mean": float(a.mean()), "std": float(a.std()), "n": len(a)}

    out: dict = {"n": len(valid)}
    for key in ["nll_clean", "nll_corrupt", "gap"]:
        out[key] = _agg(key)
    for cond in list(PATCH_SETS.keys()):
        out[f"nll_{cond}"]      = _agg(f"nll_{cond}")
        out[f"recovery_{cond}"] = _agg(f"recovery_{cond}")
    # Background control
    bg_keys = [k for k in valid[0] if k.startswith("bg_recovery_")]
    for k in bg_keys:
        out[k] = _agg(k)
    return out


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--dataset_split",     default="validation")
    ap.add_argument("--max_samples",       type=int, default=300)
    ap.add_argument("--layers",            type=int, nargs="+", default=[10, 12, 14, 16])
    ap.add_argument("--noise_std_multiplier", type=float, default=3.0)
    ap.add_argument("--max_length",        type=int, default=4096)
    ap.add_argument("--dataset_cache_dir", default=DATASET_CACHE)
    ap.add_argument("--output_json",       default="outputs/layer_causal_patching_hotpot.json")
    ap.add_argument("--run_background_control", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading model: {args.model_name_or_path}")
    model, tok = load_model(args.model_name_or_path, device)
    print(f"Model loaded. Layers: {args.layers}  noise_multiplier: {args.noise_std_multiplier}")

    print(f"Loading hotpotqa/hotpot_qa distractor ...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor",
                      cache_dir=args.dataset_cache_dir)
    val = ds[args.dataset_split]
    n_take = min(args.max_samples, len(val))
    print(f"  Split={args.dataset_split}  using {n_take}/{len(val)} examples")

    bg_rng = np.random.default_rng(args.seed)

    results: List[dict] = []
    t0 = time.time()
    n_skipped = 0
    n_used = 0

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)

    for i in range(n_take):
        ex = val[i]
        try:
            r = process_example(
                ex, i, model, tok,
                layers=args.layers,
                noise_std_multiplier=args.noise_std_multiplier,
                max_length=args.max_length,
                run_background_control=args.run_background_control,
                bg_rng=bg_rng,
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            r = {"idx": i, "skip_reason": f"exception: {exc}"}

        results.append(r)

        if "skip_reason" in r:
            n_skipped += 1
            print(f"  [{i}] SKIP: {r['skip_reason']}")
        else:
            n_used += 1
            gap = r["gap"]
            rec_multi = r.get("recovery_multi", "?")
            rec_L14   = r.get("recovery_L14", "?")
            print(f"  [{i}] gap={gap:+.3f}  rec_L14={rec_L14}  rec_multi={rec_multi}"
                  f"  sup_toks={r['n_support_tokens']}"
                  f"  t={time.time()-t0:.0f}s")

        # Incremental save every 10 examples
        if (i + 1) % 10 == 0 or (i + 1) == n_take:
            _save(args, results, n_take, n_used, n_skipped)

    _save(args, results, n_take, n_used, n_skipped)
    print(f"\nDone. used={n_used}  skipped={n_skipped}  output={args.output_json}")

    # Print summary
    agg_all = aggregate(results, "all")
    agg_eff = aggregate(results, "effective")
    print(f"\n── All examples (n={agg_all['n']}) ──")
    _print_summary(agg_all)
    print(f"\n── Effective corruption subset (gap≥{EFF_CORRUPT_THRESH}, n={agg_eff['n']}) ──")
    _print_summary(agg_eff)


def _save(args, results, n_take, n_used, n_skipped):
    agg_all = aggregate(results, "all")
    agg_eff = aggregate(results, "effective")
    payload = {
        "config": {
            "model": args.model_name_or_path,
            "max_samples": n_take,
            "layers": args.layers,
            "noise_std_multiplier": args.noise_std_multiplier,
            "max_length": args.max_length,
            "run_background_control": args.run_background_control,
            "eff_corrupt_thresh": EFF_CORRUPT_THRESH,
            "n_used": n_used,
            "n_skipped": n_skipped,
        },
        "aggregate_all":       agg_all,
        "aggregate_effective": agg_eff,
        "examples": results,
    }
    with open(args.output_json, "w") as f:
        json.dump(payload, f, indent=2)


def _print_summary(agg: dict):
    if agg.get("n", 0) == 0:
        print("  (no examples)")
        return
    gap_m = agg.get("gap", {})
    if gap_m:
        print(f"  mean_gap={gap_m['mean']:+.3f}")
    for cond in ["L10", "L12", "L14", "L16", "multi",
                 "cumul_10_12", "cumul_10_12_14"]:
        key = f"recovery_{cond}"
        v = agg.get(key)
        if v and v["n"] > 0:
            print(f"  {key:26s}  mean={v['mean']:+.3f}  std={v['std']:.3f}  n={v['n']}")


if __name__ == "__main__":
    main()
