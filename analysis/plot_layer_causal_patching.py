import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
"""
Plot causal activation patching results.

Usage
-----
  python analysis/plot_layer_causal_patching.py \\
      --input_json outputs/layer_causal_patching_hotpot.json \\
      --output_dir outputs/layer_causal_patching_figs
"""

import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BLUE = "#4878CF"
RED  = "#C44E52"
GRAY = "#888888"
GREENS = ["#2ca02c", "#98df8a", "#006400", "#00fa00"]
PALETTE = [BLUE, "#6EB5FF", "#2255AA", "#001166",
           "#FF7F0E", "#FFBB78", RED]


def bootstrap_ci(vals, n_boot=2000, ci=95, seed=42):
    """Return (mean, lo, hi) with percentile bootstrap CI."""
    arr = np.array([v for v in vals if v is not None and not np.isnan(v)],
                   dtype=np.float64)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_means, (100 - ci) / 2)
    hi = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return float(arr.mean()), float(lo), float(hi)


def load_subset(examples, effective_thresh=0.10):
    valid  = [e for e in examples if "skip_reason" not in e]
    effec  = [e for e in valid if e.get("gap", 0) >= effective_thresh]
    return valid, effec


def get_recovery(examples, key):
    return [e[key] for e in examples if e.get(key) is not None]


# ─────────────────────────────────────────────────────────────
# Figure 1: single-layer + multi recovery (bar chart)
# ─────────────────────────────────────────────────────────────

def fig1_single_vs_multi(examples, out_dir, suffix=""):
    conditions = [
        ("L10",   "L10"),
        ("L12",   "L12"),
        ("L14",   "L14"),
        ("L16",   "L16"),
        ("multi", "Multi\n{10,12,14,16}"),
    ]
    means, los, his, labels = [], [], [], []
    for cond, lbl in conditions:
        vals = get_recovery(examples, f"recovery_{cond}")
        m, lo, hi = bootstrap_ci(vals)
        means.append(m)
        los.append(m - lo)
        his.append(hi - m)
        labels.append(lbl)

    x = np.arange(len(labels))
    colors = [BLUE, "#6EB5FF", "#2255AA", "#001166", RED]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, means, yerr=[los, his], capsize=6,
                  color=colors, alpha=0.85, width=0.55, error_kw={"elinewidth": 1.5})

    ax.axhline(0.0, color=GRAY, lw=0.8, ls="--", alpha=0.5)
    ax.axhline(1.0, color=GRAY, lw=0.8, ls="--", alpha=0.5, label="Full recovery (=1)")

    for xi, m in zip(x, means):
        ax.text(xi, max(m + 0.03, 0.03), f"{m:.2f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Mean recovery fraction\n(0=no help, 1=full)", fontsize=11)
    ax.set_title(
        "Causal Recovery by Layer (Supporting-Fact Tokens)\n"
        f"n={len(examples)} examples, effective-corruption subset",
        fontsize=11
    )
    ax.set_ylim(min(0, min(means) - 0.15), max(1.15, max(means) + 0.15))
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"fig1_single_vs_multi{suffix}.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig1_single_vs_multi{suffix}")


# ─────────────────────────────────────────────────────────────
# Figure 2: cumulative recovery
# ─────────────────────────────────────────────────────────────

def fig2_cumulative(examples, out_dir, suffix=""):
    conditions = [
        ("L10",            "{10}"),
        ("cumul_10_12",    "{10,12}"),
        ("cumul_10_12_14", "{10,12,14}"),
        ("multi",          "{10,12,14,16}"),
    ]
    means, los, his, labels = [], [], [], []
    for cond, lbl in conditions:
        vals = get_recovery(examples, f"recovery_{cond}")
        m, lo, hi = bootstrap_ci(vals)
        means.append(m)
        los.append(m - lo)
        his.append(hi - m)
        labels.append(lbl)

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(x, means, yerr=[los, his], marker="o", color=RED,
                linewidth=2.5, markersize=10, capsize=6,
                label="Mean recovery (95% bootstrap CI)")
    ax.fill_between(x, [m - l for m, l in zip(means, los)],
                       [m + h for m, h in zip(means, his)],
                       alpha=0.15, color=RED)

    ax.axhline(0.0, color=GRAY, lw=0.8, ls="--", alpha=0.5)
    ax.axhline(1.0, color=GRAY, lw=0.8, ls="--", alpha=0.5, label="Full recovery (=1)")

    for xi, m in zip(x, means):
        ax.text(xi, m + 0.025, f"{m:.2f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Mean recovery fraction", fontsize=11)
    ax.set_title(
        "Cumulative Multi-Layer Recovery\n"
        f"n={len(examples)} examples, effective-corruption subset",
        fontsize=11
    )
    ax.set_ylim(min(0, min(m - l for m, l in zip(means, los))) - 0.1,
                max(1.1, max(m + h for m, h in zip(means, his))) + 0.1)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"fig2_cumulative{suffix}.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig2_cumulative{suffix}")


# ─────────────────────────────────────────────────────────────
# Figure 3: support vs background control
# ─────────────────────────────────────────────────────────────

def fig3_bg_control(examples, out_dir, suffix=""):
    """Only produced if background control data is present."""
    conditions = ["L10", "L12", "L14", "L16", "multi"]
    labels = ["L10", "L12", "L14", "L16", "Multi\n{10..16}"]

    sup_means, sup_los, sup_his = [], [], []
    bg_means,  bg_los,  bg_his  = [], [], []

    has_bg = any(f"bg_recovery_L10" in e for e in examples)
    if not has_bg:
        print("  No background control data — skipping fig3")
        return

    for cond in conditions:
        sv = get_recovery(examples, f"recovery_{cond}")
        bv = get_recovery(examples, f"bg_recovery_{cond}")
        sm, slo, shi = bootstrap_ci(sv)
        bm, blo, bhi = bootstrap_ci(bv)
        sup_means.append(sm); sup_los.append(sm - slo); sup_his.append(shi - sm)
        bg_means.append(bm);  bg_los.append(bm - blo);  bg_his.append(bhi - bm)

    x = np.arange(len(conditions))
    w = 0.33

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, sup_means, yerr=[sup_los, sup_his], width=w,
           capsize=5, color=RED,  alpha=0.8, label="Support-fact tokens")
    ax.bar(x + w / 2, bg_means,  yerr=[bg_los,  bg_his],  width=w,
           capsize=5, color=BLUE, alpha=0.8, label="Background tokens (matched n)")

    ax.axhline(0.0, color=GRAY, lw=0.8, ls="--", alpha=0.5)
    ax.axhline(1.0, color=GRAY, lw=0.8, ls="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Mean recovery fraction", fontsize=11)
    ax.set_title(
        "Support-Fact vs Background Token Patching\n"
        f"n={len(examples)} effective-corruption examples",
        fontsize=11
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"fig3_bg_control{suffix}.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig3_bg_control{suffix}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json",  required=True)
    ap.add_argument("--output_dir",  default="outputs/layer_causal_patching_figs")
    ap.add_argument("--eff_thresh",  type=float, default=0.10)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input_json) as f:
        data = json.load(f)

    examples = data.get("examples", [])
    cfg      = data.get("config", {})
    print(f"Loaded {len(examples)} examples from {args.input_json}")
    print(f"Config: {cfg}")

    _, eff_examples = load_subset(examples, args.eff_thresh)
    print(f"Effective-corruption subset (gap >= {args.eff_thresh}): n={len(eff_examples)}")

    if not eff_examples:
        print("WARNING: no effective-corruption examples — plotting all valid examples instead")
        all_valid, _ = load_subset(examples, 0)
        eff_examples = all_valid

    fig1_single_vs_multi(eff_examples, args.output_dir)
    fig2_cumulative(eff_examples, args.output_dir)
    fig3_bg_control(eff_examples, args.output_dir)

    # Also plot all valid examples for comparison
    all_valid, _ = load_subset(examples, 0)
    if len(all_valid) != len(eff_examples):
        fig1_single_vs_multi(all_valid, args.output_dir, suffix="_all")
        fig2_cumulative(all_valid, args.output_dir, suffix="_all")

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
