import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
"""Build the RULER-clean Stage-2 SFT mix as a single shuffled JSONL.

Sources (TRAIN splits only — no dev, no synthetic NIAH/VT/CWE/FWE):
  - SQuAD train          (rajpurkar/squad)
  - HotpotQA train       (lucadiliello/hotpotqa, MRQA single-passage format; flat layout — one QA per row)
  - MuSiQue train        (dgslibisey/MuSiQue, full subset)
  - LongAlpaca-12k train (Yukang/LongAlpaca-12k)

Output schema (jsonl, one example per line):
  {"input": <document/context>, "prompt": <question/instruction>, "answer": <gold answer>, "source": <str>}

Mix policy: all examples from each source, concatenated and shuffled with
seed=42. Natural proportions; no resampling. ~209k total. v1 mix.

Output files (under --output_dir):
  train.jsonl   (~95%)
  val.jsonl     (~5%, used for in-training eval / U-turn detection)
  stats.json    (per-source counts, total, char/token length quantiles)
"""

import argparse
import json
import os
import random
from pathlib import Path

from datasets import load_dataset


def _norm(s):
    return (s or "").strip()


def squad_to_records(ds):
    """SQuAD train: {context, question, answers={text:[...]}}. Use first answer."""
    out = []
    for ex in ds:
        ctx = _norm(ex.get("context"))
        q = _norm(ex.get("question"))
        ans_list = ex.get("answers", {}).get("text", [])
        if not ctx or not q or not ans_list:
            continue
        ans = _norm(ans_list[0])
        if not ans:
            continue
        out.append({"input": ctx, "prompt": q, "answer": ans, "source": "squad"})
    return out


def mrqa_hotpot_to_records(ds):
    """tau/mrqa config=hotpotqa: {context, qas:[{qid, question, answers:[{text,...}]}]}.

    MRQA layout has multiple QAs per context. Flatten to one record per QA.
    Some rows may already be flat (one QA per row) — handle both.
    """
    out = []
    for ex in ds:
        ctx = _norm(ex.get("context"))
        if not ctx:
            continue
        qas = ex.get("qas")
        if qas is None:
            # Flat layout fallback (e.g. lucadiliello/hotpotqa style)
            q = _norm(ex.get("question"))
            answers = ex.get("answers", [])
            if isinstance(answers, dict):
                answers = answers.get("text", [])
            if not q or not answers:
                continue
            ans = _norm(answers[0] if isinstance(answers[0], str) else answers[0].get("text", ""))
            if not ans:
                continue
            out.append({"input": ctx, "prompt": q, "answer": ans, "source": "hotpotqa_mrqa"})
            continue
        # Nested layout
        for qa in qas:
            q = _norm(qa.get("question"))
            ans_field = qa.get("answers") or qa.get("detected_answers") or []
            ans_text = None
            for a in ans_field:
                if isinstance(a, str):
                    ans_text = _norm(a)
                elif isinstance(a, dict):
                    ans_text = _norm(a.get("text"))
                if ans_text:
                    break
            if not q or not ans_text:
                continue
            out.append({"input": ctx, "prompt": q, "answer": ans_text, "source": "hotpotqa_mrqa"})
    return out


def musique_to_records(ds):
    """MuSiQue: {question, paragraphs:[{title, paragraph_text, is_supporting}], answer}.

    Use gold (is_supporting=True) paragraphs only — matches musique_qa_transform
    in training_utils.py (gold-only QA).
    """
    out = []
    for ex in ds:
        q = _norm(ex.get("question"))
        ans = _norm(ex.get("answer"))
        if not q or not ans:
            continue
        paragraphs = ex.get("paragraphs", [])
        gold = [p for p in paragraphs if p.get("is_supporting")]
        if not gold:
            continue
        ctx_parts = []
        for p in gold:
            title = _norm(p.get("title"))
            text = _norm(p.get("paragraph_text"))
            if title:
                ctx_parts.append(f"{title}: {text}")
            else:
                ctx_parts.append(text)
        ctx = "\n\n".join(ctx_parts)
        if not ctx:
            continue
        out.append({"input": ctx, "prompt": q, "answer": ans, "source": "musique"})
    return out


_LONGALPACA_SPLITTERS = (
    " Now the paper ends. \n",
    " Now the paper ends. ",
    " Now the review guideline ends.",
    " Now the review guideline ends. ",
)


def longalpaca_to_records(ds):
    """LongAlpaca-12k: {instruction, input, output}.

    In practice `input` is always None — the long doc + question are concatenated
    in `instruction`, ending with one of the markers in _LONGALPACA_SPLITTERS.
    We split on the marker so the long doc goes into `input` (gets compressed)
    and the trailing question goes into `prompt` (stays uncompressed for the
    decoder to attend to).

    Fallback when no marker is found: whole instruction goes to `input`,
    `prompt` is empty. Loses the question structure but keeps the example.
    """
    out = []
    n_split = 0
    n_fallback = 0
    for ex in ds:
        instruction = _norm(ex.get("instruction"))
        out_text = _norm(ex.get("output"))
        if not instruction or not out_text:
            continue

        doc, question = instruction, ""
        for marker in _LONGALPACA_SPLITTERS:
            if marker in instruction:
                head, _, tail = instruction.partition(marker)
                doc = _norm(head)
                # Drop a leading "Question:" prefix if present (style varies)
                q = _norm(tail)
                if q.lower().startswith("question:"):
                    q = _norm(q[len("question:"):])
                question = q
                break

        if question:
            n_split += 1
        else:
            n_fallback += 1

        out.append({
            "input": doc,
            "prompt": question,
            "answer": out_text,
            "source": "longalpaca",
        })
    print(f"[longalpaca]   split-on-marker: {n_split:,}, fallback (no marker): {n_fallback:,}")
    return out


SOURCES = [
    # (name, hf_name, hf_config, split, builder)
    ("squad",         "rajpurkar/squad",            None,        "train", squad_to_records),
    ("hotpotqa_mrqa", "lucadiliello/hotpotqa",      None,        "train", mrqa_hotpot_to_records),
    ("musique",       "dgslibisey/MuSiQue",         None,        "train", musique_to_records),
    ("longalpaca",    "Yukang/LongAlpaca-12k",      None,        "train", longalpaca_to_records),
]


def quantiles(values, qs=(0.5, 0.9, 0.99)):
    if not values:
        return {}
    s = sorted(values)
    n = len(s)
    return {f"q{int(q*100)}": s[min(int(q * n), n - 1)] for q in qs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_per_source", type=int, default=None,
                    help="Smoke-test cap. None = use all.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    stats_path = out_dir / "stats.json"

    if train_path.exists() and val_path.exists() and stats_path.exists():
        print(f"[skip] outputs already present in {out_dir}")
        print(json.loads(stats_path.read_text()))
        return

    all_records = []
    per_source_stats = {}

    for name, hf_name, cfg, split, builder in SOURCES:
        print(f"[{name}] loading {hf_name} (config={cfg}, split={split}) ...", flush=True)
        ds = load_dataset(hf_name, cfg, split=split, cache_dir=args.cache_dir)
        if args.max_per_source is not None:
            ds = ds.select(range(min(args.max_per_source, len(ds))))
        print(f"[{name}]   raw rows: {len(ds):,}", flush=True)
        records = builder(ds)
        print(f"[{name}]   after build: {len(records):,}", flush=True)
        char_lens = [len(r["input"]) for r in records]
        per_source_stats[name] = {
            "hf_name": hf_name,
            "hf_config": cfg,
            "split": split,
            "raw_rows": len(ds),
            "built_records": len(records),
            "input_chars_quantiles": quantiles(char_lens),
        }
        all_records.extend(records)

    print(f"[mix] total records: {len(all_records):,}", flush=True)

    rng = random.Random(args.seed)
    rng.shuffle(all_records)

    n_total = len(all_records)
    n_val = max(1, int(n_total * args.val_frac))
    val_records = all_records[:n_val]
    train_records = all_records[n_val:]

    with open(train_path, "w") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(val_path, "w") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    final_stats = {
        "seed": args.seed,
        "val_frac": args.val_frac,
        "total_records": n_total,
        "train_records": len(train_records),
        "val_records": len(val_records),
        "per_source": per_source_stats,
        "natural_percent_in_total": {
            name: round(100.0 * stat["built_records"] / max(n_total, 1), 2)
            for name, stat in per_source_stats.items()
        },
    }
    stats_path.write_text(json.dumps(final_stats, indent=2))

    print(f"[done] wrote {train_path} ({len(train_records):,}) and {val_path} ({len(val_records):,})")
    print(json.dumps(final_stats, indent=2))


if __name__ == "__main__":
    main()
