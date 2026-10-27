from transformers import Trainer, TrainerCallback
import os
import torch
import random
import numpy as np
from safetensors.torch import save_file

from transformers.trainer_utils import get_last_checkpoint
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_context_window(input_ids, length, lm_ratio=0.0):

    input_len = len(input_ids)
    assert input_len >= 1, f"Error: invalid input length ({input_len})"

    # ae
    if random.random() >= lm_ratio:
        if input_len <= length: # if shorter, keep the complete text
            return input_ids, []
        else:
            last_start = input_len - length
            random_start = random.randint(0, last_start)
            return input_ids[random_start: random_start+length], []

    # lm
    if input_len <= length:
        r = random.randint(0, input_len-1)
        return input_ids[:r+1], input_ids[r+1:]
    else:
        last_start = input_len - length
        random_start = random.randint(0, last_start)
        return input_ids[random_start: random_start+length], input_ids[random_start+length:]


def hotpotqa_pretrain_transform(examples):
    """Transform raw HotPotQA examples into {"text": ...} for pretraining."""
    texts = []
    for context in examples["context"]:
        parts = []
        for title, sents in zip(context["title"], context["sentences"]):
            para = f"{title}: {''.join(sents).strip()}"
            parts.append(para)
        texts.append("\n\n".join(parts))
    return {"text": texts}


def hotpotqa_qa_transform(examples):
    """Transform raw HotPotQA examples into {"input": ..., "prompt": ..., "answer": ...} for QA."""
    inputs = []
    for context, supporting_facts in zip(examples["context"], examples["supporting_facts"]):
        gold_titles = set(supporting_facts["title"])
        parts = []
        for title, sents in zip(context["title"], context["sentences"]):
            if title in gold_titles:
                para = f"{title}: {''.join(sents).strip()}"
                parts.append(para)
        inputs.append("\n\n".join(parts))
    return {"input": inputs, "prompt": examples["question"], "answer": examples["answer"]}


def hotpotqa_qa_transform_full(examples):
    """Transform raw HotPotQA examples into {"input": ..., "prompt": ..., "answer": ...} for QA.

    Unlike hotpotqa_qa_transform which keeps only gold paragraphs, this variant
    keeps ALL paragraphs (gold + distractors). Forces the model to distinguish
    relevant from irrelevant context, preventing the question-prior shortcut.
    """
    inputs = []
    for context in examples["context"]:
        parts = []
        for title, sents in zip(context["title"], context["sentences"]):
            para = f"{title}: {''.join(sents).strip()}"
            parts.append(para)
        inputs.append("\n\n".join(parts))
    return {"input": inputs, "prompt": examples["question"], "answer": examples["answer"]}


# =====================================================================
# Qasper inline transforms  (HF dataset → expected fields)
# =====================================================================

def _qasper_build_paper_text(abstract, full_text):
    """Concatenate abstract + all sections of a Qasper paper into a single string."""
    parts = [abstract]
    for sec_name, paragraphs in zip(full_text["section_name"], full_text["paragraphs"]):
        parts.append(f"{sec_name}\n" + "\n".join(paragraphs))
    return "\n\n".join(parts)


def qasper_pretrain_transform(examples):
    """Transform raw Qasper examples into {"text": ...} for pretraining.

    Each row is a full paper.  We concatenate abstract + all section paragraphs.
    """
    texts = []
    for abstract, full_text in zip(examples["abstract"], examples["full_text"]):
        texts.append(_qasper_build_paper_text(abstract, full_text))
    return {"text": texts}


def qasper_qa_transform(examples):
    """Transform raw Qasper examples into {"input", "prompt", "answer"} for QA.

    Flattens: 1 paper with N questions → N (input, prompt, answer) examples.
    Skips unanswerable questions.  Uses the first valid annotator answer per
    question, preferring free-form > extractive spans > yes/no.
    """
    inputs, prompts, answers = [], [], []
    for abstract, full_text, qas in zip(
        examples["abstract"], examples["full_text"], examples["qas"]
    ):
        paper_text = _qasper_build_paper_text(abstract, full_text)

        for q_idx, question in enumerate(qas["question"]):
            ans_data = qas["answers"][q_idx]
            annotators = ans_data["answer"]  # list of annotator dicts

            ans_text = None
            for annotator in annotators:
                if annotator["unanswerable"]:
                    continue
                if annotator["free_form_answer"]:
                    ans_text = annotator["free_form_answer"]
                    break
                if annotator["extractive_spans"]:
                    ans_text = ", ".join(annotator["extractive_spans"])
                    break
                if annotator["yes_no"] is not None:
                    ans_text = "Yes" if annotator["yes_no"] else "No"
                    break

            if ans_text and ans_text.strip():
                inputs.append(paper_text)
                prompts.append(question)
                answers.append(ans_text)

    return {"input": inputs, "prompt": prompts, "answer": answers}


def qasper_qa_eval_transform(examples):
    """Like qasper_qa_transform but collects ALL valid annotator answers per question.

    Returns "all_answers" as a list of lists (one list of answer strings per question).
    The "answer" field still contains the first valid answer (for compatibility).
    Also returns "evidence": per-example list of evidence paragraph strings
    (union across annotators, "FLOAT SELECTED..." table captions filtered out
    since they are not in the paper text the compressor sees).
    Standard Qasper evaluation takes max score across all annotator answers.
    """
    inputs, prompts, answers, all_answers, evidences = [], [], [], [], []
    for abstract, full_text, qas in zip(
        examples["abstract"], examples["full_text"], examples["qas"]
    ):
        paper_text = _qasper_build_paper_text(abstract, full_text)

        for q_idx, question in enumerate(qas["question"]):
            ans_data = qas["answers"][q_idx]
            annotators = ans_data["answer"]

            # Collect all valid answers + evidence from answerable annotators
            valid_answers = []
            ev_set = []
            seen_ev = set()
            for annotator in annotators:
                if annotator["unanswerable"]:
                    continue
                if annotator["free_form_answer"]:
                    valid_answers.append(annotator["free_form_answer"])
                elif annotator["extractive_spans"]:
                    valid_answers.append(", ".join(annotator["extractive_spans"]))
                elif annotator["yes_no"] is not None:
                    valid_answers.append("Yes" if annotator["yes_no"] else "No")
                for e in annotator.get("evidence", []) or []:
                    if not e or not e.strip():
                        continue
                    if e.startswith("FLOAT SELECTED"):
                        continue
                    if e not in seen_ev:
                        seen_ev.add(e)
                        ev_set.append(e)

            if valid_answers:
                inputs.append(paper_text)
                prompts.append(question)
                answers.append(valid_answers[0])  # first for compat
                all_answers.append(valid_answers)
                evidences.append(ev_set)

    return {
        "input": inputs,
        "prompt": prompts,
        "answer": answers,
        "all_answers": all_answers,
        "evidence": evidences,
    }


# =====================================================================
# MuSiQue inline transforms (HF dataset → expected fields)
#   Source: dgslibisey/MuSiQue — paragraphs=[{idx,title,paragraph_text,
#   is_supporting}], question, answer, answer_aliases, answerable.
# =====================================================================

def _musique_paragraph_str(title, paragraph_text):
    return f"{title}: {paragraph_text.strip()}"


def _musique_build_full_context(paragraphs):
    return "\n\n".join(
        _musique_paragraph_str(p["title"], p["paragraph_text"]) for p in paragraphs
    )


def musique_pretrain_transform(examples):
    """Full context (20 paragraphs per example) as raw text for pretraining."""
    texts = [_musique_build_full_context(p) for p in examples["paragraphs"]]
    return {"text": texts}


def musique_qa_transform(examples):
    """Gold-only QA: keep only is_supporting paragraphs. Mirrors hotpotqa gold."""
    inputs, prompts, answers = [], [], []
    for paragraphs, q, a, answerable in zip(
        examples["paragraphs"], examples["question"],
        examples["answer"], examples["answerable"],
    ):
        if not answerable:
            continue
        gold = [p for p in paragraphs if p["is_supporting"]]
        if not gold or not a or not a.strip():
            continue
        inputs.append("\n\n".join(
            _musique_paragraph_str(p["title"], p["paragraph_text"]) for p in gold
        ))
        prompts.append(q)
        answers.append(a)
    return {"input": inputs, "prompt": prompts, "answer": answers}


def musique_qa_transform_full(examples):
    """Full-context QA: all paragraphs (gold + distractors). For routing/selection."""
    inputs, prompts, answers = [], [], []
    for paragraphs, q, a, answerable in zip(
        examples["paragraphs"], examples["question"],
        examples["answer"], examples["answerable"],
    ):
        if not answerable or not a or not a.strip():
            continue
        inputs.append(_musique_build_full_context(paragraphs))
        prompts.append(q)
        answers.append(a)
    return {"input": inputs, "prompt": prompts, "answer": answers}


def musique_qa_eval_transform(examples):
    """Full context + per-example evidence (gold paragraph strings, verbatim
    substrings of `input`) + all_answers (answer + answer_aliases)."""
    inputs, prompts, answers, all_answers, evidences = [], [], [], [], []
    for paragraphs, q, a, aliases, answerable in zip(
        examples["paragraphs"], examples["question"], examples["answer"],
        examples["answer_aliases"], examples["answerable"],
    ):
        if not answerable or not a or not a.strip():
            continue
        context = _musique_build_full_context(paragraphs)
        ev = [
            _musique_paragraph_str(p["title"], p["paragraph_text"])
            for p in paragraphs if p["is_supporting"]
        ]
        all_ans = [a] + [x for x in (aliases or []) if x and x.strip() and x != a]
        inputs.append(context)
        prompts.append(q)
        answers.append(a)
        all_answers.append(all_ans)
        evidences.append(ev)
    return {
        "input": inputs, "prompt": prompts, "answer": answers,
        "all_answers": all_answers, "evidence": evidences,
    }


# =====================================================================
# 2WikiMultihopQA inline transforms (HF dataset → expected fields)
#   Source: xanhho/2WikiMultihopQA at revision='refs/convert/parquet'
#   (datasets>=4 compatible; no trust_remote_code needed). Splits on that
#   revision are train / validation / test.
#
#   Parquet schema (pair-encoded — different from the loading-script form):
#     context          : list of [title: str, sentences: list[str]]
#     supporting_facts : list of [title: str, sent_id: int]
#     evidences        : list of [fact: str, relation: str, entity: str]
#     question, answer : str (single-string answer)
# =====================================================================

def _twowiki_paragraph_str(title, sentences):
    return f"{title}: {''.join(sentences).strip()}"


def _twowiki_parse(val):
    """On the parquet revision, `context` / `supporting_facts` come back as
    JSON strings (nested mixed-type lists can't round-trip through pyarrow)."""
    if isinstance(val, str):
        import json
        return json.loads(val)
    return val


def twowiki_pretrain_transform(examples):
    texts = []
    for context in examples["context"]:
        context = _twowiki_parse(context)
        parts = [_twowiki_paragraph_str(title, sents) for title, sents in context]
        texts.append("\n\n".join(parts))
    return {"text": texts}


def twowiki_qa_transform(examples):
    """Gold-only QA: keep only paragraphs whose title is in supporting_facts."""
    inputs = []
    for context, sf in zip(examples["context"], examples["supporting_facts"]):
        context = _twowiki_parse(context)
        sf = _twowiki_parse(sf)
        gold_titles = {t for t, _ in sf}
        parts = [
            _twowiki_paragraph_str(title, sents)
            for title, sents in context if title in gold_titles
        ]
        inputs.append("\n\n".join(parts))
    return {"input": inputs, "prompt": examples["question"], "answer": examples["answer"]}


def twowiki_qa_transform_full(examples):
    """Full-context QA: all paragraphs (gold + distractors).

    Also emits paragraph-level structure (optional evidence masks):
      - paragraphs_text     : list[list[str]]  — each example's paragraphs as strings
      - paragraphs_evidence : list[list[int]]  — 1 if paragraph's title is in
                                                  supporting_facts, else 0
    Downstream (in stage2_tokenize_function), these are converted into a
    per-segment evidence_mask used by info_loss for attention supervision.
    """
    inputs = []
    paragraphs_text = []
    paragraphs_evidence = []
    for context, sf in zip(examples["context"], examples["supporting_facts"]):
        context = _twowiki_parse(context)
        sf = _twowiki_parse(sf)
        gold_titles = {t for t, _ in sf}
        parts = [_twowiki_paragraph_str(title, sents) for title, sents in context]
        para_ev = [int(title in gold_titles) for title, _ in context]
        inputs.append("\n\n".join(parts))
        paragraphs_text.append(parts)
        paragraphs_evidence.append(para_ev)
    return {
        "input": inputs,
        "prompt": examples["question"],
        "answer": examples["answer"],
        "paragraphs_text": paragraphs_text,
        "paragraphs_evidence": paragraphs_evidence,
    }


def twowiki_qa_transform_gold_only(examples):
    """Gold-paragraphs-only context (no distractors).

    A0 control: tests "what F1 do we get if the model sees only the supporting
    paragraphs?" — upper bound on what any selection method could achieve.
    Same output schema as the `_full` transform so it's a drop-in replacement
    in the launcher.
    """
    inputs = []
    paragraphs_text = []
    paragraphs_evidence = []
    for context, sf in zip(examples["context"], examples["supporting_facts"]):
        context = _twowiki_parse(context)
        sf = _twowiki_parse(sf)
        gold_titles = {t for t, _ in sf}
        gold_parts = [
            _twowiki_paragraph_str(title, sents)
            for title, sents in context
            if title in gold_titles
        ]
        # Edge case: no gold paragraphs found (mismatched titles). Fall back to
        # full context to avoid empty inputs.
        if not gold_parts:
            gold_parts = [_twowiki_paragraph_str(t, s) for t, s in context]
        inputs.append("\n\n".join(gold_parts))
        paragraphs_text.append(gold_parts)
        paragraphs_evidence.append([1] * len(gold_parts))   # all included paragraphs are gold
    return {
        "input": inputs,
        "prompt": examples["question"],
        "answer": examples["answer"],
        "paragraphs_text": paragraphs_text,
        "paragraphs_evidence": paragraphs_evidence,
    }


def twowiki_qa_eval_transform_gold_only(examples):
    """Eval-mode gold-only context (A0 control). Inputs only the supporting
    paragraphs; otherwise mirrors `twowiki_qa_eval_transform`."""
    inputs, prompts, answers, all_answers, evidences = [], [], [], [], []
    for context, sf, q, a in zip(
        examples["context"], examples["supporting_facts"],
        examples["question"], examples["answer"],
    ):
        if not a or not a.strip():
            continue
        context = _twowiki_parse(context)
        sf = _twowiki_parse(sf)
        gold_titles = {t for t, _ in sf}
        gold_parts = [
            _twowiki_paragraph_str(title, sents)
            for title, sents in context if title in gold_titles
        ]
        if not gold_parts:
            gold_parts = [_twowiki_paragraph_str(t, s) for t, s in context]
        inputs.append("\n\n".join(gold_parts))
        prompts.append(q)
        answers.append(a)
        all_answers.append([a])
        evidences.append(gold_parts)
    return {
        "input": inputs, "prompt": prompts, "answer": answers,
        "all_answers": all_answers, "evidences": evidences,
    }


def twowiki_qa_eval_transform(examples):
    """Full context + paragraph-level evidence (gold paragraph strings, verbatim
    substrings of `input`). 2Wiki answers are single-string; we expose [answer]
    as all_answers for API parity with qasper/musique."""
    inputs, prompts, answers, all_answers, evidences = [], [], [], [], []
    for context, sf, q, a in zip(
        examples["context"], examples["supporting_facts"],
        examples["question"], examples["answer"],
    ):
        if not a or not a.strip():
            continue
        context = _twowiki_parse(context)
        sf = _twowiki_parse(sf)
        gold_titles = {t for t, _ in sf}
        parts = [_twowiki_paragraph_str(title, sents) for title, sents in context]
        ev = [
            _twowiki_paragraph_str(title, sents)
            for title, sents in context if title in gold_titles
        ]
        inputs.append("\n\n".join(parts))
        prompts.append(q)
        answers.append(a)
        all_answers.append([a])
        evidences.append(ev)
    return {
        "input": inputs, "prompt": prompts, "answer": answers,
        "all_answers": all_answers, "evidence": evidences,
    }


def ruler_qa_eval_transform(examples):
    """Fields: context (long string), question, answer_prefix (the assistant
    continuation cue, e.g. 'The special magic number for X is'), answer (list
    of gold strings), task. Emits prompt = 'question\\nanswer_prefix' so the
    decoder continues coherently. No evidence."""
    inputs, prompts, answers, all_answers, evidences = [], [], [], [], []
    for ctx, q, prefix, ans in zip(
        examples["context"], examples["question"],
        examples["answer_prefix"], examples["answer"],
    ):
        if not ans:
            continue
        prompt = f"{q}\n{prefix}" if prefix else q
        inputs.append(ctx)
        prompts.append(prompt)
        answers.append(ans[0])
        all_answers.append(list(ans))
        evidences.append([])
    return {
        "input": inputs, "prompt": prompts, "answer": answers,
        "all_answers": all_answers, "evidence": evidences,
    }


def slimpajama_pretrain_transform(examples):
    """Pass-through for DKYoon/SlimPajama-6B (and cerebras/SlimPajama-627B):
    each row already has a 'text' field — no restructuring needed."""
    return {"text": examples["text"]}


PRETRAIN_TRANSFORMS = {
    "slimpajama": slimpajama_pretrain_transform,
    "hotpotqa": hotpotqa_pretrain_transform,
    "qasper": qasper_pretrain_transform,
    "musique": musique_pretrain_transform,
    "2wikimultihopqa": twowiki_pretrain_transform,
    # RULER: eval-only; no pretrain entry.
}

QA_TRANSFORMS = {
    "hotpotqa": hotpotqa_qa_transform,
    "hotpotqa_full": hotpotqa_qa_transform_full,
    "qasper": qasper_qa_transform,
    "musique": musique_qa_transform,
    "musique_full": musique_qa_transform_full,
    "2wikimultihopqa": twowiki_qa_transform,
    "2wikimultihopqa_full": twowiki_qa_transform_full,
    "2wikimultihopqa_gold_only": twowiki_qa_transform_gold_only,
    # RULER: eval-only.
}

QA_EVAL_TRANSFORMS = {
    "qasper": qasper_qa_eval_transform,
    "musique": musique_qa_eval_transform,
    "musique_full": musique_qa_eval_transform,
    "2wikimultihopqa": twowiki_qa_eval_transform,
    "2wikimultihopqa_full": twowiki_qa_eval_transform,
    "2wikimultihopqa_gold_only": twowiki_qa_eval_transform_gold_only,
    "ruler": ruler_qa_eval_transform,
}


# =====================================================================
# Dataset-loading helpers (handle HF quirks per dataset)
# =====================================================================

# Substring → canonical transform name used by registries. Order matters:
# more-specific keys must come before less-specific ones (e.g.
# "mrqa-hotpotqa" before "hotpotqa", "2wikimultihopqa" before "hotpotqa").
_DATASET_TRANSFORM_KEYS = (
    "2wikimultihopqa",
    "hotpotqa",        # hotpotqa/hotpot_qa distractor (10-paragraph)
    "qasper",
    "musique",
    "slimpajama",
    "ruler",
)


def resolve_transform_name(dataset_name, dataset_config=None):
    """Auto-detect transform short name from HF dataset_name (and optional config).

    Backward-compatible: callers that don't pass dataset_config still work.
    """
    haystack = dataset_name.lower()
    if dataset_config:
        haystack = f"{haystack}-{dataset_config.lower()}"
    for key in _DATASET_TRANSFORM_KEYS:
        if key in haystack:
            return key
    return None


def resolve_load_kwargs(dataset_name):
    """Dataset-specific kwargs for `datasets.load_dataset`.

    Datasets that ship a Python loading script (qasper, 2Wiki, tau/mrqa)
    can no longer auto-run scripts under datasets>=4; we use HF's
    auto-parquet conversion mirror at revision='refs/convert/parquet'.
    """
    kwargs = {}
    # Local JSONL path (no HF dataset_name) — no special kwargs needed.
    if dataset_name is None:
        return kwargs
    lower = dataset_name.lower()
    if (
        "qasper" in lower
        or "2wikimultihopqa" in lower
        or "2wiki" in lower
    ):
        kwargs["revision"] = "refs/convert/parquet"
    return kwargs


def resolve_splits(dataset):
    """Return (train_split, eval_split) names, tolerating dataset-specific
    conventions (`dev` instead of `validation`, `test`-only benchmarks, etc.).
    Returns (None, eval_split) when no train split exists (e.g. RULER)."""
    names = set(dataset.keys())
    train = "train" if "train" in names else None
    if "validation" in names:
        eval_split = "validation"
    elif "dev" in names:
        eval_split = "dev"
    elif "test" in names:
        eval_split = "test"
    else:
        raise ValueError(
            f"No eval split found in dataset (available: {sorted(names)}). "
            "Expected one of: validation, dev, test."
        )
    return train, eval_split


def get_pretrain_transform(name):
    """Look up a pretrain transform by short name (e.g. 'hotpotqa', 'qasper')."""
    if name not in PRETRAIN_TRANSFORMS:
        raise ValueError(f"Unknown pretrain transform '{name}'. Available: {list(PRETRAIN_TRANSFORMS)}")
    return PRETRAIN_TRANSFORMS[name]


def get_qa_transform(name):
    """Look up a QA transform by short name (e.g. 'hotpotqa', 'qasper')."""
    if name not in QA_TRANSFORMS:
        raise ValueError(f"Unknown QA transform '{name}'. Available: {list(QA_TRANSFORMS)}")
    return QA_TRANSFORMS[name]


def get_qa_eval_transform(name):
    """Look up a QA eval transform (with all answers). Falls back to standard QA transform."""
    if name in QA_EVAL_TRANSFORMS:
        return QA_EVAL_TRANSFORMS[name]
    return get_qa_transform(name)


def apply_subset_manifest(dataset, manifest_path, split_names=("train", "validation")):
    """Filter a DatasetDict in place using a subset manifest JSON.

    The manifest is produced by prepare_hotpot_subset.py and contains, per
    split, a list of integer indices into the original dataset and the matching
    example IDs. We .select() by indices and verify the IDs still line up to
    catch silent dataset-version drift.

    Args:
        dataset: a DatasetDict (or dict) with at least the listed split_names.
        manifest_path: path to the JSON manifest file.
        split_names: which splits to filter. Missing splits are left untouched.

    Returns:
        The same dict with the listed splits replaced by filtered versions.
    """
    import json
    import os

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"subset_manifest not found: {manifest_path}. "
            "Run prepare_hotpot_subset.py to create it."
        )
    with open(manifest_path) as f:
        manifest = json.load(f)

    meta = manifest.get("meta", {})
    print(f"[subset] Applying manifest: {manifest_path}")
    print(f"[subset]   dataset={meta.get('dataset')} config={meta.get('config')} seed={meta.get('seed')}")

    for split in split_names:
        if split not in dataset:
            continue
        if split not in manifest:
            print(f"[subset]   split '{split}' not in manifest, leaving unchanged")
            continue
        entry = manifest[split]
        indices = entry["indices"]
        expected_ids = entry.get("ids")

        src = dataset[split]
        if max(indices) >= len(src):
            raise ValueError(
                f"subset manifest indices out of range for split '{split}': "
                f"max index {max(indices)} >= dataset size {len(src)}"
            )

        filtered = src.select(indices)

        # Verify the first/last example IDs still match so we catch
        # dataset-version drift between manifest build time and now.
        if expected_ids and "id" in src.column_names:
            actual_ids = [filtered[i]["id"] for i in (0, len(filtered) // 2, len(filtered) - 1)]
            expected_check = [expected_ids[0], expected_ids[len(expected_ids) // 2], expected_ids[-1]]
            if actual_ids != expected_check:
                raise ValueError(
                    f"subset manifest ID mismatch on split '{split}': "
                    f"expected {expected_check}, got {actual_ids}. "
                    "The underlying dataset may have been updated since the manifest was built. "
                    "Re-run prepare_hotpot_subset.py with --force."
                )

        dataset[split] = filtered
        print(f"[subset]   {split}: {len(src)} → {len(filtered)} examples")

    return dataset


# =====================================================================
# Tokenize helpers for SeDeM hidden-state compressor
# =====================================================================

def stage1_tokenize_function(examples, model, lm_ratio=0.0):
    """
    Windowed prefix-starved reconstruction for the hidden-state compressor.

    Memory is computed from the FULL context (matching inference), but the
    decoder only sees a short random window of k tokens for teacher forcing.
    The first token of the window must be predicted from memory alone —
    forcing the compressor to produce genuinely useful memory tokens.

    This eliminates the teacher-forcing shortcut: with the full context in
    the decoder input, the model could predict x_t from x_{<t} alone without
    using memory.  With a short window, there is too little local context to
    bypass memory.

    We produce:
      - input_ids          : FULL context  (backbone → compressor → memory)
      - prompt_answer_ids  : [mem_placeholders] + [ae_token] + window_tokens
      - labels             : [-100]*prompt_len  + window_tokens
      - teacher_input_ids  : FULL context for distillation (teacher sees everything)
      - teacher_offset     : window start position in full context
                             (for aligning student and teacher predictions)
    """
    text_output = model.tokenizer(
        examples["text"], truncation=False, padding=False,
        return_attention_mask=False,
    )
    text_output["prompt_answer_ids"] = []
    text_output["labels"] = []
    text_output["teacher_input_ids"] = []
    text_output["teacher_offset"] = []

    max_len = model.training_args.model_max_length
    window_size = getattr(model.training_args, 'pretrain_window_size', 128)
    memonly_fraction = getattr(model.training_args, 'memonly_fraction', 0.0)

    # One-time log for memonly configuration
    if memonly_fraction > 0 and not getattr(model, '_memonly_logged', False):
        print(f"[memonly] fraction={memonly_fraction}, short_window=4-16, "
              f"normal_window={window_size}")
        model._memonly_logged = True

    for idx in range(len(text_output["input_ids"])):
        a, b = sample_context_window(text_output["input_ids"][idx], max_len, lm_ratio=lm_ratio)
        length_a = len(a)
        num_segments = model.compute_num_segments(length_a)
        total_mem_length = num_segments * model.memory_slots

        text_output["input_ids"][idx] = a  # FULL context → compressor

        # ---- random window for decoder ----
        # With probability memonly_fraction, use a very short window (4-16 tokens)
        # to force the decoder to predict from memory alone. This creates gradient
        # pressure for document-specific content encoding in memory tokens.
        if memonly_fraction > 0 and random.random() < memonly_fraction:
            k = random.randint(4, 16)
            k = min(k, length_a)
        else:
            k = min(window_size, length_a)
        if length_a > k:
            start = random.randint(0, length_a - k)
        else:
            start = 0
        window = a[start : start + k]

        # decoder side: memory placeholders + ae_token + window tokens
        mem_placeholder_id = model.mem_id_start
        prompt_ids = [mem_placeholder_id] * total_mem_length + [model.ae_token_id]
        answer_ids = window + [model.eos_id]

        text_output["prompt_answer_ids"].append(prompt_ids + answer_ids)
        labels = [-100] * len(prompt_ids) + answer_ids
        text_output["labels"].append(labels)

        # teacher sees FULL context for distillation
        teacher_ids = a + [model.eos_id]
        text_output["teacher_input_ids"].append(teacher_ids)
        # teacher_offset: teacher logit at position `start` predicts a[start+1],
        # aligning with student logit at position prompt_len predicting the same.
        text_output["teacher_offset"].append(start)

        assert len(text_output["prompt_answer_ids"][-1]) == len(labels)

    return text_output


QA_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about academic papers. "
    "Give short, direct answers."
)


def stage2_tokenize_function(examples, model):
    """
    For QA fine-tuning with the hidden-state compressor.

    Each example has "input" (context), "prompt" (question), "answer".
    The question stays UNCOMPRESSED -- only the context is compressed.

    Uses chat template for instruct models (system + user + assistant), UNLESS
    `teacher_prompt` is also present in the row — in that case the prompt is
    treated as already chat-templated (the prep applied the model's template),
    and `teacher_prompt` is used to build `teacher_input_ids` + `teacher_offset`
    for KL distillation against a full-context teacher.
    """
    ctx_max = model.training_args.model_max_length
    text_output = model.tokenizer(
        examples["input"], max_length=ctx_max, truncation=True,
        padding=False, return_attention_mask=False, add_special_tokens=False,
    )

    # Check if tokenizer has a chat template actually configured
    has_chat_template = (
        hasattr(model.tokenizer, 'apply_chat_template')
        and getattr(model.tokenizer, 'chat_template', None) is not None
    )

    # Pre-templated pipeline (e.g. arl_arena Stage 2). When set, `prompt` is
    # used raw (no internal chat-template wrap) and `teacher_prompt` is used
    # to build the distillation teacher input.
    _has_teacher = "teacher_prompt" in examples

    text_output["prompt_answer_ids"] = []
    text_output["labels"] = []
    if _has_teacher:
        text_output["teacher_input_ids"] = []
        text_output["teacher_offset"] = []

    # ---- evidence plumbing ----
    # If the transform supplied per-paragraph structure (paragraphs_text +
    # paragraphs_evidence), compute a per-segment evidence_mask using the
    # tokenizer. Used downstream by info_loss for attention supervision.
    _has_paragraphs = ("paragraphs_text" in examples and "paragraphs_evidence" in examples)
    if _has_paragraphs:
        text_output["evidence_mask"] = []
        # Tokenize the "\n\n" separator once to know how many tokens it adds
        # between paragraphs in the joined input.
        _sep_ids = model.tokenizer("\n\n", add_special_tokens=False)["input_ids"]
        _sep_len = len(_sep_ids)
        _segment_length = model.memory_slots * model.compression_factor
        # Debug print on the first few examples
        _DEBUG_EVIDENCE_FIRST_N = 3
        _debug_count = 0

    for idx in range(len(text_output["input_ids"])):
        length = len(text_output["input_ids"][idx])
        num_segments = model.compute_num_segments(length)
        total_mem_length = num_segments * model.memory_slots

        mem_placeholder_id = model.mem_id_start
        question = examples["prompt"][idx]
        answer = examples["answer"][idx]

        # ---- Compute per-segment evidence mask ----
        if _has_paragraphs:
            paragraphs_text = examples["paragraphs_text"][idx]
            paragraphs_evidence = examples["paragraphs_evidence"][idx]
            evidence_mask = [0] * num_segments
            cur_pos = 0
            for p_idx, (para_str, is_ev) in enumerate(
                zip(paragraphs_text, paragraphs_evidence)
            ):
                # Tokenize this paragraph (no special tokens, matches transform's "\n\n".join)
                para_token_count = len(
                    model.tokenizer(para_str, add_special_tokens=False)["input_ids"]
                )
                para_start = cur_pos
                para_end = cur_pos + para_token_count
                # Account for "\n\n" separator AFTER this paragraph (unless last)
                cur_pos = para_end + (_sep_len if p_idx + 1 < len(paragraphs_text) else 0)
                # Clip to actual context length (truncation may have cut content)
                para_end_clipped = min(para_end, length)
                if para_start >= length:
                    break  # this paragraph is fully truncated, all later ones too
                if not is_ev:
                    continue
                s_start = para_start // _segment_length
                s_end = (para_end_clipped - 1) // _segment_length + 1
                s_end = min(s_end, num_segments)
                for s in range(s_start, s_end):
                    evidence_mask[s] = 1
            text_output["evidence_mask"].append(evidence_mask)
            # Debug print: show paragraph→segment mapping for a few examples
            if _debug_count < _DEBUG_EVIDENCE_FIRST_N:
                n_para = len(paragraphs_text)
                n_evidence_para = sum(paragraphs_evidence)
                n_evidence_seg = sum(evidence_mask)
                print(
                    f"  [evidence_plumb idx={idx}] paragraphs={n_para}, "
                    f"evidence_paragraphs={n_evidence_para}, "
                    f"ctx_tokens={length}, num_segments={num_segments}, "
                    f"evidence_segments={n_evidence_seg}, "
                    f"mask[:min(num_segments,16)]={evidence_mask[:min(num_segments, 16)]}"
                )
                _debug_count += 1

        if _has_teacher:
            # Prep already chat-templated the prompt; use raw to avoid
            # double-wrap.
            chat_prompt_ids = model.tokenizer(
                question, add_special_tokens=False, truncation=False,
            )["input_ids"]
        elif has_chat_template:
            # Format question with chat template (system + user + generation prompt)
            messages = [
                {"role": "system", "content": QA_SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {question}"},
            ]
            chat_kwargs = dict(tokenize=False, add_generation_prompt=True)
            # Disable thinking for Qwen3 models
            try:
                chat_prompt = model.tokenizer.apply_chat_template(
                    messages, enable_thinking=False, **chat_kwargs,
                )
            except TypeError:
                chat_prompt = model.tokenizer.apply_chat_template(
                    messages, **chat_kwargs,
                )
            chat_prompt_ids = model.tokenizer(
                chat_prompt, add_special_tokens=False, truncation=False,
            )["input_ids"]
        else:
            # Fallback: raw question tokens
            chat_prompt_ids = model.tokenizer(
                question, add_special_tokens=False, truncation=False,
            )["input_ids"]

        answer_ids = model.tokenizer(
            answer, add_special_tokens=False, truncation=False,
        )["input_ids"] + [model.eos_id]

        prompt_ids = (
            [mem_placeholder_id] * total_mem_length
            + [model.ft_token_id]
            + chat_prompt_ids
        )

        text_output["prompt_answer_ids"].append(prompt_ids + answer_ids)
        labels = [-100] * len(prompt_ids) + answer_ids
        text_output["labels"].append(labels)

        assert len(text_output["prompt_answer_ids"][-1]) == len(labels)

        if _has_teacher:
            teacher_prompt = examples["teacher_prompt"][idx]
            teacher_prompt_ids = model.tokenizer(
                teacher_prompt, add_special_tokens=False, truncation=False,
            )["input_ids"]
            # Cap teacher_prompt at ctx_max-len(answer_ids) so teacher_input_ids
            # stays within model_max_length (decoder forward sees this).
            t_budget = ctx_max - len(answer_ids)
            if len(teacher_prompt_ids) > t_budget:
                teacher_prompt_ids = teacher_prompt_ids[-t_budget:]
            teacher_input_ids = teacher_prompt_ids + answer_ids
            text_output["teacher_input_ids"].append(teacher_input_ids)
            text_output["teacher_offset"].append(len(teacher_prompt_ids))

    return text_output


class SeDeMCollator:
    """Data collator that also handles the optional teacher_input_ids field."""
    def __init__(self, pad_token_id, pad_to_multiple_of=None):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, examples):
        input_ids = [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in examples]
        labels = [torch.tensor(ex["labels"], dtype=torch.long) for ex in examples]
        prompt_answer_ids = [torch.tensor(ex["prompt_answer_ids"], dtype=torch.long) for ex in examples]

        input_ids = self._pad(input_ids, fill_value=self.pad_token_id)
        prompt_answer_ids = self._pad(prompt_answer_ids, fill_value=self.pad_token_id)
        labels = self._pad(labels)

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "prompt_answer_ids": prompt_answer_ids,
        }

        # distillation teacher input (optional)
        if "teacher_input_ids" in examples[0] and examples[0]["teacher_input_ids"]:
            teacher = [torch.tensor(ex["teacher_input_ids"], dtype=torch.long) for ex in examples]
            batch["teacher_input_ids"] = self._pad(teacher, fill_value=self.pad_token_id)

        # teacher_offset for completion-style distillation alignment
        if "teacher_offset" in examples[0]:
            batch["teacher_offset"] = torch.tensor(
                [ex["teacher_offset"] for ex in examples], dtype=torch.long
            )

        # evidence mask: per-segment 0/1 from supporting_facts.
        # Variable num_segments per example → pad to max with zeros.
        if "evidence_mask" in examples[0] and examples[0]["evidence_mask"] is not None:
            ev_lists = [ex["evidence_mask"] for ex in examples]
            max_segs = max(len(x) for x in ev_lists)
            ev_padded = torch.zeros(len(ev_lists), max_segs, dtype=torch.long)
            for i, ev in enumerate(ev_lists):
                if len(ev) > 0:
                    ev_padded[i, : len(ev)] = torch.tensor(ev, dtype=torch.long)
            batch["evidence_mask"] = ev_padded

        return batch

    def _pad(self, sequences, fill_value=-100):
        max_length = max(len(x) for x in sequences)
        if self.pad_to_multiple_of:
            max_length = ((max_length - 1) // self.pad_to_multiple_of + 1) * self.pad_to_multiple_of
        padded = torch.full((len(sequences), max_length), fill_value, dtype=torch.long)
        for i, seq in enumerate(sequences):
            padded[i, : len(seq)] = seq
        return padded


