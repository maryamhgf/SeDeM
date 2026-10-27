"""Shared helpers for the analysis scripts: dataset loading with evidence spans,
evidence-to-token masks, segment-wise compression, and model loading."""

import math
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset

from sedem.modeling import SeDeMModel


def _build_paper_text(abstract: str, full_text: dict) -> str:
    parts = [abstract]
    for sec_name, paragraphs in zip(full_text["section_name"], full_text["paragraphs"]):
        parts.append(f"{sec_name}\n" + "\n".join(paragraphs))
    return "\n\n".join(parts)


def _infer_answer_type(annotator: dict) -> Optional[str]:
    if annotator.get("unanswerable"):
        return None
    if annotator.get("free_form_answer"):
        return "free_form"
    if annotator.get("extractive_spans"):
        return "extractive"
    if annotator.get("yes_no") is not None:
        return "yes_no"
    return None


def load_qasper_with_evidence(dataset_name: str, cache_dir: str) -> List[dict]:
    """Load all QASPER validation questions that have a non-empty answer
    AND at least one evidence span.
    """
    load_kwargs = dict(cache_dir=cache_dir)
    if "qasper" in dataset_name:
        load_kwargs["revision"] = "refs/convert/parquet"
    ds = load_dataset(dataset_name, **load_kwargs)["validation"]

    out = []
    for paper in ds:
        paper_id = paper.get("id", paper.get("paper_id", ""))
        paper_text = _build_paper_text(paper["abstract"], paper["full_text"])
        qas = paper["qas"]
        for q_idx, question in enumerate(qas["question"]):
            annotators = qas["answers"][q_idx]["answer"]
            answer_text = None
            answer_type = None
            evidence_texts: List[str] = []
            for ann in annotators:
                if ann.get("evidence"):
                    evidence_texts.extend(ann["evidence"])
                if answer_text is None:
                    atype = _infer_answer_type(ann)
                    if atype is None:
                        continue
                    if atype == "free_form":
                        answer_text = ann["free_form_answer"]
                    elif atype == "extractive":
                        answer_text = ", ".join(ann["extractive_spans"])
                    elif atype == "yes_no":
                        answer_text = "Yes" if ann["yes_no"] else "No"
                    answer_type = atype
            if answer_text and evidence_texts:
                out.append({
                    "paper_id": paper_id,
                    "paper_text": paper_text,
                    "question": question,
                    "answer": answer_text,
                    "answer_type": answer_type,
                    "evidence_texts": list(dict.fromkeys(evidence_texts)),
                })
    return out


def load_hotpotqa_with_evidence(
    dataset_name: str, cache_dir: str, include_distractors: bool = True,
) -> List[dict]:
    """Load HotpotQA validation questions. Evidence = supporting-fact sentences.

    `include_distractors=True` (default): paper_text contains BOTH gold and
    distractor paragraphs (matches the `hotpotqa_full` transform). This gives
    non-trivial segment-level discrimination — evidence segments are a small
    subset of all segments.

    `include_distractors=False`: gold paragraphs only (matches the model's
    training input). Useful for sanity-check that scoring works, but every
    segment becomes an evidence segment so segment-level AP collapses to 1.0.
    """
    ds = load_dataset(dataset_name, "distractor", cache_dir=cache_dir)["validation"]

    out = []
    for ex in ds:
        answer = ex["answer"]
        if not answer or not answer.strip():
            continue

        ctx_titles = ex["context"]["title"]
        ctx_sentences = ex["context"]["sentences"]
        gold_titles = set(ex["supporting_facts"]["title"])
        gold_sent_ids = ex["supporting_facts"]["sent_id"]

        title_to_sents = dict(zip(ctx_titles, ctx_sentences))
        kept_paragraphs = []
        evidence_texts = []
        for title in ctx_titles:
            sents = title_to_sents[title]
            if not include_distractors and title not in gold_titles:
                continue
            para = f"{title}: {''.join(sents).strip()}"
            kept_paragraphs.append(para)
            if title in gold_titles:
                for sf_title, sf_sid in zip(ex["supporting_facts"]["title"],
                                             gold_sent_ids):
                    if sf_title == title and 0 <= sf_sid < len(sents):
                        sent = sents[sf_sid].strip()
                        if sent:
                            evidence_texts.append(sent)

        if not kept_paragraphs or not evidence_texts:
            continue

        paper_text = "\n\n".join(kept_paragraphs)
        out.append({
            "paper_id": ex["id"],
            "paper_text": paper_text,
            "question": ex["question"],
            "answer": answer,
            "answer_type": ex.get("type", "unknown"),
            "evidence_texts": list(dict.fromkeys(evidence_texts)),
        })
    return out


def stratified_pool(
    examples: List[dict], n: int, seed: int, oversample_factor: int = 4,
) -> "dict[str, Tuple[int, List[dict]]]":
    """Return per-answer-type candidate pools, each with its quota and
    `oversample_factor`x candidates to draw from.

    Caller is expected to iterate type-by-type, taking up to `quota` valid
    analyses per type. This preserves the type quotas even when many candidates
    get skipped at analysis time (e.g., evidence outside truncated context).
    """
    rng = random.Random(seed)

    by_type: dict = {}
    for ex in examples:
        by_type.setdefault(ex["answer_type"], []).append(ex)

    types = sorted(by_type.keys())
    per_type_quota = [n // len(types)] * len(types)
    for i in range(n % len(types)):
        per_type_quota[i] += 1

    pools = {}
    for t, quota in zip(types, per_type_quota):
        pool = list(by_type[t])
        rng.shuffle(pool)
        pools[t] = (quota, pool[: quota * oversample_factor])
    return pools


# =====================================================================
# Evidence → token-position mask
# =====================================================================

def build_evidence_mask(
    tokenizer,
    context_ids: List[int],
    evidence_texts: List[str],
    min_span_len: int = 5,
    head_match_len: int = 8,
    fuzzy_min_overlap: float = 0.6,
) -> np.ndarray:
    """Locate each evidence text inside the tokenized context.

    Strategy (in order):
      1. Strict head+tail token-id match on the first/last `head_match_len` tokens
         (handles the common case of byte-identical text in the paper).
      2. Strict head-only match on the first `head_match_len` tokens (handles
         the case where the evidence text in QASPER's annotation is a prefix of
         a longer source span).
      3. Fuzzy: best-overlap window — among all candidate windows of the same
         length, take the one whose token-id overlap with `ev_ids` is highest;
         accept if overlap >= `fuzzy_min_overlap`. Handles minor whitespace /
         punctuation tokenization differences.
    """
    mask = np.zeros(len(context_ids), dtype=bool)
    ctx_arr = np.asarray(context_ids, dtype=np.int64)

    for ev_text in evidence_texts:
        ev_ids = tokenizer(ev_text, add_special_tokens=False, truncation=False)["input_ids"]
        L = len(ev_ids)
        if L < min_span_len or L > len(context_ids):
            continue

        ev_arr = np.asarray(ev_ids, dtype=np.int64)

        # 1) Strict head + tail match.
        head_n = min(head_match_len, L)
        tail_n = min(head_match_len, L)
        ev_head = ev_arr[:head_n]
        ev_tail = ev_arr[-tail_n:]
        found = False
        for start in range(len(context_ids) - L + 1):
            if (np.array_equal(ctx_arr[start:start + head_n], ev_head) and
                np.array_equal(ctx_arr[start + L - tail_n:start + L], ev_tail)):
                mask[start:start + L] = True
                found = True
                break
        if found:
            continue

        # 2) Head-only strict match.
        for start in range(len(context_ids) - head_n + 1):
            if np.array_equal(ctx_arr[start:start + head_n], ev_head):
                end = min(start + L, len(context_ids))
                mask[start:end] = True
                found = True
                break
        if found:
            continue

        # 3) Fuzzy best-overlap window.
        ev_set = set(ev_ids)
        best_overlap = 0.0
        best_start = -1
        # Coarse stride to keep this fast for long contexts; refine after.
        stride = max(1, L // 8)
        for start in range(0, len(context_ids) - L + 1, stride):
            win = context_ids[start:start + L]
            overlap = sum(1 for t in win if t in ev_set) / L
            if overlap > best_overlap:
                best_overlap = overlap
                best_start = start
        if best_overlap >= fuzzy_min_overlap and best_start >= 0:
            mask[best_start:best_start + L] = True

    return mask


def compress_and_capture_attention(
    model, input_ids: torch.Tensor
) -> Tuple[torch.Tensor, Optional[List[np.ndarray]], List[Tuple[int, int]]]:
    """Run the compressor segment-by-segment and also capture per-slot
    cross-attention over source tokens if the compressor uses cross-attention.

    For ChunkedGated-style compressors that do chunk-local pooling instead of
    cross-attention, attention capture returns None. Segment boundaries are
    deterministic either way.
    """
    total_length = input_ids.size(1)
    num_segments = model.compute_num_segments(total_length)
    segment_length = math.ceil(total_length / num_segments)
    hidden_states = model.extract_or_load_cached_hidden_states(input_ids, num_segments, segment_length)

    has_xattn = hasattr(model.compressor, "cross_attn") and hasattr(model.compressor, "queries")

    mem_list = []
    attn_list: List[np.ndarray] = []
    boundaries: List[Tuple[int, int]] = []
    prev_mem = None

    if has_xattn:
        xattn = model.compressor.cross_attn
        queries_all = model.compressor.queries.expand(1, -1, -1)

    for seg_idx in range(num_segments):
        start = seg_idx * segment_length
        end = min((seg_idx + 1) * segment_length, total_length)
        boundaries.append((start, end))
        seg_hidden = hidden_states[:, start:end, :].float()

        comp_out = model.compressor(seg_hidden, prev_mem=prev_mem, return_attn_stats=True)
        seg_mem = comp_out[0] if isinstance(comp_out, tuple) else comp_out
        mem_list.append(seg_mem)

        if has_xattn:
            if prev_mem is not None:
                kv = torch.cat([prev_mem.detach(), seg_hidden], dim=1)
                prev_len = prev_mem.size(1)
            else:
                kv = seg_hidden
                prev_len = 0
            q_normed = xattn.norm_q(queries_all)
            kv_normed = xattn.norm_kv(kv)
            _, attn_w = xattn.attn(
                q_normed, kv_normed, kv_normed,
                need_weights=True, average_attn_weights=False,
            )
            attn_to_ctx = attn_w[0].mean(dim=0)[:, prev_len:]
            attn_list.append(attn_to_ctx.float().cpu().numpy())

        prev_mem = seg_mem

    memory_raw = torch.cat(mem_list, dim=1)
    return memory_raw, (attn_list if has_xattn else None), boundaries


def load_model(model_args, training_args, device):
    model_args.train = True  # ensures LoRA wrapping and correct _is_training branch
    lora_config = None
    if model_args.lora_decoder:
        from peft import LoraConfig
        target_modules = [m.strip() for m in model_args.lora_target_modules.split(",")]
        lora_config = LoraConfig(
            r=model_args.lora_r, lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=target_modules, bias="none", task_type="CAUSAL_LM",
        )
    model = SeDeMModel(model_args, training_args, lora_config=lora_config)
    model.eval()
    if hasattr(model, "decoder"):
        model.decoder.gradient_checkpointing_disable()
    model = model.to(device)
    return model
