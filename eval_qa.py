"""
Evaluate a fine-tuned SeDeMModel QA model.

Generates answers on the eval set and reports:
  - ROUGE-L F1, token-level F1, exact match
  - Time to first token (TTFT), total generation time
  - Peak GPU memory usage

Usage:
    python eval_qa.py \
        --model_name_or_path meta-llama/Llama-3.2-3B \
        --compressor_type learnable_query \
        --compressor_hidden_dim 512 \
        --compressor_num_heads 8 \
        --memory_slots_per_segment 128 \
        --compression_factor 4 \
        --extraction_layer -1 \
        --lora_decoder True \
        --lora_r 64 \
        --lora_alpha 128 \
        --lora_dropout 0.05 \
        --lora_target_modules "q_proj,k_proj,v_proj,o_proj" \
        --restore_from /path/to/model.safetensors \
        --dataset_name allenai/qasper \
        --dataset_transform qasper \
        --eval_output_file eval_results.json \
        --max_new_tokens 512 \
        --bf16 True \
        --output_dir /tmp/eval_dummy
"""

import transformers
import transformers.utils.import_utils
import transformers.trainer

import json
import re
import time
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
from datasets import load_dataset

from safetensors.torch import load_file as safe_load_file
from sedem.modeling import (
    SeDeMModel,
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from sedem.data import (
    get_qa_eval_transform,
    apply_subset_manifest,
    resolve_transform_name,
    resolve_load_kwargs,
    resolve_splits,
)
from sedem.evidence_labels_qasper import (
    evidence_labels_from_qasper_spans,
    select_top_k_blocks,
)
import random as _random


# =====================================================================
# Eval-specific arguments
# =====================================================================

@dataclass
class EvalArguments:
    max_new_tokens: int = field(
        default=512, metadata={"help": "Maximum tokens to generate per example"}
    )
    max_eval_samples: Optional[int] = field(
        default=None, metadata={"help": "Limit eval to N examples (None = all)"}
    )
    eval_output_file: str = field(
        default="eval_results.json", metadata={"help": "Path to save JSON results"}
    )
    eval_split: str = field(
        default="validation", metadata={"help": "Dataset split to evaluate on"}
    )
    memory_ablation: str = field(
        default="none",
        metadata={"help": "Counterfactual memory ablation for causal analysis. "
                  "'none': normal evaluation. "
                  "'zero': zero out all memory tokens (tests if decoder depends on memory at all). "
                  "'random': replace memory with random vectors of matched norm (tests if decoder uses content vs just position). "
                  "'shuffle': randomly permute memory tokens across positions (tests if segment-position mapping matters). "
                  "'cross': use previous example's memory for current example (tests if decoder uses document-specific content). "
                  "'cross': use previous example's memory for current example (tests if decoder uses document-specific content)."}
    )
    memory_diagnostics: bool = field(
        default=False,
        metadata={"help": "Record per-example memory-slot diversity statistics (pairwise cosine, effective rank)."}
    )
    selection_mode: str = field(
        default="off",
        metadata={"help": "Selection selection mode for decompress-inject eval. "
                  "'off': inject all decompressed blocks (no selection). "
                  "'label': ceiling — pick top-K blocks from evidence-positive segments. "
                  "'random': floor — pick K blocks uniformly at random. "
                  "'learned_pooled': use the trained pooled-query selector, requires "
                  "--selector_checkpoint. "
                  "'learned': use the trained late-interaction selector, requires "
                  "--selector_checkpoint."}
    )
    selector_checkpoint: str = field(
        default="",
        metadata={"help": "Path to selector safetensors (required for r1_learned/r2_learned)."}
    )
    selector_config: str = field(
        default="",
        metadata={"help": "Path to selector_config.json (produced by train_selector.py). "
                  "If empty, inferred as selector_checkpoint's sibling 'selector_config.json'."}
    )
    context_max_length: int = field(
        default=4096,
        metadata={"help": "Max tokens per context before truncation. Default 4096 (historical). "
                  "Increase to 8192 for long papers (Qasper) to recover evidence past the cutoff. "
                  "With inject_decompressed + selection, decoder cost stays flat, only compressor cost grows."}
    )
    top_k: int = field(
        default=8,
        metadata={"help": "K for 'label' / 'random' selection_mode (number of blocks to inject). "
                  "If K >= total blocks, all blocks are kept. "
                  "Named top_k to avoid clashing with ModelArguments.selection_top_k, "
                  "which configures the MemoryRetrievalSystem during training."}
    )
    selection_seed: int = field(
        default=0,
        metadata={"help": "Seed for random selection within 'label' (picks K positives at random) "
                  "and 'random' (picks K blocks at random). Repeating with the same seed "
                  "and dataset reproduces the exact selection."}
    )
    selection_block_size_mode: str = field(
        default="rate",
        metadata={"help": "Block granularity for LEARNED selection (r1_learned/r2_learned). "
                  "'rate' (default): block_size = compression_factor (sub-segment blocks; "
                  "preserves existing behavior). 'segment': block_size = memory_slots (1 block = 1 "
                  "full segment), required to match selectors TRAINED at paragraph/segment "
                  "granularity (hotpot/twowiki/musique cross-dataset). No effect when "
                  "selection_mode='off'."}
    )
    ruler_task: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated RULER task names to keep (e.g. 'qa_2' for HotpotQA, "
                  "'qa_1,qa_2' for SQuAD+HotpotQA). Only applied when the loaded split has a "
                  "'task' column (RULER datasets like simonjegou/ruler). Default None disables "
                  "filtering — no behavior change for non-RULER datasets."}
    )
    parallel: bool = field(
        default=False,
        metadata={"help": "Enable parallel-segment compression in SeDeMModel.compress_segments "
                  "(Option A: batch segments along the B dim, one backbone forward). "
                  "Safe only for stateless compressors (SegmentCompressor always; "
                  "LearnableQueryCompressor when ignore_prev_mem=True). Default False preserves "
                  "the sequential loop. Output is identical to sequential up to FP noise; "
                  "the only difference is TTFT (faster) and peak GPU memory (higher)."}
    )
    batched_compressor: bool = field(
        default=False,
        metadata={"help": "Enable batched-compressor fast path on top of --parallel "
                  "(Option C): collapse the per-segment compressor for-loop into ONE "
                  "batched compressor call. Implies --parallel. Safe for stateless "
                  "compressors (SeDeM / SegmentCompressor with no_z_global+"
                  "linear_writer); output is bit-exact mathematically, with cuDNN/cuBLAS "
                  "dispatch fp drift ~1e-6 fp32. Default False reproduces the sequential "
                  "compressor loop exactly. Use --verify_batched_n N to A/B the first N "
                  "examples and assert max_abs_diff == 0 before trusting the speedup."}
    )
    verify_batched_n: int = field(
        default=0,
        metadata={"help": "When > 0 and --batched_compressor is set, run both sequential and "
                  "batched compressor on the first N examples and assert max_abs_diff == 0. "
                  "Default 0 disables. Useful as a one-shot bit-exactness check at the "
                  "start of an eval run before trusting the speedup."}
    )
    compile_decoder: bool = field(
        default=False,
        metadata={"help": "Apply torch.compile(mode='default', dynamic=True) to each decoder "
                  "layer. mode='default' does Inductor kernel fusion only (no CUDA Graphs), "
                  "safe under the two-phase forward. Targets per-step kernel-launch overhead "
                  "at batch=1. First example pays a 5-10 min compile warmup. Output bit-exact "
                  "(no numerical reordering, just fused kernels). Pairs with --parallel + "
                  "--batched_compressor on long-answer datasets where the warmup amortizes. Also "
                  "raises torch._dynamo.config.cache_size_limit to 256 so KV-cache structural "
                  "variation per decode step doesn't fall back to eager. Named to avoid "
                  "collision with TrainingArguments.compile_decoder."}
    )
    compile_compressor: bool = field(
        default=False,
        metadata={"help": "Apply torch.compile(mode='default', dynamic=True) to model.compressor "
                  "(the SegmentCompressor / SeDeM compressor). Targets the compressor "
                  "kernel-launch overhead on the compress hotpath. Pairs with --parallel + "
                  "--batched_compressor (parallel calls the compressor once per segment; batched_compressor "
                  "calls it once across all segments). dynamic=True handles per-segment shape "
                  "variation. compressor_is_stateless() unwraps OptimizedModule via _orig_mod "
                  "so the parallel fast path is still taken. Bit-exact (Inductor fuses kernels "
                  "without reordering reductions). First example pays compile warmup. Default off."}
    )
    kv_cache: bool = field(
        default=False,
        metadata={"help": "Enable the cached two-phase decoder path: Phase 1 caches text K,V; "
                  "Phase 2 caches [decomp|text] K,V across decode steps. Sets env var "
                  "SEDEM_KV_CACHE=1 which is read inside generate_with_injection. Output "
                  "equivalent to the non-cached path modulo numerical noise. Default off "
                  "preserves the uncached path."}
    )
    shape_stable_kv: bool = field(
        default=False,
        metadata={"help": "When --kv_cache is on, allocate StaticCache(max_cache_len) "
                  "instead of DynamicCache so dynamo sees the SAME past_key_values shape "
                  "across all decode steps. Eliminates per-step compile-thrash. Bit-exact "
                  "vs DynamicCache modulo bf16 reduction-order. Default off."}
    )
    verify_shape_stable_bitexact: int = field(
        default=0,
        metadata={"help": "If >0 and --shape_stable_kv is on, on first N examples run BOTH "
                  "the StaticCache and DynamicCache paths and compare argmax tokens at each "
                  "decode step. Aborts on mismatch."}
    )
    backbone_early_exit: bool = field(
        default=False,
        metadata={"help": "Stop the compress-phase backbone forward at extraction_layer "
                  "(skip layers ℓ..L-1 whose outputs are never read). Bit-exact since "
                  "layer ℓ output is a pure function of layers 0..ℓ-1. For 1B with ℓ=8/16 "
                  "saves ~50%% of backbone compute on the compress hotpath; for ℓ=14/16 "
                  "saves ~12.5%%. Default off."}
    )
    verify_backbone_early_exit: int = field(
        default=0,
        metadata={"help": "If >0 and --backbone_early_exit is on, on first N backbone calls "
                  "A/B against the full forward and assert max_abs_diff == 0 (strict bit-exact). "
                  "Aborts on drift. Default 0 disables."}
    )
    efficient_decompress: bool = field(
        default=False,
        metadata={"help": "Decompress ONLY the selection-selected blocks instead of "
                  "decompressing all M_total memory slots (full reconstruction) and then "
                  "discarding all but the chosen K blocks. Gathers the chosen blocks' raw "
                  "memory slots and runs a single decompressor call. F1-preserving for a per-slot "
                  "decompressor (Decompressor: shared MLP applied independently per slot, "
                  "no cross-slot signal, no position embeddings) — output matches decompress-all "
                  "modulo fp32 GEMM reduction-order (~1e-6, washed out by the bf16 backbone cast). "
                  "Only affects selection modes "
                  "that pass selected_block_indices (label/random/learned); the 'full' mode "
                  "is unchanged. Default off. Recommend pairing with "
                  "--verify_efficient_decompress on first use."}
    )
    verify_efficient_decompress: int = field(
        default=0,
        metadata={"help": "If >0 and --efficient_decompress is on, on the first N decompress calls "
                  "A/B against the decompress-all-then-subset path and assert max_abs_diff == 0 "
                  "(strict bit-exact). Aborts on drift (e.g. a non-per-slot decompressor). "
                  "Default 0 disables."}
    )
    concise_prompt: bool = field(
        default=False,
        metadata={"help": "Answer-format lever (default OFF). "
                  "Wrap the question with --concise_prompt_template before tokenizing so the "
                  "model is instructed to answer briefly / yes-no / Unanswerable. Pure zero-shot "
                  "prompt change (no QASPER fine-tuning, no train/inference ft mismatch)."}
    )
    concise_prompt_template: str = field(
        default=("Answer the question in as few words as possible. "
                 "For yes/no questions, answer only \"Yes\" or \"No\". "
                 "If the answer is not in the paper, answer \"Unanswerable\". "
                 "Question: {question}"),
        metadata={"help": "Template applied when --concise_prompt is on. Must contain '{question}'. "
                  "Kept question-last so generation still starts immediately after the question "
                  "tokens, matching the Stage-2 training format."}
    )
    answer_postproc: bool = field(
        default=False,
        metadata={"help": "Post-processing lever (default OFF). "
                  "Before scoring, trim each prediction to first line, strip a leading 'Answer:', "
                  "cut to first sentence, and canonicalize yes/no answers to a single word for "
                  "yes/no-style questions. Pure text transform; does not change generation."}
    )


# =====================================================================
# Metrics (no external dependencies)
# =====================================================================

def _normalize(text):
    """SQuAD/MRQA-canonical normalization: lowercase, strip punctuation,
    remove articles (a/an/the), collapse whitespace."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    return ' '.join(text.split())


def _lcs_length(x, y):
    """Length of the longest common subsequence of two lists."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    # Space-optimized: only keep two rows
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def compute_rouge_l(prediction, reference):
    """ROUGE-L F1 at word level."""
    pred_tokens = _normalize(prediction).split()
    ref_tokens = _normalize(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_token_f1(prediction, reference):
    """SQuAD-style token-level F1."""
    pred_tokens = _normalize(prediction).split()
    ref_tokens = _normalize(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = set(pred_tokens) & set(ref_tokens)
    num_common = sum(min(pred_tokens.count(w), ref_tokens.count(w)) for w in common)
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_exact_match(prediction, reference):
    """Exact match after normalization."""
    return _normalize(prediction) == _normalize(reference)


# ---- Tier-1b answer post-processing (flag-gated; default OFF) ---------------
_YN_QUESTION = re.compile(r'^(is|are|does|do|did|can|could|was|were|has|have|had|'
                          r'should|shall|will|would|may|might|must)\b', re.I)
_YN_ANSWER = re.compile(r'^\s*(yes|no)\b[\s,.:;!?]*', re.I)


def postprocess_answer(text, question):
    """Trim a verbose prediction to its scoring-relevant head (see --answer_postproc).

    Steps: first line -> strip leading 'Answer:' -> first sentence -> yes/no
    canonicalization (only when the question is yes/no-style and the prediction
    opens with yes/no). Returns the original text unchanged if it is already short.
    """
    if not text:
        return text
    t = text.strip().split('\n')[0].strip()                      # first line
    t = re.sub(r'^\s*(answer|a)\s*[:\-]\s*', '', t, flags=re.I)  # strip 'Answer:'
    m = re.search(r'[.!?](\s|$)', t)                             # first sentence
    if m:
        t = t[:m.start() + 1].strip()
    if _YN_QUESTION.match((question or '').strip()):             # yes/no canon
        ma = _YN_ANSWER.match(t)
        if ma:
            t = ma.group(1).capitalize()
    return t


# =====================================================================
# Generation
# =====================================================================

def _agg(xs):
    """Mean / p50 / p90 / p99 / min / max over per-example values."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
    ys = sorted(xs)
    m = sum(ys) / len(ys)
    def pct(p):
        k = max(0, min(len(ys) - 1, int(round((p / 100.0) * (len(ys) - 1)))))
        return ys[k]
    return {
        "mean": round(m, 2),
        "p50": round(pct(50), 2),
        "p90": round(pct(90), 2),
        "p99": round(pct(99), 2),
        "min": round(ys[0], 2),
        "max": round(ys[-1], 2),
    }


def _kv_cache_stats(past):
    """Walk past_key_values (tuple-of-tuples or DynamicCache) and return
    (total_bytes, seq_len). seq_len is taken from the first layer's key
    tensor at dim -2 (HF convention: [B, n_kv_heads, T, head_dim]).
    Returns (0, 0) if past is None or empty.
    """
    if past is None:
        return 0, 0
    tensors = []
    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        for t in list(past.key_cache) + list(past.value_cache):
            if isinstance(t, torch.Tensor) and t.numel() > 0:
                tensors.append(t)
    else:
        for layer in past:
            if layer is None:
                continue
            for t in layer:
                if isinstance(t, torch.Tensor) and t.numel() > 0:
                    tensors.append(t)
    if not tensors:
        return 0, 0
    total_bytes = sum(t.numel() * t.element_size() for t in tensors)
    seq_len = int(tensors[0].shape[-2])
    return int(total_bytes), seq_len


_EXAMPLE_COUNT = 0  # global counter for verbose debugging

@torch.no_grad()
def generate_answer(model, context_text, question_text, max_new_tokens, device,
                    memory_ablation="none", memory_diagnostics=False,
                    selection_mode="off", selection_top_k=8,
                    selection_evidence=None, selection_rng=None,
                    selector_head=None, context_max_length=4096,
                    selection_block_size_mode="rate",
                    concise_prompt=False, concise_prompt_template=None):
    """
    Compress context and generate an answer using greedy decoding.

    Args:
        memory_ablation: Counterfactual ablation mode.
            'none'    - normal evaluation
            'zero'    - zero out all memory tokens
            'random'  - replace with random vectors of matched per-token norm
            'shuffle' - randomly permute memory tokens across all positions
            'cross'   - use previous example's memory (stored in _PREV_MEMORY)

    Returns dict with: text, ttft_ms, total_time_ms, num_tokens,
                       num_memory_tokens, context_tokens
    """
    global _EXAMPLE_COUNT
    _EXAMPLE_COUNT += 1
    _verbose = _EXAMPLE_COUNT <= 3  # detailed debug for first 3 examples

    # 1. Tokenize
    context_ids = model.tokenizer(
        context_text, max_length=context_max_length, truncation=True,
        padding=False, return_attention_mask=False, add_special_tokens=False,
    )["input_ids"]
    # Tier-1a concise-answer prompt: wrap the question (default off).
    if concise_prompt and concise_prompt_template:
        question_text = concise_prompt_template.format(question=question_text)
    # Format question with chat template for instruct models
    has_chat_template = (hasattr(model.tokenizer, 'chat_template')
                         and model.tokenizer.chat_template is not None)
    if has_chat_template:
        from sedem.data import QA_SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": QA_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question_text}"},
        ]
        chat_kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            chat_prompt = model.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **chat_kwargs,
            )
        except TypeError:
            chat_prompt = model.tokenizer.apply_chat_template(
                messages, **chat_kwargs,
            )
        question_ids = model.tokenizer(
            chat_prompt, add_special_tokens=False, truncation=False,
        )["input_ids"]
    else:
        question_ids = model.tokenizer(
            question_text, truncation=False, padding=False,
            return_attention_mask=False, add_special_tokens=False,
        )["input_ids"]

    context_len = len(context_ids)
    input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)

    if _verbose:
        print(f"\n  [DEBUG example {_EXAMPLE_COUNT}] context_tokens={context_len}, "
              f"question_tokens={len(question_ids)}")

    # Reset peak memory stats so each example gives clean per-example numbers.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    # 2. Compress context (timed)
    torch.cuda.synchronize()
    _t_comp = time.perf_counter()
    memory_tokens = model.compress_segments(input_ids)  # (1, M_total, D)
    torch.cuda.synchronize()
    compress_ms = (time.perf_counter() - _t_comp) * 1000.0
    M_total = memory_tokens.size(1)

    # ---- Memory diversity diagnostics ----
    mem_diversity = None
    if memory_diagnostics:
        with torch.no_grad():
            mem_f = memory_tokens[0].float()  # (M, D)
            mem_normed = F.normalize(mem_f, dim=-1)
            cos_matrix = mem_normed @ mem_normed.T  # (M, M)
            M = cos_matrix.size(0)
            mask = ~torch.eye(M, dtype=torch.bool, device=cos_matrix.device)
            off_diag = cos_matrix[mask]
            # Effective rank via singular values
            _, svs, _ = torch.svd(mem_f)
            svs_norm = svs / (svs.sum() + 1e-8)
            eff_rank = torch.exp(-(svs_norm * (svs_norm + 1e-10).log()).sum()).item()
            mem_diversity = {
                "num_tokens": M,
                "mean_pairwise_cos": round(off_diag.mean().item(), 4),
                "max_pairwise_cos": round(off_diag.max().item(), 4),
                "min_pairwise_cos": round(off_diag.min().item(), 4),
                "effective_rank": round(eff_rank, 2),
                "norm_mean": round(mem_f.norm(dim=-1).mean().item(), 4),
                "norm_std": round(mem_f.norm(dim=-1).std().item(), 4),
            }

    # ---- Counterfactual memory ablations ----
    if memory_ablation != "none":
        if memory_ablation == "zero":
            memory_tokens = torch.zeros_like(memory_tokens)
        elif memory_ablation == "random":
            # Match per-token norms but randomize direction
            norms = memory_tokens.norm(dim=-1, keepdim=True)  # (1, M, 1)
            rand_vecs = torch.randn_like(memory_tokens)
            rand_vecs = rand_vecs / rand_vecs.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            memory_tokens = rand_vecs * norms
        elif memory_ablation == "shuffle":
            # Randomly permute memory tokens across all positions
            perm = torch.randperm(M_total, device=device)
            memory_tokens = memory_tokens[:, perm, :]
        elif memory_ablation == "cross":
            # Use previous example's memory; store current for next
            if not hasattr(generate_answer, '_prev_memory'):
                generate_answer._prev_memory = None
            if generate_answer._prev_memory is not None:
                prev = generate_answer._prev_memory
                # Handle size mismatch: pad or truncate to match current M_total
                prev_M = prev.size(1)
                if prev_M >= M_total:
                    memory_tokens = prev[:, :M_total, :]
                else:
                    memory_tokens = torch.cat([
                        prev,
                        torch.zeros(1, M_total - prev_M, prev.size(2),
                                    device=device, dtype=prev.dtype)
                    ], dim=1)
            # Store current (real) memory for next example
            generate_answer._prev_memory = model.compress_segments(input_ids).clone()
        else:
            raise ValueError(f"Unknown memory_ablation='{memory_ablation}'")

        if _EXAMPLE_COUNT <= 1:
            print(f"  [ABLATION] memory_ablation='{memory_ablation}', "
                  f"M_total={M_total}, "
                  f"mem_norm={memory_tokens.float().norm(dim=-1).mean().item():.4f}")

    if _verbose:
        mem_norms = memory_tokens.float().norm(dim=-1)  # (1, M_total)
        # Pairwise cosine similarity among memory tokens
        mem_normed = torch.nn.functional.normalize(memory_tokens[0].float(), dim=-1)
        cos_sim = mem_normed @ mem_normed.T
        M = cos_sim.size(0)
        mask = ~torch.eye(M, dtype=torch.bool, device=cos_sim.device)
        intra_cos = cos_sim[mask].mean().item()
        print(f"  [DEBUG] raw memory (before norm): M_total={M_total}, "
              f"norm_mean={mem_norms.mean().item():.2f}, "
              f"norm_min={mem_norms.min().item():.2f}, "
              f"norm_max={mem_norms.max().item():.2f}, "
              f"intra_cos={intra_cos:.4f}")

    # 3. Build prompt and embed
    use_injection = getattr(model, '_inject_decompressed', False)

    # ---- Decompress-inject: pass raw (possibly ablated) memory to helper ----
    if use_injection:
        prompt_tensor = torch.tensor([question_ids], dtype=torch.long, device=device)
        # ---- Block selection (label / random / learned) ----
        selected_block_indices = None
        selection_stats = None
        if selection_mode in ("label", "random"):
            positive_mask, _, num_blocks, map_stats = evidence_labels_from_qasper_spans(
                context_text,
                selection_evidence or [],
                model.tokenizer,
                model,
                max_length=context_max_length,
            )
            selected_block_indices, sel_stats = select_top_k_blocks(
                positive_mask, num_blocks, selection_top_k,
                mode=selection_mode,
                rng=selection_rng if selection_rng is not None else _random.Random(0),
            )
            selected_block_indices = selected_block_indices.to(device)
            selection_stats = {**map_stats, **sel_stats, "mode": selection_mode}
            if _verbose:
                print(f"  [RETRIEVAL] mode={selection_mode}, K={selection_top_k}, "
                      f"num_blocks={num_blocks}, "
                      f"n_positive_blocks={map_stats['num_positive_blocks']}, "
                      f"k_selected={sel_stats['k_selected']}, "
                      f"from_positives={sel_stats['n_from_positives']}")
        elif selection_mode in ("learned_pooled", "learned"):
            # Learned selector scores blocks; top-K by score. Also compute
            # the label positive mask so we can report recall@K vs evidence
            # (diagnostic only — doesn't affect selection).
            head = selector_head
            assert head is not None, (
                f"selection_mode={selection_mode!r} requires --selector_checkpoint "
                "(loaded into selector_head)."
            )
            M_total, D = memory_tokens.size(1), memory_tokens.size(2)
            num_segments = M_total // model.memory_slots
            # Block granularity: 'segment' (block_size=memory_slots, 1 block=1 full
            # segment) matches selectors TRAINED at paragraph/segment granularity
            # (hotpot/twowiki/musique cross-dataset). 'rate' (default) = sub-segment.
            block_size = (model.memory_slots if selection_block_size_mode == "segment"
                          else model.compression_factor)
            blocks_per_segment = model.memory_slots // block_size
            num_blocks = num_segments * blocks_per_segment
            blocks = memory_tokens.view(1, num_segments, model.memory_slots, D).view(
                1, num_blocks, block_size, D,
            ).float()

            # selector scores -> top-K blocks
            learned_idx_set = set()
            top_scores = None
            q_ids_t = torch.tensor([question_ids], dtype=torch.long, device=device).clamp(
                max=model.original_vocab_size - 1,
            )
            with torch.no_grad():
                # Query reads only extraction_layer. Flag-gated, default OFF:
                #  --backbone_early_exit ON (no layer-mixing) → bit-exact early-exit
                #  dispatcher (skips ℓ..L-1); else ORIGINAL full forward (preserved).
                if getattr(model, "_backbone_early_exit", False) and not getattr(model, "_mix_extraction_layers", False):
                    question_hs = model.extract_hidden_states(q_ids_t).float()
                else:
                    q_out = model.backbone(q_ids_t, output_hidden_states=True)
                    question_hs = q_out.hidden_states[model.extraction_layer].float()
                # Cross-model: lift enc_dim → dec_dim so q_hs matches block space.
                if hasattr(model, "compressor") and hasattr(model.compressor, "input_proj"):
                    question_hs = model.compressor.input_proj(question_hs)
                scores = head(question_hs, blocks).squeeze(0)  # (num_blocks,)
                K = min(selection_top_k, num_blocks)
                top_scores, top_idx = scores.topk(K)
            learned_idx_set = set(top_idx.tolist())

            union_idx = sorted(learned_idx_set)
            selected_block_indices = torch.tensor(union_idx, dtype=torch.long, device=device)

            # Diagnostics: evidence recall@K if we have evidence labels
            pos_mask, _, _, map_stats = evidence_labels_from_qasper_spans(
                context_text, selection_evidence or [], model.tokenizer, model, max_length=context_max_length,
            )
            pos_set = set(torch.nonzero(pos_mask).flatten().tolist())
            picked = set(selected_block_indices.tolist())
            n_pos_picked = len(pos_set.intersection(picked))
            selection_stats = {
                **map_stats,
                "mode": selection_mode,
                "k_requested": selection_top_k,
                "k_selected": len(picked),
                "n_from_positives": n_pos_picked,
                "recall_at_k": (n_pos_picked / max(len(pos_set), 1)),
                "top_score": float(top_scores.max().item()) if top_scores is not None else 0.0,
                "mean_score": float(scores.mean().item()),
            }
            if _verbose:
                print(f"  [RETRIEVAL] mode={selection_mode}, K={selection_top_k}, "
                      f"num_blocks={num_blocks}, "
                      f"n_positives={map_stats['num_positive_blocks']}, "
                      f"picks_from_pos={n_pos_picked}/{len(picked)}, "
                      f"recall@K={selection_stats['recall_at_k']:.3f}")
        elif selection_mode == "full":
            # Pass all blocks — selected_block_indices stays None → generate_with_injection
            # injects all decompressed slots, no filtering. Option B: full-paper context.
            M_total, D = memory_tokens.size(1), memory_tokens.size(2)
            num_segments = M_total // model.memory_slots
            block_size = model.compression_factor
            blocks_per_segment = model.memory_slots // block_size
            num_blocks = num_segments * blocks_per_segment
            pos_mask, _, _, map_stats = evidence_labels_from_qasper_spans(
                context_text, selection_evidence or [], model.tokenizer, model, max_length=context_max_length,
            )
            selection_stats = {
                **map_stats,
                "mode": "full",
                "k_requested": num_blocks,
                "k_selected": num_blocks,
                "n_from_positives": int(pos_mask.sum().item()),
                "recall_at_k": 1.0,
            }
            if _verbose:
                print(f"  [RETRIEVAL] mode=full, num_blocks={num_blocks} (all injected)")
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        # Pass raw memory — generate_with_injection handles decompression + normalization.
        # return_timing=True asks the function to return a dict with real prefill_ms /
        # decode_ms instead of just the text (bit-exact: only adds sync+perf_counter).
        gen_out = model.generate_with_injection(
            memory_tokens, prompt_tensor, max_new_tokens,
            selected_block_indices=selected_block_indices,
            return_timing=True,
        )
        torch.cuda.synchronize()
        t_end = time.perf_counter()
        gen_total_ms = (t_end - t_start) * 1000.0
        text = gen_out["text"]
        prefill_ms_real = float(gen_out["prefill_ms"])
        decode_ms_real = float(gen_out["decode_ms"])
        decoded_tokens = int(gen_out["num_tokens_generated"])
        # Capture peak memory after the full generation. KV stats are unavailable
        # from this code path (cache is internal); return zeros so downstream
        # aggregation doesn't crash.
        peak_gpu_mb_di = 0.0
        peak_gpu_reserved_mb_di = 0.0
        if torch.cuda.is_available():
            peak_gpu_mb_di = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            peak_gpu_reserved_mb_di = torch.cuda.max_memory_reserved(device) / (1024 * 1024)
        decode_tok_per_s = (decoded_tokens / (decode_ms_real / 1000.0)) if decode_ms_real > 0 and decoded_tokens > 0 else 0.0
        return {
            "text": text,
            "ttft_ms": compress_ms + prefill_ms_real,    # real TTFT = compress + true prefill
            "compress_ms": compress_ms,
            "prefill_ms": prefill_ms_real,                # measured inside generate_with_injection
            "decode_ms": decode_ms_real,                  # measured inside generate_with_injection
            "total_time_ms": compress_ms + gen_total_ms,
            "num_tokens": decoded_tokens,
            "num_memory_tokens": M_total,
            "context_tokens": context_len,
            "kv_cache_bytes": 0,
            "kv_cache_tokens": 0,
            "peak_gpu_mb": round(peak_gpu_mb_di, 1),
            "peak_gpu_reserved_mb": round(peak_gpu_reserved_mb_di, 1),
            "decode_tok_per_s": round(decode_tok_per_s, 2),
            "mem_diversity": mem_diversity,
            "selection_stats": selection_stats,
        }

    # Prefix mode: prepend memory tokens to decoder input
    prompt_ids = (
        [model.mem_id_start] * M_total
        + [model.ft_token_id]
        + question_ids
    )
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # 4. Embed (replaces mem placeholders with actual memory tokens)
    decoder_embs = model.embed_decoder_input_with_prefix(prompt_tensor, memory_tokens)

    if _verbose:
        mem_emb_norms = decoder_embs[0, :M_total, :].float().norm(dim=-1)
        ft_emb_norm = decoder_embs[0, M_total, :].float().norm().item()
        q_emb_norms = decoder_embs[0, M_total+1:, :].float().norm(dim=-1)
        print(f"  [DEBUG] decoder_embs: "
              f"mem_norm_mean={mem_emb_norms.mean().item():.4f}, "
              f"mem_norm_min={mem_emb_norms.min().item():.4f}, "
              f"mem_norm_max={mem_emb_norms.max().item():.4f}, "
              f"ft_norm={ft_emb_norm:.4f}, "
              f"question_norm_mean={q_emb_norms.mean().item():.4f}")

        # Compare: regular text embedding norms (what the decoder expects)
        sample_text_embs = model.backbone.model.embed_tokens(
            torch.tensor([question_ids[:10]], device=device)
        )
        print(f"  [DEBUG] text_embedding_norms (first 10 question tokens): "
              f"mean={sample_text_embs.float().norm(dim=-1).mean().item():.4f}")

    # 5. Prefill — record TTFT
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    out = model.decoder(inputs_embeds=decoder_embs, use_cache=True)

    logits = out.logits[:, -1, :model.original_vocab_size]
    past_kv = out.past_key_values

    torch.cuda.synchronize()
    t_first = time.perf_counter()
    prefill_ms = (t_first - t_start) * 1000.0
    # TTFT (paper definition): wall time from start of compression to first token logit.
    ttft_ms = compress_ms + prefill_ms

    # Snapshot KV-cache size RIGHT after prefill — what the decoder must hold for autoregressive decode.
    kv_cache_bytes, kv_cache_tokens = _kv_cache_stats(past_kv)

    # (KV inject hooks already disabled before prefill)

    if _verbose:
        # Top-10 predictions for the first token
        probs = torch.softmax(logits[0].float(), dim=-1)
        topk = probs.topk(10)
        top_tokens = [model.tokenizer.decode([tid]) for tid in topk.indices.tolist()]
        top_probs = topk.values.tolist()
        print(f"  [DEBUG] first token predictions:")
        for t, p in zip(top_tokens, top_probs):
            print(f"    p={p:.4f}  '{t}'")

        # Check: what fraction of attention/logit probability is "reasonable"
        # vs the model just using its prior
        entropy = -(probs * (probs + 1e-10).log()).sum().item()
        max_entropy = np.log(model.original_vocab_size)
        print(f"  [DEBUG] logit entropy: {entropy:.2f} / {max_entropy:.2f} "
              f"(normalized: {entropy/max_entropy:.4f})")

    # 6. Greedy decode
    next_id = logits.argmax(dim=-1)  # (1,)
    generated_ids = []
    if next_id.item() != model.eos_id:
        generated_ids.append(next_id.item())

        for step_i in range(max_new_tokens - 1):
            next_emb = model.backbone.model.embed_tokens(next_id).unsqueeze(1)
            out = model.decoder(
                inputs_embeds=next_emb,
                past_key_values=past_kv,
                use_cache=True,
            )
            logits = out.logits[:, -1, :model.original_vocab_size]
            past_kv = out.past_key_values
            next_id = logits.argmax(dim=-1)

            if next_id.item() == model.eos_id:
                break
            generated_ids.append(next_id.item())

    torch.cuda.synchronize()
    t_end = time.perf_counter()
    decode_ms = (t_end - t_first) * 1000.0
    # total_ms here = compress + prefill + decode (full wall time for one example)
    total_ms = compress_ms + prefill_ms + decode_ms

    peak_gpu_mb = 0.0
    peak_gpu_reserved_mb = 0.0
    if torch.cuda.is_available():
        peak_gpu_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        peak_gpu_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 * 1024)

    decoded_tokens = len(generated_ids)
    decode_tok_per_s = (decoded_tokens / (decode_ms / 1000.0)) if decode_ms > 0 and decoded_tokens > 0 else 0.0

    text = model.tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {
        "text": text,
        "ttft_ms": ttft_ms,                   # compress + prefill (paper definition)
        "compress_ms": compress_ms,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "total_time_ms": total_ms,
        "num_tokens": decoded_tokens,
        "num_memory_tokens": M_total,
        "context_tokens": context_len,
        "kv_cache_bytes": int(kv_cache_bytes),
        "kv_cache_tokens": int(kv_cache_tokens),
        "peak_gpu_mb": round(peak_gpu_mb, 1),
        "peak_gpu_reserved_mb": round(peak_gpu_reserved_mb, 1),
        "decode_tok_per_s": round(decode_tok_per_s, 2),
        "mem_diversity": mem_diversity,
    }


# =====================================================================
# Model loading
# =====================================================================

def load_model(model_args, training_args, device):
    """Load SeDeMModel with LoRA and fine-tuned checkpoint."""
    # Force train=True so LoRA is applied and checkpoint loaded via _init_training
    model_args.train = True

    lora_config = None
    if model_args.lora_decoder:
        from peft import LoraConfig
        target_modules = [m.strip() for m in model_args.lora_target_modules.split(",")]
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

    model = SeDeMModel(model_args, training_args, lora_config=lora_config)

    # ---- DEBUG: verify checkpoint loaded correctly ----
    print("\n" + "=" * 60)
    print("DEBUG: Model state after checkpoint loading")
    print("=" * 60)

    # 1) mem_scale (critical for memory visibility)
    eff_scale = torch.clamp(model.mem_scale, min=0.01).item()
    raw_scale = model.mem_scale.item()
    print(f"  mem_scale: raw={raw_scale:.6f}, effective={eff_scale:.6f}")
    print(f"  _normalize_mem: {model._normalize_mem}")

    # 2) Compressor gates
    xattn = getattr(model.compressor, 'cross_attn', None)
    if xattn is not None and getattr(xattn, 'gated', False):
        raw_gate = getattr(xattn, 'raw_gate', None)
        if raw_gate is not None:
            g_max = getattr(xattn, 'g_max', 0.5)
            gate_val = g_max * torch.tanh(raw_gate).item()
            print(f"  xattn_gate: raw={raw_gate.item():.6f}, effective={gate_val:.6f}")
    ffn = getattr(model.compressor, 'mlp', None)
    if ffn is not None:
        print(f"  ffn: standard residual (ungated)")

    # 3) Compressor topic norms
    if hasattr(model.compressor, 'topics'):
        q = model.compressor.topics
        print(f"  topics: shape={q.shape}, norm={q.float().norm(dim=-1).mean().item():.4f}")

    # 4) LoRA weights loaded?
    if model_args.lora_decoder:
        lora_norms = []
        for name, p in model.decoder.named_parameters():
            if 'lora_' in name and p.requires_grad:
                lora_norms.append((name.split('.')[-3] + '.' + name.split('.')[-1], p.float().norm().item()))
        if lora_norms:
            print(f"  LoRA params: {len(lora_norms)} tensors")
            # Show a few
            for name, norm in lora_norms[:4]:
                print(f"    {name}: norm={norm:.4f}")
            # Check if any are still at init (near zero for lora_B)
            zero_count = sum(1 for _, n in lora_norms if n < 1e-6)
            if zero_count > 0:
                print(f"    WARNING: {zero_count}/{len(lora_norms)} LoRA params have near-zero norm!")
        else:
            print("  WARNING: No LoRA parameters found!")

    # 5) special_token_embed
    print(f"  special_token_embed: norm={model.special_token_embed.weight.float().norm().item():.4f}")
    print(f"  mem_norm: weight_mean={model.mem_norm.weight.float().mean().item():.4f}, "
          f"bias_mean={model.mem_norm.bias.float().mean().item():.6f}")
    print("=" * 60 + "\n")

    # Switch to eval mode
    model.eval()
    if hasattr(model, 'decoder'):
        model.decoder.gradient_checkpointing_disable()

    model = model.to(device)
    return model


# =====================================================================
# Dataset loading
# =====================================================================

def load_eval_dataset(data_args, eval_args):
    """Load and transform the evaluation split."""
    if data_args.debug_data:
        from datasets import Dataset
        raw = {
            "input": ["The quick brown fox jumps over the lazy dog. " * 20] * 8,
            "prompt": ["What does the fox do?"] * 8,
            "answer": ["jumps over the lazy dog"] * 8,
        }
        return Dataset.from_dict(raw)

    load_kwargs = dict(cache_dir=data_args.dataset_cache_dir)
    load_kwargs.update(resolve_load_kwargs(data_args.dataset_name))

    # Local-JSONL branch (no HF dataset_name set) — rows are already in
    # tokenize-ready format (e.g., niah_traj qa output: {id, input, prompt,
    # answer, ...}). Skip transform resolution; rows go straight through.
    if data_args.dataset_name is None and data_args.data_path:
        eval_file = data_args.eval_data_path or data_args.data_path
        dataset = load_dataset("json", data_files={"eval": eval_file})
        # No transform — assume rows already have {input, prompt, answer, ...}
        split = "eval"
        split_ds = dataset[split]
        if eval_args.ruler_task and "task" in split_ds.column_names:
            keep_tasks = {t.strip() for t in eval_args.ruler_task.split(",") if t.strip()}
            n_before = len(split_ds)
            split_ds = split_ds.filter(lambda x: x["task"] in keep_tasks)
            print(f"[ruler_task] filter {sorted(keep_tasks)}: {n_before} → {len(split_ds)} examples")
        if eval_args.max_eval_samples > 0 and len(split_ds) > eval_args.max_eval_samples:
            split_ds = split_ds.select(range(eval_args.max_eval_samples))
        return split_ds

    dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config,
        **load_kwargs,
    )

    # Optional: filter to a stratified subset from a prebuilt manifest
    if data_args.subset_manifest:
        dataset = apply_subset_manifest(dataset, data_args.subset_manifest)

    # Resolve transform
    transform_name = data_args.dataset_transform or resolve_transform_name(
        data_args.dataset_name, data_args.dataset_config,
    )
    if transform_name is None:
        raise ValueError(
            f"Cannot auto-detect transform for '{data_args.dataset_name}'. "
            "Set --dataset_transform explicitly."
        )

    qa_transform = get_qa_eval_transform(transform_name)
    # eval_split = explicit override > auto-resolved (validation→dev→test)
    split = eval_args.eval_split
    if split not in dataset:
        _, split = resolve_splits(dataset)
    split_ds = dataset[split]
    if eval_args.ruler_task and "task" in split_ds.column_names:
        keep_tasks = {t.strip() for t in eval_args.ruler_task.split(",") if t.strip()}
        n_before = len(split_ds)
        split_ds = split_ds.filter(lambda x: x["task"] in keep_tasks)
        print(f"[ruler_task] filter {sorted(keep_tasks)}: {n_before} → {len(split_ds)} examples")
    eval_dataset = split_ds.map(
        qa_transform, batched=True,
        remove_columns=split_ds.column_names,
    )

    if eval_args.max_eval_samples is not None:
        n = min(eval_args.max_eval_samples, len(eval_dataset))
        eval_dataset = eval_dataset.select(range(n))

    return eval_dataset


# =====================================================================
# Main
# =====================================================================

def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, EvalArguments)
    )
    model_args, data_args, training_args, eval_args = parser.parse_args_into_dataclasses()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reset peak memory tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ---- DEBUG: Inspect checkpoint keys before loading ----
    if training_args.restore_from:
        print(f"\nDEBUG: Inspecting checkpoint: {training_args.restore_from}")
        ckpt_state = safe_load_file(training_args.restore_from)
        ckpt_keys = sorted(ckpt_state.keys())
        print(f"  Total keys: {len(ckpt_keys)}")
        # Categorize keys
        compressor_keys = [k for k in ckpt_keys if k.startswith('compressor.')]
        lora_keys = [k for k in ckpt_keys if 'lora_' in k]
        mem_keys = [k for k in ckpt_keys if k.startswith('mem_')]
        special_keys = [k for k in ckpt_keys if k.startswith('special_')]
        other_keys = [k for k in ckpt_keys if k not in compressor_keys + lora_keys + mem_keys + special_keys]
        print(f"  compressor: {len(compressor_keys)} keys")
        print(f"  LoRA: {len(lora_keys)} keys")
        print(f"  mem_norm/scale: {len(mem_keys)} keys => {mem_keys}")
        print(f"  special_token: {len(special_keys)} keys")
        if other_keys:
            print(f"  OTHER (unexpected): {other_keys}")
        # Print mem_scale value from checkpoint
        if 'mem_scale' in ckpt_state:
            print(f"  checkpoint mem_scale value: {ckpt_state['mem_scale'].item():.6f}")
        else:
            print("  WARNING: mem_scale NOT in checkpoint!")
        # Print gate values from checkpoint
        for k in ckpt_keys:
            if 'raw_gate' in k:
                print(f"  checkpoint {k}: {ckpt_state[k].item():.6f}")
        del ckpt_state  # free memory

    # Load model
    print("\nLoading model...")
    model = load_model(model_args, training_args, device)

    # Toggle parallel-segment compress_segments fast path (default off).
    # --batched_compressor implies --parallel (batched_compressor lives inside compress_segments_parallel).
    if eval_args.parallel or eval_args.batched_compressor:
        model._parallel_segments = True
        comp_cls = type(getattr(model, 'compressor', None)).__name__
        stateless = bool(getattr(model, 'compressor_is_stateless', lambda: False)())
        print(f"[parallel] Enabled parallel-segment compression "
              f"(compressor={comp_cls}, stateless={stateless}). "
              f"Output identical to sequential up to FP noise.")
        if not stateless:
            print(f"[parallel] Note: compressor is not stateless; runtime will "
                  f"fall back to sequential per-call (no harm, just no speedup).")
    if eval_args.batched_compressor:
        model._batched_compressor_fast = True
        if eval_args.verify_batched_n > 0:
            model._verify_batched_remaining = int(eval_args.verify_batched_n)
            print(f"[batched_compressor] Verifier ON: will A/B sequential vs batched on "
                  f"first {eval_args.verify_batched_n} examples and assert max_abs_diff == 0.")
        print(f"[batched_compressor] Enabled batched-compressor fast path (one compressor call "
              f"across all segments). Math-identical to sequential; fp ULP drift ~1e-6.")

    # Bit-exact perf knobs (don't change numerics, just kernel selection).
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    if eval_args.compile_decoder:
        # Raise dynamo cache to fit all KV-cache structural variants without
        # falling back to eager.
        import torch._dynamo as _dynamo
        _dynamo.config.cache_size_limit = 256
        _dynamo.config.accumulated_cache_size_limit = 1024
        print(f"[dynamo] cache_size_limit={_dynamo.config.cache_size_limit}, "
              f"accumulated_cache_size_limit={_dynamo.config.accumulated_cache_size_limit}")
        # Wrap decoder layers. mode='default' = Inductor kernel fusion only (no CUDA
        # Graphs), dynamic=True handles prefill vs decode shape change.
        if hasattr(model.decoder, 'model') and hasattr(model.decoder.model, 'model'):
            _inner = model.decoder.model.model
        else:
            _inner = model.decoder.model
        for _i, _layer in enumerate(_inner.layers):
            _inner.layers[_i] = torch.compile(_layer, mode='default', dynamic=True)
        print(f"[compile_decoder] compiled {len(_inner.layers)} decoder layers "
              f"(mode=default, dynamic=True). First example pays compile warmup.")

    # Compile the compressor (TTFT lever, bit-exact). compressor_is_stateless
    # in sedem.modeling.py unwraps OptimizedModule via _orig_mod so
    # the --parallel fast path still classifies it as stateless.
    if eval_args.compile_compressor:
        if not eval_args.compile_decoder:
            # Ensure dynamo cache is large enough even without decoder compile.
            import torch._dynamo as _dynamo_c
            _dynamo_c.config.cache_size_limit = max(_dynamo_c.config.cache_size_limit, 256)
            _dynamo_c.config.accumulated_cache_size_limit = max(
                _dynamo_c.config.accumulated_cache_size_limit, 1024
            )
        _comp_cls = type(model.compressor).__name__
        model.compressor = torch.compile(model.compressor, mode='default', dynamic=True)
        print(f"[compile_compressor] wrapped model.compressor ({_comp_cls}) with "
              f"torch.compile(mode='default', dynamic=True). First example pays compile "
              f"warmup; bit-exact thereafter (Inductor fuses kernels without reordering "
              f"reductions). Parallel fast path still active.")

    # Cached two-phase decoder path (env-gated inside generate_with_injection).
    if eval_args.kv_cache:
        import os as _os_kv
        _os_kv.environ["SEDEM_KV_CACHE"] = "1"
        print("[kv_cache] SEDEM_KV_CACHE=1 (cached two-phase decoder forward enabled)")
    # Shape-stable KV cache via StaticCache; only meaningful when kv_cache is on.
    if eval_args.shape_stable_kv:
        if not eval_args.kv_cache:
            raise SystemExit("--shape_stable_kv requires --kv_cache (StaticCache only meaningful inside the cached two-phase path)")
        model._shape_stable_kv = True
        print("[shape_stable_kv] model._shape_stable_kv=True (StaticCache; shape-stable across "
              "decode steps; bit-exact vs DynamicCache modulo bf16 reduction-order)")
    if eval_args.verify_shape_stable_bitexact > 0 and not eval_args.shape_stable_kv:
        raise SystemExit("--verify_shape_stable_bitexact requires --shape_stable_kv")
    # Backbone early-exit: skip layers ℓ..L-1 in compress-phase backbone forward.
    if eval_args.backbone_early_exit:
        model._backbone_early_exit = True
        _layer = int(model.extraction_layer)
        _total = int(model.backbone.config.num_hidden_layers)
        _pct = 100.0 * (_total - _layer) / max(_total, 1)
        print(f"[backbone_early_exit] model._backbone_early_exit=True "
              f"(ℓ={_layer}/{_total}, skip {_total-_layer} layers = ~{_pct:.1f}% saved). Bit-exact.")
    if eval_args.verify_backbone_early_exit > 0:
        if not eval_args.backbone_early_exit:
            raise SystemExit("--verify_backbone_early_exit requires --backbone_early_exit")
        model._verify_backbone_early_exit_remaining = int(eval_args.verify_backbone_early_exit)
        print(f"[backbone_early_exit verifier] will A/B-check first "
              f"{eval_args.verify_backbone_early_exit} backbone calls vs full forward "
              "(strict max_abs_diff == 0).")
    # Efficient decompress: only decompress selection-selected blocks.
    if eval_args.efficient_decompress:
        model._efficient_decompress = True
        print("[efficient_decompress] model._efficient_decompress=True (decompress only selected "
              "blocks; bit-exact for per-slot Decompressor).")
    if eval_args.verify_efficient_decompress > 0:
        if not eval_args.efficient_decompress:
            raise SystemExit("--verify_efficient_decompress requires --efficient_decompress")
        model._verify_efficient_decom_remaining = int(eval_args.verify_efficient_decompress)
        print(f"[efficient_decompress verifier] will A/B-check first "
              f"{eval_args.verify_efficient_decompress} decompress calls vs decompress-all-then-subset "
              "(strict max_abs_diff == 0).")

    # Load dataset
    print("Loading evaluation dataset...")
    eval_dataset = load_eval_dataset(data_args, eval_args)

    # ---- Optional: load trained selector head ----
    selector_head = None
    if eval_args.selection_mode in ("learned_pooled", "learned"):
        assert eval_args.selector_checkpoint, (
            f"selection_mode={eval_args.selection_mode!r} requires --selector_checkpoint"
        )
        from sedem.selector import load_selector_config, build_selector_from_config, resolve_selector_checkpoint
        eval_args.selector_checkpoint = resolve_selector_checkpoint(eval_args.selector_checkpoint)
        ret_cfg = load_selector_config(eval_args.selector_checkpoint, eval_args.selector_config or None)
        print(f"  selector_config: {ret_cfg}")
        _want = "pooled" if eval_args.selection_mode == "learned_pooled" else "late_interaction"
        assert ret_cfg.get("selector_type") == _want, (
            f"Config says selector_type={ret_cfg.get('selector_type')!r}, but selection_mode="
            f"{eval_args.selection_mode!r} expects {_want!r}"
        )
        selector_head = build_selector_from_config(ret_cfg, model.dim)
        _state = safe_load_file(eval_args.selector_checkpoint)
        missing, unexpected = selector_head.load_state_dict(_state, strict=True)
        selector_head.to(device).float().eval()
        for p in selector_head.parameters():
            p.requires_grad_(False)
        print(f"  loaded selector weights "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    print(f"Evaluation dataset: {len(eval_dataset)} examples")

    # Run evaluation
    results = []
    all_mem_diversity = []  # collect per-example memory diversity
    selection_rng = _random.Random(eval_args.selection_seed)
    for idx in tqdm(range(len(eval_dataset)), desc="Evaluating"):
        example = eval_dataset[idx]
        evidence = example.get("evidence", None)

        gen = generate_answer(
            model,
            example["input"],
            example["prompt"],
            eval_args.max_new_tokens,
            device,
            memory_ablation=eval_args.memory_ablation,
            memory_diagnostics=eval_args.memory_diagnostics,
            selection_mode=eval_args.selection_mode,
            selection_top_k=eval_args.top_k,
            selection_evidence=evidence,
            selection_rng=selection_rng,
            selector_head=selector_head,
            context_max_length=eval_args.context_max_length,
            selection_block_size_mode=eval_args.selection_block_size_mode,
            concise_prompt=eval_args.concise_prompt,
            concise_prompt_template=eval_args.concise_prompt_template,
        )

        raw_prediction = gen["text"]
        prediction = raw_prediction
        # Tier-1b post-processing (default off → prediction unchanged).
        if eval_args.answer_postproc:
            prediction = postprocess_answer(raw_prediction, example["prompt"])

        # Score against all annotator answers (max over references)
        all_refs = example.get("all_answers", None)
        if all_refs is None or len(all_refs) == 0:
            all_refs = [example["answer"]]

        rouge_l = max(compute_rouge_l(prediction, ref) for ref in all_refs)
        token_f1 = max(compute_token_f1(prediction, ref) for ref in all_refs)
        em = any(compute_exact_match(prediction, ref) for ref in all_refs)

        results.append({
            "id": idx,
            "question": example["prompt"],
            "reference": all_refs[0],
            "all_references": all_refs,
            "prediction": prediction,
            "raw_prediction": raw_prediction,
            "rouge_l_f1": round(rouge_l, 4),
            "token_f1": round(token_f1, 4),
            "exact_match": em,
            "ttft_ms": round(gen["ttft_ms"], 1),
            "compress_ms": round(gen.get("compress_ms", 0.0), 1),
            "prefill_ms": round(gen.get("prefill_ms", 0.0), 1),
            "decode_ms": round(gen.get("decode_ms", 0.0), 1),
            "generation_time_ms": round(gen["total_time_ms"], 1),
            "num_generated_tokens": gen["num_tokens"],
            "context_tokens": gen["context_tokens"],
            "num_memory_tokens": gen["num_memory_tokens"],
            "kv_cache_bytes": int(gen.get("kv_cache_bytes", 0)),
            "kv_cache_tokens": int(gen.get("kv_cache_tokens", 0)),
            "peak_gpu_mb": gen.get("peak_gpu_mb", 0.0),
            "peak_gpu_reserved_mb": gen.get("peak_gpu_reserved_mb", 0.0),
            "decode_tok_per_s": gen.get("decode_tok_per_s", 0.0),
            "selection_stats": gen.get("selection_stats"),
        })

        # Collect and print memory diversity
        if gen.get("mem_diversity"):
            all_mem_diversity.append(gen["mem_diversity"])
        if gen.get("mem_diversity") and idx < 5:
            md = gen["mem_diversity"]
            print(f"  [MEM DIVERSITY] tokens={md['num_tokens']}, "
                  f"cos_mean={md['mean_pairwise_cos']}, "
                  f"cos_max={md['max_pairwise_cos']}, "
                  f"eff_rank={md['effective_rank']}, "
                  f"norm={md['norm_mean']}±{md['norm_std']}")

        # Print each example
        print(f"\n--- Example {idx+1}/{len(eval_dataset)} ---")
        print(f"  Q: {example['prompt']}")
        print(f"  A (refs): {all_refs}")
        print(f"  A (pred): {prediction}")
        print(f"  ROUGE-L={rouge_l:.4f}, F1={token_f1:.4f}, EM={em}")

        # Periodic aggregate progress
        if (idx + 1) % 50 == 0:
            avg_rouge = np.mean([r["rouge_l_f1"] for r in results])
            avg_f1 = np.mean([r["token_f1"] for r in results])
            print(f"\n  === Running avg [{idx+1}/{len(eval_dataset)}]: "
                  f"ROUGE-L={avg_rouge:.4f}, F1={avg_f1:.4f} ===")

    # Aggregate metrics
    peak_mem_mb = 0.0
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    aggregated = {
        "rouge_l_f1": round(float(np.mean([r["rouge_l_f1"] for r in results])), 4),
        "token_f1": round(float(np.mean([r["token_f1"] for r in results])), 4),
        "exact_match": round(float(np.mean([r["exact_match"] for r in results])), 4),
        "avg_ttft_ms": round(float(np.mean([r["ttft_ms"] for r in results])), 1),
        "avg_generation_time_ms": round(float(np.mean([r["generation_time_ms"] for r in results])), 1),
        "avg_generated_tokens": round(float(np.mean([r["num_generated_tokens"] for r in results])), 1),
        "peak_gpu_memory_mb": round(peak_mem_mb, 1),
        "num_examples": len(results),
        # --- per-example timing breakdowns ---
        "compress_ms": _agg([r.get("compress_ms", 0.0) for r in results]),
        "prefill_ms": _agg([r.get("prefill_ms", 0.0) for r in results]),
        "ttft_ms":    _agg([r.get("ttft_ms", 0.0) for r in results]),
        "decode_ms":  _agg([r.get("decode_ms", 0.0) for r in results]),
        "total_ms":   _agg([r.get("generation_time_ms", 0.0) for r in results]),
        "peak_gpu_mb":           _agg([r.get("peak_gpu_mb", 0.0) for r in results]),
        "peak_gpu_reserved_mb":  _agg([r.get("peak_gpu_reserved_mb", 0.0) for r in results]),
        "kv_cache_bytes":        _agg([r.get("kv_cache_bytes", 0) for r in results]),
        "kv_cache_tokens":       _agg([r.get("kv_cache_tokens", 0) for r in results]),
        "decode_tok_per_s":      _agg([r.get("decode_tok_per_s", 0.0) for r in results if r.get("decode_tok_per_s", 0.0) > 0]),
        "input_ctx_tokens":      _agg([r.get("context_tokens", 0) for r in results]),
        "num_mem_tokens":        _agg([r.get("num_memory_tokens", 0) for r in results]),
        "generated_tokens":      _agg([r.get("num_generated_tokens", 0) for r in results]),
    }

    # Selection summary: avg picks from positives, recall@K for learned modes.
    if eval_args.selection_mode in ("label", "random", "learned_pooled", "learned", "full"):
        ret_stats = [r["selection_stats"] for r in results if r.get("selection_stats")]
        if ret_stats:
            base = {
                "mode": eval_args.selection_mode,
                "top_k_requested": eval_args.top_k,
                "avg_num_blocks": round(float(np.mean([s["num_blocks"] for s in ret_stats])), 1),
                "avg_num_positive_blocks": round(float(np.mean([s["num_positive_blocks"] for s in ret_stats])), 1),
                "avg_n_from_positives": round(float(np.mean([s["n_from_positives"] for s in ret_stats])), 2),
                "avg_k_selected": round(float(np.mean([s["k_selected"] for s in ret_stats])), 2),
                "examples_with_zero_positives": int(sum(1 for s in ret_stats if s["num_positive_blocks"] == 0)),
                "examples_with_unmatched_evidence": int(sum(1 for s in ret_stats if s["n_unmatched"] > 0)),
                "total_unmatched_evidence": int(sum(s["n_unmatched"] for s in ret_stats)),
            }
            if eval_args.selection_mode in ("learned_pooled", "learned", "full"):
                # Only meaningful where #positives > 0
                recalls = [s["recall_at_k"] for s in ret_stats
                           if s.get("recall_at_k") is not None and s["num_positive_blocks"] > 0]
                if recalls:
                    base["avg_recall_at_k"] = round(float(np.mean(recalls)), 4)
                if eval_args.selection_mode in ("learned_pooled", "learned"):
                    base["checkpoint"] = eval_args.selector_checkpoint
            aggregated["selection"] = base

    # Save results
    output = {
        "config": {
            "model_name": model_args.model_name_or_path,
            "checkpoint": training_args.restore_from,
            "dataset": data_args.dataset_name,
            "dataset_transform": data_args.dataset_transform,
            "eval_split": eval_args.eval_split,
            "max_new_tokens": eval_args.max_new_tokens,
            "compressor_type": model_args.compressor_type,
            "memory_slots": training_args.memory_slots_per_segment,
            "compression_factor": training_args.compression_factor,
            "lora_r": model_args.lora_r if model_args.lora_decoder else None,
            "memory_ablation": eval_args.memory_ablation,
            "selection_mode": eval_args.selection_mode,
            "top_k": eval_args.top_k,
            "selection_seed": eval_args.selection_seed,
            "selector_checkpoint": eval_args.selector_checkpoint or None,
            "concise_prompt": eval_args.concise_prompt,
            "concise_prompt_template": eval_args.concise_prompt_template if eval_args.concise_prompt else None,
            "answer_postproc": eval_args.answer_postproc,
        },
        "aggregated": aggregated,
        "examples": results,
    }

    output_path = eval_args.eval_output_file
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for k, v in aggregated.items():
        print(f"  {k:30s}: {v}")

    # Memory diversity summary
    if all_mem_diversity:
        print("\n" + "=" * 60)
        print("MEMORY DIVERSITY (averaged over all examples)")
        print("=" * 60)
        for key in ["mean_pairwise_cos", "max_pairwise_cos", "min_pairwise_cos",
                     "effective_rank", "norm_mean", "norm_std"]:
            vals = [d[key] for d in all_mem_diversity if key in d]
            if vals:
                print(f"  {key:<25s}: {np.mean(vals):.4f} (std={np.std(vals):.4f})")
        avg_eff_rank = np.mean([d["effective_rank"] for d in all_mem_diversity])
        avg_cos = np.mean([d["mean_pairwise_cos"] for d in all_mem_diversity])
        if avg_cos > 0.9:
            print("  >>> WARNING: memory tokens are nearly identical (collapsed) — "
                  "compressor is not producing diverse representations!")
        if avg_eff_rank < 5:
            print(f"  >>> WARNING: effective rank is very low ({avg_eff_rank:.1f}) — "
                  "memory lives in a low-dimensional subspace.")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
