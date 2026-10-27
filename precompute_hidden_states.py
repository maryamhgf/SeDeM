"""
Precompute and cache backbone hidden states for all training/eval examples.

This eliminates the backbone forward passes during training — the compressor
and decoder are the only modules that run. Gives ~10-30x speedup for epochs
where the cache is warm.

Single layer:
    python precompute_hidden_states.py \
        --model_name_or_path meta-llama/Llama-3.2-3B \
        --extraction_layer -1 \
        --hidden_states_cache_dir /path/to/hs_cache \
        ...

Multiple layers (one backbone pass saves all layers):
    python precompute_hidden_states.py \
        --model_name_or_path meta-llama/Llama-3.2-3B \
        --extraction_layers "0,4,7,11,17,23,29,-1" \
        --hidden_states_cache_dir /path/to/hs_cache \
        ...
    Saves to: /path/to/hs_cache/layer_{L}/{model_tag}/{hash}.pt
"""

import os
import math
import hashlib
import torch
import transformers
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class CacheArguments:
    model_name_or_path: str = field(metadata={"help": "backbone model"})
    hidden_states_cache_dir: str = field(metadata={"help": "output directory for cached hidden states"})
    extraction_layer: int = field(default=-1, metadata={"help": "which layer to extract (-1 = last). Ignored if --extraction_layers is set."})
    extraction_layers: Optional[str] = field(default=None, metadata={"help": "comma-separated list of layers to extract in one pass (e.g., '0,4,7,-1'). Saves to layer-specific subdirs."})
    memory_slots_per_segment: int = field(default=32, metadata={"help": "memory tokens per segment"})
    compression_factor: int = field(default=4, metadata={"help": "compression rate"})
    model_max_length: int = field(default=4096, metadata={"help": "max context length"})
    dataset_name: str = field(default=None, metadata={"help": "HuggingFace dataset name"})
    dataset_config: str = field(default=None, metadata={"help": "HuggingFace dataset config"})
    dataset_cache_dir: str = field(default=None, metadata={"help": "HF datasets cache dir"})
    dataset_transform: str = field(default=None, metadata={"help": "transform: 'qasper' or 'hotpotqa'"})
    data_path: str = field(default=None, metadata={"help": "local JSONL path (alternative to dataset_name)"})
    bf16: bool = field(default=True, metadata={"help": "use bf16 for backbone"})
    splits: str = field(default="train,validation", metadata={"help": "comma-separated splits to cache"})


def compute_num_segments(total_length, memory_slots, compression_factor):
    return math.ceil(total_length / (memory_slots * compression_factor))


def get_cache_key(input_ids):
    """Deterministic cache key from token IDs.

    Must match the key computation in SeDeMModel.extract_or_load_cached_hidden_states.
    Uses first+last 64 tokens + length, as a torch.long tensor's raw bytes.
    """
    ids_tensor = torch.tensor(input_ids, dtype=torch.long)
    T = ids_tensor.size(0)
    key_tokens = torch.cat([ids_tensor[:64], ids_tensor[-64:], torch.tensor([T], dtype=torch.long)])
    key_bytes = key_tokens.numpy().tobytes()
    return hashlib.sha256(key_bytes).hexdigest()[:16]


def main():
    parser = transformers.HfArgumentParser(CacheArguments)
    args = parser.parse_args_into_dataclasses()[0]

    model_tag = args.model_name_or_path.replace("/", "_")

    # Determine layers to extract
    if args.extraction_layers:
        layers = [int(x.strip()) for x in args.extraction_layers.split(",")]
        multi_layer = True
    else:
        layers = [args.extraction_layer]
        multi_layer = False

    # Build per-layer cache directories
    # Multi-layer: {base}/layer_{L}/{model_tag}/
    # Single-layer (backward compat): {base}/{model_tag}/
    cache_dirs = {}
    for layer in layers:
        if multi_layer:
            d = os.path.join(args.hidden_states_cache_dir, f"layer_{layer}", model_tag)
        else:
            d = os.path.join(args.hidden_states_cache_dir, model_tag)
        os.makedirs(d, exist_ok=True)
        cache_dirs[layer] = d

    print(f"Layers to cache: {layers}")
    for layer, d in cache_dirs.items():
        print(f"  layer {layer:>3d} -> {d}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    # Load backbone
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    print(f"Loading backbone: {args.model_name_or_path} ({dtype})")
    backbone = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=dtype,
        attn_implementation="eager",
    )
    backbone.eval()
    backbone.cuda()
    for p in backbone.parameters():
        p.requires_grad = False

    # Load dataset
    from sedem.data import get_qa_transform, get_pretrain_transform

    if args.dataset_name:
        load_kwargs = dict(cache_dir=args.dataset_cache_dir)
        if "qasper" in args.dataset_name:
            load_kwargs["revision"] = "refs/convert/parquet"
        dataset = load_dataset(args.dataset_name, args.dataset_config, **load_kwargs)
    elif args.data_path:
        dataset = load_dataset("json", data_files={"train": args.data_path})
    else:
        raise ValueError("Must set --dataset_name or --data_path")

    # Apply transform — use QA transform for finetuning data (produces "input" field),
    # or pretrain transform (produces "text" field) for pretraining data.
    transform_name = args.dataset_transform
    data_transform = None
    text_field = "input"  # default for QA
    if transform_name:
        try:
            data_transform = get_qa_transform(transform_name)
            text_field = "input"
        except KeyError:
            data_transform = get_pretrain_transform(transform_name)
            text_field = "text"

    splits = [s.strip() for s in args.splits.split(",")]

    for split_name in splits:
        if split_name not in dataset:
            print(f"Split '{split_name}' not found, skipping.")
            continue

        split_data = dataset[split_name]
        if data_transform:
            split_data = split_data.map(data_transform, batched=True,
                                        remove_columns=split_data.column_names)

        print(f"\n=== Processing {split_name} ({len(split_data)} examples, field='{text_field}') ===")

        cached = 0
        skipped = 0

        for i in tqdm(range(len(split_data)), desc=f"Caching {split_name}"):
            example = split_data[i]
            context = example[text_field]

            # Tokenize context
            tokens = tokenizer.encode(context, add_special_tokens=False,
                                      truncation=True, max_length=args.model_max_length)
            if len(tokens) == 0:
                continue

            cache_key = get_cache_key(tokens)

            # Check which layers still need caching for this example
            layers_to_compute = []
            for layer in layers:
                cache_path = os.path.join(cache_dirs[layer], f"{cache_key}.pt")
                if not os.path.exists(cache_path):
                    layers_to_compute.append(layer)

            if not layers_to_compute:
                skipped += 1
                continue

            # Compute hidden states — one forward pass, extract all needed layers
            total_length = len(tokens)
            num_segments = compute_num_segments(total_length, args.memory_slots_per_segment,
                                                args.compression_factor)
            segment_length = math.ceil(total_length / num_segments)

            input_ids = torch.tensor([tokens], dtype=torch.long, device="cuda")

            # Accumulate per-layer hidden states across segments
            per_layer_hidden = {layer: [] for layer in layers_to_compute}

            with torch.no_grad():
                for seg_idx in range(num_segments):
                    start = seg_idx * segment_length
                    end = min((seg_idx + 1) * segment_length, total_length)
                    seg_ids = input_ids[:, start:end]
                    outputs = backbone(input_ids=seg_ids, output_hidden_states=True)
                    for layer in layers_to_compute:
                        per_layer_hidden[layer].append(
                            outputs.hidden_states[layer].cpu()
                        )

            # Save each layer
            for layer in layers_to_compute:
                hidden_states = torch.cat(per_layer_hidden[layer], dim=1)  # (1, T, D)
                cache_path = os.path.join(cache_dirs[layer], f"{cache_key}.pt")
                torch.save(hidden_states.to(torch.bfloat16), cache_path)

            cached += 1

            if (cached + skipped) <= 3 or cached % 500 == 0:
                print(f"  Cached {cached} (layers={layers_to_compute}), skipped {skipped}")

        print(f"  Done: {cached} cached, {skipped} already existed")

    # Print summary for each layer
    print(f"\n=== Cache summary ===")
    for layer in layers:
        d = cache_dirs[layer]
        pt_files = [f for f in os.listdir(d) if f.endswith('.pt')]
        total_size_gb = sum(os.path.getsize(os.path.join(d, f)) for f in pt_files) / (1024**3)
        print(f"  Layer {layer:>3d}: {len(pt_files)} files, {total_size_gb:.1f} GB  [{d}]")


if __name__ == "__main__":
    main()
