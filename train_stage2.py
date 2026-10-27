"""
QA fine-tuning with the hidden-state compressor.

Objective: given compressed context (memory tokens) and an UNCOMPRESSED
question, generate the answer.

Data format (jsonl): each line has {"input": ..., "prompt": ..., "answer": ...}

Usage example:
    python train_stage2.py \
        --model_name_or_path meta-llama/Llama-3.2-3B \
        --compressor_type pooled_linear \
        --compressor_hidden_dim 512 \
        --memory_slots_per_segment 128 \
        --compression_factor 4 \
        --restore_from ./output_hs_pretrain/model.safetensors \
        --qa_weight 1.0 \
        --output_dir ./output_hs_qa \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --learning_rate 5e-5 \
        --num_train_epochs 3 \
        --bf16 True \
        --overwrite_output_dir
"""

import transformers
from datasets import load_dataset

from sedem.modeling import (
    SeDeMModel,
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from sedem.trainer import run_training
from sedem.data import (
    stage2_tokenize_function,
    SeDeMCollator,
    get_qa_transform,
    apply_subset_manifest,
    resolve_transform_name,
    resolve_load_kwargs,
    resolve_splits,
)


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print(model_args)
    print(data_args)

    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    assert (training_args.memory_slots_per_segment & (training_args.memory_slots_per_segment - 1)) == 0, (
        "memory_slots_per_segment must be a power of 2"
    )

    # ---- LoRA config for decoder (optional) ----------------------------
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
        print(f"LoRA config: r={model_args.lora_r}, alpha={model_args.lora_alpha}, "
              f"dropout={model_args.lora_dropout}, targets={target_modules}")

    # ---- model ----------------------------------------------------------
    model = SeDeMModel(model_args, training_args, lora_config=lora_config)

    # ---- data -----------------------------------------------------------
    if data_args.debug_data:
        raw = {
            "input": ["The quick brown fox jumps over the lazy dog. " * 20] * 32,
            "prompt": ["What does the fox do?"] * 32,
            "answer": ["jumps over the lazy dog"] * 32,
        }
        from datasets import Dataset
        train_dataset = Dataset.from_dict(raw)
        eval_dataset = Dataset.from_dict(raw)
    elif data_args.dataset_name:
        # Load directly from HuggingFace Hub
        # Some HF datasets (e.g. allenai/qasper) still use loading scripts
        # which are no longer supported in datasets>=4. Use the parquet
        # conversion branch for those.
        load_kwargs = dict(cache_dir=data_args.dataset_cache_dir)
        load_kwargs.update(resolve_load_kwargs(data_args.dataset_name))
        dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config,
            **load_kwargs,
        )
        # Optional: filter to a stratified subset from a prebuilt manifest
        if data_args.subset_manifest:
            dataset = apply_subset_manifest(dataset, data_args.subset_manifest)
        # Resolve the transform: explicit flag > auto-detect from dataset_name
        transform_name = data_args.dataset_transform or resolve_transform_name(
            data_args.dataset_name, data_args.dataset_config,
        )
        if transform_name is None:
            raise ValueError(
                f"Cannot auto-detect transform for dataset '{data_args.dataset_name}'. "
                "Please set --dataset_transform explicitly."
            )
        qa_transform = get_qa_transform(transform_name)
        train_split, eval_split = resolve_splits(dataset)
        if train_split is None:
            raise ValueError(
                f"Dataset '{data_args.dataset_name}' has no train split — it is "
                "eval-only. QA fine-tuning requires a train split."
            )
        train_dataset = dataset[train_split].map(
            qa_transform, batched=True,
            remove_columns=dataset[train_split].column_names,
        )
        eval_dataset = dataset[eval_split].map(
            qa_transform, batched=True,
            remove_columns=dataset[eval_split].column_names,
        )
    else:
        train_file = data_args.data_path
        eval_file = data_args.eval_data_path or train_file.replace("train", "val")
        dataset = load_dataset(
            "json",
            data_files={"train": train_file, "eval": eval_file},
        )
        train_dataset = dataset["train"]
        eval_dataset = dataset["eval"]

    train_dataset = train_dataset.map(
        stage2_tokenize_function,
        batched=True,
        fn_kwargs={"model": model},
    )
    eval_dataset = eval_dataset.map(
        stage2_tokenize_function,
        batched=True,
        fn_kwargs={"model": model},
    )

    data_collator = SeDeMCollator(model.pad_token_id)
    run_training(model, train_dataset, eval_dataset, training_args, data_collator)


if __name__ == "__main__":
    main()
