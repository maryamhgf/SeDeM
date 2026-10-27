from transformers import Trainer, TrainerCallback
import os
import torch
import random
import numpy as np
from safetensors.torch import save_file

from transformers.trainer_utils import get_last_checkpoint
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loss component keys that SeDeMModel.forward() may return
_LOSS_KEYS = (
    # loss components returned by SeDeMModel.forward()
    "loss_next_token", "loss_distill", "loss_decomp", "loss_tok",
    # memory diagnostics (logged, not optimized)
    "attn_past_mass", "inter_seg_cosine", "intra_slot_cosine",
    "cross_slot_mixer_gate", "cross_slot_attn_entropy", "cross_slot_attn_entropy_norm", "cross_slot_attn_diag_mass",
    "attn_alloc_cosine", "attn_effective_rank",
    "slot_mass_mean", "slot_mass_std", "slot_mass_min", "slot_mass_max", "slot_near_dead_frac",
    "erank_per_seg", "erank_global", "mem_token_norm_mean", "mem_token_norm_std",
)


class SeDeMTrainer(Trainer):
    """
    HuggingFace Trainer subclass that:
      1. Extracts per-component losses from the model output dict.
      2. Logs them to wandb (or any other configured reporter) during
         both training and evaluation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # accumulators for per-component train losses (between logging steps)
        self._train_loss_accum = {}
        self._train_loss_count = 0

    # ---- training -------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"]

        # accumulate individual loss components for logging
        for k in _LOSS_KEYS:
            if k in outputs:
                v = outputs[k]
                val = v.mean().item() if torch.is_tensor(v) else v
                self._train_loss_accum[k] = self._train_loss_accum.get(k, 0.0) + val
        # also pass through dynamic mix_alpha_L{layer} keys from ExtractionLayerMixer
        for k in outputs:
            if k.startswith("mix_alpha_"):
                v = outputs[k]
                val = v.mean().item() if torch.is_tensor(v) else v
                self._train_loss_accum[k] = self._train_loss_accum.get(k, 0.0) + val
        self._train_loss_count += 1

        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None, **kwargs):
        # inject averaged per-component losses into the train log
        if self._train_loss_count > 0 and "loss" in logs:
            for k, total in self._train_loss_accum.items():
                logs[k] = round(total / self._train_loss_count, 6)
            self._train_loss_accum = {}
            self._train_loss_count = 0
        super().log(logs, start_time=start_time, **kwargs)

    # ---- optimizer (separate param group for gate parameters) -------------
    def create_optimizer(self):
        """
        Override to give raw_gate parameters their own param group with
        higher LR and no weight decay.  Gates scale gradients into the
        MHA/FFN branches; a higher LR ensures they move off their initial
        value quickly, avoiding the dead-branch effect.
        """
        model = self.model
        gate_params = []
        other_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'raw_gate' in name:
                gate_params.append(param)
            else:
                other_params.append(param)

        gate_lr_mult = 10.0  # gates get 10x the base LR
        optimizer_grouped_parameters = [
            {
                "params": other_params,
                "lr": self.args.learning_rate,
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": gate_params,
                "lr": self.args.learning_rate * gate_lr_mult,
                "weight_decay": 0.0,  # no WD on gates
            },
        ]

        all_trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        print(f"Optimizer: {len(all_trainable)} trainable params")

        if gate_params:
            gate_names = [n for n, p in model.named_parameters()
                          if p.requires_grad and 'raw_gate' in n]
            print(f"Optimizer: gate params ({len(gate_params)}) at "
                  f"LR={self.args.learning_rate * gate_lr_mult:.1e} (no WD): {gate_names}")
            print(f"Optimizer: other params ({len(other_params)}) at "
                  f"LR={self.args.learning_rate:.1e}")

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args, model
        )
        # Remove default lr from kwargs since we set it per-group
        optimizer_kwargs.pop("lr", None)
        self.optimizer = optimizer_cls(
            optimizer_grouped_parameters, **optimizer_kwargs
        )

        # Verify: count params per optimizer group and check dtypes
        for gi, group in enumerate(self.optimizer.param_groups):
            dtypes = set()
            for p in group['params']:
                dtypes.add(str(p.dtype))
            print(f"Optimizer group {gi}: {len(group['params'])} params, "
                  f"lr={group['lr']:.1e}, dtypes={dtypes}")

        # Verify: check if compressor params are in the optimizer
        optimizer_param_ids = set()
        for group in self.optimizer.param_groups:
            for p in group['params']:
                optimizer_param_ids.add(id(p))
        raw_model = model.module if hasattr(model, 'module') else model
        if hasattr(raw_model, 'compressor'):
            comp_in_opt = 0
            comp_total = 0
            for name, p in raw_model.compressor.named_parameters():
                if p.requires_grad:
                    comp_total += 1
                    if id(p) in optimizer_param_ids:
                        comp_in_opt += 1
                    else:
                        print(f"  WARNING: compressor.{name} NOT in optimizer! id={id(p)}")
            print(f"Optimizer: {comp_in_opt}/{comp_total} compressor params in optimizer")

        return self.optimizer

    # ---- save (only trainable params, avoids tied-weight safetensors error)
    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        # unwrap DataParallel / DDP if needed
        model = self.model.module if hasattr(self.model, "module") else self.model
        trainable_keys = {
            n for n, p in model.named_parameters() if p.requires_grad
        }
        full_sd = model.state_dict()
        # ALWAYS persist the full compressor (params + buffers), frozen or not.
        # It is the memory-maker and is required at inference. In decoupled-memory
        # / freeze_compressor mode the compressor is frozen (requires_grad=False),
        # so without this it is silently dropped from the checkpoint — producing a
        # fresh-random compressor (random slots) at Stage-2 restore and at eval.
        # (For non-frozen runs the compressor is already trainable → no-op.)
        for k in full_sd:
            if k.startswith('compressor.'):
                trainable_keys.add(k)
        state_dict = {
            k: v for k, v in full_sd.items() if k in trainable_keys
        }
        save_path = os.path.join(output_dir, "model.safetensors")
        save_file(state_dict, save_path)
        print(f"[Checkpoint saved] {save_path}")

    # ---- evaluation -----------------------------------------------------
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Single forward pass: get loss + per-component losses without
        # the redundant second forward that caused OOM.
        with torch.no_grad():
            inputs = self._prepare_inputs(inputs)
            outputs = model(**inputs)

        loss = outputs["loss"].mean().detach()

        if not hasattr(self, "_eval_loss_accum"):
            self._eval_loss_accum = {}
            self._eval_loss_count = 0

        for k in _LOSS_KEYS:
            if k in outputs:
                v = outputs[k]
                val = v.mean().item() if torch.is_tensor(v) else v
                self._eval_loss_accum[k] = self._eval_loss_accum.get(k, 0.0) + val
        self._eval_loss_count += 1

        return loss, None, None

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        self._eval_loss_accum = {}
        self._eval_loss_count = 0

        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # inject per-component eval losses
        if self._eval_loss_count > 0:
            for k, total in self._eval_loss_accum.items():
                metrics[f"{metric_key_prefix}_{k}"] = round(
                    total / self._eval_loss_count, 6
                )
        self.log_metrics(metric_key_prefix, metrics)
        return metrics


def run_training(model, train_dataset, eval_dataset, training_args, data_collator=None):

    last_checkpoint = None
    overwrite = getattr(training_args, "overwrite_output_dir", False)
    if os.path.isdir(training_args.output_dir) and not overwrite:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Set --output_dir to a fresh path to train from scratch."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            print(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or pass an explicit `--resume_from_checkpoint`."
            )

    # Only drop the collator when batch_size=1 AND no custom collator was passed
    if data_collator is None and max(training_args.per_device_train_batch_size, training_args.per_device_eval_batch_size) == 1:
        data_collator = None

    # print training_args at local_rank 0
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    if local_rank == 0:
        print(training_args)

    # Use SeDeMTrainer for per-loss wandb logging; falls back gracefully
    # when report_to is not wandb.
    trainer = SeDeMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    checkpoint = None

    if training_args.resume_from_checkpoint is not None:
        # --resume_from_checkpoint True  → auto-detect latest checkpoint
        if training_args.resume_from_checkpoint == "True":
            checkpoint = last_checkpoint
        else:
            checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    print(f"Loaded from the checkpoint: {checkpoint}")

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    metrics = trainer.evaluate()
    trainer.save_metrics("eval", metrics)


