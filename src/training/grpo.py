"""GRPO training — Group Relative Policy Optimization from SFT checkpoint."""

import argparse
import os

import torch
import yaml
from datasets import Dataset, load_from_disk
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from src.data.spider_loader import SPIDER_DATA_DIR, load_spider_split
from src.rewards.composite import make_phase1_rewards, make_phase2_rewards
from src.training.utils import find_latest_checkpoint, setup_slurm_signal_handler


class _PeftResumeGRPOTrainer(GRPOTrainer):
    """GRPOTrainer with PEFT checkpoint resume support for transformers 4.47.1.

    Context:
      When AutoModelForCausalLM.from_pretrained(peft_dir) is called,
      transformers >= 4.35 injects LoRA layers into the BASE model in-place
      and sets _hf_peft_config_loaded = True. The result is a base
      PreTrainedModel (e.g. LlamaForCausalLM), NOT a PeftModel instance.

      transformers 4.47.1's Trainer._load_from_checkpoint gates its PEFT
      branch on _is_peft_model() (= isinstance PeftModel/PeftMixedModel),
      which returns False for this in-place-injected setup, so resume falls
      through to load_sharded_checkpoint and crashes on adapter-only
      checkpoints ("Can't find a checkpoint index").

      We detect adapter-only checkpoints by filesystem and route through
      model.load_adapter(), which is the same API transformers PR #24274
      uses for its native PeftModel branch. load_adapter correctly strips
      the "base_model.model." prefix that PEFT save_pretrained uses —
      calling set_peft_model_state_dict directly would silently load zero
      weights because it does not strip that prefix.
    """

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        _model = model if model is not None else self.model

        has_adapter = (
            any(
                os.path.exists(os.path.join(resume_from_checkpoint, f))
                for f in ("adapter_model.safetensors", "adapter_model.bin")
            )
            and os.path.exists(os.path.join(resume_from_checkpoint, "adapter_config.json"))
        )
        has_full = any(
            os.path.exists(os.path.join(resume_from_checkpoint, f))
            for f in (
                "model.safetensors",
                "model.safetensors.index.json",
                "pytorch_model.bin",
                "pytorch_model.bin.index.json",
            )
        )

        if has_adapter and not has_full and hasattr(_model, "load_adapter"):
            # Determine the active adapter name (default for single-adapter setups).
            # On transformers' PeftAdapterMixin, `active_adapters` is a BOUND METHOD
            # returning List[str]; on PEFT's PeftModel it is a list/str PROPERTY.
            # Passing the method object as adapter_name would poison an nn.ModuleDict
            # key later in the load, so resolve both shapes explicitly.
            aa = getattr(_model, "active_adapters", None)
            if callable(aa):
                try:
                    aa = aa()
                except Exception:
                    aa = None

            if aa:
                active = aa[0] if isinstance(aa, (list, tuple)) else aa
            else:
                aa_single = getattr(_model, "active_adapter", None)
                if callable(aa_single):
                    try:
                        aa_single = aa_single()
                    except Exception:
                        aa_single = None
                active = aa_single or "default"

            if not isinstance(active, str):
                # Last-resort guard — never pass a non-string to load_adapter /
                # delete_adapter, since it ends up as a ModuleDict key downstream.
                active = "default"

            # load_adapter refuses to overwrite an existing adapter under the
            # same name (raises ValueError), so evict the stale one first. The
            # weights are about to be replaced anyway; this is safe.
            #
            # `PeftModel.delete_adapter` is the idiomatic path when available.
            # The transformers 4.47.1 PeftAdapterMixin path does NOT expose
            # delete_adapter — in that case we pop the config entry directly.
            # PEFT's subsequent `inject_adapter_in_model` call handles
            # existing LoraLayers via `update_layer`, and `load_adapter` then
            # overwrites their weights with the checkpoint state_dict, so we
            # do not need to manually prune the injected modules.
            if (
                getattr(_model, "_hf_peft_config_loaded", False)
                and hasattr(_model, "peft_config")
                and active in _model.peft_config
            ):
                if hasattr(_model, "delete_adapter"):
                    _model.delete_adapter(active)
                else:
                    _model.peft_config.pop(active, None)

            # is_trainable=True sets peft_config.inference_mode=False so
            # gradients flow; without it, resume would silently continue in
            # eval mode and train nothing.
            _model.load_adapter(
                resume_from_checkpoint,
                adapter_name=active,
                is_trainable=True,
            )
            print(
                f"Loaded PEFT adapter (via load_adapter, adapter_name={active!r}) "
                f"from {resume_from_checkpoint}"
            )
            return

        # Full-weight checkpoint or no adapter files — use default behavior.
        return super()._load_from_checkpoint(resume_from_checkpoint, model)

    def _load_optimizer_and_scheduler(self, checkpoint):
        """Graceful degrade on optimizer-state mismatch.

        Context:
          PEFT's `inject_adapter_in_model` (invoked from our
          `_load_from_checkpoint` overload via `load_adapter`) calls
          `LoraLayer.update_layer` for every target module. `update_layer`
          assigns fresh `nn.Linear` instances to `lora_A[adapter_name]` and
          `lora_B[adapter_name]`, replacing the ones that were present
          right after `GRPOTrainer.__init__` loaded the SFT warm-start.
          The trainable-parameter set seen by the optimizer therefore has
          new Python identities after resume, and bitsandbytes' 8-bit
          `paged_adamw_8bit` optimizer rejects a saved state_dict whose
          `param_groups` do not match by count/shape, raising
          `ValueError: loaded state dict contains a parameter group that
          doesn't match the size of optimizer's group`.

        Resolution:
          We accept the cost of losing Adam moment estimates (~20-50 steps
          of stabilization) in exchange for a correct resume. Model
          weights, `global_step`, RNG state, and LR scheduler progress are
          still restored by the surrounding machinery; only the optimizer
          state_dict is dropped. This has a negligible effect on long GRPO
          runs and is strictly preferable to a hard failure.
        """
        try:
            return super()._load_optimizer_and_scheduler(checkpoint)
        except (ValueError, RuntimeError) as e:
            msg = str(e)
            # Only swallow the specific mismatch classes we understand.
            if (
                "parameter group" in msg
                or "doesn't match the size" in msg
                or "size mismatch" in msg
            ):
                step = getattr(getattr(self, "state", None), "global_step", "?")
                print(
                    f"[GRPOTrainer] Skipping optimizer state restore: {e}\n"
                    f"[GRPOTrainer] Cause: PEFT adapter re-injection during "
                    f"resume changed parameter identity; saved state no "
                    f"longer matches current optimizer's param_groups.\n"
                    f"[GRPOTrainer] Resuming training at step {step} with "
                    f"fresh Adam moment estimates. Expect roughly 20-50 "
                    f"warmup steps before moments stabilize. Model weights, "
                    f"LR schedule, and RNG state remain restored."
                )
                return
            raise


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_grpo_dataset(config: dict) -> Dataset:
    """
    Load filtered training data for GRPOTrainer.
    Dataset must have 'prompt' column (list of message dicts).
    Extra columns (gold_sql, db_id, db_path) are passed to reward functions as **kwargs.
    """
    data_dir = config.get("data_dir", SPIDER_DATA_DIR)

    if config.get("use_filtered") and config.get("filtered_data_path"):
        filtered_path = config["filtered_data_path"]
        if os.path.exists(filtered_path):
            return load_from_disk(filtered_path)
        else:
            print(f"Filtered data not found at {filtered_path}, loading raw...")

    return load_spider_split("train", data_dir=data_dir)


def _purge_optimizer_state(checkpoint_dir: str) -> None:
    """Remove optimizer/scheduler state files from a GRPO resume checkpoint.

    Root cause:
      PEFT's inject_adapter_in_model, called from _load_from_checkpoint via
      load_adapter -> update_layer, replaces lora_A[name] and lora_B[name]
      with fresh nn.Linear instances. This changes parameter identity.
      The bitsandbytes 8-bit optimizer (paged_adamw_8bit) serializes state
      against parameter indices and rejects any saved state_dict whose
      param_group sizes no longer match the current optimizer.

      Two separate code paths in transformers 4.47.1 can hit this error:
        1. _load_optimizer_and_scheduler (handled by our override in this file)
        2. A direct self.optimizer.load_state_dict call inside
           _inner_training_loop via the accelerate optimizer wrapper

      Overriding only path 1 is not enough — path 2 still fires.
      Removing the files before trainer.train() is called causes both paths
      to skip optimizer loading (each guards on os.path.isfile), so neither
      can fail. Model weights, global_step, RNG state, and LR scheduler
      are restored from the checkpoint; only Adam moment estimates are lost,
      which re-stabilize within ~20-50 training steps.

    Files are renamed to .bak rather than deleted so they can be inspected.
    """
    # Only purge the bitsandbytes optimizer state — it stores per-parameter
    # momentum keyed by parameter identity, which changes after PEFT's
    # inject_adapter_in_model replaces lora_A/lora_B with fresh nn.Linear
    # instances. The LR scheduler state (scheduler.pt) is purely step-count
    # bookkeeping with no parameter-identity dependency, so it is safe to
    # restore and must NOT be purged (removing it restarts LR warmup on every
    # resume, wasting ~88 warmup steps per job boundary).
    for fname in ("optimizer.pt", "scaler.pt"):
        src = os.path.join(checkpoint_dir, fname)
        if os.path.exists(src):
            bak = src + ".bak"
            os.rename(src, bak)
            print(f"  Moved {fname} → {os.path.basename(bak)} "
                  f"(incompatible with post-PEFT-reinject optimizer; "
                  f"Adam moments will reset for this resume)")


def train(config_path: str, warm_start: str | None = None):
    """
    Main GRPO training loop.
    Warm start from SFT checkpoint is mandatory.
    """
    config = load_config(config_path)

    # Resolve model path — warm start from SFT checkpoint
    model_path = warm_start or config.get("sft_checkpoint")
    if not model_path:
        raise ValueError(
            "GRPO requires warm start from SFT checkpoint. "
            "Pass --warm-start or set sft_checkpoint in config."
        )
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"SFT checkpoint not found at {model_path}. "
            "Run SFT training first (sbatch jobs/run_sft.sh)."
        )
    print(f"Loading from SFT checkpoint: {model_path}")

    # Build reward functions
    grpo_cfg = config["grpo"]
    reward_phase = grpo_cfg.get("reward_phase", 2)
    if reward_phase == 1:
        reward_funcs = make_phase1_rewards()
        print("Using Phase 1 rewards (execution + syntax)")
    else:
        reward_funcs = make_phase2_rewards()
        print("Using Phase 2 rewards (execution + syntax + schema + format + exec_success)")

    # LoRA config
    lora_cfg = config["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
    )

    # GRPO training config
    train_cfg = config["training"]
    model_init_kwargs = {"torch_dtype": torch.bfloat16}
    training_args = GRPOConfig(
        output_dir=config["output_dir"],
        model_init_kwargs=model_init_kwargs,
        # GRPO-specific (v0.14.0 supported params only)
        num_generations=grpo_cfg["num_generations"],
        beta=grpo_cfg["beta"],
        max_completion_length=grpo_cfg["max_completion_length"],
        temperature=grpo_cfg["temperature"],
        # vLLM
        use_vllm=config.get("use_vllm", False),
        # Training
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        max_grad_norm=train_cfg["max_grad_norm"],
        num_train_epochs=train_cfg["num_train_epochs"],
        optim=train_cfg["optim"],
        bf16=train_cfg["bf16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg.get("save_total_limit", 5),
        report_to=train_cfg.get("report_to", "none"),
        run_name=train_cfg.get("run_name", "grpo"),
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = build_grpo_dataset(config)
    print(f"Training on {len(dataset)} examples with {grpo_cfg['num_generations']} generations each")

    # If warm-starting from an SFT checkpoint that already has LoRA adapters,
    # don't pass peft_config — GRPOTrainer would apply a second fresh LoRA on top,
    # resulting in two stacked adapters (one trained SFT, one random).
    has_existing_lora = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    if has_existing_lora:
        print("Detected existing LoRA in checkpoint — skipping peft_config to avoid double adapters")
        peft_config = None

    # Create trainer (subclass handles PEFT checkpoint resume on transformers 4.47.1)
    trainer = _PeftResumeGRPOTrainer(
        model=model_path,
        processing_class=tokenizer,
        args=training_args,
        reward_funcs=reward_funcs,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    # Slurm graceful shutdown — save checkpoint on SIGUSR1/SIGTERM
    setup_slurm_signal_handler(trainer, config["output_dir"])

    # Auto-resume from latest checkpoint if available
    resume_ckpt = find_latest_checkpoint(config["output_dir"])
    if resume_ckpt:
        print(f"Resuming from checkpoint: {resume_ckpt}")
        _purge_optimizer_state(resume_ckpt)
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # Save
    best_path = os.path.join(config["output_dir"], "best")
    trainer.save_model(best_path)
    print(f"Saved GRPO model to {best_path}")


def smoke_test():
    """
    Minimal GRPO run — 5 steps on 10 examples.
    Verifies GRPOTrainer + LoRA + reward functions work end-to-end.
    """
    from src.data.spider_loader import load_spider_split
    from transformers import AutoTokenizer, BitsAndBytesConfig

    print("=== GRPO Smoke Test ===")

    # Check available VRAM and use 4-bit quantization if < 40GB
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {vram_gb:.1f} GB")

    # Set tokenizer padding
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token

    # Tiny dataset
    dataset = load_spider_split("train")
    dataset = dataset.select(range(min(10, len(dataset))))

    # Minimal rewards (syntax only — fast, no DB needed)
    from src.rewards.syntax import syntax_reward

    peft_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
    )

    # Let GRPOTrainer handle model loading — pass string, not pre-loaded model
    model_init_kwargs = {"torch_dtype": torch.bfloat16}
    if vram_gb < 40:
        print("Using 4-bit quantization for smaller GPU")
        model_init_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    training_args = GRPOConfig(
        output_dir="checkpoints/smoke_test",
        model_init_kwargs=model_init_kwargs,
        num_generations=2,
        max_completion_length=64,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        max_steps=5,
        learning_rate=1e-5,
        bf16=True,
        gradient_checkpointing=False,
        logging_steps=1,
        save_steps=999,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        processing_class=tokenizer,
        args=training_args,
        reward_funcs=[syntax_reward],
        train_dataset=dataset,
        peft_config=peft_config,
    )

    trainer.train()
    print("=== Smoke test PASSED ===")


def main():
    parser = argparse.ArgumentParser(description="GRPO training on Spider")
    parser.add_argument("--config", default="configs/grpo.yaml", help="Path to GRPO config")
    parser.add_argument("--warm-start", default=None, help="Path to SFT checkpoint")
    parser.add_argument("--smoke-test", action="store_true", help="Run minimal smoke test")
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test()
    else:
        train(args.config, args.warm_start)


if __name__ == "__main__":
    main()
