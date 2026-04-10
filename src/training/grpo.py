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

    # Create trainer
    trainer = GRPOTrainer(
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
