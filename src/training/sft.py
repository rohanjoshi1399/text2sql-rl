"""SFT warm-up — supervised fine-tuning with LoRA on filtered Spider training data."""

import argparse
import os

import torch
import yaml
from datasets import Dataset, load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from src.data.spider_loader import SPIDER_DATA_DIR, load_spider_split
from src.training.utils import find_latest_checkpoint, setup_slurm_signal_handler


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_sft_datasets(config: dict) -> tuple[Dataset, Dataset]:
    """Load filtered training data and dev data."""
    data_dir = config.get("data_dir", SPIDER_DATA_DIR)

    # Training data
    if config.get("use_filtered") and config.get("filtered_data_path"):
        filtered_path = config["filtered_data_path"]
        if os.path.exists(filtered_path):
            train_ds = load_from_disk(filtered_path)
        else:
            print(f"Filtered data not found at {filtered_path}, loading raw...")
            train_ds = load_spider_split("train", data_dir=data_dir)
    else:
        train_ds = load_spider_split("train", data_dir=data_dir)

    # Eval data
    eval_ds = load_spider_split(config.get("eval_split", "dev"), data_dir=data_dir)

    return train_ds, eval_ds


def make_formatting_func(tokenizer):
    """
    Create SFT formatting function.
    Concatenates prompt messages + gold_sql as assistant response
    into a single string using the chat template.
    """

    def formatting_func(examples):
        # TRL 0.14.0 always passes batched examples (dict of lists).
        # Must always return a list of strings.
        results = []
        for prompt, gold_sql in zip(examples["prompt"], examples["gold_sql"]):
            messages = prompt + [{"role": "assistant", "content": gold_sql}]
            results.append(tokenizer.apply_chat_template(messages, tokenize=False))
        return results

    return formatting_func


def train(config_path: str):
    """Main SFT training loop."""
    config = load_config(config_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model (use flash_attention_2 if available, otherwise sdpa)
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
        print("flash-attn not installed, using SDPA attention")

    # Use 4-bit quantization if configured or if VRAM < 40GB
    load_in_4bit = config.get("load_in_4bit", False)
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb < 40:
            load_in_4bit = True
            print(f"GPU has {vram_gb:.0f}GB VRAM — enabling 4-bit quantization")

    model_kwargs = {
        "attn_implementation": attn_impl,
        "device_map": "auto",
    }
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        **model_kwargs,
    )

    # LoRA config
    lora_cfg = config["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type=lora_cfg["task_type"],
    )

    # Build datasets
    train_ds, eval_ds = build_sft_datasets(config)
    print(f"Training on {len(train_ds)} examples, evaluating on {len(eval_ds)}")

    # Training config
    train_cfg = config["training"]
    training_args = SFTConfig(
        output_dir=config["output_dir"],
        max_seq_length=train_cfg["max_seq_length"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
        optim=train_cfg["optim"],
        bf16=train_cfg["bf16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=train_cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=train_cfg["eval_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        report_to=train_cfg.get("report_to", "none"),
        run_name=train_cfg.get("run_name", "sft"),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        formatting_func=make_formatting_func(tokenizer),
    )

    # Slurm graceful shutdown — save checkpoint on SIGUSR1/SIGTERM
    setup_slurm_signal_handler(trainer, config["output_dir"])

    # Auto-resume from latest checkpoint if available
    resume_ckpt = find_latest_checkpoint(config["output_dir"])
    if resume_ckpt:
        print(f"Resuming from checkpoint: {resume_ckpt}")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # Save best checkpoint
    best_path = os.path.join(config["output_dir"], "best")
    trainer.save_model(best_path)
    tokenizer.save_pretrained(best_path)
    print(f"Saved best SFT model to {best_path}")


def main():
    parser = argparse.ArgumentParser(description="SFT training on Spider")
    parser.add_argument("--config", default="configs/sft.yaml", help="Path to SFT config")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
