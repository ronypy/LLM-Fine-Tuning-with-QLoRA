"""
==============================================================================
 Training Script — LoRA / QLoRA Fine-Tuning with SFTTrainer
==============================================================================

This is the heart of the fine-tuning pipeline.  It:
    1. Loads the base model in quantised form (4-bit QLoRA by default)
    2. Attaches LoRA adapter layers on top of the frozen base
    3. Trains only the adapter weights (~0.5% of total parameters)
    4. Logs metrics to Weights & Biases
    5. Saves both the adapter and the merged full model

KEY CONCEPTS YOU'LL LEARN HERE:
    • QLoRA:  Quantise the base model to 4-bit, then train 16-bit LoRA
              adapters on top.  This lets you fine-tune a 7B model on a
              single 24 GB GPU (or even 16 GB with small batch sizes).
    • PEFT:   The Hugging Face library that wraps LoRA (and other PEFT
              methods) into a clean `get_peft_model()` API.
    • SFTTrainer:  A Trainer subclass from the `trl` library optimised
              for Supervised Fine-Tuning of language models.

USAGE:
    python src/train.py --config configs/lora_config.yaml
==============================================================================
"""

import argparse
import os
import sys
from datetime import datetime

import torch
import yaml
from datasets import load_from_disk
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from rich.console import Console
from rich.panel import Panel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
# NOTE: In trl v0.14+, SFTConfig replaces TrainingArguments for SFTTrainer.
# SFTConfig extends TrainingArguments with SFT-specific params like max_seq_length.
from trl import SFTTrainer, SFTConfig

# ─── Rich console for pretty output ─────────────────────────────────────────
console = Console()


# ==============================================================================
#  1. CONFIGURATION
# ==============================================================================
def load_config(config_path: str) -> dict:
    """Load and return the YAML configuration dictionary."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    console.print(f"[green]✓[/green] Loaded config from [bold]{config_path}[/bold]")
    return config


# ==============================================================================
#  2. QUANTISATION SETUP  (BitsAndBytes)
# ==============================================================================
def create_bnb_config(model_cfg: dict) -> BitsAndBytesConfig:
    """
    Create a BitsAndBytesConfig for 4-bit or 8-bit quantisation.

    HOW QLORA QUANTISATION WORKS:
        Normal fp16 weights use 16 bits per parameter.
        A 7B model → 7 × 10⁹ × 2 bytes ≈ 14 GB VRAM.

        With 4-bit NF4 quantisation:
        Each weight is stored as a 4-bit integer using a Normal Float
        distribution (NF4), which is information-theoretically optimal
        for normally-distributed weights.

        7 × 10⁹ × 0.5 bytes ≈ 3.5 GB  (+ some overhead ≈ 4–5 GB total)

        The quantised weights are "frozen" — we never update them.
        Instead, we attach small float16 LoRA matrices and train those.

    DOUBLE QUANTISATION (use_nested_quant):
        Quantises the quantisation constants themselves, saving ~0.4 GB.
        Negligible accuracy impact; always worth enabling.

    Args:
        model_cfg: The 'model' section of the YAML config.

    Returns:
        A BitsAndBytesConfig object to pass to from_pretrained().
    """
    quant_bits = model_cfg.get("quantization_bits", 4)

    # Map string dtype names → torch dtype objects
    compute_dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    compute_dtype = compute_dtype_map.get(
        model_cfg.get("bnb_4bit_compute_dtype", "float16"),
        torch.float16,
    )

    if quant_bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,                    # Enable 4-bit loading
            bnb_4bit_quant_type=model_cfg.get(    # NF4 vs FP4
                "bnb_4bit_quant_type", "nf4"
            ),
            bnb_4bit_compute_dtype=compute_dtype, # Compute precision
            bnb_4bit_use_double_quant=model_cfg.get(  # Double quantisation
                "use_nested_quant", True
            ),
        )
        console.print("[green]✓[/green] Created 4-bit NF4 quantisation config")
    elif quant_bits == 8:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        console.print("[green]✓[/green] Created 8-bit quantisation config")
    else:
        # No quantisation — use full precision
        console.print("[yellow]⚠[/yellow] No quantisation (full precision)")
        return None

    return bnb_config


# ==============================================================================
#  3. MODEL LOADING
# ==============================================================================
def load_model_and_tokenizer(
    model_cfg: dict,
    bnb_config: BitsAndBytesConfig = None,
):
    """
    Load the base model (quantised) and its tokeniser.

    WHAT HAPPENS UNDER THE HOOD:
        1. AutoModelForCausalLM.from_pretrained() downloads the model weights
           from the Hugging Face Hub (cached in ~/.cache/huggingface/).
        2. If quantization_config is provided, bitsandbytes replaces the
           nn.Linear layers with quantised equivalents on the fly.
        3. device_map="auto" distributes layers across available GPUs
           (or CPU if no GPU).

    Args:
        model_cfg:  The 'model' section of the YAML config.
        bnb_config: Optional BitsAndBytesConfig for quantisation.

    Returns:
        (model, tokenizer) tuple.
    """
    model_name = model_cfg["name"]
    console.print(f"\n[bold blue]🤖 Loading model:[/bold blue] {model_name}")

    # ── Load the tokeniser ───────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )

    # Many models (Llama, Mistral) don't define a pad token.
    # Without one, the Trainer crashes on batched inputs.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        console.print("   [dim]pad_token ← eos_token (not defined by model)[/dim]")

    # ── Load the model ───────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,  # None → full precision
        device_map="auto",               # Automatic GPU placement
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        torch_dtype=torch.float16,       # Default dtype for non-quantised layers
    )

    # Disable caching — incompatible with gradient checkpointing
    model.config.use_cache = False

    # Some models need this for stable training
    model.config.pretraining_tp = 1

    num_params = sum(p.numel() for p in model.parameters())
    console.print(f"   Total parameters: {num_params / 1e9:.2f}B")

    return model, tokenizer


# ==============================================================================
#  4. LORA ADAPTER SETUP
# ==============================================================================
def setup_lora(model, lora_cfg: dict):
    """
    Attach LoRA adapters to the frozen base model.

    HOW LORA WORKS:
        For each target layer (e.g. q_proj), LoRA adds two small matrices:

            W' = W + (α/r) · B @ A

        where:
            W ∈ R^{d×k}  — original frozen weight
            A ∈ R^{d×r}  — down-projection (initialised randomly)
            B ∈ R^{r×k}  — up-projection   (initialised to zero)
            r             — rank (much smaller than d, k)
            α             — scaling factor

        At init, B=0 so W' = W (no change).  During training, A and B learn
        the task-specific "delta" to apply on top of the frozen base.

    PARAMETER COUNT:
        Original q_proj for 7B model: 4096 × 4096 = 16.8M params
        LoRA adapter (r=16):          4096 × 16 + 16 × 4096 = 131K params
        That's 131K / 16.8M ≈ 0.78% of the original — massive savings!

    Args:
        model:    The loaded (quantised) base model.
        lora_cfg: The 'lora' section of the YAML config.

    Returns:
        The model with LoRA adapters attached (only adapters are trainable).
    """
    console.print("\n[bold blue]🔧 Attaching LoRA adapters[/bold blue]")

    # Step 1: Prepare the quantised model for k-bit training
    # This does two things:
    #   a) Casts the layer norms to float32 for numerical stability
    #   b) Enables gradient computation on the input embeddings
    model = prepare_model_for_kbit_training(model)

    # Step 2: Define the LoRA configuration
    peft_config = LoraConfig(
        r=lora_cfg.get("r", 16),                                 # Rank
        lora_alpha=lora_cfg.get("lora_alpha", 32),               # Scaling
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),         # Dropout
        target_modules=lora_cfg.get("target_modules", [          # Which layers
            "q_proj", "k_proj", "v_proj", "o_proj",
        ]),
        bias=lora_cfg.get("bias", "none"),                       # Bias training
        task_type=TaskType.CAUSAL_LM,                            # Task type
    )

    # Step 3: Wrap the model — this "freezes" all base parameters
    # and marks only the LoRA parameters as trainable
    model = get_peft_model(model, peft_config)

    # Print a summary of trainable vs total parameters
    model.print_trainable_parameters()
    # Example output:
    #   trainable params: 13,631,488 || all params: 3,514,900,480
    #   || trainable%: 0.3878

    return model, peft_config


# ==============================================================================
#  5. TRAINING ARGUMENTS
# ==============================================================================
def create_training_args(train_cfg: dict, wandb_cfg: dict) -> SFTConfig:
    """
    Build an SFTConfig from the YAML config.

    WHY SFTConfig INSTEAD OF TrainingArguments?
        In trl v0.14+, SFTConfig replaced TrainingArguments for SFTTrainer.
        SFTConfig *extends* TrainingArguments with SFT-specific parameters
        like max_seq_length, dataset_text_field, and packing — so you get
        everything in one object.

    IMPORTANT SETTINGS EXPLAINED:
        • gradient_accumulation_steps:
              Simulates a larger batch size without using more VRAM.
              effective_batch = per_device_batch × gradient_accumulation
              = 4 × 4 = 16 in our defaults.

        • gradient_checkpointing:
              Instead of storing all intermediate activations (for the
              backward pass), recompute them on the fly.  Trades ~30% more
              compute for ~40% less VRAM.

        • optim = "paged_adamw_32bit":
              Uses paged memory for the optimiser states.  Prevents OOM
              spikes during uneven memory usage.

    Args:
        train_cfg: The 'training' section of the YAML config.
        wandb_cfg: The 'wandb' section of the YAML config.

    Returns:
        An SFTConfig object (extends TrainingArguments).
    """
    # Set up Weights & Biases reporting
    report_to = "wandb" if wandb_cfg.get("enabled", True) else "none"

    if wandb_cfg.get("enabled", True):
        os.environ["WANDB_PROJECT"] = wandb_cfg.get("project", "llm-text-to-sql")
        run_name = wandb_cfg.get("run_name", "qlora-run")
        # Append timestamp for unique run names
        run_name = f"{run_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        os.environ["WANDB_RUN_NAME"] = run_name

    # SFTConfig extends TrainingArguments with SFT-specific params.
    # max_seq_length and dataset_text_field now go HERE, not in SFTTrainer().
    training_args = SFTConfig(
        output_dir=train_cfg.get("output_dir", "./results"),
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=train_cfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        weight_decay=train_cfg.get("weight_decay", 0.001),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        # NOTE: warmup_ratio is deprecated in transformers v5.2+.
        # We use warmup_steps instead.
        warmup_steps=train_cfg.get("warmup_steps", 100),
        fp16=train_cfg.get("fp16", True),
        bf16=train_cfg.get("bf16", False),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        logging_steps=train_cfg.get("logging_steps", 25),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 200),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        eval_strategy=train_cfg.get("eval_strategy", "steps"),
        eval_steps=train_cfg.get("eval_steps", 200),
        optim=train_cfg.get("optim", "paged_adamw_32bit"),
        seed=train_cfg.get("seed", 42),
        group_by_length=train_cfg.get("group_by_length", True),
        report_to=report_to,
        remove_unused_columns=False,
        # ── SFT-specific parameters (new in trl v0.14+) ─────────────────
        max_seq_length=train_cfg.get("max_seq_length", 512),
        dataset_text_field="text",  # Column with formatted prompts
    )

    console.print("[green]✓[/green] Created SFTConfig")
    return training_args


# ==============================================================================
#  6. TRAINING LOOP
# ==============================================================================
def train(config: dict, data_dir: str = None):
    """
    Full QLoRA training pipeline.

    PIPELINE OVERVIEW:
        ┌─────────────┐    ┌─────────────┐    ┌──────────────┐
        │  Load Model  │ →  │  Attach LoRA │ →  │  Load Data   │
        │  (4-bit)     │    │  adapters    │    │  (processed) │
        └─────────────┘    └─────────────┘    └──────────────┘
              │                                       │
              └───────────────┬───────────────────────┘
                              ▼
                    ┌──────────────────┐
                    │   SFT Trainer    │
                    │  (train loop)    │
                    └──────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
            ┌──────────────┐  ┌──────────────────┐
            │ Save Adapter │  │  Merge & Save    │
            │ (small)      │  │  Full Model      │
            └──────────────┘  └──────────────────┘

    Args:
        config:   Parsed YAML config dictionary.
        data_dir: Path to pre-processed dataset (from data_prep.py).
                  If None, runs data_prep on the fly.
    """
    model_cfg = config["model"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]
    data_cfg = config["data"]
    wandb_cfg = config.get("wandb", {})

    console.print(Panel.fit(
        "[bold]🚀 Starting QLoRA Fine-Tuning Pipeline[/bold]",
        border_style="blue",
    ))

    # ── Step 1: Create quantisation config ───────────────────────────────────
    bnb_config = create_bnb_config(model_cfg)

    # ── Step 2: Load model and tokeniser ─────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(model_cfg, bnb_config)

    # ── Step 3: Attach LoRA adapters ─────────────────────────────────────────
    model, peft_config = setup_lora(model, lora_cfg)

    # ── Step 4: Load the dataset ─────────────────────────────────────────────
    if data_dir and os.path.exists(data_dir):
        console.print(f"\n[bold blue]📂 Loading pre-processed data:[/bold blue] {data_dir}")
        from datasets import load_from_disk
        dataset = load_from_disk(data_dir)
    else:
        console.print("\n[bold blue]📦 Processing data on-the-fly[/bold blue]")
        # Import and run data_prep inline
        from data_prep import prepare_data
        dataset = prepare_data(config)

    console.print(f"   Train: {len(dataset['train']):,} examples")
    console.print(f"   Val:   {len(dataset['validation']):,} examples")

    # ── Step 5: Create training arguments ────────────────────────────────────
    training_args = create_training_args(train_cfg, wandb_cfg)

    # ── Step 6: Initialise the SFT Trainer ───────────────────────────────────
    #
    # SFTTrainer (Supervised Fine-Tuning Trainer) from the `trl` library is
    # a subclass of HF Trainer that:
    #   a) Handles the causal LM loss automatically
    #   b) Supports packing multiple short examples into one sequence
    #   c) Integrates cleanly with PEFT / LoRA
    #
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        # NOTE: In trl v0.14+, `tokenizer` was renamed to `processing_class`.
        # If you're on an older trl, change back to `tokenizer=tokenizer`.
        processing_class=tokenizer,
        args=training_args,
        # max_seq_length and dataset_text_field are now inside SFTConfig
        # (the `args` object above), NOT passed here.
    )

    # ── Step 7: Train! ───────────────────────────────────────────────────────
    console.print("\n[bold green]🏋️ Training started …[/bold green]\n")
    trainer.train()
    console.print("\n[bold green]✓ Training complete![/bold green]")

    # ── Step 8: Save the adapter ─────────────────────────────────────────────
    #
    # Saving only the adapter weights is fast and tiny (~50 MB for r=16).
    # You can reload them later with:
    #   base_model = AutoModelForCausalLM.from_pretrained(...)
    #   model = PeftModel.from_pretrained(base_model, adapter_path)
    #
    adapter_path = os.path.join(train_cfg["output_dir"], "adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    console.print(f"[green]💾 Adapter saved to {adapter_path}[/green]")

    # ── Step 9: Merge adapter into base model and save ───────────────────────
    #
    # For inference, it's convenient to have a single merged model.
    # This "bakes in" the LoRA weights:  W_merged = W_base + (α/r) · B @ A
    # The merged model can be loaded without PEFT and served with vLLM.
    #
    console.print("\n[yellow]⏳[/yellow] Merging adapter into base model …")
    merged_model = model.merge_and_unload()  # Returns a standard HF model
    merged_path = os.path.join(train_cfg["output_dir"], "merged_model")
    merged_model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    console.print(f"[green]💾 Merged model saved to {merged_path}[/green]")

    # ── Step 10: Log final metrics ───────────────────────────────────────────
    if wandb_cfg.get("enabled", True):
        import wandb
        wandb.finish()
        console.print("[green]✓[/green] W&B run finalised")

    console.print(Panel.fit(
        "[bold green]🎉 Fine-tuning pipeline complete![/bold green]\n"
        f"  Adapter:      {adapter_path}\n"
        f"  Merged model: {merged_path}",
        border_style="green",
    ))


# ==============================================================================
#  CLI ENTRY POINT
# ==============================================================================
def main():
    """
    Command-line interface.

    Examples:
        # Train with pre-processed data
        python src/train.py --config configs/lora_config.yaml --data data/processed

        # Train with on-the-fly data processing
        python src/train.py --config configs/lora_config.yaml
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune an LLM with QLoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/lora_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to pre-processed dataset directory (from data_prep.py)",
    )

    args = parser.parse_args()
    config = load_config(args.config)
    train(config, data_dir=args.data)


if __name__ == "__main__":
    main()
