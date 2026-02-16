"""
==============================================================================
 Data Preparation — Text-to-SQL Dataset Processing
==============================================================================

This module handles everything related to loading, formatting, and tokenising
the training data for our LLM fine-tuning pipeline.

KEY CONCEPTS YOU'LL LEARN HERE:
    1. Loading datasets from the Hugging Face Hub with `datasets`
    2. Formatting raw examples into *instruction-style prompts*
    3. Tokenising text for causal language models
    4. Splitting a single-split dataset into train / val / test

DATASET: b-mc2/sql-create-context
    Each row contains:
        - question : natural-language question about a database
        - context  : the CREATE TABLE schema(s) relevant to the question
        - answer   : the gold SQL query

USAGE:
    # Process and save to disk
    python src/data_prep.py --config configs/lora_config.yaml --output data/processed

    # Quick peek (5 examples)
    python src/data_prep.py --config configs/lora_config.yaml --peek
==============================================================================
"""

import argparse
import os
import sys
from typing import Dict, List, Optional

import yaml
from datasets import Dataset, DatasetDict, load_dataset
from rich.console import Console
from rich.table import Table
from transformers import AutoTokenizer

# Rich console for pretty output ─────────────────────────────────────────────
console = Console()


# ─── 1. CONFIGURATION LOADING ───────────────────────────────────────────────
def load_config(config_path: str) -> dict:
    """
    Load the YAML configuration file.

    WHY YAML?
        We centralise every hyper-parameter in one file so experiments are
        reproducible.  You can version-control this file alongside your code
        and always know *exactly* which settings produced a given model.

    Args:
        config_path: Path to the YAML config (e.g. configs/lora_config.yaml).

    Returns:
        A nested dictionary mirroring the YAML structure.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)  # safe_load prevents arbitrary code execution
    console.print(f"[green]✓[/green] Loaded config from [bold]{config_path}[/bold]")
    return config


# ─── 2. PROMPT FORMATTING ───────────────────────────────────────────────────
def format_prompt(example: Dict[str, str], template: str) -> Dict[str, str]:
    """
    Convert a raw dataset row into an instruction-style prompt.

    WHY INSTRUCTION-STYLE?
        Modern LLMs are trained to follow instructions.  By structuring our
        fine-tuning data the same way, the model learns to map
        (question + schema) → SQL answer in a natural, predictable format.

    The template comes from the YAML config and looks like:

        Below is a question about a database along with the schema.
        Write the SQL query that answers the question.

        ### Question:
        {question}

        ### Schema:
        {context}

        ### SQL:
        {answer}

    Args:
        example:  A single row from the dataset with keys
                  'question', 'context', 'answer'.
        template: The prompt template string with {placeholders}.

    Returns:
        The example dict with a new 'text' key containing the formatted prompt.
    """
    # .format() replaces {question}, {context}, {answer} with actual values
    example["text"] = template.format(
        question=example["question"],
        context=example["context"],
        answer=example["answer"],
    )
    return example


def format_dataset(dataset: Dataset, template: str) -> Dataset:
    """
    Apply the prompt template to every example in the dataset.

    Uses `Dataset.map()` which:
        • Is lazy & memory-efficient (processes in batches under the hood)
        • Supports multiprocessing via `num_proc`
        • Caches results to disk automatically (won't reprocess on re-runs)

    Args:
        dataset:  A Hugging Face `Dataset` object.
        template: The prompt template string.

    Returns:
        The dataset with an added 'text' column.
    """
    console.print("[yellow]⏳[/yellow] Formatting prompts …")

    formatted = dataset.map(
        lambda ex: format_prompt(ex, template),
        desc="Formatting prompts",  # Progress bar label
    )

    console.print(f"[green]✓[/green] Formatted {len(formatted):,} examples")
    return formatted


# ─── 3. TOKENISATION ────────────────────────────────────────────────────────
def tokenize_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_seq_length: int,
) -> Dataset:
    """
    Tokenise the 'text' column for causal language modelling.

    KEY DETAILS:
        • Truncation: cuts prompts longer than max_seq_length tokens so we
          don't run out of GPU memory.
        • Padding: pads shorter prompts to max_seq_length so every batch
          has uniform dimensions.  The model ignores padded positions via
          the attention mask.
        • Labels: for causal LM, labels == input_ids.  The loss is computed
          only on non-padded tokens.

    Args:
        dataset:        Dataset with a 'text' column.
        tokenizer:      The model's tokeniser.
        max_seq_length: Maximum number of tokens per example.

    Returns:
        The dataset with 'input_ids', 'attention_mask', and 'labels' columns.
    """
    console.print(f"[yellow]⏳[/yellow] Tokenising (max_seq_length={max_seq_length}) …")

    def _tokenize(examples: Dict[str, List[str]]) -> Dict[str, List]:
        """
        Inner function applied to batches of examples.

        We set return_tensors=None so we get plain Python lists — the
        Trainer converts them to tensors later with proper collation.
        """
        tokenized = tokenizer(
            examples["text"],
            truncation=True,          # Cut at max_seq_length
            padding="max_length",     # Pad short sequences
            max_length=max_seq_length,
            return_tensors=None,      # Return lists, not tensors
        )
        # For causal LM:  labels = input_ids  (shifted internally by the model)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized = dataset.map(
        _tokenize,
        batched=True,     # Process in batches for speed
        desc="Tokenising",
    )

    console.print(f"[green]✓[/green] Tokenised {len(tokenized):,} examples")
    return tokenized


# ─── 4. DATASET SPLITTING ───────────────────────────────────────────────────
def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    """
    Split a single-split dataset into train / validation / test.

    WHY SPLIT OURSELVES?
        The `b-mc2/sql-create-context` dataset ships as one giant 'train'
        split.  We need a held-out validation set (for early stopping /
        hyper-parameter tuning) and a test set (for final evaluation).

    HOW IT WORKS:
        1. First split off the test set.
        2. Then split the remainder into train and validation.

    Args:
        dataset:     The full dataset.
        train_ratio: Fraction for training (default 0.8).
        val_ratio:   Fraction for validation (default 0.1).
        test_ratio:  Fraction for testing (default 0.1).
        seed:        Random seed for reproducibility.

    Returns:
        A DatasetDict with keys 'train', 'validation', 'test'.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Split ratios must sum to 1.0"
    )

    # Step 1: carve out the test set
    # train_test_split returns a DatasetDict with 'train' and 'test' keys
    train_val_test = dataset.train_test_split(
        test_size=test_ratio,
        seed=seed,
    )

    # Step 2: split the remaining data into train and validation
    # val_ratio relative to (train + val) portion
    relative_val_ratio = val_ratio / (train_ratio + val_ratio)
    train_val = train_val_test["train"].train_test_split(
        test_size=relative_val_ratio,
        seed=seed,
    )

    # Assemble DatasetDict with standard key names
    splits = DatasetDict(
        {
            "train": train_val["train"],
            "validation": train_val["test"],  # the "test" of this sub-split is our val
            "test": train_val_test["test"],
        }
    )

    # Pretty-print split sizes
    table = Table(title="Dataset Splits")
    table.add_column("Split", style="cyan")
    table.add_column("Examples", style="green", justify="right")
    for split_name, split_data in splits.items():
        table.add_row(split_name, f"{len(split_data):,}")
    console.print(table)

    return splits


# ─── 5. MAIN PIPELINE ───────────────────────────────────────────────────────
def prepare_data(
    config: dict,
    output_dir: Optional[str] = None,
    peek: bool = False,
) -> DatasetDict:
    """
    End-to-end data preparation pipeline.

    STEPS:
        1. Load dataset from the Hugging Face Hub
        2. (Optional) Sub-sample for quick experiments
        3. Format prompts
        4. Split into train / val / test
        5. Tokenise
        6. (Optional) Save to disk for later re-use

    Args:
        config:     Parsed YAML config dictionary.
        output_dir: If provided, save the processed dataset here.
        peek:       If True, print 5 example prompts and exit.

    Returns:
        A DatasetDict ready for training.
    """
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    # ── Step 1: Load from Hub ────────────────────────────────────────────────
    console.print(
        f"\n[bold blue]📦 Loading dataset:[/bold blue] {data_cfg['dataset_name']}"
    )
    raw_dataset = load_dataset(data_cfg["dataset_name"], split="train")
    console.print(f"   Total examples: {len(raw_dataset):,}")

    # ── Step 2: Sub-sample (for quick experiments) ───────────────────────────
    subset_fraction = data_cfg.get("subset_fraction", 1.0)
    if subset_fraction < 1.0:
        num_examples = int(len(raw_dataset) * subset_fraction)
        raw_dataset = raw_dataset.shuffle(seed=42).select(range(num_examples))
        console.print(
            f"   [dim]Using {subset_fraction:.0%} subset → {len(raw_dataset):,} examples[/dim]"
        )

    # ── Step 3: Format prompts ───────────────────────────────────────────────
    template = data_cfg["prompt_template"]
    formatted = format_dataset(raw_dataset, template)

    # ── Peek mode: show a few examples and exit ───────────────────────────
    if peek:
        console.print("\n[bold magenta]👀 Sample prompts:[/bold magenta]\n")
        for i in range(min(5, len(formatted))):
            console.print(f"[dim]{'─' * 60}[/dim]")
            console.print(formatted[i]["text"])
        return None

    # ── Step 4: Split ────────────────────────────────────────────────────────
    splits = split_dataset(
        formatted,
        train_ratio=data_cfg.get("train_ratio", 0.8),
        val_ratio=data_cfg.get("val_ratio", 0.1),
        test_ratio=data_cfg.get("test_ratio", 0.1),
        seed=train_cfg.get("seed", 42),
    )

    # ── Step 5: Tokenise ─────────────────────────────────────────────────────
    console.print(
        f"\n[bold blue]🔤 Loading tokeniser:[/bold blue] {model_cfg['name']}"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"],
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    # Many instruct models don't set a pad token — use EOS as a safe fallback
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        console.print("   [dim]pad_token not set → using eos_token[/dim]")

    max_seq_length = train_cfg.get("max_seq_length", 512)
    tokenized_splits = DatasetDict(
        {
            name: tokenize_dataset(split, tokenizer, max_seq_length)
            for name, split in splits.items()
        }
    )

    # ── Step 6: Save to disk (optional) ──────────────────────────────────────
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        tokenized_splits.save_to_disk(output_dir)
        console.print(
            f"\n[green]💾 Saved processed dataset to {output_dir}[/green]"
        )

    return tokenized_splits


# ─── CLI ENTRY POINT ─────────────────────────────────────────────────────────
def main():
    """
    Command-line interface for data preparation.

    Examples:
        # Full pipeline — process and save
        python src/data_prep.py --config configs/lora_config.yaml --output data/processed

        # Quick look at formatted prompts
        python src/data_prep.py --config configs/lora_config.yaml --peek
    """
    parser = argparse.ArgumentParser(
        description="Prepare text-to-SQL dataset for fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/lora_config.yaml",
        help="Path to YAML config file (default: configs/lora_config.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory to save the processed dataset (default: don't save)",
    )
    parser.add_argument(
        "--peek",
        action="store_true",
        help="Print 5 sample prompts and exit (no tokenisation)",
    )

    args = parser.parse_args()

    # Load config and run the pipeline
    config = load_config(args.config)
    prepare_data(config, output_dir=args.output, peek=args.peek)


if __name__ == "__main__":
    main()
