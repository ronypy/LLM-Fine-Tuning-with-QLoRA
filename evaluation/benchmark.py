"""
==============================================================================
 Evaluation & Benchmarking — Measure Your Fine-Tuned Model
==============================================================================

This module evaluates your text-to-SQL model using multiple complementary
metrics.  Each metric captures a different aspect of quality:

    1. PERPLEXITY       — How "surprised" is the model by the test data?
                          Lower = better.  Measures language modelling quality.

    2. EXACT MATCH      — Does the generated SQL exactly match the gold query?
                          Strict but interpretable.

    3. EXECUTION ACCURACY — Is the SQL syntactically valid?  (Parsed by sqlparse.)
                          Checks if the output is at least runnable.

    4. ROUGE-L          — Longest Common Subsequence overlap between
                          generated and gold SQL.  Good for partial credit.

    5. BLEU             — n-gram precision with brevity penalty.
                          Standard MT metric, also useful for code.

USAGE:
    # Evaluate the merged model on the test split
    python evaluation/benchmark.py \\
        --model results/merged_model \\
        --test-data data/processed \\
        --config configs/lora_config.yaml

    # Evaluate with a running vLLM server instead
    python evaluation/benchmark.py \\
        --mode vllm --port 8000 \\
        --test-data data/processed \\
        --config configs/lora_config.yaml
==============================================================================
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import sqlparse
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ==============================================================================
#  PROMPT TEMPLATE  (must match training)
# ==============================================================================
PROMPT_TEMPLATE = """Below is a question about a database along with the schema.
Write the SQL query that answers the question.

### Question:
{question}

### Schema:
{context}

### SQL:
"""


# ==============================================================================
#  1. PERPLEXITY
# ==============================================================================
def compute_perplexity(
    model,
    tokenizer,
    texts: List[str],
    batch_size: int = 8,
    max_length: int = 512,
) -> float:
    """
    Compute perplexity of the model on a list of texts.

    WHAT IS PERPLEXITY?
        Perplexity = exp(average negative log-likelihood per token)

        Intuitively, it measures how many equally-likely tokens the model
        is choosing between at each step.  A perplexity of 10 means the
        model is, on average, as uncertain as if it had to pick from
        10 equally-likely options.

        Lower perplexity = the model is more confident and accurate.

    HOW WE COMPUTE IT:
        1. Tokenise each text
        2. Run a forward pass to get the loss (cross-entropy) — this is
           the average negative log-likelihood per token
        3. Perplexity = exp(mean loss across all texts)

    Args:
        model:      The loaded HF model.
        tokenizer:  The corresponding tokenizer.
        texts:      List of full prompt+answer strings.
        batch_size: Number of texts per forward pass.
        max_length: Maximum token length.

    Returns:
        The perplexity value (float).
    """
    console.print("[yellow]⏳[/yellow] Computing perplexity …")

    model.eval()  # Set to evaluation mode (disables dropout)
    total_loss = 0.0
    total_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        # Tokenise the batch
        encodings = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)

        with torch.no_grad():  # No gradient tracking needed for evaluation
            outputs = model(**encodings, labels=encodings["input_ids"])
            # outputs.loss is the mean cross-entropy over all tokens in the batch

        # Count non-padding tokens (padding tokens have attention_mask = 0)
        num_tokens = encodings["attention_mask"].sum().item()
        total_loss += outputs.loss.item() * num_tokens
        total_tokens += num_tokens

    # Average loss → perplexity
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    console.print(f"[green]✓[/green] Perplexity: {perplexity:.2f}")
    return perplexity


# ==============================================================================
#  2. EXACT MATCH
# ==============================================================================
def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    """
    Exact string match between predicted and gold SQL queries.

    We normalise both strings first:
        • Lowercase
        • Strip whitespace
        • Remove trailing semicolons

    This is strict — even a difference in column ordering counts as wrong.
    That's why we also compute softer metrics (ROUGE, BLEU) below.

    Args:
        predictions: List of generated SQL queries.
        references:  List of gold SQL queries.

    Returns:
        Exact match accuracy as a fraction (0.0 to 1.0).
    """
    def normalise(sql: str) -> str:
        """Normalise SQL for fair comparison."""
        return sql.strip().lower().rstrip(";").strip()

    matches = sum(
        1 for pred, ref in zip(predictions, references)
        if normalise(pred) == normalise(ref)
    )
    accuracy = matches / len(predictions) if predictions else 0.0

    console.print(f"[green]✓[/green] Exact Match: {accuracy:.2%} ({matches}/{len(predictions)})")
    return accuracy


# ==============================================================================
#  3. EXECUTION ACCURACY  (SQL Syntax Validation)
# ==============================================================================
def compute_execution_accuracy(predictions: List[str]) -> float:
    """
    Check how many generated SQL queries are syntactically valid.

    We use `sqlparse` to parse each query.  If parsing succeeds and
    produces at least one statement, we count it as valid.

    NOTE: This does NOT run the SQL against a database — it only checks
    syntax.  A query could be syntactically valid but semantically wrong
    (e.g. referencing a non-existent table).

    For full execution accuracy, you'd need to:
        1. Set up a SQLite database with the schema
        2. Execute both gold and predicted queries
        3. Compare the result sets

    Args:
        predictions: List of generated SQL queries.

    Returns:
        Fraction of syntactically valid queries (0.0 to 1.0).
    """
    valid = 0
    for sql in predictions:
        try:
            # sqlparse.parse returns a list of Statement objects
            parsed = sqlparse.parse(sql.strip())
            # Check that we got at least one non-empty statement
            if parsed and str(parsed[0]).strip():
                # Additional check: first token should be a SQL keyword
                first_token = parsed[0].tokens[0]
                if first_token.ttype in (
                    sqlparse.tokens.Keyword.DML,   # SELECT, INSERT, etc.
                    sqlparse.tokens.Keyword.DDL,   # CREATE, ALTER, etc.
                    sqlparse.tokens.Keyword,
                ) or str(first_token).strip().upper() in (
                    "SELECT", "INSERT", "UPDATE", "DELETE",
                    "CREATE", "ALTER", "DROP", "WITH",
                ):
                    valid += 1
        except Exception:
            # If parsing crashes, it's definitely not valid SQL
            continue

    accuracy = valid / len(predictions) if predictions else 0.0
    console.print(
        f"[green]✓[/green] Execution Accuracy (syntax): {accuracy:.2%} ({valid}/{len(predictions)})"
    )
    return accuracy


# ==============================================================================
#  4. ROUGE-L
# ==============================================================================
def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute ROUGE-L score between predictions and references.

    WHAT IS ROUGE-L?
        ROUGE-L uses the Longest Common Subsequence (LCS) between the
        predicted and reference texts.

        Precision = LCS length / predicted length
        Recall    = LCS length / reference length
        F1        = harmonic mean of precision and recall

        Unlike exact match, ROUGE-L gives partial credit for queries
        that are mostly correct but have minor differences.

    Args:
        predictions: List of generated strings.
        references:  List of gold strings.

    Returns:
        Dict with 'rouge_l_precision', 'rouge_l_recall', 'rouge_l_f1'.
    """
    import evaluate

    rouge = evaluate.load("rouge")
    results = rouge.compute(
        predictions=predictions,
        references=references,
        rouge_types=["rougeL"],
    )

    scores = {
        "rouge_l": results["rougeL"],
    }

    console.print(f"[green]✓[/green] ROUGE-L F1: {scores['rouge_l']:.4f}")
    return scores


# ==============================================================================
#  5. BLEU
# ==============================================================================
def compute_bleu(predictions: List[str], references: List[str]) -> float:
    """
    Compute BLEU score between predictions and references.

    WHAT IS BLEU?
        BLEU (Bilingual Evaluation Understudy) measures n-gram precision:
        how many n-grams in the prediction appear in the reference.

        It includes a "brevity penalty" to penalise very short outputs.

        Originally designed for machine translation, but works well for
        any text generation task including code.

        BLEU range: 0.0 (no overlap) to 1.0 (identical).

    Args:
        predictions: List of generated strings.
        references:  List of gold strings.

    Returns:
        Corpus-level BLEU score (float).
    """
    import evaluate

    bleu = evaluate.load("bleu")

    # BLEU expects references as list of lists (multiple refs per prediction)
    refs_wrapped = [[ref] for ref in references]

    results = bleu.compute(
        predictions=predictions,
        references=refs_wrapped,
    )

    score = results["bleu"]
    console.print(f"[green]✓[/green] BLEU: {score:.4f}")
    return score


# ==============================================================================
#  6. GENERATE PREDICTIONS
# ==============================================================================
def generate_predictions(
    model,
    tokenizer,
    questions: List[str],
    schemas: List[str],
    max_new_tokens: int = 256,
    temperature: float = 0.1,
) -> List[str]:
    """
    Generate SQL predictions for a list of questions and schemas.

    Args:
        model:          Loaded HF model.
        tokenizer:      Corresponding tokenizer.
        questions:      List of natural-language questions.
        schemas:        List of SQL schemas (CREATE TABLE statements).
        max_new_tokens: Maximum tokens to generate per example.
        temperature:    Sampling temperature.

    Returns:
        List of generated SQL queries.
    """
    console.print(f"[yellow]⏳[/yellow] Generating predictions for {len(questions)} examples …")

    predictions = []
    for i, (question, schema) in enumerate(zip(questions, schemas)):
        prompt = PROMPT_TEMPLATE.format(question=question, context=schema)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        sql = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        predictions.append(sql)

        # Progress indicator every 50 examples
        if (i + 1) % 50 == 0:
            console.print(f"   [{i + 1}/{len(questions)}] generated")

    console.print(f"[green]✓[/green] Generated {len(predictions)} predictions")
    return predictions


# ==============================================================================
#  7. FULL BENCHMARK PIPELINE
# ==============================================================================
def run_benchmark(
    model_path: str,
    test_data_path: str,
    config_path: str,
    output_path: Optional[str] = None,
    max_examples: int = 200,
    use_vllm: bool = False,
    vllm_port: int = 8000,
) -> Dict[str, float]:
    """
    Run the full evaluation benchmark.

    PIPELINE:
        1. Load the test dataset
        2. Generate predictions (local model or vLLM server)
        3. Compute all metrics
        4. Print results table
        5. (Optional) Save results to JSON and log to W&B

    Args:
        model_path:     Path to merged model or adapter.
        test_data_path: Path to processed dataset (from data_prep.py).
        config_path:    Path to YAML config.
        output_path:    If provided, save results JSON to this path.
        max_examples:   Max test examples to evaluate (for speed).
        use_vllm:       If True, send requests to a running vLLM server.
        vllm_port:      Port of the vLLM server.

    Returns:
        Dictionary of metric name → value.
    """
    import yaml

    console.print(Panel.fit(
        "[bold]📊 Running Evaluation Benchmark[/bold]",
        border_style="blue",
    ))

    # ── Load config ──────────────────────────────────────────────────────────
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # ── Load test data ───────────────────────────────────────────────────────
    if os.path.exists(test_data_path):
        from datasets import load_from_disk
        dataset = load_from_disk(test_data_path)
        test_set = dataset["test"]
    else:
        # Fall back: load from Hub and split
        console.print("[yellow]⚠[/yellow] Pre-processed data not found; loading from Hub …")
        from datasets import load_dataset
        raw = load_dataset(config["data"]["dataset_name"], split="train")
        splits = raw.train_test_split(test_size=0.1, seed=42)
        test_set = splits["test"]

    # Limit examples for speed
    if len(test_set) > max_examples:
        test_set = test_set.shuffle(seed=42).select(range(max_examples))
        console.print(f"   Using {max_examples} test examples (capped for speed)")

    questions = test_set["question"]
    schemas = test_set["context"]
    gold_answers = test_set["answer"]
    full_texts = test_set["text"] if "text" in test_set.column_names else None

    # ── Generate predictions ─────────────────────────────────────────────────
    if use_vllm:
        # Query a running vLLM server
        console.print(f"[blue]Using vLLM server on port {vllm_port}[/blue]")
        import requests

        predictions = []
        for q, s in zip(questions, schemas):
            prompt = PROMPT_TEMPLATE.format(question=q, context=s)
            resp = requests.post(
                f"http://localhost:{vllm_port}/v1/completions",
                json={"model": "default", "prompt": prompt, "max_tokens": 256, "temperature": 0.1},
            )
            predictions.append(resp.json()["choices"][0]["text"].strip())
    else:
        # Load model locally
        from transformers import AutoModelForCausalLM, AutoTokenizer

        console.print(f"\n[bold blue]🤖 Loading model:[/bold blue] {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        predictions = generate_predictions(model, tokenizer, questions, schemas)

    # ── Compute metrics ──────────────────────────────────────────────────────
    console.print("\n[bold blue]📏 Computing metrics …[/bold blue]\n")
    results = {}

    # Perplexity (only for local model, needs full texts)
    if not use_vllm and full_texts:
        results["perplexity"] = compute_perplexity(model, tokenizer, full_texts)

    # Exact match
    results["exact_match"] = compute_exact_match(predictions, gold_answers)

    # Execution accuracy
    results["execution_accuracy"] = compute_execution_accuracy(predictions)

    # ROUGE-L
    try:
        rouge_scores = compute_rouge(predictions, gold_answers)
        results.update(rouge_scores)
    except Exception as e:
        console.print(f"[red]ROUGE computation failed: {e}[/red]")

    # BLEU
    try:
        results["bleu"] = compute_bleu(predictions, gold_answers)
    except Exception as e:
        console.print(f"[red]BLEU computation failed: {e}[/red]")

    # ── Print results table ──────────────────────────────────────────────────
    console.print()
    table = Table(title="Evaluation Results", border_style="blue")
    table.add_column("Metric", style="cyan", justify="left")
    table.add_column("Value", style="green", justify="right")

    for metric, value in results.items():
        if isinstance(value, float):
            if metric == "perplexity":
                table.add_row(metric, f"{value:.2f}")
            else:
                table.add_row(metric, f"{value:.4f}")
        else:
            table.add_row(metric, str(value))

    console.print(table)

    # ── Show sample predictions ──────────────────────────────────────────────
    console.print("\n[bold]📝 Sample Predictions:[/bold]\n")
    for i in range(min(5, len(predictions))):
        console.print(f"[cyan]Q:[/cyan] {questions[i]}")
        console.print(f"[green]Gold:[/green] {gold_answers[i]}")
        console.print(f"[yellow]Pred:[/yellow] {predictions[i]}")
        console.print(f"[dim]{'─' * 60}[/dim]")

    # ── Save results ─────────────────────────────────────────────────────────
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"\n[green]💾 Results saved to {output_path}[/green]")

    # ── Log to W&B (optional) ────────────────────────────────────────────────
    try:
        import wandb

        if wandb.run is not None or config.get("wandb", {}).get("enabled", False):
            wandb.init(
                project=config.get("wandb", {}).get("project", "llm-text-to-sql"),
                name="evaluation",
                job_type="eval",
            )
            wandb.log(results)
            wandb.finish()
            console.print("[green]✓[/green] Results logged to W&B")
    except ImportError:
        pass

    return results


# ==============================================================================
#  CLI ENTRY POINT
# ==============================================================================
def main():
    """
    Command-line interface for evaluation.

    Examples:
        # Evaluate locally
        python evaluation/benchmark.py \\
            --model results/merged_model \\
            --test-data data/processed \\
            --config configs/lora_config.yaml \\
            --output evaluation/results.json

        # Evaluate with vLLM server
        python evaluation/benchmark.py \\
            --mode vllm --port 8000 \\
            --test-data data/processed \\
            --config configs/lora_config.yaml
    """
    parser = argparse.ArgumentParser(
        description="Benchmark your fine-tuned text-to-SQL model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="results/merged_model",
        help="Path to the merged model (not needed if --mode vllm)",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/processed",
        help="Path to pre-processed dataset directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/lora_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=200,
        help="Max test examples to evaluate (default: 200)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["local", "vllm"],
        default="local",
        help="Evaluation mode: 'local' or 'vllm'",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="vLLM server port (default: 8000)",
    )

    args = parser.parse_args()

    run_benchmark(
        model_path=args.model,
        test_data_path=args.test_data,
        config_path=args.config,
        output_path=args.output,
        max_examples=args.max_examples,
        use_vllm=(args.mode == "vllm"),
        vllm_port=args.port,
    )


if __name__ == "__main__":
    main()
