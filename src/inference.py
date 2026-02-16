"""
==============================================================================
 Inference / Serving — Run Your Fine-Tuned Model
==============================================================================

This module provides multiple ways to serve your fine-tuned text-to-SQL model:

    1. LOCAL MODE    — Load the merged model with HF `pipeline`, interactive REPL
    2. VLLM MODE     — Launch a high-throughput vLLM server, call via HTTP
    3. FASTAPI MODE  — Custom REST API wrapping the HF pipeline

KEY CONCEPTS YOU'LL LEARN HERE:
    • HF pipeline()       : one-liner inference for any HF model
    • vLLM                : PagedAttention-based serving (2–4× faster than HF)
    • LoRA adapter loading: reload just the adapter on top of the base model
    • Prompt engineering   : feeding the same instruction template at inference

USAGE:
    # Interactive REPL (loads merged model locally)
    python src/inference.py --mode local --model results/merged_model

    # Launch FastAPI server
    python src/inference.py --mode api --model results/merged_model --port 8000

    # Launch vLLM serving
    python src/inference.py --mode vllm --model results/merged_model --port 8000
==============================================================================
"""

import argparse
import os
import sys
from typing import Optional

import torch
from rich.console import Console
from rich.panel import Panel

console = Console()


# ==============================================================================
#  PROMPT TEMPLATE
# ==============================================================================
# This MUST match the template used during training (from lora_config.yaml).
# If the prompt format differs, the model will produce garbage.
PROMPT_TEMPLATE = """Below is a question about a database along with the schema.
Write the SQL query that answers the question.

### Question:
{question}

### Schema:
{context}

### SQL:
"""
# NOTE: We intentionally omit the {answer} part — the model generates it.


# ==============================================================================
#  1. LOCAL INFERENCE  (Hugging Face pipeline)
# ==============================================================================
def load_local_model(model_path: str, use_adapter: bool = False, base_model: str = None):
    """
    Load a fine-tuned model for local inference.

    TWO LOADING STRATEGIES:
        a) Merged model (default):
           The adapter weights have been baked into the base model.
           Load it like any other HF model — no PEFT dependency needed.

        b) Adapter-only (use_adapter=True):
           Load the original base model, then overlay the LoRA adapter.
           This is useful when you want to swap adapters without storing
           multiple full-size models.

    Args:
        model_path:  Path to the merged model or adapter directory.
        use_adapter: If True, load base_model + adapter separately.
        base_model:  Required if use_adapter=True; the base model ID.

    Returns:
        (model, tokenizer) tuple.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    console.print(f"[bold blue]🤖 Loading model from:[/bold blue] {model_path}")

    if use_adapter:
        # ── Strategy B: base + adapter ───────────────────────────────────────
        from peft import PeftModel

        if not base_model:
            raise ValueError("--base-model is required when --use-adapter is set")

        console.print(f"   Base model: {base_model}")
        console.print(f"   Adapter:    {model_path}")

        # Load the base model (optionally quantised for memory savings)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        # Overlay the LoRA adapter — this modifies the model in-place
        model = PeftModel.from_pretrained(model, model_path)
        # Merge for faster inference (optional but recommended)
        model = model.merge_and_unload()

        tokenizer = AutoTokenizer.from_pretrained(base_model)
    else:
        # ── Strategy A: merged model ────────────────────────────────────────
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",        # Auto GPU/CPU placement
            torch_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    console.print("[green]✓[/green] Model loaded successfully")
    return model, tokenizer


def generate_sql(
    model,
    tokenizer,
    question: str,
    schema: str,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    top_p: float = 0.95,
) -> str:
    """
    Generate a SQL query from a natural-language question and schema.

    GENERATION PARAMETERS EXPLAINED:
        • max_new_tokens: Maximum tokens to generate (not counting the prompt).
          SQL queries are usually short, so 256 is generous.

        • temperature: Controls randomness.
          0.0 = greedy (always pick the most likely token)
          1.0 = sample from the full distribution
          For code generation, low temperature (0.1) is best — we want
          deterministic, correct SQL.

        • top_p (nucleus sampling): Only consider tokens whose cumulative
          probability exceeds top_p.  Filters out unlikely tokens.

    Args:
        model:          The loaded HF model.
        tokenizer:      The corresponding tokenizer.
        question:       Natural-language question (e.g. "How many employees?")
        schema:         SQL schema (CREATE TABLE statements).
        max_new_tokens: Max generated tokens.
        temperature:    Sampling temperature (0.0 = greedy).
        top_p:          Nucleus sampling threshold.

    Returns:
        The generated SQL query string.
    """
    # Format the prompt using the same template as training
    prompt = PROMPT_TEMPLATE.format(question=question, context=schema)

    # Tokenise the prompt
    inputs = tokenizer(
        prompt,
        return_tensors="pt",       # Return PyTorch tensors
        truncation=True,
        max_length=512,
    ).to(model.device)             # Move to the same device as the model

    # Generate with the model
    with torch.no_grad():          # No need to track gradients for inference
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,  # Greedy if temperature=0
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the NEW tokens (skip the prompt)
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    sql = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return sql


def interactive_repl(model, tokenizer):
    """
    Interactive Read-Eval-Print Loop for text-to-SQL.

    Type your question and schema, get a SQL query back.
    Type 'quit' or 'exit' to stop.
    """
    console.print(Panel.fit(
        "[bold green]🗣️  Interactive Text-to-SQL[/bold green]\n"
        "Enter a question and schema to generate SQL.\n"
        "Type [bold]quit[/bold] or [bold]exit[/bold] to stop.",
        border_style="green",
    ))

    while True:
        console.print("\n[bold cyan]Question:[/bold cyan] ", end="")
        question = input().strip()

        if question.lower() in ("quit", "exit", "q"):
            console.print("[dim]Bye![/dim]")
            break

        console.print("[bold cyan]Schema (paste, then press Enter twice):[/bold cyan]")
        schema_lines = []
        while True:
            line = input()
            if line == "":
                break
            schema_lines.append(line)
        schema = "\n".join(schema_lines)

        if not schema:
            console.print("[red]Please provide a schema.[/red]")
            continue

        console.print("\n[yellow]⏳ Generating …[/yellow]")
        sql = generate_sql(model, tokenizer, question, schema)

        console.print(f"\n[bold green]Generated SQL:[/bold green]")
        console.print(Panel(sql, border_style="green"))


# ==============================================================================
#  2. VLLM SERVING
# ==============================================================================
def launch_vllm_server(model_path: str, port: int = 8000, gpu_memory_utilization: float = 0.9):
    """
    Launch a vLLM OpenAI-compatible API server.

    WHAT IS VLLM?
        vLLM uses PagedAttention — a technique inspired by virtual memory
        in operating systems.  Instead of pre-allocating contiguous memory
        for each sequence's KV cache, it allocates small "pages" on demand.

        This gives:
        • 2–4× higher throughput than HF generate()
        • Near-zero memory waste
        • Continuous batching (new requests don't wait for the batch to finish)

    The server exposes an OpenAI-compatible endpoint:
        POST http://localhost:{port}/v1/completions
        POST http://localhost:{port}/v1/chat/completions

    You can call it with the `openai` Python client or plain `requests`.

    Args:
        model_path:             Path to the merged model.
        port:                   Port to serve on.
        gpu_memory_utilization: Fraction of GPU memory vLLM can use (0.0–1.0).
    """
    console.print(Panel.fit(
        f"[bold blue]🚀 Launching vLLM server[/bold blue]\n"
        f"  Model: {model_path}\n"
        f"  Port:  {port}\n"
        f"  GPU memory utilisation: {gpu_memory_utilization:.0%}",
        border_style="blue",
    ))

    # vLLM provides a CLI entry point that starts the server
    # We invoke it programmatically via subprocess for more control
    import subprocess

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--dtype", "float16",
    ]

    console.print(f"[dim]Running: {' '.join(cmd)}[/dim]")
    console.print(f"\n[green]Server will be available at:[/green] http://localhost:{port}/v1/completions")
    console.print("[dim]Press Ctrl+C to stop.[/dim]\n")

    # Start the server (blocks until Ctrl+C)
    subprocess.run(cmd)


def call_vllm_api(
    prompt: str,
    port: int = 8000,
    max_tokens: int = 256,
    temperature: float = 0.1,
) -> str:
    """
    Send a completion request to a running vLLM server.

    This is a convenience function you can use from notebooks or other scripts
    to query the vLLM server programmatically.

    Args:
        prompt:      The full formatted prompt string.
        port:        Port the vLLM server is running on.
        max_tokens:  Maximum new tokens to generate.
        temperature: Sampling temperature.

    Returns:
        The generated text.
    """
    import requests

    response = requests.post(
        f"http://localhost:{port}/v1/completions",
        json={
            "model": "default",      # vLLM uses this as the model identifier
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )
    response.raise_for_status()
    return response.json()["choices"][0]["text"].strip()


# ==============================================================================
#  3. FASTAPI SERVING
# ==============================================================================
def create_fastapi_app(model_path: str):
    """
    Create a FastAPI application wrapping the HF model.

    ENDPOINTS:
        POST /generate
            Body: {"question": "...", "schema": "..."}
            Returns: {"sql": "SELECT ...", "latency_ms": 42.5}

        GET /health
            Returns: {"status": "ok", "model": "..."}

    WHY FASTAPI?
        • Auto-generated OpenAPI docs at /docs
        • Async support for high concurrency
        • Type validation via Pydantic
        • Easy to deploy behind nginx / Kubernetes

    Args:
        model_path: Path to the merged model or adapter directory.

    Returns:
        A FastAPI app instance.
    """
    import time

    from fastapi import FastAPI
    from pydantic import BaseModel

    # ── Define request/response schemas ──────────────────────────────────────
    class SQLRequest(BaseModel):
        """Input schema for the /generate endpoint."""
        question: str           # The natural-language question
        schema_text: str        # The SQL CREATE TABLE schema(s)
        max_new_tokens: int = 256
        temperature: float = 0.1

    class SQLResponse(BaseModel):
        """Output schema for the /generate endpoint."""
        sql: str                # The generated SQL query
        latency_ms: float       # How long generation took

    class HealthResponse(BaseModel):
        """Output schema for the /health endpoint."""
        status: str
        model: str

    # ── Create the FastAPI app ───────────────────────────────────────────────
    app = FastAPI(
        title="Text-to-SQL API",
        description="Generate SQL queries from natural language using a fine-tuned LLM",
        version="1.0.0",
    )

    # We load the model at startup and keep it in memory
    # Using a dictionary to hold mutable state (closures can't rebind outer vars)
    state = {"model": None, "tokenizer": None}

    @app.on_event("startup")
    async def startup():
        """Load the model when the server starts."""
        console.print("[yellow]⏳[/yellow] Loading model for API serving …")
        model, tokenizer = load_local_model(model_path)
        state["model"] = model
        state["tokenizer"] = tokenizer
        console.print("[green]✓[/green] Model loaded — API is ready!")

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        return HealthResponse(
            status="ok",
            model=model_path,
        )

    @app.post("/generate", response_model=SQLResponse)
    async def generate(request: SQLRequest):
        """
        Generate a SQL query from a question and schema.

        This endpoint:
            1. Formats the prompt using the training template
            2. Runs model.generate()
            3. Returns the SQL and latency
        """
        start = time.time()

        sql = generate_sql(
            model=state["model"],
            tokenizer=state["tokenizer"],
            question=request.question,
            schema=request.schema_text,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
        )

        latency_ms = (time.time() - start) * 1000

        return SQLResponse(sql=sql, latency_ms=round(latency_ms, 1))

    return app


def launch_fastapi_server(model_path: str, host: str = "0.0.0.0", port: int = 8000):
    """
    Launch the FastAPI server using Uvicorn.

    Args:
        model_path: Path to the merged model.
        host:       Host to bind to.
        port:       Port to serve on.
    """
    import uvicorn

    console.print(Panel.fit(
        f"[bold blue]🌐 Launching FastAPI server[/bold blue]\n"
        f"  Model:  {model_path}\n"
        f"  URL:    http://{host}:{port}\n"
        f"  Docs:   http://{host}:{port}/docs",
        border_style="blue",
    ))

    # Create the app and inject the model path via a module-level variable
    # (Uvicorn needs a string import path, but we work around it here)
    app = create_fastapi_app(model_path)
    uvicorn.run(app, host=host, port=port)


# ==============================================================================
#  CLI ENTRY POINT
# ==============================================================================
def main():
    """
    Command-line interface for inference / serving.

    Examples:
        # Interactive local REPL
        python src/inference.py --mode local --model results/merged_model

        # FastAPI server
        python src/inference.py --mode api --model results/merged_model --port 8000

        # vLLM high-throughput server
        python src/inference.py --mode vllm --model results/merged_model --port 8000

        # Load adapter instead of merged model
        python src/inference.py --mode local --model results/adapter \\
            --use-adapter --base-model mistralai/Mistral-7B-v0.1
    """
    parser = argparse.ArgumentParser(
        description="Run inference with your fine-tuned text-to-SQL model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["local", "api", "vllm"],
        default="local",
        help="Serving mode: 'local' (REPL), 'api' (FastAPI), 'vllm' (vLLM server)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the merged model or adapter directory",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for API/vLLM server (default: 8000)",
    )
    parser.add_argument(
        "--use-adapter",
        action="store_true",
        help="Load as adapter on top of base model (instead of merged)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model ID (required if --use-adapter is set)",
    )

    args = parser.parse_args()

    if args.mode == "local":
        # ── Interactive REPL ─────────────────────────────────────────────────
        model, tokenizer = load_local_model(
            args.model,
            use_adapter=args.use_adapter,
            base_model=args.base_model,
        )
        interactive_repl(model, tokenizer)

    elif args.mode == "api":
        # ── FastAPI server ───────────────────────────────────────────────────
        launch_fastapi_server(args.model, port=args.port)

    elif args.mode == "vllm":
        # ── vLLM server ─────────────────────────────────────────────────────
        launch_vllm_server(args.model, port=args.port)


if __name__ == "__main__":
    main()
