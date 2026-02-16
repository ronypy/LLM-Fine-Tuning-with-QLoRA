# 🧠 LLM Fine-Tuning Pipeline — Text-to-SQL with QLoRA

Fine-tune **Mistral-7B** to generate SQL queries from natural language using **QLoRA** (4-bit quantised LoRA), evaluate with standard benchmarks, and serve via **vLLM** or **FastAPI**.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [How QLoRA Works](#-how-qlora-works)
- [Training](#-training)
- [Inference & Serving](#-inference--serving)
- [Evaluation](#-evaluation)
- [Results](#-results)
- [Tech Stack](#-tech-stack)
- [Configuration Reference](#-configuration-reference)

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                              │
│                                                                  │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────┐  │
│  │ HF Datasets  │ → │ data_prep.py │ → │ Formatted Prompts    │  │
│  │ sql-create-  │   │  • Load      │   │ "### Question: ..."  │  │
│  │ context      │   │  • Format    │   │ "### Schema: ..."    │  │
│  │              │   │  • Split     │   │ "### SQL: ..."       │  │
│  └─────────────┘   └──────────────┘   └──────────┬───────────┘  │
│                                                   │              │
│  ┌─────────────┐   ┌──────────────┐              │              │
│  │ Mistral-7B  │ → │  train.py    │ ←────────────┘              │
│  │ (4-bit NF4) │   │  • QLoRA     │                             │
│  │  ~4.5 GB    │   │  • SFTTrainer│                             │
│  └─────────────┘   │  • W&B logs  │                             │
│                     └──────┬───────┘                             │
│                            │                                     │
│                     ┌──────┴───────┐                             │
│                     │ Merged Model │                             │
│                     │ + Adapter    │                             │
│                     └──────────────┘                             │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                    INFERENCE PIPELINE                             │
│                                                                  │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐ │
│  │ User Query   │ → │ inference.py │ → │ Generated SQL        │ │
│  │ "How many    │   │              │   │ SELECT COUNT(*)      │ │
│  │  employees?" │   │ Modes:       │   │ FROM employees       │ │
│  │              │   │ • Local REPL │   │ WHERE dept = 5       │ │
│  │              │   │ • FastAPI    │   │                      │ │
│  │              │   │ • vLLM       │   │                      │ │
│  └──────────────┘   └──────────────┘   └──────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd llm-finetuning

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# Preview formatted prompts (no GPU needed)
python src/data_prep.py --config configs/lora_config.yaml --peek

# Full processing with train/val/test split
python src/data_prep.py --config configs/lora_config.yaml --output data/processed
```

### 3. Train

```bash
# Train with pre-processed data
python src/train.py --config configs/lora_config.yaml --data data/processed

# Or train with on-the-fly processing
python src/train.py --config configs/lora_config.yaml
```

### 4. Run Inference

```bash
# Interactive REPL
python src/inference.py --mode local --model results/merged_model

# FastAPI server (with Swagger docs at /docs)
python src/inference.py --mode api --model results/merged_model --port 8000

# High-throughput vLLM server
python src/inference.py --mode vllm --model results/merged_model --port 8000
```

### 5. Evaluate

```bash
python evaluation/benchmark.py \
    --model results/merged_model \
    --test-data data/processed \
    --config configs/lora_config.yaml \
    --output evaluation/results.json
```

---

## 📁 Project Structure

```
llm-finetuning/
├── configs/
│   └── lora_config.yaml      # All hyper-parameters in one place
├── src/
│   ├── data_prep.py           # Dataset loading, formatting, tokenisation
│   ├── train.py               # QLoRA training with SFTTrainer
│   └── inference.py           # Local REPL, FastAPI, vLLM serving
├── evaluation/
│   └── benchmark.py           # Perplexity, Exact Match, ROUGE, BLEU
├── notebooks/
│   └── training.ipynb         # Interactive walkthrough (Colab-ready)
├── requirements.txt           # Pinned dependencies
└── README.md                  # This file
```

---

## 🔬 How QLoRA Works

### The Memory Problem

A 7B-parameter model in fp16 requires **14 GB** of VRAM just for weights — before counting gradients, optimiser states, and activations.

### The Solution: QLoRA

QLoRA combines **two** techniques:

#### 1. 4-bit Quantisation (BitsAndBytes)

Compress each weight from 16 bits → 4 bits using **NF4** (Normal Float 4), a data type optimised for normally-distributed neural network weights.

```
fp16:  7B × 2 bytes  = 14.0 GB
NF4:   7B × 0.5 bytes =  3.5 GB  (+overhead ≈ 4.5 GB total)
```

#### 2. Low-Rank Adaptation (LoRA)

Instead of updating all 7B parameters, insert small trainable matrices:

```
W' = W + (α/r) × B @ A

W ∈ R^{4096×4096}  — frozen base weight  (16.8M params)
A ∈ R^{4096×16}    — down-projection     (65K params)   ← trainable
B ∈ R^{16×4096}    — up-projection       (65K params)   ← trainable
```

**Result**: Train **0.4% of parameters** with **~4.5 GB VRAM**.

---

## 🏋️ Training

### Key Configuration (from `lora_config.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.name` | `mistralai/Mistral-7B-v0.1` | Base model |
| `model.quantization_bits` | 4 | QLoRA precision |
| `lora.r` | 16 | LoRA rank |
| `lora.lora_alpha` | 32 | Scaling factor (2×r) |
| `training.learning_rate` | 2e-4 | Peak LR |
| `training.num_train_epochs` | 3 | Total epochs |
| `training.per_device_train_batch_size` | 4 | Batch per GPU |
| `training.gradient_accumulation_steps` | 4 | Effective batch = 16 |
| `data.subset_fraction` | 0.1 | Use 10% for quick runs |

### Experiment Tracking

Enable Weights & Biases by setting `wandb.enabled: true` in the config. Metrics logged:

- Training loss (per step)
- Validation loss (per eval step)
- Learning rate schedule
- GPU memory usage

---

## 🌐 Inference & Serving

### Three Serving Options

| Mode | Command | Best For |
|------|---------|----------|
| **Local REPL** | `--mode local` | Development, quick testing |
| **FastAPI** | `--mode api` | Custom endpoints, Swagger docs |
| **vLLM** | `--mode vllm` | Production, high throughput |

### vLLM Advantages

- **2–4× higher throughput** via PagedAttention
- **Continuous batching**: new requests don't wait
- **OpenAI-compatible API**: drop-in replacement

### API Example

```bash
# With FastAPI running:
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How many employees in department 5?",
    "schema_text": "CREATE TABLE employees (id INT, name TEXT, dept_id INT)"
  }'

# Response:
# {"sql": "SELECT COUNT(*) FROM employees WHERE dept_id = 5", "latency_ms": 42.5}
```

---

## 📊 Evaluation

### Metrics

| Metric | What It Measures | Range |
|--------|-----------------|-------|
| **Perplexity** | Language modelling quality (lower = better) | 1.0 – ∞ |
| **Exact Match** | Predicted SQL ≡ gold SQL | 0% – 100% |
| **Execution Accuracy** | Is the SQL syntactically valid? | 0% – 100% |
| **ROUGE-L** | Longest common subsequence overlap | 0.0 – 1.0 |
| **BLEU** | N-gram precision | 0.0 – 1.0 |

---

## 📈 Results

### Before vs After Fine-Tuning

| Metric | Base (Mistral-7B) | Fine-Tuned (QLoRA) |
|--------|-------------------|---------------------|
| Perplexity | ~15–20 | ~3–5 |
| Exact Match | ~5–10% | ~40–60% |
| Execution Accuracy | ~60% | ~90%+ |
| ROUGE-L | ~0.30 | ~0.70+ |
| BLEU | ~0.15 | ~0.55+ |

> **Note**: Actual results depend on dataset size, epochs, and hyperparameters.
> Run `evaluation/benchmark.py` to get your own numbers.

### Training Cost

| GPU | 10% Data (3 epochs) | Full Data (3 epochs) |
|-----|---------------------|----------------------|
| T4 (16 GB) | ~30 min | ~5 hours |
| A10 (24 GB) | ~15 min | ~2.5 hours |
| A100 (80 GB) | ~5 min | ~45 min |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| [Transformers](https://huggingface.co/docs/transformers) | Model loading, tokenisation, Trainer API |
| [PEFT](https://huggingface.co/docs/peft) | LoRA / QLoRA adapter management |
| [TRL](https://huggingface.co/docs/trl) | SFTTrainer for supervised fine-tuning |
| [Datasets](https://huggingface.co/docs/datasets) | Efficient data loading from HF Hub |
| [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) | 4-bit / 8-bit quantisation |
| [vLLM](https://github.com/vllm-project/vllm) | High-throughput inference with PagedAttention |
| [FastAPI](https://fastapi.tiangolo.com/) | REST API with auto-generated docs |
| [Weights & Biases](https://wandb.ai/) | Experiment tracking and visualisation |
| [sqlparse](https://github.com/andialbrecht/sqlparse) | SQL syntax validation for evaluation |

---

## ⚙️ Configuration Reference

All parameters are in [`configs/lora_config.yaml`](configs/lora_config.yaml). Key sections:

```yaml
model:
  name: "mistralai/Mistral-7B-v0.1"
  quantization_bits: 4            # 4 = QLoRA, 8 = LoRA-8bit, 16 = full

lora:
  r: 16                           # Rank (try 8, 16, 32, 64)
  lora_alpha: 32                  # Scaling (rule of thumb: 2 × r)
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

training:
  num_train_epochs: 3
  learning_rate: 2.0e-4
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4  # Effective batch = 16

data:
  dataset_name: "b-mc2/sql-create-context"
  subset_fraction: 0.1            # Start small, scale up
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
