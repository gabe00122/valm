# VALM

A JAX-based research framework for online reinforcement learning with Large Language Models.

## Overview

This project explores efficient multi-step RL for LLMs and investigates value approximation strategies for language tasks. The project features:

- **Custom Qwen3 Implementation**: A from-scratch JAX/Flax implementation optimized for RL training with integrated value networks
- **Parameter-Efficient Fine-Tuning**: LoRA support for attention and MLP layers
- **Fast Rust Environments**: Wordle and Arithmetic environments implemented in Rust via PyO3
- **Flexible Training**: Support for both online RL and offline value network pre-training
- **Visualization Tools**: Episode viewer for debugging and analysis

## Installation

### Prerequisites

- Python 3.13+
- Rust (for building environments)
- CUDA-compatible GPU (recommended)

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd vaml

# Install with uv (recommended)
uv sync
```

This will compile the Rust environments and install all Python dependencies.

### Download Base Model

```bash
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 \
    --local-dir ./base-models/Qwen/Qwen3-4B-Instruct-2507 \
    --exclude "*.bin"
```

## Quick Start

### Train with Online RL

```bash
uv run vaml train configs/test.json
```

### Train Value Network (Offline)

```bash
# First, build offline data
uv run vaml build-offline configs/offline.json ./offline_data 100 100

# Then train value network
uv run vaml train-value configs/value-net.json ./offline_data
```

### Evaluate Models

```bash
# Evaluate an OpenRouter model
uv run vaml eval openrouter openrouter/meta-llama/llama-3.3-8b-instruct:free --env wordle

# Evaluate a trained checkpoint
uv run vaml eval checkpoint <experiment-name> --episodes 100
```

## Environments

### Wordle
Guess a 5-letter word in 6 tries. Feedback: G=Green (correct), Y=Yellow (wrong position), -=Grey (not in word).

### Arithmetic
Solve arithmetic expressions (+, -, *, /) with numbers up to 10,000.

---

**Note**: This project is in early stages of development.
