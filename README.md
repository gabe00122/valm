# VALM

Research framework for LLM finetuning with value approximation written in jax.

Trains a Qwen3 4b from 1% to 99% on wordle in 11 hours on a single 5090 gpu!
The model weights used for continuous batching are the same used for model updates saving vram and allowing training to happen on a single consumer gpu.

Experiemental results/writeup here: https://gabrielkeith.dev/posts/valm

## Overview

- **Custom Qwen3 Implementation**: A from-scratch JAX/Flax implementation, clean example of Qwen3 in jax using cudnn attention
- **Continuous Batched Inference**: Simple and efficient continous batching, using jax primitives like vmap and a jax.lax.while_loop to generate multiple tokens on the gpu without python overhead. ~8,200 tokens/s output on a RTX 5090 with 128 slots.
- **PPO and GRPO**: Clean PPO implementation with value function warmup and a GRPO implementation with interleved continous batching by group id
- **Novel approximation**: Value network architecture that reads latents from the base model at multiple layers of the base model to improve value approximation accuracy.
- **Experiement Manage Tools**: Episode viewer for debugging and analysis, checkpointing, detailed logging
- **Textual Environments in Rust**: The environment time is trivial compared to the inference python would have been fast enough. The real reason for writing them in rust was because it was fun.

## Limitations
- Only Qwen3 is supported, the LLM implementation is within the project (and based on my gridworld model implementation https://github.com/gabe00122/mapox-trainer). This means to support a new model I'd need to program the architecture into the framework.
- No prefill, the raw token decode is very fast but prompt tokens are handle in the same loop as output tokens
- The only environment is Wordle for now, more environments are planned.

## Installation

### Prerequisites

- Python 3.14+
- Rust (for building environments)
- Nvidia GPU for cudnn (it would be easy to switch this to XLA attention in attention.py)

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd valm

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
uv run valm pipeline configs/test.json
```

For PPO this kicks off three stages
1) Data collection to a offline_data directory (by default 20,000 games)
2) Warms the value function up on the offline data with a frozen policy
3) Trains a lora and value network with online learning based on the value net from step 2

GRPO skips both these steps and goes strait into online learning

### Evaluate Models

```bash
# Evaluate an OpenRouter model
uv run valm eval openrouter openrouter/google/gemma-4-26b-a4b-it --env wordle

# Evaluate a trained checkpoint
uv run valm eval checkpoint <experiment-name> --episodes 100
```

## Environments

### Wordle
Guess a 5-letter word in 6 tries. Feedback: G=Green (correct), Y=Yellow (wrong position), -=Grey (not in word).

The reward function is split into two components, both designed so the maximum return is 1.0.

Partial credit, granted once per slot, the first time a slot is revealed:
* +0.025 the first time slot *i* is yellow or green
* +0.025 the first time slot *i* is green

Each of the 5 slots can contribute at most 0.05, so partial credit tops out at 0.25.

Terminal bonus: +0.75 when the word is solved.

Partial rewards are applied at turn boundaries while the terminal reward is always at the end of the episode.

## Web Viewer

By default every episode during training is saved in results/ along with checkpoints and metrics
These can be viewed by launching the svelte dev server and the fastapi episode data server
You need some training data in the results folder to use the web viewer

```bash

# Run the fastapi server
uv run fastapi dev ./python/valm/server/episode_apy.py

# Run the svelte server
cd ux
npm run dev

```
