# Qwen INT8 KV Cache Quantization

Hands-on practice implementing INT8 KV cache quantization (dynamic and static) for LLM inference serving.

## Model

- **Qwen2.5-7B-Instruct** ([Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct))
- bf16 weights, ~14GB on disk
- Architecture: 28 hidden layers, GQA with 28 query heads / 4 KV heads, head_dim 128

Pulled directly on the training/inference pod (not committed to this repo):

```bash
hf download Qwen/Qwen2.5-7B-Instruct --local-dir models/Qwen2.5-7B-Instruct
```

## Pod Specs

| Resource | Spec |
| --- | --- |
| GPU | 1x A100 PCIe |
| vCPU | 18 (AMD EPYC 9374F 32-Core Processor) |
| Memory | 215 GB |
| Container disk | 50 GB |

## Contents

- `analyze_kv_granularity.py` — samples KV cache activations (K/V per layer) from calibration prompts, assembles them into the `[block_num, block_size, 2, head, head_dim]` layout matching the actual paged KV cache convention, and computes an outlier ratio (max-group / median-group max-abs) for each candidate INT8 quantization granularity (per-tensor, per-token, per-channel, per-head) to determine which is actually justified. Outputs a bar chart comparing all four across the sampled layers.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install torch transformers accelerate huggingface_hub datasets matplotlib
```

## Usage

```bash
python analyze_kv_granularity.py \
    --model models/Qwen2.5-7B-Instruct \
    --data calibration_prompts.jsonl \
    --layers 0,13,27 \
    --block-size 16 \
    --out kv_granularity_analysis.png
```

`calibration_prompts.jsonl`: one JSON object per line, `{"text": "<prompt>"}`.
