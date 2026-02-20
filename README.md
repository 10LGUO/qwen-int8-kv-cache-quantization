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
- `paged_attention_ref.py` — pure-PyTorch reproduction of the FLASH_ATTN paged attention kernel (block-table gather, GQA expansion, end-anchored causal mask, fp32 softmax), with a verification harness against the real `flash_attn_varlen_func` at Qwen2.5-7B geometry.
- `calibrate_kv_scales.py` — derives static per-channel `[num_kv_heads, head_size]` INT8 scales for all 28 layers from calibration prompts; output consumed by the vLLM fork via `VLLM_KV_INT8_SCALES`.
- `eval_humaneval.py` — HumanEval pass@1 harness through the vLLM engine; mode (bf16 / int8) selected by env vars.
- `vllm/` — submodule pointing at the `kv-int8-quant` branch of a vLLM fork. Key changes there:
  - `vllm/v1/attention/backends/pytorch_paged_ref.py` — PyTorch paged attention (bf16 + INT8 variants) swappable into `FlashAttentionImpl.forward` via `VLLM_PYTORCH_PAGED_ATTN=1` / `VLLM_PYTORCH_PAGED_ATTN_INT8=1`; INT8 write path quantizes K/V per-channel and stores genuine int8 codes in the paged cache.
  - `tests/kernels/attention/test_flash_attn.py` — 80 added test cases running both PyTorch variants over the kernel suite's own parameter matrix.

## Results

HumanEval pass@1 (greedy, 164 problems), Qwen2.5-7B-Instruct, RTX PRO 4500 Blackwell 32GB:

| KV cache | pass@1 | |
| --- | --- | --- |
| bf16 (stock FlashAttention kernel) | 130/164 | 79.3% |
| INT8 static per-channel (this repo) | 129/164 | 78.7% |

- **0.6-point cost** for halving KV cache bytes. 9 tasks flipped: 5 lost, 4 *gained* by int8 — near-symmetric flips indicate noise-level perturbation, not systematic degradation.
- Scales were calibrated on AIME (math) prompts and evaluated on code — static per-channel scales generalize across distributions, consistent with the granularity analysis (K's outlier channels are intrinsic, not input-dependent).
- Design validated end-to-end: per-channel K scales are **folded into Q** before the QK contraction (they cannot be factored out of the dot product), V scales applied after the PV contraction — the same math a real int8 kernel needs.
- Correctness chain: PyTorch reproduction ≈ kernel (unit tests, 40 configs) → serving A/B token-identical at bf16 → INT8 divergence bounded (40 configs) → end-to-end eval above.

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
