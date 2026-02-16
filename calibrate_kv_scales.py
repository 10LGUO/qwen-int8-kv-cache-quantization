"""
Derive static per-channel INT8 KV cache scales for every layer of
Qwen2.5-7B-Instruct from calibration prompts.

Output: torch.save dict {layer_idx: {"k_scale": [num_kv_heads, head_size],
"v_scale": ...}} -- the file consumed by the vLLM fork's
VLLM_KV_INT8_SCALES env var (vllm/v1/attention/backends/pytorch_paged_ref.py).

Usage (on the GPU pod):
    python calibrate_kv_scales.py \
        --model models/Qwen2.5-7B-Instruct \
        --data calibration_prompts.jsonl \
        --out kv_int8_scales.pt
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_layer_kv(past, l):
    if hasattr(past, "layers"):  # transformers>=5.13
        return past.layers[l].keys, past.layers[l].values
    if hasattr(past, "key_cache"):
        return past.key_cache[l], past.value_cache[l]
    return past[l]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="kv_int8_scales.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device)
    model.eval()
    num_layers = model.config.num_hidden_layers

    prompts = []
    with open(args.data) as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line)["text"])

    k_max = [None] * num_layers
    v_max = [None] * num_layers
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        past = model(**inputs, use_cache=True).past_key_values
        for l in range(num_layers):
            k, v = get_layer_kv(past, l)  # [1, num_kv_heads, seq_len, head_size]
            k_amax = k[0].abs().amax(dim=1).float()  # [num_kv_heads, head_size]
            v_amax = v[0].abs().amax(dim=1).float()
            k_max[l] = k_amax if k_max[l] is None else torch.maximum(k_max[l], k_amax)
            v_max[l] = v_amax if v_max[l] is None else torch.maximum(v_max[l], v_amax)

    scales = {
        l: {
            "k_scale": (k_max[l].clamp_min(1e-8) / 127.0).cpu(),
            "v_scale": (v_max[l].clamp_min(1e-8) / 127.0).cpu(),
        }
        for l in range(num_layers)
    }
    torch.save(scales, args.out)
    print(f"Saved per-channel scales for {num_layers} layers to {args.out}")
    print(f"k_scale shape: {tuple(scales[0]['k_scale'].shape)}")
    print(f"max k_scale overall: {max(s['k_scale'].max().item() for s in scales.values()):.5f}")


if __name__ == "__main__":
    main()
