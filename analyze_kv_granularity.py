"""
Sample KV cache activations from Qwen2.5-7B-Instruct and analyze their
distribution to decide WHICH quantization granularity (per-tensor,
per-token, per-head, per-channel) is actually justified -- rather than
assuming one. See concepts/kv-cache-int8-quantization.md.

Two analyses, per layer, per K/V:
  1. Outlier-structure inspection: group values along each candidate axis
     and look at how much the per-group max varies. High variation along
     an axis = that axis carries real outlier structure = worth its own
     scale. Low variation = safe to share one scale across that axis.
  2. Direct empirical quant error: simulate symmetric int8 quantize/
     dequantize under each candidate granularity and measure relative
     reconstruction error. This is the ground-truth signal -- whichever
     granularity minimizes error is the one to pick.

Usage (on the GPU pod):
    python analyze_kv_granularity.py \
        --model models/Qwen2.5-7B-Instruct \
        --data calibration_prompts.jsonl \
        --layers 0,6,13,20,27
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def quant_error(x: torch.Tensor, reduce_dims: tuple[int, ...]) -> float:
    """Simulate symmetric int8 quant/dequant with per-group max-abs scale,
    where groups are everything NOT in reduce_dims (reduce_dims are the
    dims collapsed into a single shared scale)."""
    scale = x.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(1e-8) / 127.0
    x_int8 = (x / scale).round().clamp(-128, 127)
    x_hat = x_int8 * scale
    err = (x - x_hat).float()
    rel_err = err.norm() / x.float().norm().clamp_min(1e-8)
    return rel_err.item()


def outlier_ratio(x: torch.Tensor, group_dim: int) -> float:
    """Ratio of max-over-groups to median-over-groups of each group's max-abs.
    High ratio => this axis has a few groups (e.g. a few channels) that
    dominate => real per-axis structure worth quantizing on."""
    other_dims = tuple(d for d in range(x.dim()) if d != group_dim)
    group_max = x.abs().amax(dim=other_dims)  # [size of group_dim]
    return (group_max.max() / group_max.median().clamp_min(1e-8)).item()


@torch.no_grad()
def analyze_layer(k: torch.Tensor, v: torch.Tensor, label: str):
    """k, v: [seq_len, num_kv_heads, head_dim] for one calibration example,
    already concatenated across all examples along seq_len."""
    print(f"\n--- {label} ---")
    for name, t in (("K", k), ("V", v)):
        seq_len, num_heads, head_dim = t.shape

        # axis dims within this [seq_len, head, head_dim] tensor:
        # 0=token, 1=head, 2=channel(head_dim)
        print(f"  [{name}] shape (seq_len={seq_len}, heads={num_heads}, head_dim={head_dim})")

        # --- outlier-structure ratios ---
        r_channel = outlier_ratio(t, group_dim=2)
        r_head = outlier_ratio(t, group_dim=1)
        r_token = outlier_ratio(t, group_dim=0)
        print(f"    outlier ratio (max-group / median-group max-abs):"
              f" per-channel={r_channel:.1f}x  per-head={r_head:.1f}x  per-token={r_token:.1f}x")
        print(f"    -> higher ratio = more outlier structure on that axis = more benefit from quantizing on it")

        # --- direct quant error under each granularity ---
        # per-tensor: single scale for whole tensor -> reduce all dims
        e_tensor = quant_error(t, reduce_dims=(0, 1, 2))
        # per-token: scale per token, shared across head+channel
        e_token = quant_error(t, reduce_dims=(1, 2))
        # per-head: scale per head, shared across token+channel
        e_head = quant_error(t, reduce_dims=(0, 2))
        # per-channel: scale per (head, channel), shared across token -- this is
        # the granularity the wiki article settles on
        e_channel = quant_error(t, reduce_dims=(0,))
        print(f"    relative L2 quant error: per-tensor={e_tensor:.4f}  per-token={e_token:.4f}"
              f"  per-head={e_head:.4f}  per-channel={e_channel:.4f}")
        print(f"    -> lower error is better; compare against the cost of that granularity "
              f"(per-token can't be static, per-tensor is 1 scale vs per-channel's {num_heads * head_dim})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="jsonl file with {'text': ...} per line")
    ap.add_argument("--layers", default="", help="comma-separated layer indices to analyze; default = evenly spaced 5 layers")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device)
    model.eval()

    num_layers = model.config.num_hidden_layers
    if args.layers:
        layer_ids = [int(x) for x in args.layers.split(",")]
    else:
        layer_ids = sorted(set(int(x) for x in torch.linspace(0, num_layers - 1, 5)))

    prompts = []
    with open(args.data) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line)["text"])

    # accumulate per-layer K/V across all calibration prompts, concatenated along seq_len
    per_layer_k = {l: [] for l in layer_ids}
    per_layer_v = {l: [] for l in layer_ids}

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            out = model(**inputs, use_cache=True)
            past = out.past_key_values
            # past_key_values API has churned across transformers versions:
            # older versions -> plain tuple of (key, value) per layer, subscriptable directly.
            # ~4.48-5.x -> DynamicCache with .key_cache/.value_cache lists (some also expose
            #              to_legacy_cache() to convert back to a tuple).
            # transformers>=5.13 -> DynamicCache.layers[l] is a DynamicLayer with .keys/.values tensors.
            if hasattr(past, "layers"):
                get_kv = lambda l: (past.layers[l].keys, past.layers[l].values)
            elif hasattr(past, "key_cache") and hasattr(past, "value_cache"):
                get_kv = lambda l: (past.key_cache[l], past.value_cache[l])
            elif hasattr(past, "to_legacy_cache"):
                legacy = past.to_legacy_cache()
                get_kv = lambda l: legacy[l]
            else:
                get_kv = lambda l: past[l]
            for l in layer_ids:
                k, v = get_kv(l)  # [batch=1, num_kv_heads, seq_len, head_dim]
                per_layer_k[l].append(k[0].transpose(0, 1).cpu())  # -> [seq_len, num_kv_heads, head_dim]
                per_layer_v[l].append(v[0].transpose(0, 1).cpu())

    for l in layer_ids:
        k_cat = torch.cat(per_layer_k[l], dim=0)
        v_cat = torch.cat(per_layer_v[l], dim=0)
        analyze_layer(k_cat, v_cat, label=f"Layer {l}")


if __name__ == "__main__":
    main()
