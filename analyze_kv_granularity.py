"""
Sample KV cache activations from Qwen2.5-7B-Instruct and analyze their
distribution to decide WHICH quantization granularity (per-tensor,
per-token, per-head, per-channel) is actually justified -- rather than
assuming one. See concepts/kv-cache-int8-quantization.md.

Per layer, the sampled K/V activations are assembled into a single tensor
matching the actual KV cache memory layout: [block_num, block_size, 2,
head, head_dim] (2 = K/V), the same convention used by the serving engine.

For each candidate granularity, we group the tensor along the dims that
would each get their own scale, take the max-abs value within each group,
and compute the outlier ratio = max(group-max) / median(group-max):
  - high ratio -> groups along this axis have very different magnitudes,
    so a shared scale (i.e. NOT splitting on this axis) would force small
    groups to share range with a big outlier group -> real benefit to
    giving this axis its own per-group scale.
  - low ratio  -> groups along this axis already look alike -> little to
    gain from specializing a scale per group on this axis.
  - per-tensor is the trivial single-group case (ratio == 1.0 always) --
    it's the "no specialization" baseline the other three are compared
    against.

Usage (on the GPU pod):
    python analyze_kv_granularity.py \
        --model models/Qwen2.5-7B-Instruct \
        --data calibration_prompts.jsonl \
        --layers 0,13,27 \
        --block-size 16 \
        --out kv_granularity_analysis.png
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GRANULARITIES = {
    # name -> dims of [block_num, block_size, 2, head, head_dim] that are
    # kept as separate groups (each such group gets its own scale)
    "per_tensor": (),
    "per_token": (0, 1),
    "per_channel": (3, 4),
    "per_head": (3,),
}


def group_outlier_ratio(t: torch.Tensor, group_dims: tuple) -> float:
    all_dims = tuple(range(t.dim()))
    other_dims = tuple(d for d in all_dims if d not in group_dims)
    group_max = t.abs().amax(dim=other_dims)
    flat = group_max.flatten().float()
    return (flat.max() / flat.median().clamp_min(1e-8)).item()


def get_layer_kv(past, l):
    # past_key_values API has churned across transformers versions:
    # older versions -> plain tuple of (key, value) per layer, subscriptable directly.
    # ~4.48-5.x -> DynamicCache with .key_cache/.value_cache lists (some also expose
    #              to_legacy_cache() to convert back to a tuple).
    # transformers>=5.13 -> DynamicCache.layers[l] is a DynamicLayer with .keys/.values tensors.
    if hasattr(past, "layers"):
        return past.layers[l].keys, past.layers[l].values
    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        return past.key_cache[l], past.value_cache[l]
    if hasattr(past, "to_legacy_cache"):
        return past.to_legacy_cache()[l]
    return past[l]


@torch.no_grad()
def build_layer_cache(per_layer_k, per_layer_v, l, block_size):
    """Concatenate all calibration examples' K/V for layer l along the token
    axis, then reshape into [block_num, block_size, 2, head, head_dim]."""
    k_cat = torch.cat(per_layer_k[l], dim=0)  # [total_tokens, num_kv_heads, head_dim]
    v_cat = torch.cat(per_layer_v[l], dim=0)

    total_tokens = k_cat.shape[0]
    block_num = total_tokens // block_size
    if block_num == 0:
        raise ValueError(
            f"Only {total_tokens} tokens sampled for layer {l}, fewer than block_size={block_size}. "
            "Use more/longer calibration prompts or a smaller --block-size."
        )
    truncated = block_num * block_size
    k_cat, v_cat = k_cat[:truncated], v_cat[:truncated]

    cache = torch.stack([k_cat, v_cat], dim=1)  # [total_tokens, 2, num_kv_heads, head_dim]
    num_kv_heads, head_dim = cache.shape[2], cache.shape[3]
    cache = cache.view(block_num, block_size, 2, num_kv_heads, head_dim)
    return cache


def analyze_layer(cache: torch.Tensor, label: str) -> dict:
    print(f"\n--- {label} ---  cache shape [block_num={cache.shape[0]}, block_size={cache.shape[1]}, "
          f"2, head={cache.shape[3]}, head_dim={cache.shape[4]}]")
    ratios = {}
    for name, group_dims in GRANULARITIES.items():
        ratio = group_outlier_ratio(cache, group_dims)
        ratios[name] = ratio
        print(f"    {name:12s} outlier ratio (max-group / median-group max-abs): {ratio:.1f}x")
    print("    -> higher ratio = more outlier structure on that axis = more benefit from quantizing on it")
    return ratios


def plot_comparison(results: dict, out_path: str):
    """results: {layer_label: {granularity: ratio}}"""
    granularities = list(GRANULARITIES.keys())
    layer_labels = list(results.keys())

    x = range(len(granularities))
    width = 0.8 / max(len(layer_labels), 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, layer_label in enumerate(layer_labels):
        ratios = [results[layer_label][g] for g in granularities]
        offsets = [xi + i * width for xi in x]
        ax.bar(offsets, ratios, width=width, label=layer_label)

    ax.set_yscale("log")
    ax.set_ylabel("outlier ratio (max-group / median-group), log scale")
    ax.set_xlabel("quantization granularity")
    ax.set_xticks([xi + width * (len(layer_labels) - 1) / 2 for xi in x])
    ax.set_xticklabels(granularities)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="per_tensor baseline")
    ax.legend()
    ax.set_title("KV cache outlier ratio by quantization granularity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved comparison chart to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="jsonl file with {'text': ...} per line")
    ap.add_argument("--layers", default="", help="comma-separated layer indices to analyze; default = evenly spaced 5 layers")
    ap.add_argument("--block-size", type=int, default=16, help="tokens per block, matches the paged KV cache convention")
    ap.add_argument("--out", default="kv_granularity_analysis.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager"
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

    per_layer_k = {l: [] for l in layer_ids}
    per_layer_v = {l: [] for l in layer_ids}

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            out = model(**inputs, use_cache=True)
            past = out.past_key_values
            for l in layer_ids:
                k, v = get_layer_kv(past, l)  # [batch=1, num_kv_heads, seq_len, head_dim]
                per_layer_k[l].append(k[0].transpose(0, 1).cpu())  # -> [seq_len, num_kv_heads, head_dim]
                per_layer_v[l].append(v[0].transpose(0, 1).cpu())

    results = {}
    for l in layer_ids:
        cache = build_layer_cache(per_layer_k, per_layer_v, l, args.block_size)
        label = f"Layer {l}"
        results[label] = analyze_layer(cache, label)

    plot_comparison(results, args.out)


if __name__ == "__main__":
    main()
