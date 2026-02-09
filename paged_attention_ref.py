"""
Pure-PyTorch reproduction of the paged attention kernel used by vLLM for
Qwen2.5-7B-Instruct decode (FLASH_ATTN backend -> flash_attn_varlen_func
with block_table).

Purpose: de-risk INT8 KV cache quantization by first having a numerically
validated PyTorch model of the kernel. Quant/dequant experiments get
injected here and measured, before writing any CUDA.

The behavioral spec comes from the kernel's own unit test
(vllm/tests/kernels/attention/test_flash_attn.py::test_varlen_with_paged_kv):

  - query is packed varlen: [total_query_tokens, num_q_heads, head_size],
    per-sequence boundaries given by query_lens (kernel: cu_seqlens_q).
  - key_cache / value_cache: [num_blocks, block_size, num_kv_heads,
    head_size]. This is the kernel-facing view; vLLM's engine stores one
    fused tensor [num_blocks, 2, block_size, num_kv_heads, head_size] and
    splits it (flash_attn.py: kv_cache.transpose(1, 2).split(head_size)).
  - block_tables: [num_seqs, max_blocks_per_seq] int32 -- the indirection
    layer that makes it "paged": logical block i of a sequence lives at
    physical block block_tables[seq, i].
  - causal masking is anchored to the END of the kv sequence: the last
    query token sees all kv_len tokens, query token j (0-based from the
    start of this sequence's queries) sees the first
    kv_len - query_len + 1 + j tokens. Decode is query_len == 1: the
    single query sees everything.
  - GQA: q heads outnumber kv heads; kv head h serves q heads
    [h * ratio, (h+1) * ratio).
  - softmax in fp32, output cast back to the value dtype.

Run on a GPU pod inside the vllm venv:
    python paged_attention_ref.py
"""

import torch


def paged_attention_pytorch(
    query: torch.Tensor,        # [total_q_tokens, num_q_heads, head_size]
    key_cache: torch.Tensor,    # [num_blocks, block_size, num_kv_heads, head_size]
    value_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_size]
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,  # [num_seqs, max_blocks_per_seq]
    scale: float,
) -> torch.Tensor:
    """Reproduces flash_attn_varlen_func(causal=True, block_table=...).

    One python-level loop over sequences; everything inside is vectorized.
    """
    _, block_size, num_kv_heads, head_size = key_cache.shape
    num_q_heads = query.shape[1]
    group_size = num_q_heads // num_kv_heads

    outputs = []
    q_start = 0
    for seq_idx, (q_len, kv_len) in enumerate(zip(query_lens, kv_lens)):
        q = query[q_start : q_start + q_len]  # [q_len, num_q_heads, head_size]
        q_start += q_len

        # --- paged gather: block table indirection, then trim padding ---
        num_blocks_used = (kv_len + block_size - 1) // block_size
        physical_blocks = block_tables[seq_idx, :num_blocks_used].long()
        # [num_blocks_used, block_size, kv_heads, head_size] -> flat tokens
        k = key_cache[physical_blocks].reshape(-1, num_kv_heads, head_size)[:kv_len]
        v = value_cache[physical_blocks].reshape(-1, num_kv_heads, head_size)[:kv_len]

        # --- GQA: expand kv heads to match q heads ---
        if group_size > 1:
            k = k.repeat_interleave(group_size, dim=1)  # [kv_len, num_q_heads, hs]
            v = v.repeat_interleave(group_size, dim=1)

        # --- scores: [num_q_heads, q_len, kv_len], accumulated in fp32 ---
        scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * scale

        # --- causal mask anchored to the end of the kv sequence ---
        # query j (0-based) attends to kv positions <= kv_len - q_len + j
        q_pos = torch.arange(q_len, device=query.device).unsqueeze(1)
        kv_pos = torch.arange(kv_len, device=query.device).unsqueeze(0)
        visible = kv_pos <= (kv_len - q_len) + q_pos  # [q_len, kv_len]
        scores = scores.masked_fill(~visible.unsqueeze(0), float("-inf"))

        # --- fp32 softmax, then back to value dtype for the PV matmul ---
        probs = torch.softmax(scores, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", probs, v)
        outputs.append(out)

    return torch.cat(outputs, dim=0)


def split_vllm_kv_cache(kv_cache: torch.Tensor):
    """Engine's fused KV tensor -> (key_cache, value_cache) in the
    kernel-facing [num_blocks, block_size, kv_heads, head_size] layout.

    Layout verified empirically on the live engine (vLLM 0.24, FLASH_ATTN
    backend, Qwen2.5-7B): runner.kv_caches[layer] is
    [num_blocks, 2, block_size, num_kv_heads, head_size], K at index 0 and
    V at index 1 of dim 1.
    """
    key_cache, value_cache = kv_cache.unbind(dim=1)
    return key_cache, value_cache


@torch.inference_mode()
def verify_against_kernel():
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    torch.set_default_device("cuda")
    torch.manual_seed(0)

    # Qwen2.5-7B geometry: 28 q heads / 4 kv heads, head_dim 128, block 16.
    num_q_heads, num_kv_heads, head_size, block_size = 28, 4, 128, 16
    num_blocks = 2048
    scale = head_size**-0.5
    dtype = torch.bfloat16

    configs = {
        "decode-only (batch of single-token queries)": [(1, 523), (1, 37), (1, 2011)],
        "mixed prefill + decode": [(1, 1328), (5, 18), (129, 463)],
        "chunked prefill": [(129, 463), (256, 512)],
    }

    all_ok = True
    for name, seq_lens in configs.items():
        query_lens = [x[0] for x in seq_lens]
        kv_lens = [x[1] for x in seq_lens]
        num_seqs = len(seq_lens)
        max_blocks_per_seq = (max(kv_lens) + block_size - 1) // block_size

        query = torch.randn(sum(query_lens), num_q_heads, head_size, dtype=dtype)
        key_cache = torch.randn(
            num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
        )
        value_cache = torch.randn_like(key_cache)
        block_tables = torch.randint(
            0, num_blocks, (num_seqs, max_blocks_per_seq), dtype=torch.int32
        )

        kernel_out = flash_attn_varlen_func(
            q=query,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=torch.tensor(
                [0] + query_lens, dtype=torch.int32
            ).cumsum(dim=0, dtype=torch.int32),
            seqused_k=torch.tensor(kv_lens, dtype=torch.int32),
            max_seqlen_q=max(query_lens),
            max_seqlen_k=max(kv_lens),
            softmax_scale=scale,
            causal=True,
            block_table=block_tables,
        )

        ours = paged_attention_pytorch(
            query, key_cache, value_cache, query_lens, kv_lens, block_tables, scale
        )

        max_diff = (kernel_out - ours).abs().max().item()
        try:
            # same tolerances as vllm's own test for bf16
            torch.testing.assert_close(kernel_out, ours, atol=1.5e-2, rtol=1e-2)
            status = "OK"
        except AssertionError:
            status = "MISMATCH"
            all_ok = False
        print(f"[{status}] {name}: max abs diff = {max_diff:.2e}")

    if not all_ok:
        raise SystemExit(1)
    print("\nPyTorch reproduction matches flash_attn_varlen_func on all configs.")


if __name__ == "__main__":
    verify_against_kernel()
