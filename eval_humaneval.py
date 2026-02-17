"""
HumanEval pass@1 (greedy) for Qwen2.5-7B-Instruct through vLLM.

Engine mode is controlled by env vars (set outside):
  baseline: (none)
  int8:     VLLM_PYTORCH_PAGED_ATTN_INT8=1 VLLM_KV_INT8_SCALES=...

Usage:
    python eval_humaneval.py --model <path> --out results_base.json [--limit 164]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

# The repo root contains the vllm/ submodule checkout, which shadows the
# installed vllm package when this script's directory lands on sys.path.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _here]

from datasets import load_dataset
from vllm import LLM, SamplingParams

INSTRUCTION = (
    "Complete the following Python function. Reply with ONLY the complete "
    "function implementation (including the signature) in a ```python code "
    "block, no explanations.\n\n```python\n{prompt}```"
)


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


def run_one(code: str, test: str, entry_point: str, dataset_prompt: str) -> bool:
    if f"def {entry_point}" not in code:
        code = dataset_prompt + "\n" + code
    program = code + "\n\n" + test + f"\n\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
        path = f.name
    try:
        r = subprocess.run(
            ["python", path], capture_output=True, timeout=10
        )
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=164)
    args = ap.parse_args()

    ds = load_dataset("openai/openai_humaneval", split="test").select(
        range(args.limit)
    )

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": INSTRUCTION.format(prompt=ex["prompt"])}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for ex in ds
    ]
    outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=512))

    results = []
    for ex, out in zip(ds, outputs):
        code = extract_code(out.outputs[0].text)
        passed = run_one(code, ex["test"], ex["entry_point"], ex["prompt"])
        results.append({"task_id": ex["task_id"], "passed": passed})

    n_pass = sum(r["passed"] for r in results)
    summary = {
        "pass@1": n_pass / len(results),
        "passed": n_pass,
        "total": len(results),
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"pass@1 = {n_pass}/{len(results)} = {n_pass / len(results):.3f}")


if __name__ == "__main__":
    main()
