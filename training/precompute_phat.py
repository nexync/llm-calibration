"""
Precompute empirical p_hat for every prompt in the training set using vLLM
offline inference. Outputs a JSON file mapping dataset index (int) -> p_hat
(float) that DynamicBufferSampler can load as a warm-start.

Usage:
    python3 training/precompute_phat.py \
        --model /path/to/model \
        --data   data/math/math_wager_detailed_train.parquet \
        --output data/math/train_phat.json \
        --n 16 \
        --temperature 0.7 \
        --top_p 0.95 \
        --max_tokens 512
"""

import argparse
import json
import datasets
import numpy as np
from math_verify import parse, verify
from vllm import LLM, SamplingParams


def build_prompt(row: dict) -> str:
    messages = row["prompt"]
    # Minimal chat-template rendering: concatenate user turns.
    # For the wager-detailed prompt the first (and only) user message is
    # the full problem statement with the wager instructions.
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "user":
            parts.append(f"User: {content}")
        else:
            parts.append(content)
    return "\n\n".join(parts) + "\n\nAssistant:"


def extract_confidence(text: str) -> float | None:
    import re
    m = re.match(r"\s*Confidence:\s*(\d+)", text)
    if m:
        v = int(m.group(1))
        return max(0, min(100, v)) / 100.0
    return None


def _extract_boxed(text: str) -> str | None:
    import re
    matches = list(re.finditer(r"\\boxed\{", text))
    if not matches:
        return None
    start = matches[-1].end()
    depth, i = 1, start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start : i - 1].strip() if depth == 0 else None


def _check_correct(pred: str, gold: str) -> bool:
    try:
        return verify(parse(pred, parsing_timeout=None), parse(gold, parsing_timeout=None))
    except Exception:
        return pred.strip() == gold.strip()


def extract_is_correct(text: str, ground_truths: list[str]) -> bool:
    if not ground_truths:
        return False
    pred = _extract_boxed(text)
    if pred is None:
        return False
    return any(_check_correct(pred, gt) for gt in ground_truths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()

    print(f"Loading dataset from {args.data}")
    if args.data.endswith(".parquet"):
        ds = datasets.load_dataset("parquet", data_files=args.data)["train"]
    else:
        ds = datasets.load_dataset("json", data_files=args.data)["train"]

    print(f"Dataset size: {len(ds)}")

    prompts = [build_prompt(row) for row in ds]

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.85,
    )

    sampling_params = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    print(f"Running inference with n={args.n} rollouts per prompt...")
    outputs = llm.generate(prompts, sampling_params)

    results: dict[int, float] = {}
    for idx, (row, output) in enumerate(zip(ds, outputs)):
        rm = row.get("reward_model") or {}
        gts = rm.get("ground_truth") or []
        if isinstance(gts, str):
            gts = [gts]

        corrects = [float(extract_is_correct(c.text, gts)) for c in output.outputs]
        p_hat = float(np.mean(corrects)) if corrects else 0.0
        results[idx] = p_hat

        if idx % 500 == 0:
            print(f"  [{idx}/{len(ds)}] p_hat={p_hat:.2f}")
            # checkpoint so a late crash doesn't lose everything
            with open(args.output, "w") as f:
                json.dump(results, f)

    print(f"\nSaving p_hat estimates to {args.output}")
    with open(args.output, "w") as f:
        json.dump(results, f)

    p_hats = list(results.values())
    print(f"Done. mean={np.mean(p_hats):.3f}, std={np.std(p_hats):.3f}")
    bins = np.histogram(p_hats, bins=10, range=(0, 1))[0]
    print("Distribution across decile bins:", bins.tolist())


if __name__ == "__main__":
    main()
