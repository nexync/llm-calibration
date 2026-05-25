"""
Preprocess MATH data into verl-compatible parquets.

For training (parquet input):
  - math_standard_train.parquet  standard GRPO (correctness reward only)
  - math_wager_train.parquet     wager GRPO (model declares Confidence: X in thinking)

For validation (jsonl input):
  - math500_standard_val.parquet
  - math500_wager_val.parquet

Prompt format matches the existing eval pipeline (no system prompt, \\boxed{X} answer format).

Usage:
    # training parquets
    python training/preprocess.py --input data/math/math_train.parquet

    # math500 val parquets
    python training/preprocess.py --input data/math/math500_test.jsonl --split val
"""

import argparse
import json
import os

import pandas as pd

MATH_FORMAT_INSTRUCTION = (
    "After your reasoning, provide your final mathematical answer string X "
    "in the format of \\boxed{X}."
)

WAGER_INSTRUCTION = (
    "The very first line of your response must be:\n\n"
    "Confidence: X\n\n"
    "where X is an integer between 0 and 100 representing your confidence (as a "
    "percentage) that you will solve this problem correctly.\n\n"
    "Your reward is computed using a proper scoring rule: it is maximized in "
    "expectation when X equals your true empirical probability of solving this "
    "problem correctly. Declaring X too high or too low relative to your actual "
    "capability both reduce your expected reward.\n\n"
    "If you do not begin your response with 'Confidence: X', you will receive a "
    "score of -1 for this problem regardless of whether your answer is correct.\n\n"
    "For example, a correctly formatted response to a problem looks like:\n\n"
    "Confidence: 72\n\n"
    "[your reasoning here]\n\n"
    "\\boxed{your answer}"
)


WAGER_COT_INSTRUCTION = (
    "Before declaring your confidence, you may write up to 100 words of planning "
    "or solution sketch. Then, on its own line, write:\n\n"
    "Confidence: X\n\n"
    "where X is an integer between 0 and 100 representing your confidence (as a "
    "percentage) that you will solve this problem correctly.\n\n"
    "Your reward is computed using a proper scoring rule: it is maximized in "
    "expectation when X equals your true empirical probability of solving this "
    "problem correctly. Declaring X too high or too low relative to your actual "
    "capability both reduce your expected reward.\n\n"
    "If you do not write 'Confidence: X' within your first 100 words, you will "
    "receive a score of -1 for this problem regardless of whether your answer is "
    "correct. After declaring your confidence, solve the problem fully.\n\n"
    "You may refer to the following verbal labels when reasoning about your confidence:\n\n"
    "0–10: \"Almost no chance\"\n"
    "10–20: \"Highly unlikely\"\n"
    "20–30: \"Chances are slight\"\n"
    "30–40: \"Unlikely\"\n"
    "40–50: \"Less than even\"\n"
    "50–60: \"Better than even\"\n"
    "60–70: \"Likely\"\n"
    "70–80: \"Very good chance\"\n"
    "80–90: \"Highly likely\"\n"
    "90–100: \"Almost certain\"\n\n"
    "For example, a correctly formatted response looks like:\n\n"
    "[up to 100 words of planning]\n\n"
    "Confidence: 72\n\n"
    "[your full reasoning here]\n\n"
    "\\boxed{your answer}"
)


WAGER_INSTRUCTION_DETAILED = (
    "The very first line of your response must be:\n\n"
    "Confidence: X\n\n"
    "where X is an integer between 0 and 100 representing your confidence (as a "
    "percentage) that you will solve this problem correctly.\n\n"
    "Your reward is computed using a proper scoring rule: it is maximized in "
    "expectation when X equals your true empirical probability of solving this "
    "problem correctly. Declaring X too high or too low relative to your actual "
    "capability both reduce your expected reward.\n\n"
    "If you do not begin your response with 'Confidence: X', you will receive a "
    "score of -1 for this problem regardless of whether your answer is correct.\n\n"
    "You may refer to the following verbal labels when reasoning about your confidence:\n\n"
    "0–10: \"Almost no chance\"\n"
    "10–20: \"Highly unlikely\"\n"
    "20–30: \"Chances are slight\"\n"
    "30–40: \"Unlikely\"\n"
    "40–50: \"Less than even\"\n"
    "50–60: \"Better than even\"\n"
    "60–70: \"Likely\"\n"
    "70–80: \"Very good chance\"\n"
    "80–90: \"Highly likely\"\n"
    "90–100: \"Almost certain\"\n\n"
    "For example, a correctly formatted response to a problem looks like:\n\n"
    "Confidence: 72\n\n"
    "[your reasoning here]\n\n"
    "\\boxed{your answer}"
)


def build_standard_prompt(question: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": f"{question}\n\n{MATH_FORMAT_INSTRUCTION}",
        }
    ]


def build_wager_prompt(question: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": f"{question}\n\n{MATH_FORMAT_INSTRUCTION}\n\n{WAGER_INSTRUCTION}",
        }
    ]


def build_wager_detailed_prompt(question: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": f"{question}\n\n{MATH_FORMAT_INSTRUCTION}\n\n{WAGER_INSTRUCTION_DETAILED}",
        }
    ]


def build_wager_cot_prompt(question: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": f"{question}\n\n{MATH_FORMAT_INSTRUCTION}\n\n{WAGER_COT_INSTRUCTION}",
        }
    ]


def load_input(path):
    if path.endswith(".jsonl"):
        rows = []
        with open(path) as f:
            for line in f:
                rows.append(json.loads(line))
        return rows
    else:
        return pd.read_parquet(path).to_dict("records")


def get_token_length(prompt: list[dict], tokenizer) -> int:
    text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/math/math_train.parquet")
    parser.add_argument("--output_dir", default="data/math")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--max_prompt_length", type=int, default=None,
                        help="Skip examples whose wager prompt exceeds this many tokens")
    parser.add_argument("--tokenizer", default=None,
                        help="Tokenizer path/name (required when --max_prompt_length is set)")
    args = parser.parse_args()

    tokenizer = None
    if args.max_prompt_length is not None:
        if args.tokenizer is None:
            parser.error("--tokenizer is required when --max_prompt_length is set")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"Loading {args.input}...")
    records = load_input(args.input)
    if args.max_examples:
        records = records[: args.max_examples]
    print(f"Loaded {len(records)} examples")

    standard_rows = []
    wager_rows = []
    wager_detailed_rows = []
    wager_cot_rows = []
    skipped = 0

    for item in records:
        question = item["problem"]
        answer = item["answer"]
        meta = {
            "subject": item.get("subject", ""),
            "level": int(item.get("level", 0)),
        }

        wager_prompt = build_wager_prompt(question)
        wager_detailed_prompt = build_wager_detailed_prompt(question)
        wager_cot_prompt = build_wager_cot_prompt(question)
        if tokenizer is not None:
            # detailed wager prompt is longest, so only need to check that one
            if get_token_length(wager_detailed_prompt, tokenizer) > args.max_prompt_length:
                skipped += 1
                continue

        standard_rows.append({
            "data_source": "math",
            "prompt": build_standard_prompt(question),
            "reward_model": {"ground_truth": answer},
            "extra_info": meta,
        })

        wager_rows.append({
            "data_source": "math",
            "prompt": wager_prompt,
            "reward_model": {"ground_truth": answer},
            "extra_info": meta,
        })

        wager_detailed_rows.append({
            "data_source": "math",
            "prompt": wager_detailed_prompt,
            "reward_model": {"ground_truth": answer},
            "extra_info": meta,
        })

        wager_cot_rows.append({
            "data_source": "math",
            "prompt": wager_cot_prompt,
            "reward_model": {"ground_truth": answer},
            "extra_info": meta,
        })

    if skipped:
        print(f"Skipped {skipped} examples exceeding max_prompt_length={args.max_prompt_length}")

    os.makedirs(args.output_dir, exist_ok=True)

    standard_path = os.path.join(args.output_dir, f"math_standard_{args.split}.parquet")
    wager_path = os.path.join(args.output_dir, f"math_wager_{args.split}.parquet")
    wager_detailed_path = os.path.join(args.output_dir, f"math_wager_detailed_{args.split}.parquet")
    wager_cot_path = os.path.join(args.output_dir, f"math_wager_cot_{args.split}.parquet")

    pd.DataFrame(standard_rows).to_parquet(standard_path, index=False)
    pd.DataFrame(wager_rows).to_parquet(wager_path, index=False)
    pd.DataFrame(wager_detailed_rows).to_parquet(wager_detailed_path, index=False)
    pd.DataFrame(wager_cot_rows).to_parquet(wager_cot_path, index=False)

    print(f"Saved {len(standard_rows)} rows to {standard_path}")
    print(f"Saved {len(wager_rows)} rows to {wager_path}")
    print(f"Saved {len(wager_detailed_rows)} rows to {wager_detailed_path}")
    print(f"Saved {len(wager_cot_rows)} rows to {wager_cot_path}")


if __name__ == "__main__":
    main()
