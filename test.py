"""
Quick sanity check: does the base model follow the wager confidence format?
Run on a GPU node: python test.py [--model PATH] [--n N]
"""
import argparse
import re
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="/scratch/gpfs/DANQIC/jeff/models/olmo3-7b-instruct")
parser.add_argument("--parquet", default="data/math/math_wager_detailed_train.parquet")
parser.add_argument("--n", type=int, default=10)
parser.add_argument("--max-new-tokens", type=int, default=512)
args = parser.parse_args()

MODEL_PATH = args.model
WAGER_PARQUET = args.parquet
N_SAMPLES = args.n
MAX_NEW_TOKENS = args.max_new_tokens

CONF_RE = re.compile(r"^\s*Confidence:\s*([0-9]{1,3})", re.IGNORECASE | re.MULTILINE)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda"
)
model.eval()

df = pd.read_parquet(WAGER_PARQUET)
declared = 0

for i in range(N_SAMPLES):
    prompt = df["prompt"].iloc[i].tolist()
    text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)

    m = CONF_RE.search(response)
    if m:
        declared += 1

    print(f"[{i}] confidence_declared={bool(m)}  confidence={m.group(1) if m else 'None'}")
    print(repr(response[:300]))
    print()

print(f"Declared: {declared}/{N_SAMPLES}")
