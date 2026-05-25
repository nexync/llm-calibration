# LLM Calibration — Project State

**Goal:** Train an LLM to estimate its own capability *before* generating a long reasoning chain, so that the declared confidence is well-calibrated to the empirical probability of getting the answer right.

---

## High-level idea

We use a **wager prompt** on MATH problems: before reasoning, the model must write `Confidence: X` (an integer 0–100) as the very first line of its response. Reward is a **proper scoring rule**:

```
correct:   2R - R²      (maximised when R = true p)
incorrect: -R²          (penalises overconfidence)
missing:   -1.0         (hard penalty)
```

This incentivises the model to report its true empirical solve-rate. On top of the GRPO reward we layer two supervised signals.

---

## Training setup

| Item | Value |
|---|---|
| Base model | `llama3.2-3b-instruct` (path: `/scratch/gpfs/DANQIC/jeff/models/llama3.2-3b-instruct`) |
| Framework | [verl](./verl/) — custom fork with calibration losses added |
| Task | MATH (train split → `data/math/math_wager_detailed_train.parquet`) |
| Batch | 256 prompts × 16 rollouts per prompt |
| Steps | 150 total training steps |
| Hardware | 4× GPU node (SLURM, `pli-c` partition) |
| Infra | Princeton HPC (`/scratch/gpfs/DANQIC/jeff/`) |

**Always use** `math_wager_detailed_{train,val}.parquet` — the simple prompt causes whole-number confidence clustering.

---

## Key code locations

| File | Role |
|---|---|
| `training/preprocess.py` | Builds parquet datasets; defines the three prompt variants (standard, wager, wager-detailed) |
| `training/eval_utils.py` | Reward functions: `grpo_reward_standard`, `grpo_reward_wager` |
| `verl/verl/trainer/ppo/ray_trainer.py` | Main training loop; lines ~1570–1755 compute per-prompt Brier/correlation metrics and populate `conf_positions`, `calib_token_ids`, `calib_weights`, `conf_mse_weights` for the auxiliary losses |
| `verl/verl/workers/utils/losses.py` | `ppo_loss()`: applies `suppr_coef` and `calib_coef` auxiliary losses on top of standard GRPO loss |
| `verl/verl/workers/config/actor.py` | `ActorConfig` dataclass; lines 166–171 define `suppr_coef`, `calib_coef`, `calib_k`, decay fields |
| `verl/verl/utils/dataset/rl_dataset.py` | `RLHFDataset.__getitem__` — adds `data_index` field (stable dataset row integer) used by `DynamicBufferSampler` |
| `training/dynamic_buffer_sampler.py` | `DynamicBufferSampler` — curriculum sampler that stratifies batches by p_hat decile bin (exp8+) |
| `training/precompute_phat.py` | Offline vLLM inference to warm-start the dynamic buffer with base-model p_hat estimates; run once before exp8 on 1 GPU |
| `training/scripts/exp7.sh` | SLURM array script for exp7 (6 runs) |
| `training/scripts/exp8.sh` | SLURM array script for exp8 (6 runs); uses dynamic buffer for runs 1-3 |
| `training/scripts/exp9.sh` | SLURM array script for exp9 CoT-sketch (2 runs) |

---

## Auxiliary losses (implemented in `losses.py`)

### Suppression loss (`suppr_coef`)
Pushes **down** the log-probability of the confidence token the model declared, weighted by the squared miscalibration error:

```
suppr_loss = sum_pair  (R - p_hat)² * log P(c_R)
```

### Calibration loss (`calib_coef`)
Pushes **up** the log-probability of tokens in a Gaussian window of width `calib_k` centred on `round(p_hat * 100)`:

```
calib_loss = -sum_pair  sum_i  a_i * log P(c_i)
```

Both losses are normalised by `global_batch_size` (number of pairs) to match the per-token GRPO normalisation.

**Filtering:** `calib_filtering_steps` — after this many steps, skip the calib signal for prompts where the batch is entirely correct or entirely wrong (p_hat ∈ {0, 1}), avoiding absorbing states. **Superseded in exp8 by the dynamic buffer sampler** (see below).

**Decay:** `suppr_coef_decay_steps` / `calib_coef_decay_steps` — linearly decay coefficient to 0 over this many steps (0 = no decay).

---

## Calibration metrics (logged to wandb)

| Metric | Meaning |
|---|---|
| `reward/brier_per_prompt` | Mean (R − p_hat)² across prompts; target ≤ 0.01 |
| `reward/corr_R_phat` | Pearson correlation of mean R vs p_hat; target ≥ 0.85 |
| `reward/confidence_std` | Within-prompt std of declared R (diversity) |
| `reward/between_prompt_R_std` | Across-prompt std of mean R |
| `reward/confidence_declared` | Fraction of rollouts where model declared a confidence |
| `reward/is_correct` | Fraction correct (accuracy) |

Exp7 best run (`comb_lr2x`): Brier ~0.109, corr ~0.5. Still real headroom to target values.

---

## Experiment history

| Exp | Date | Key changes | Key findings |
|---|---|---|---|
| 0 | 0417 | Initial | Bug: interleaved rollouts; no temperature on val |
| 1 | 0420 | Fixed training/val; added supervised loss | GRPO > DAPO > DR. Bug: loss in wrong file (dp_actor.py, not losses.py) |
| 2 | 0423 | Fixed loss placement; 3 conditions: detailed prompt, suppr, calib | `.sum()→.mean()` aggregation issue |
| 3 | 0424 | Fixed loss scaling; added ε=0.05 penalty vs R=0 absorbing state | Wrong normalisation by another factor of #sequences |
| 4 | 0427 | Detailed prompt; full ablation (baseline, calib, suppr, comb) | Detailed prompt improves diversity; suppr drives away miscal; calib still attracted to R=0 |
| 5 | 0429 | Clipped suppression loss | Attracting states shift to p=0.1, p=0.9; empirical caps bimodal at 0/1 |
| 6 | 0501 | 150-step training; decay ablations (calib decay=30) | Baseline run added for comparison |
| 7 | 0511 | `calib_filtering_steps=50`: suppress calib signal at extreme-accuracy batches after step 50; LR ablation (1e-6 vs 2e-6) | Best: `comb_lr2x` (Brier 0.109). Higher LR helps. `calib_filtering` goldilocks problem identified. |
| 8 | 0525 | Dynamic buffer sampler: stratify each batch uniformly across p_hat decile bins; warm-started from base model p_hat; `calib_filtering_steps=0` | Results pending |
| 9 | 0525 | Allow model 100-word sketch before declaring confidence; tests calibration/token tradeoff | Results pending |

### Exp 7 runs

| ID | Name | LR | suppr_coef | calib_coef | decay | Brier@150 |
|---|---|---|---|---|---|---|
| 0 | calib | 1e-6 | 0.0 | 0.002 | none | ~0.18 |
| 1 | comb | 1e-6 | 0.02 | 0.002 | none | ~0.15 |
| 2 | calib_lr2x | 2e-6 | 0.0 | 0.002 | none | ~0.14 |
| 3 | comb_lr2x | 2e-6 | 0.02 | 0.002 | none | **0.109** |
| 4 | calib_lr2x_decay200 | 2e-6 | 0.0 | 0.002 | calib→0 by step 200 | ~0.13 |
| 5 | comb_lr2x_decay200 | 2e-6 | 0.02 | 0.002 | both→0 by step 200 | ~0.12 |

### Exp 8 runs (current — `main` branch)

| ID | Name | LR | suppr_coef | calib_coef | calib_decay | calib_filtering | dynamic buffer |
|---|---|---|---|---|---|---|---|
| 0 | baseline | 2e-6 | 0.02 | 0.002 | none | -1 (off) | no |
| 1 | dynbuf_rl_only | 2e-6 | 0.0 | 0.0 | none | 0 | yes |
| 2 | dynbuf_calib | 2e-6 | 0.0 | 0.002 | none | 0 | yes |
| 3 | dynbuf_comb | 2e-6 | 0.02 | 0.002 | none | 0 | yes |
| 4 | twophase_comb | 2e-6 | 0.02 | 0.002 | →0 by step 75 | 0 | no |
| 5 | twophase_calib | 2e-6 | 0.0 | 0.002 | →0 by step 75 | 0 | no |

**Pre-requisite for runs 1-3**: run `training/precompute_phat.py` on 1 GPU to generate `data/math/train_phat.json` before submitting.

### Dynamic buffer design (runs 1-3)

**Problem**: MATH difficulty is bimodal — most prompts have p_hat ≈ 0 or 1 for Llama 3B. `calib_filtering_steps` created a goldilocks problem: filter extremes → lose signal at 0/100; don't filter → intermediate calibration signal drowned out.

**Solution**: `DynamicBufferSampler` (`training/dynamic_buffer_sampler.py`) bins all training prompts into 10 p_hat decile bins and samples each batch with ~equal slots per bin. p_hat per prompt is updated via EMA (α=0.3) after every training step using rollout `is_correct` results. Warm-started from base model estimates so stratification is active from step 0.

### Two-phase calib design (runs 4-5)

**Motivation**: exp7 showed `comb_lr2x_decay200` was worse overall Brier than `comb_lr2x` but correctly clustered confidence mass near 0 for hard problems — suggesting calib decay lets RL+suppr clean up the extremes. However, decay=200 left ~25% calib signal at step 150, not enough time for RL/suppr to finish the job.

**Design**: Use `calib_filtering_steps=0` (never apply calib to all-correct or all-wrong batches) and `calib_decay=75` (calib reaches 0 by step 75, halfway through training). Phase 1 (steps 0–75): calib pushes intermediate-difficulty prompts toward correct intermediate confidences. Phase 2 (steps 75–150): RL + suppr handle the extreme-accuracy prompts without calib interference. Runs 4 vs 5 isolate the contribution of suppr.

### Exp 9 runs (CoT-sketch)

**Motivation**: Confidence declared at t=0 (before any reasoning) is pure capability estimation. Confidence declared at t=1 (after full reasoning) is essentially answer-checking. The key question is: how many extra tokens do we need to generate to meaningfully improve calibration? This is a point on a continuous axis.

**Design**: New prompt (`WAGER_COT_INSTRUCTION`) allows the model up to 100 words of planning/sketch before writing `Confidence: X`. Reward function (`grpo_reward_wager_cot`) searches for the confidence declaration within the first 100 whitespace-delimited words only; anything later scores -1. Uses `math_wager_cot_{train,val}.parquet`. `calib_filtering_steps=-1` (disabled).

**Pre-requisite**: generate cot parquets (run from `main`):
```bash
python3 training/preprocess.py --input data/math/math_train.parquet --split train
python3 training/preprocess.py --input data/math/math500_test.jsonl --split val
```

| ID | Name | LR | suppr_coef | calib_coef |
|---|---|---|---|---|
| 0 | rl_only | 2e-6 | 0.0 | 0.0 |
| 1 | comb | 2e-6 | 0.02 | 0.002 |

---

## Open questions / next steps

- Does dynamic buffer stratification break the bimodal collapse without the goldilocks tradeoff?
- Does `dynbuf_comb` beat the exp7 `comb_lr2x` baseline?
- Once Llama 3B is working well: extend to **Qwen** and **OLMo** to test generality.
- Downstream applications: budget allocation and pass@k simulation (see `applications/`).

---

## Data files

```
data/math/
  math_standard_train.parquet        # correctness-only prompt
  math_wager_train.parquet           # simple wager prompt (avoid: causes clustering)
  math_wager_detailed_train.parquet  # detailed wager prompt with verbal labels ← USE THIS
  math_wager_detailed_val.parquet
  math_standard_val.parquet
  math500_test.jsonl                 # held-out eval
  train_phat.json                    # base model p_hat per training prompt (generate via precompute_phat.py)
```
