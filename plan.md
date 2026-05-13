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
| `training/scripts/exp7.sh` | Latest SLURM array script (6 runs) |

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

**Filtering:** `calib_filtering_steps` — after this many steps, skip the calib signal for prompts where the batch is entirely correct or entirely wrong (p_hat ∈ {0, 1}), avoiding absorbing states.

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

Current runs sit at Brier ~0.11–0.22 and corr ~0.4–0.6. Real headroom to target values.

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
| 7 | 0511 | `calib_filtering_steps=50`: suppress calib signal at extreme-accuracy batches after step 50; LR ablation (1e-6 vs 2e-6) | Results pending |

### Exp 7 runs (current)

| ID | Name | LR | suppr_coef | calib_coef | decay |
|---|---|---|---|---|---|
| 0 | calib | 1e-6 | 0.0 | 0.002 | none |
| 1 | comb | 1e-6 | 0.02 | 0.002 | none |
| 2 | calib_lr2x | 2e-6 | 0.0 | 0.002 | none |
| 3 | comb_lr2x | 2e-6 | 0.02 | 0.002 | none |
| 4 | calib_lr2x_decay200 | 2e-6 | 0.0 | 0.002 | calib decays over 200 steps |
| 5 | comb_lr2x_decay200 | 2e-6 | 0.02 | 0.002 | both decay over 200 steps |

---

## Open questions / next steps

- Do Exp 7 results confirm that `calib_filtering_steps` breaks the p=0,1 absorbing states?
- Does higher LR (2e-6) improve convergence speed without hurting calibration?
- Does calib decay help or hurt long-run calibration?
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
```
