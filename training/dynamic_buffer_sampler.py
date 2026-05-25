"""
DynamicBufferSampler: curriculum sampler that stratifies training prompts by
empirical difficulty (p_hat) so that each batch contains roughly equal
representation across confidence bins.

p_hat for each prompt is estimated from rollout results and updated via EMA
after every training step. On cold start (before any prompt has been seen),
the sampler falls back to uniform random sampling. A warm-start file produced
by training/precompute_phat.py can be provided to skip the cold-start phase.

Usage (verl config):
    data.sampler.class_path=training/dynamic_buffer_sampler.py
    data.sampler.class_name=DynamicBufferSampler
    data.sampler.warmstart_path=<path_to_phat_json>   # optional
    data.sampler.ema_alpha=0.3
    data.sampler.n_bins=10
    data.dataloader_num_workers=0   # required for curriculum samplers
"""

import json
import logging
import random
from collections import defaultdict
from collections.abc import Iterator, Sized

import numpy as np
from omegaconf import DictConfig

from verl.experimental.dataset.sampler import AbstractCurriculumSampler

logger = logging.getLogger(__name__)

N_UNKNOWN = -1  # sentinel bin for unseen prompts


class DynamicBufferSampler(AbstractCurriculumSampler):
    """
    Maintains a running EMA of p_hat per training prompt and samples each
    batch with equal slots across p_hat bins, breaking the bimodal class
    imbalance that accumulates when the model's capability distribution is
    skewed toward 0 / 1.
    """

    def __init__(self, data_source: Sized, data_config: DictConfig):
        super().__init__(data_source, data_config)
        self.n = len(data_source)
        self.ema_alpha: float = data_config.sampler.get("ema_alpha", 0.3)
        self.n_bins: int = data_config.sampler.get("n_bins", 10)

        # p_hat table: index -> float in [0,1] or None (unseen)
        self.p_hat: list[float | None] = [None] * self.n

        warmstart = data_config.sampler.get("warmstart_path", None)
        if warmstart:
            self._load_warmstart(warmstart)

        # rng for reproducibility
        seed = data_config.get("seed", 42)
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------
    # warm-start
    # ------------------------------------------------------------------

    def _load_warmstart(self, path: str) -> None:
        """Load pre-computed p_hat estimates keyed by dataset index (int)."""
        try:
            with open(path) as f:
                raw = json.load(f)
            loaded = 0
            for k, v in raw.items():
                idx = int(k)
                if 0 <= idx < self.n:
                    self.p_hat[idx] = float(v)
                    loaded += 1
            logger.info("DynamicBufferSampler: loaded %d p_hat estimates from %s", loaded, path)
        except Exception as e:
            logger.warning("DynamicBufferSampler: failed to load warmstart from %s: %s", path, e)

    # ------------------------------------------------------------------
    # binning helpers
    # ------------------------------------------------------------------

    def _bin(self, p: float | None) -> int:
        """Map p_hat to bin index 0..(n_bins-1), or N_UNKNOWN."""
        if p is None:
            return N_UNKNOWN
        return min(int(p * self.n_bins), self.n_bins - 1)

    def _build_bin_lists(self) -> dict[int, list[int]]:
        bins: dict[int, list[int]] = defaultdict(list)
        for idx, p in enumerate(self.p_hat):
            bins[self._bin(p)].append(idx)
        return bins

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[int]:
        bins = self._build_bin_lists()

        unknown = bins.pop(N_UNKNOWN, [])
        known_bins = [v for v in bins.values() if v]

        n_known_bins = len(known_bins)

        if n_known_bins == 0:
            # cold start: all unseen — pure shuffle
            idxs = list(range(self.n))
            self.rng.shuffle(idxs)
            yield from idxs
            return

        # Allocate slots: split dataset into equal-size chunks per bin plus
        # a proportional unknown chunk so cold-start prompts stay explored.
        # Each known bin gets one "slot unit"; unknown gets one slot unit too
        # (if non-empty), so unknown prompts don't starve.
        n_slot_groups = n_known_bins + (1 if unknown else 0)
        slots_per_group = max(1, self.n // n_slot_groups)

        order: list[int] = []

        for bucket in known_bins:
            shuffled = bucket.copy()
            self.rng.shuffle(shuffled)
            # tile to fill the slot quota
            tiled: list[int] = []
            while len(tiled) < slots_per_group:
                tiled.extend(shuffled)
            order.extend(tiled[:slots_per_group])

        if unknown:
            self.rng.shuffle(unknown)
            tiled_u: list[int] = []
            while len(tiled_u) < slots_per_group:
                tiled_u.extend(unknown)
            order.extend(tiled_u[:slots_per_group])

        self.rng.shuffle(order)
        yield from order

    def __len__(self) -> int:
        return self.n

    # ------------------------------------------------------------------
    # update (called by ray_trainer after each step)
    # ------------------------------------------------------------------

    def update(self, batch) -> None:
        """
        Extract per-prompt p_hat from this step's rollout results and apply
        EMA update to the p_hat table.

        Expects batch.non_tensor_batch to contain:
          - "data_index": integer dataset index, one entry per rollout
          - "is_correct": binary correctness, one entry per rollout
        """
        ntb = batch.non_tensor_batch
        if "data_index" not in ntb or "is_correct" not in ntb:
            return

        data_indices = ntb["data_index"]
        is_correct = np.array(ntb["is_correct"], dtype=float)

        # group rollouts by dataset index
        groups: dict[int, list[float]] = defaultdict(list)
        for idx, correct in zip(data_indices, is_correct):
            groups[int(idx)].append(float(correct))

        alpha = self.ema_alpha
        for idx, corrects in groups.items():
            if not (0 <= idx < self.n):
                continue
            new_p = float(np.mean(corrects))
            old_p = self.p_hat[idx]
            if old_p is None:
                self.p_hat[idx] = new_p
            else:
                self.p_hat[idx] = alpha * new_p + (1 - alpha) * old_p

    # ------------------------------------------------------------------
    # serialisation for StatefulDataLoader checkpoint/resume
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {"p_hat": self.p_hat}

    def load_state_dict(self, state: dict) -> None:
        self.p_hat = state["p_hat"]
