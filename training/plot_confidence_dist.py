import json
import collections
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse
import re
from pathlib import Path
from PIL import Image
import imageio



def plot_confidence_dist(rows, jsonl_name, out_path):
    # Blue: declared confidence (per rollout, normalized by total rollouts)
    declared_counts = collections.Counter()
    n_rollouts = len(rows)
    for row in rows:
        conf = row.get("declared_confidence")
        if conf is not None:
            declared_counts[round(conf * 100)] += 1

    # Orange: empirical accuracy per prompt (one count per group, normalized by n_prompts).
    # Each group's p_hat = mean(is_correct), binned to integer percentage.
    groups = defaultdict(list)
    for row in rows:
        groups[row["input"]].append(row)
    empirical_counts = collections.Counter()
    n_prompts = len(groups)
    for rollouts in groups.values():
        p_hat = sum(r.get("is_correct", 0) for r in rollouts) / len(rollouts)
        empirical_counts[round(p_hat * 100)] += 1

    xs = list(range(0, 101))
    declared_ys = [declared_counts.get(x, 0) / n_rollouts for x in xs]
    empirical_ys = [empirical_counts.get(x, 0) / n_prompts for x in xs]

    width = 0.4
    _, ax = plt.subplots(figsize=(14, 5))
    ax.bar([x - width / 2 for x in xs], declared_ys, width=width, color="steelblue",
           edgecolor="none", label=f"declared R (per rollout, n={n_rollouts})")
    ax.bar([x + width / 2 for x in xs], empirical_ys, width=width, color="darkorange",
           edgecolor="none", label=f"empirical p_hat (per prompt, n={n_prompts})")
    ax.set_xlabel("Confidence (%)")
    ax.set_ylabel("Normalized count")
    ax.set_xlim(-1, 101)
    ax.set_xticks(range(0, 101, 5))
    ax.set_title(f"Declared confidence vs empirical accuracy — {jsonl_name}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved to {out_path}")


def plot_brier_dist(rows, jsonl_name, out_path):
    groups = defaultdict(list)
    for row in rows:
        groups[row["input"]].append(row)

    brier_scores = []
    for rollouts in groups.values():
        confs = [r.get("declared_confidence") or 0.0 for r in rollouts]
        p_hat = sum(r["is_correct"] for r in rollouts) / len(rollouts)
        scores = [(c - p_hat) ** 2 for c in confs]
        brier_scores.append(sum(scores) / len(scores))

    n_prompts = len(brier_scores)
    mean_brier = sum(brier_scores) / n_prompts

    bucket_counts = collections.Counter()
    for bs in brier_scores:
        bucket = round(bs * 100) / 100
        bucket_counts[bucket] += 1

    xs = [i / 100 for i in range(101)]
    ys = [bucket_counts.get(x, 0) / n_prompts for x in xs]

    _, ax = plt.subplots(figsize=(14, 5))
    ax.bar(xs, ys, width=0.008, color="steelblue", edgecolor="none", align="center")
    ax.axvline(mean_brier, color="tomato", linewidth=1.5, linestyle="--",
               label=f"mean = {mean_brier:.4f}")
    ax.set_xlabel("Per-prompt Brier score")
    ax.set_ylabel(f"Normalized count (/ {n_prompts} prompts)")
    ax.set_xlim(0, 1)
    ax.set_xticks([i / 10 for i in range(11)])
    ax.set_title(f"Per-prompt Brier score distribution — {jsonl_name}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved to {out_path}  (mean Brier = {mean_brier:.4f}, n={n_prompts} prompts)")


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def create_animation_from_directory(directory, output_dir, plot_type="both"):
    jsonl_files = sorted(
        Path(directory).glob("*.jsonl"),
        key=lambda p: int(re.match(r"(\d+)", p.name).group(1)) if re.match(r"(\d+)", p.name) else 0
    )

    if not jsonl_files:
        print(f"No .jsonl files found in {directory}")
        return

    os.makedirs(output_dir, exist_ok=True)
    frame_dir = os.path.join(output_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)

    confidence_frames = []
    brier_frames = []

    for i, jsonl_path in enumerate(jsonl_files):
        print(f"Processing {jsonl_path.name} ({i+1}/{len(jsonl_files)})")
        rows = load_jsonl(str(jsonl_path))
        stem = jsonl_path.stem

        if plot_type in ("confidence", "both"):
            conf_frame_path = os.path.join(frame_dir, f"confidence_{stem}.png")
            plot_confidence_dist(rows, jsonl_path.name, conf_frame_path)
            confidence_frames.append(conf_frame_path)

        if plot_type in ("brier", "both"):
            brier_frame_path = os.path.join(frame_dir, f"brier_{stem}.png")
            plot_brier_dist(rows, jsonl_path.name, brier_frame_path)
            brier_frames.append(brier_frame_path)

    if plot_type in ("confidence", "both"):
        confidence_gif = os.path.join(output_dir, "confidence_animation.gif")
        _create_gif(confidence_frames, confidence_gif)

    if plot_type in ("brier", "both"):
        brier_gif = os.path.join(output_dir, "brier_animation.gif")
        _create_gif(brier_frames, brier_gif)


def _create_gif(frame_paths, output_path):
    images = [Image.open(path) for path in frame_paths]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=500,
        loop=0
    )
    print(f"Saved animation to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="path to val generations jsonl file or directory containing .jsonl files")
    parser.add_argument("--plot", choices=["confidence", "brier", "both"], default="both")
    parser.add_argument("--output", default="./visualizations/exp_default")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        create_animation_from_directory(args.input, args.output, args.plot)
    else:
        os.makedirs(args.output, exist_ok=True)
        jsonl_name = os.path.basename(args.input)
        stem = jsonl_name.replace(".jsonl", "")

        rows = load_jsonl(args.input)

        if args.plot in ("confidence", "both"):
            plot_confidence_dist(rows, jsonl_name,
                                 os.path.join(args.output, f"confidence_dist_{stem}.png"))
        if args.plot in ("brier", "both"):
            plot_brier_dist(rows, jsonl_name,
                            os.path.join(args.output, f"brier_dist_{stem}.png"))


if __name__ == "__main__":
    main()
