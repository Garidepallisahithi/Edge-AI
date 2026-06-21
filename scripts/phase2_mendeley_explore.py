#!/usr/bin/env python3
import argparse, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_pickle(pkl_path: Path):
    with pkl_path.open("rb") as f:
        return pickle.load(f)


def collect_samples(base_dir: Path, max_files: int = 500):
    """Load up to max_files samples for exploration."""
    pkl_files = sorted(base_dir.rglob("*.pkl"))
    samples = []
    for p in pkl_files[:max_files]:
        try:
            data, metadata = load_pickle(p)
            samples.append((data, metadata))
        except Exception as e:
            print(f"[WARN] Failed to load {p}: {e}")
    return samples


def summarize_labels(samples):
    labels = []
    for _, metadata in samples:
        if isinstance(metadata, dict) and "label" in metadata:
            labels.append(str(metadata["label"]))
    return pd.Series(labels).value_counts()


def plot_feature_distribution(samples, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Collect all data arrays
    arrays = [s[0] for s in samples if hasattr(s[0], "shape")]
    if not arrays:
        print("[WARN] No arrays found for plotting")
        return
    data = np.vstack(arrays)
    df = pd.DataFrame(
        data,
        columns=[
            "volt",
            "current",
            "soc",
            "max_single_volt",
            "min_single_volt",
            "max_temp",
            "min_temp",
            "timestamp",
        ],
    )
    # Plot distributions
    for col in df.columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], bins=50, kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(out_dir / f"{col}_distribution.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Explore Mendeley dataset")
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="Path to extracted Train/Test folders"
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Where to save figures")
    args = parser.parse_args()

    # Load Train samples
    train_dir = args.data_dir / "Train" / "Train"
    test_dir = args.data_dir / "Test"
    print("[INFO] Loading Train samples...")
    train_samples = collect_samples(train_dir)
    print(f"[INFO] Loaded {len(train_samples)} Train samples")

    print("[INFO] Loading Test samples...")
    test_samples = collect_samples(test_dir)
    print(f"[INFO] Loaded {len(test_samples)} Test samples")

    # Summarize labels
    label_counts = summarize_labels(train_samples)
    print("[SUMMARY] Train label distribution:")
    print(label_counts)

    # Plot feature distributions
    plot_feature_distribution(train_samples, args.out_dir)
    print(f"[SUCCESS] Figures saved to {args.out_dir}")


if __name__ == "__main__":
    main()
