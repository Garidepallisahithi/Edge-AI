#!/usr/bin/env python3
import argparse, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

def load_pickle(pkl_path: Path):
    with pkl_path.open("rb") as f:
        return pickle.load(f)

def collect_samples(base_dir: Path, max_files: int = 500):
    pkl_files = sorted(base_dir.rglob("*.pkl"))
    samples = []
    for p in pkl_files[:max_files]:
        try:
            data, metadata = load_pickle(p)
            samples.append((data, metadata))
        except Exception as e:
            print(f"[WARN] Failed to load {p}: {e}")
    return samples

def extract_features(samples):
    rows = []
    for data, metadata in samples:
        if not hasattr(data, "shape"):
            continue
        # basic statistical features
        feat = {
            "volt_mean": np.mean(data[:,0]),
            "volt_std": np.std(data[:,0]),
            "current_mean": np.mean(data[:,1]),
            "current_std": np.std(data[:,1]),
            "temp_max": np.max(data[:,5]),
            "temp_min": np.min(data[:,6]),
            "soc_mean": np.mean(data[:,2]),
            "label": metadata.get("label", None) if isinstance(metadata, dict) else None
        }
        rows.append(feat)
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(description="Phase 3 - Feature engineering + baseline model")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Path to extracted Train/Test folders")
    parser.add_argument("--out-csv", type=Path, required=True,
                        help="Where to save features CSV")
    parser.add_argument("--metrics-out", type=Path, required=True,
                        help="Where to save metrics report")
    args = parser.parse_args()

    train_dir = args.data_dir / "Train" / "Train"
    test_dir  = args.data_dir / "Test"

    print("[INFO] Loading Train samples...")
    train_samples = collect_samples(train_dir)
    print(f"[INFO] Loaded {len(train_samples)} Train samples")

    print("[INFO] Loading Test samples...")
    test_samples = collect_samples(test_dir)
    print(f"[INFO] Loaded {len(test_samples)} Test samples")

    # Extract features
    train_df = extract_features(train_samples)
    test_df  = extract_features(test_samples)

    # Save features
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(args.out_csv, index=False)
    print(f"[SUCCESS] Features saved to {args.out_csv}")

    # Train IsolationForest
    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"].astype(str)

    model = IsolationForest(random_state=42)
    model.fit(X_train)

    # Predict on test
    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"].astype(str)

    preds = model.predict(X_test)
    # Map IsolationForest outputs (-1 anomaly, 1 normal) to dataset labels
    preds = np.where(preds == -1, "10", "00")
    
    # --- FIX: remove rows with missing labels ---
    mask = y_test.notna() & (y_test != "None")
    y_test = y_test[mask]
    preds = preds[mask]

    if len(y_test) == 0:
        print("[WARN] No labeled test samples available for metrics")
        report = "No labeled test samples available"
        cm = []
    else:
        # Metrics
        report = classification_report(y_test, preds, digits=4)
        cm = confusion_matrix(y_test, preds)

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_out.open("w") as f:
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n")

    print(f"[SUCCESS] Metrics saved to {args.metrics_out}")

if __name__ == "__main__":
    main()
