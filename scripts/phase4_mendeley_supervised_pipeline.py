#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 2 or b.size < 2:
        return np.nan
    if np.nanstd(a) == 0.0 or np.nanstd(b) == 0.0:
        return np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(a, b)[0, 1]
    return float(c) if np.isfinite(c) else np.nan


CHANNELS = [
    "volt",
    "current",
    "soc",
    "max_single_volt",
    "min_single_volt",
    "max_temp",
    "min_temp",
    "timestamp",
]

NORMAL_LABELS = {"00", "0", "normal", "healthy"}
ABNORMAL_LABELS = {"10", "1", "abnormal", "anomaly", "fault"}

FEATURE_STATS = [
    "mean",
    "std",
    "min",
    "max",
    "median",
    "q10",
    "q90",
    "range",
    "iqr",
    "slope",
    "delta",
    "energy",
]


@dataclass
class Paths:
    data_dir: Path
    work_dir: Path
    reports_dir: Path
    models_dir: Path

    @property
    def features_csv(self) -> Path:
        return self.work_dir / "mendeley_features_all.csv"

    @property
    def split_csv(self) -> Path:
        return self.work_dir / "mendeley_internal_split.csv"

    @property
    def summary_json(self) -> Path:
        return self.reports_dir / "data_summary.json"

    @property
    def metrics_json(self) -> Path:
        return self.reports_dir / "metrics.json"

    @property
    def model_comparison_csv(self) -> Path:
        return self.reports_dir / "model_comparison.csv"

    @property
    def internal_pred_csv(self) -> Path:
        return self.reports_dir / "internal_test_predictions.csv"

    @property
    def official_pred_csv(self) -> Path:
        return self.reports_dir / "official_test_predictions.csv"

    @property
    def plots_dir(self) -> Path:
        return self.reports_dir / "plots"

    @property
    def best_model_path(self) -> Path:
        return self.models_dir / "best_model.joblib"

    @property
    def logreg_path(self) -> Path:
        return self.models_dir / "logreg.joblib"

    @property
    def rf_path(self) -> Path:
        return self.models_dir / "random_forest.joblib"

    @property
    def model_selection_json(self) -> Path:
        return self.models_dir / "model_selection.json"


def ensure_dirs(paths: Paths) -> None:
    paths.work_dir.mkdir(parents=True, exist_ok=True)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    paths.plots_dir.mkdir(parents=True, exist_ok=True)


def path_split_type(p: Path) -> str:
    parts = [x.lower() for x in p.parts]
    if "train" in parts:
        return "train"
    if "test" in parts:
        return "test"
    return "unknown"


def iter_pkl_files(base_dir: Path, split_name: Optional[str] = None) -> List[Path]:
    files = sorted(base_dir.rglob("*.pkl"))
    if split_name is None:
        return files
    split_name = split_name.lower()
    return [p for p in files if split_name in [x.lower() for x in p.parts]]


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def unwrap_scalar(value):
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        if value.size == 1:
            return unwrap_scalar(value.reshape(-1)[0])
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return unwrap_scalar(value[0])
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return value.decode(errors="ignore")
    return value


def extract_metadata_label(metadata) -> Optional[int]:
    if metadata is None:
        return None

    candidates = []
    if isinstance(metadata, dict):
        for k, v in metadata.items():
            if str(k).lower() in {"label", "labels", "y", "target", "class"}:
                candidates.append(v)
    else:
        for attr in ["label", "labels", "y", "target", "class"]:
            if hasattr(metadata, attr):
                candidates.append(getattr(metadata, attr))

        if hasattr(metadata, "__dict__"):
            for k, v in metadata.__dict__.items():
                if str(k).lower() in {"label", "labels", "y", "target", "class"}:
                    candidates.append(v)

    for raw in candidates:
        raw = unwrap_scalar(raw)
        if raw is None:
            continue
        txt = str(raw).strip().lower()
        if txt in NORMAL_LABELS:
            return 0
        if txt in ABNORMAL_LABELS:
            return 1
        try:
            num = int(float(txt))
            if num == 0:
                return 0
            if num == 1:
                return 1
        except Exception:
            pass
    return None


def extract_metadata_mile(metadata) -> Optional[str]:
    if metadata is None:
        return None

    keys = ["mile", "mileage", "mile_info", "mileinfo"]
    if isinstance(metadata, dict):
        for k, v in metadata.items():
            if str(k).lower() in keys:
                return str(unwrap_scalar(v))
    else:
        for attr in keys:
            if hasattr(metadata, attr):
                return str(unwrap_scalar(getattr(metadata, attr)))
        if hasattr(metadata, "__dict__"):
            for k, v in metadata.__dict__.items():
                if str(k).lower() in keys:
                    return str(unwrap_scalar(v))
    return None


def as_2d_float_array(data: object, sample_path: Path) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"{sample_path}: expected 2D array, got shape {arr.shape}")
    if arr.shape == (8, 256):
        arr = arr.T
    if arr.shape != (256, 8):
        raise ValueError(f"{sample_path}: expected shape (256, 8), got {arr.shape}")
    return arr


def safe_channel_features(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {s: np.nan for s in FEATURE_STATS}

    q10 = float(np.nanpercentile(x, 10))
    q90 = float(np.nanpercentile(x, 90))
    q25 = float(np.nanpercentile(x, 25))
    q75 = float(np.nanpercentile(x, 75))

    if x.size >= 2:
        idx = np.arange(x.size, dtype=float)
        try:
            slope = float(np.polyfit(idx, x, 1)[0])
        except Exception:
            slope = 0.0
        delta = float(x[-1] - x[0])
    else:
        slope = 0.0
        delta = 0.0

    return {
        "mean": float(np.nanmean(x)),
        "std": float(np.nanstd(x)),
        "min": float(np.nanmin(x)),
        "max": float(np.nanmax(x)),
        "median": float(np.nanmedian(x)),
        "q10": q10,
        "q90": q90,
        "range": float(np.nanmax(x) - np.nanmin(x)),
        "iqr": float(q75 - q25),
        "slope": slope,
        "delta": delta,
        "energy": float(np.nanmean(np.square(x))),
    }


def extract_sequence_features(arr: np.ndarray) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    for idx, channel in enumerate(CHANNELS):
        channel_feats = safe_channel_features(arr[:, idx])
        for stat_name, val in channel_feats.items():
            feats[f"{channel}__{stat_name}"] = val

    # A few cross-channel features that are cheap and often useful.
    volt = arr[:, 0]
    current = arr[:, 1]
    temp_max = arr[:, 5]
    temp_min = arr[:, 6]
    feats["temp_spread_mean"] = float(np.nanmean(temp_max - temp_min))
    feats["volt_current_corr"] = safe_corr(volt, current)
    feats["temp_current_corr"] = safe_corr(temp_max, current)
    return feats


def build_rows(base_dir: Path, split_name: str) -> List[Dict]:
    rows: List[Dict] = []
    pkl_files = iter_pkl_files(base_dir, split_name=split_name)
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found for split '{split_name}' under {base_dir}")

    for p in pkl_files:
        sample = load_pickle(p)
        if not isinstance(sample, tuple) or len(sample) != 2:
            raise ValueError(f"{p}: expected tuple(data, metadata), got {type(sample)}")
        data, metadata = sample
        arr = as_2d_float_array(data, p)
        feats = extract_sequence_features(arr)
        row = {
            "sample_id": p.stem,
            "sample_path": str(p),
            "split": split_name,
            "label_raw": None,
            "label": None,
            "mile": extract_metadata_mile(metadata),
        }
        if split_name == "train":
            label = extract_metadata_label(metadata)
            if label is None:
                raise ValueError(f"{p}: could not extract a train label from metadata.")
            row["label_raw"] = str(
                unwrap_scalar(
                    metadata.get("label")
                    if isinstance(metadata, dict) and "label" in metadata
                    else (
                        metadata.get("Label")
                        if isinstance(metadata, dict) and "Label" in metadata
                        else (
                            getattr(metadata, "label", None)
                            if hasattr(metadata, "label")
                            else (
                                getattr(metadata, "Label", None)
                                if hasattr(metadata, "Label")
                                else "NA"
                            )
                        )
                    )
                )
            )
            row["label"] = int(label)
        rows.append({**row, **feats})
    return rows


def build_feature_table(paths: Paths) -> pd.DataFrame:
    train_rows = build_rows(paths.data_dir, "train")
    test_rows = build_rows(paths.data_dir, "test")

    df = pd.DataFrame(train_rows + test_rows)
    feature_cols = [
        c
        for c in df.columns
        if "__" in c or c in {"temp_spread_mean", "volt_current_corr", "temp_current_corr"}
    ]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Replace infinities with NaN so downstream imputer can handle them
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Optional: quick median imputation to keep the CSV free of NaNs.
    # If you prefer to let the training pipeline impute, comment out the next 3 lines.
    from sklearn.impute import SimpleImputer

    imp = SimpleImputer(strategy="median")
    df[feature_cols] = imp.fit_transform(df[feature_cols])

    paths.work_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.features_csv, index=False)

    summary = {
        "train_rows": int((df["split"] == "train").sum()),
        "test_rows": int((df["split"] == "test").sum()),
        "train_label_distribution": df.loc[df["split"] == "train", "label"]
        .value_counts(dropna=False)
        .astype(int)
        .to_dict(),
        "feature_count": len(feature_cols),
        "feature_csv": str(paths.features_csv),
    }
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    with paths.summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return df


def verify_dataset(paths: Paths) -> None:
    train_files = iter_pkl_files(paths.data_dir, "train")
    test_files = iter_pkl_files(paths.data_dir, "test")
    print(f"[INFO] train .pkl files: {len(train_files)}")
    print(f"[INFO] test  .pkl files: {len(test_files)}")

    if not train_files:
        raise FileNotFoundError("No train files found.")
    if not test_files:
        raise FileNotFoundError("No test files found.")

    train_sample = load_pickle(train_files[0])
    test_sample = load_pickle(test_files[0])

    for name, sample in [("train", train_sample), ("test", test_sample)]:
        if not isinstance(sample, tuple) or len(sample) != 2:
            raise ValueError(f"{name} sample is not (data, metadata)")
        data, metadata = sample
        arr = as_2d_float_array(data, train_files[0] if name == "train" else test_files[0])
        print(f"[OK] {name} sample shape: {arr.shape}")
        lbl = extract_metadata_label(metadata)
        mile = extract_metadata_mile(metadata)
        print(f"[INFO] {name} sample label: {lbl}")
        print(f"[INFO] {name} sample mile: {mile}")

    print("[SUCCESS] Dataset structure verified.")


def make_internal_splits(
    df: pd.DataFrame, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labeled = df[(df["split"] == "train") & (df["label"].notna())].copy()
    labeled["label"] = labeled["label"].astype(int)

    train_df, temp_df = train_test_split(
        labeled,
        test_size=0.40,
        stratify=labeled["label"],
        random_state=seed,
        shuffle=True,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=seed,
        shuffle=True,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, val_df, test_df


def feature_columns(df: pd.DataFrame) -> List[str]:
    excluded = {"sample_id", "sample_path", "split", "label_raw", "label", "mile"}
    return [c for c in df.columns if c not in excluded]


def make_pipelines(seed: int) -> Dict[str, Pipeline]:
    logreg = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed)),
        ]
    )

    rf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=600,
                    random_state=seed,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                    min_samples_leaf=1,
                ),
            ),
        ]
    )

    return {
        "logistic_regression": logreg,
        "random_forest": rf,
    }


def probability_scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return np.asarray(proba)[:, 1]
    raise AttributeError("Model does not support predict_proba().")


def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, Dict[str, float]]:
    thresholds = np.unique(np.concatenate([np.linspace(0.05, 0.95, 91), y_prob]))
    best_thr = 0.5
    best_f1 = -1.0
    best_stats = {}

    for thr in thresholds:
        pred = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
            best_stats = {
                "threshold": float(thr),
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
                "f1": float(f1),
                "accuracy": float(accuracy_score(y_true, pred)),
            }
    return best_thr, best_stats


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict:
    y_pred = (y_prob >= threshold).astype(int)
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["normal", "abnormal"],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None,
        "pr_auc": (
            float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None
        ),
        "confusion_matrix": cm,
        "classification_report": report,
    }
    return metrics


def plot_confusion_matrix(cm: List[List[int]], out_path: Path, title: str) -> None:
    arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(arr, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1], labels=["normal", "abnormal"])
    ax.set_yticks([0, 1], labels=["normal", "abnormal"])

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, str(arr[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, title: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, title: str) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_probability_hist(
    y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, title: str
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(y_prob[y_true == 0], bins=25, alpha=0.7, label="normal")
    ax.hist(y_prob[y_true == 1], bins=25, alpha=0.7, label="abnormal")
    ax.set_xlabel("Anomaly probability")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_threshold_sweep(
    y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, title: str
) -> None:
    thresholds = np.linspace(0.05, 0.95, 91)
    f1s = []
    for thr in thresholds:
        pred = (y_prob >= thr).astype(int)
        f1s.append(f1_score(y_true, pred, zero_division=0))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(thresholds, f1s)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_predictions(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: Optional[float],
    out_path: Path,
    model_name: str,
) -> None:
    out = df[["sample_id", "split"]].copy()
    if "label" in df.columns:
        out["label"] = df["label"]
    out["anomaly_probability"] = y_prob
    out["anomaly_score"] = y_prob
    if threshold is not None:
        out["prediction"] = (y_prob >= threshold).astype(int)
    out["model"] = model_name
    out.sort_values("anomaly_probability", ascending=False, inplace=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)


def cmd_verify(paths: Paths) -> None:
    verify_dataset(paths)


def cmd_build(paths: Paths) -> pd.DataFrame:
    verify_dataset(paths)
    df = build_feature_table(paths)
    print(f"[SUCCESS] Features saved to {paths.features_csv}")
    print(df.head(3).to_string())
    return df


def cmd_split(paths: Paths, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not paths.features_csv.exists():
        raise FileNotFoundError(f"Feature CSV not found: {paths.features_csv}. Run build first.")
    df = pd.read_csv(paths.features_csv)
    train_df, val_df, test_df = make_internal_splits(df, seed=seed)

    train_df["internal_split"] = "train"
    val_df["internal_split"] = "val"
    test_df["internal_split"] = "test"

    manifest = pd.concat([train_df, val_df, test_df], ignore_index=True)
    manifest.to_csv(paths.split_csv, index=False)

    print("[SUCCESS] Internal split created.")
    print(manifest["internal_split"].value_counts())
    print("\nLabel counts by split:")
    print(manifest.groupby("internal_split")["label"].value_counts())
    return train_df, val_df, test_df


def fit_and_select_model(paths: Paths, seed: int) -> Dict:
    if not paths.features_csv.exists():
        raise FileNotFoundError(f"Feature CSV not found: {paths.features_csv}. Run build first.")

    df = pd.read_csv(paths.features_csv)
    if not paths.split_csv.exists():
        train_df, val_df, test_df = cmd_split(paths, seed=seed)
    else:
        split_df = pd.read_csv(paths.split_csv)
        train_df = split_df[split_df["internal_split"] == "train"].copy()
        val_df = split_df[split_df["internal_split"] == "val"].copy()
        test_df = split_df[split_df["internal_split"] == "test"].copy()

    feature_cols = feature_columns(df)
    X_train = train_df[feature_cols]
    y_train = train_df["label"].astype(int).values
    X_val = val_df[feature_cols]
    y_val = val_df["label"].astype(int).values
    X_test = test_df[feature_cols]
    y_test = test_df["label"].astype(int).values

    model_pipes = make_pipelines(seed=seed)
    comparison_rows = []
    trained_models = {}

    for name, pipe in model_pipes.items():
        pipe.fit(X_train, y_train)
        trained_models[name] = pipe

        val_prob = probability_scores(pipe, X_val)
        best_thr, thr_stats = tune_threshold(y_val, val_prob)
        val_metrics = evaluate_predictions(y_val, val_prob, best_thr)

        comparison_rows.append(
            {
                "model": name,
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "val_roc_auc": val_metrics["roc_auc"],
                "val_pr_auc": val_metrics["pr_auc"],
                "best_threshold": best_thr,
            }
        )

        joblib.dump(pipe, paths.models_dir / f"{name}.joblib")

    comp_df = pd.DataFrame(comparison_rows).sort_values(["val_f1", "val_pr_auc"], ascending=False)
    comp_df.to_csv(paths.model_comparison_csv, index=False)

    best_name = comp_df.iloc[0]["model"]
    best_threshold = float(comp_df.iloc[0]["best_threshold"])
    best_model = trained_models[best_name]

    # Refit best model on train + val for final inference package.
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)
    X_trainval = trainval_df[feature_cols]
    y_trainval = trainval_df["label"].astype(int).values
    final_model = make_pipelines(seed=seed)[best_name]
    final_model.fit(X_trainval, y_trainval)
    joblib.dump(final_model, paths.best_model_path)

    # Evaluate on held-out internal test using threshold from validation.
    test_prob = probability_scores(final_model, X_test)
    test_metrics = evaluate_predictions(y_test, test_prob, best_threshold)
    val_prob = probability_scores(best_model, X_val)
    val_metrics = evaluate_predictions(y_val, val_prob, best_threshold)

    metrics = {
        "best_model": best_name,
        "best_threshold": best_threshold,
        "validation": val_metrics,
        "internal_test": test_metrics,
        "train_count": int(len(train_df)),
        "val_count": int(len(val_df)),
        "test_count": int(len(test_df)),
    }

    with paths.metrics_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with paths.model_selection_json.open("w", encoding="utf-8") as f:
        json.dump({"best_model": best_name, "best_threshold": best_threshold}, f, indent=2)

    save_predictions(
        val_df,
        val_prob,
        best_threshold,
        paths.reports_dir / "validation_predictions.csv",
        best_name,
    )
    save_predictions(test_df, test_prob, best_threshold, paths.internal_pred_csv, best_name)

    # Plots for the best model on internal test.
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        paths.plots_dir / "confusion_matrix_best_model.png",
        f"{best_name} confusion matrix",
    )
    plot_roc_curve(
        y_test, test_prob, paths.plots_dir / "roc_curve_best_model.png", f"{best_name} ROC curve"
    )
    plot_pr_curve(
        y_test, test_prob, paths.plots_dir / "pr_curve_best_model.png", f"{best_name} PR curve"
    )
    plot_probability_hist(
        y_test,
        test_prob,
        paths.plots_dir / "probability_hist_best_model.png",
        f"{best_name} anomaly probability",
    )
    plot_threshold_sweep(
        y_val,
        val_prob,
        paths.plots_dir / "threshold_sweep_best_model.png",
        f"{best_name} threshold sweep",
    )

    # Summary markdown for quick submission notes.

    summary_md = paths.reports_dir / "summary.md"
    with summary_md.open("w", encoding="utf-8") as f:
        f.write("# Phase 4 Summary\n\n")
        f.write(f"- Best model: **{best_name}**\n")
        f.write(f"- Best threshold: **{best_threshold:.4f}**\n")
        f.write(f"- Validation F1: **{val_metrics['f1']:.4f}**\n")
        f.write(f"- Internal test F1: **{test_metrics['f1']:.4f}**\n")

        # Safely format ROC-AUC which may be None
        roc_auc_val = test_metrics.get("roc_auc")
        roc_auc_str = f"{roc_auc_val:.4f}" if roc_auc_val is not None else "NA"
        f.write(f"- Internal test ROC-AUC: **{roc_auc_str}**\n")

    print("[SUCCESS] Model training and evaluation completed.")
    print(comp_df.to_string(index=False))
    print(json.dumps(metrics, indent=2))
    return metrics


def cmd_infer(paths: Paths) -> None:
    if not paths.best_model_path.exists():
        raise FileNotFoundError(f"Best model not found: {paths.best_model_path}. Run train first.")
    if not paths.features_csv.exists():
        raise FileNotFoundError(f"Feature CSV not found: {paths.features_csv}. Run build first.")

    df = pd.read_csv(paths.features_csv)
    off = df[df["split"] == "test"].copy()
    feature_cols = feature_columns(df)
    X_off = off[feature_cols]

    model = joblib.load(paths.best_model_path)
    y_prob = probability_scores(model, X_off)

    out = off[["sample_id", "split"]].copy()
    out["anomaly_probability"] = y_prob
    out["anomaly_score"] = y_prob
    out["rank"] = out["anomaly_probability"].rank(ascending=False, method="first").astype(int)
    out.sort_values("anomaly_probability", ascending=False, inplace=True)
    out.to_csv(paths.official_pred_csv, index=False)

    fig_path = paths.plots_dir / "official_test_probability_hist.png"
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(y_prob, bins=25)
    ax.set_xlabel("Anomaly probability")
    ax.set_ylabel("Count")
    ax.set_title("Official test anomaly probabilities")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    print(f"[SUCCESS] Official test predictions saved to {paths.official_pred_csv}")
    print(f"[SUCCESS] Official test plot saved to {fig_path}")


def cmd_run_all(paths: Paths, seed: int) -> None:
    cmd_verify(paths)
    cmd_build(paths)
    cmd_split(paths, seed=seed)
    fit_and_select_model(paths, seed=seed)
    cmd_infer(paths)
    print("[SUCCESS] Full phase 4 pipeline completed.")


def parse_args():
    parser = argparse.ArgumentParser(description="Supervised Mendeley anomaly pipeline.")
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="Path to Mendeley extracted dataset root."
    )
    parser.add_argument(
        "--work-dir", type=Path, required=True, help="Folder for processed data artifacts."
    )
    parser.add_argument(
        "--reports-dir", type=Path, required=True, help="Folder for reports and plots."
    )
    parser.add_argument("--models-dir", type=Path, required=True, help="Folder for trained models.")
    parser.add_argument("--seed", type=int, default=42)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify")
    sub.add_parser("build")
    sub.add_parser("split")
    sub.add_parser("train")
    sub.add_parser("infer")
    sub.add_parser("run-all")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = Paths(
        data_dir=args.data_dir,
        work_dir=args.work_dir,
        reports_dir=args.reports_dir,
        models_dir=args.models_dir,
    )
    ensure_dirs(paths)

    if args.command == "verify":
        cmd_verify(paths)
    elif args.command == "build":
        cmd_build(paths)
    elif args.command == "split":
        cmd_split(paths, seed=args.seed)
    elif args.command == "train":
        fit_and_select_model(paths, seed=args.seed)
    elif args.command == "infer":
        cmd_infer(paths)
    elif args.command == "run-all":
        cmd_run_all(paths, seed=args.seed)
    else:
        raise ValueError(f"Unknown command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
