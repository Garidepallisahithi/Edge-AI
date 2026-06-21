#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
import os

os.environ.setdefault("JOBLIB_TEMP_FOLDER", "/tmp")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

LABEL_COL = "label"
SPLIT_COL = "internal_split"
ID_COLS = {"sample_id", "sample_path", "split", "label_raw", "label", "mile", "internal_split"}


# Keep feature order broad and safe; script reads all non-metadata columns.
def get_feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in ID_COLS]


@dataclass
class Paths:
    processed_dir: Path
    phase4_model_dir: Path
    phase5_model_dir: Path
    reports_dir: Path

    @property
    def split_csv(self) -> Path:
        return self.processed_dir / "mendeley_internal_split.csv"

    @property
    def phase5_plots(self) -> Path:
        return self.reports_dir / "plots"

    @property
    def phase5_explanations(self) -> Path:
        return self.reports_dir / "explanations"

    @property
    def metrics_json(self) -> Path:
        return self.reports_dir / "metrics.json"

    @property
    def uncertainty_json(self) -> Path:
        return self.reports_dir / "uncertainty.json"

    @property
    def calibration_json(self) -> Path:
        return self.reports_dir / "calibration.json"

    @property
    def model_selection_json(self) -> Path:
        return self.reports_dir / "model_selection.json"

    @property
    def official_predictions_csv(self) -> Path:
        return self.reports_dir / "official_test_predictions.csv"

    @property
    def internal_predictions_csv(self) -> Path:
        return self.reports_dir / "internal_test_predictions_calibrated.csv"

    @property
    def summary_md(self) -> Path:
        return self.reports_dir / "summary_phase5.md"


def ensure_dirs(paths: Paths) -> None:
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    paths.phase5_plots.mkdir(parents=True, exist_ok=True)
    paths.phase5_explanations.mkdir(parents=True, exist_ok=True)
    paths.phase5_model_dir.mkdir(parents=True, exist_ok=True)


def load_split_df(paths: Paths) -> pd.DataFrame:
    if not paths.split_csv.exists():
        raise FileNotFoundError(f"Missing split CSV: {paths.split_csv}")
    df = pd.read_csv(paths.split_csv)
    if SPLIT_COL not in df.columns:
        raise ValueError(f"{paths.split_csv} must contain '{SPLIT_COL}'")
    if LABEL_COL not in df.columns:
        raise ValueError(f"{paths.split_csv} must contain '{LABEL_COL}'")
    return df


def get_split_frames(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df[SPLIT_COL] == "train"].copy()
    val_df = df[df[SPLIT_COL] == "val"].copy()
    test_df = df[df[SPLIT_COL] == "test"].copy()
    official_df = df[df["split"] == "test"].copy()
    return train_df, val_df, test_df, official_df


def load_base_model(paths: Paths):
    rf_path = paths.phase4_model_dir / "random_forest.joblib"
    best_path = paths.phase4_model_dir / "best_model.joblib"

    if rf_path.exists():
        return joblib.load(rf_path), rf_path

    if best_path.exists():
        warnings.warn(
            "random_forest.joblib not found. Falling back to best_model.joblib. "
            "For cleaner calibration, keep random_forest.joblib from phase 4."
        )
        return joblib.load(best_path), best_path

    raise FileNotFoundError(
        f"Neither {rf_path} nor {best_path} exists. Phase 4 training artifacts are missing."
    )


def get_classifier(model):
    if hasattr(model, "named_steps"):
        if "clf" in model.named_steps:
            return model.named_steps["clf"]
        last_key = list(model.named_steps.keys())[-1]
        return model.named_steps[last_key]
    return model


def predict_proba_safe(model, X: pd.DataFrame) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise AttributeError("Model does not support predict_proba().")
    probs = model.predict_proba(X)
    probs = np.asarray(probs)
    if probs.ndim != 2 or probs.shape[1] < 2:
        raise ValueError(f"Unexpected predict_proba shape: {probs.shape}")
    return probs[:, 1]


def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, Dict[str, float]]:
    thresholds = np.unique(np.concatenate([np.linspace(0.05, 0.95, 91), y_prob]))
    best_thr = 0.5
    best_f1 = -1.0
    best_stats: Dict[str, float] = {}

    for thr in thresholds:
        pred = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
            best_stats = {
                "threshold": float(thr),
                "accuracy": float(accuracy_score(y_true, pred)),
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
                "f1": float(f1),
            }
    return best_thr, best_stats


def evaluate_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None,
        "pr_auc": (
            float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["normal", "abnormal"],
            output_dict=True,
            zero_division=0,
        ),
    }


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        mask = bin_ids == i
        if not np.any(mask):
            continue
        bin_acc = np.mean(y_true[mask])
        bin_conf = np.mean(y_prob[mask])
        ece += (np.sum(mask) / n) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_uncertainty_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float, confidence_cutoff: float = 0.8
) -> Dict:
    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
    y_pred = (y_prob >= threshold).astype(int)
    confidence = np.maximum(y_prob, 1 - y_prob)
    selective_mask = confidence >= confidence_cutoff

    out = {
        "threshold": float(threshold),
        "confidence_cutoff": float(confidence_cutoff),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "ece": float(ece_score(y_true, y_prob, n_bins=10)),
        "avg_confidence": float(np.mean(confidence)),
        "coverage_at_confidence_cutoff": float(np.mean(selective_mask)),
        "selective_accuracy_at_confidence_cutoff": (
            float(accuracy_score(y_true[selective_mask], y_pred[selective_mask]))
            if np.any(selective_mask)
            else None
        ),
        "selective_f1_at_confidence_cutoff": (
            float(f1_score(y_true[selective_mask], y_pred[selective_mask], zero_division=0))
            if np.any(selective_mask)
            else None
        ),
    }
    return out


def plot_confidence_hist(confidence: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(confidence, bins=25)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_reliability_curve(
    y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, title: str
) -> None:
    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(mean_pred, frac_pos, marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_probability_hist(
    y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, title: str
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(y_prob[y_true == 0], bins=25, alpha=0.7, label="normal")
    ax.hist(y_prob[y_true == 1], bins=25, alpha=0.7, label="abnormal")
    ax.set_xlabel("Anomaly probability")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_feature_importance(importance_df: pd.DataFrame, out_path: Path, title: str) -> None:
    top = importance_df.head(20).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance"])
    ax.set_title(title)
    ax.set_xlabel("Permutation importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def compute_global_feature_importance(
    model, X_val: pd.DataFrame, y_val: np.ndarray, seed: int
) -> pd.DataFrame:
    """
    Compute global feature importance using permutation importance on a sampled subset.
    Falls back to model.feature_importances_ when permutation_importance fails or is too heavy.
    """
    import warnings
    from sklearn.inspection import permutation_importance

    # Combine X and y to sample rows consistently
    df_val = X_val.copy()
    df_val["__label__"] = y_val

    # Reduce memory footprint: sample at most 500 rows
    sample_size = min(500, len(df_val))
    df_sample = df_val.sample(n=sample_size, random_state=seed)

    X_sample = df_sample.drop(columns="__label__")
    y_sample = df_sample["__label__"].values

    try:
        # Permutation importance with limited repeats and single-threaded to avoid OOM
        result = permutation_importance(
            model,
            X_sample,
            y_sample,
            scoring="f1",
            n_repeats=3,
            random_state=seed,
            n_jobs=1,
        )

        df = (
            pd.DataFrame(
                {
                    "feature": X_sample.columns,
                    "importance": result.importances_mean,
                    "std": result.importances_std,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    except Exception as exc:
        # Fallback: use model.feature_importances_ if available (e.g., RandomForest)
        warnings.warn(
            f"Permutation importance failed ({exc}); falling back to model.feature_importances_."
        )
        if hasattr(model, "feature_importances_"):
            import numpy as _np

            fi = _np.asarray(model.feature_importances_)
            df = (
                pd.DataFrame(
                    {
                        "feature": X_sample.columns,
                        "importance": fi,
                        "std": _np.zeros_like(fi),
                    }
                )
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )
        else:
            # Last resort: return zeros so downstream code can continue
            df = (
                pd.DataFrame(
                    {
                        "feature": X_sample.columns,
                        "importance": [0.0] * len(X_sample.columns),
                        "std": [0.0] * len(X_sample.columns),
                    }
                )
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )

    return df


def ensure_model_outputs(df: pd.DataFrame, model, calibrator, threshold: float) -> pd.DataFrame:
    """
    Ensure df has anomaly_probability, confidence, and prediction columns.
    - model: base estimator or pipeline (used if calibrator is None)
    - calibrator: calibrated classifier (CalibratedClassifierCV) or None
    - threshold: decision threshold for positive class
    """
    clf = calibrator if calibrator is not None else model

    # Try to infer the exact feature names the model expects
    model_features = None

    # 1) calibrated wrapper may not expose feature_names_in_, try underlying estimator
    try:
        # If clf is CalibratedClassifierCV, its .estimator or calibrated_classifiers_ may exist.
        base_candidate = (
            getattr(clf, "estimator", None) or getattr(clf, "base_estimator", None) or clf
        )
    except Exception:
        base_candidate = clf

    # 2) If pipeline, get the final estimator
    try:
        base_candidate = get_classifier(base_candidate)
    except Exception:
        pass

    # 3) read feature_names_in_ if available
    if hasattr(base_candidate, "feature_names_in_"):
        model_features = list(getattr(base_candidate, "feature_names_in_"))

    # 4) If still None, try clf directly
    if model_features is None and hasattr(clf, "feature_names_in_"):
        model_features = list(getattr(clf, "feature_names_in_"))

    # 5) Final fallback: use non-ID columns from df
    if not model_features:
        model_features = [c for c in df.columns if c not in ID_COLS.union({LABEL_COL, SPLIT_COL})]

    # Keep only columns that exist in df (defensive)
    feature_cols = [c for c in model_features if c in df.columns]
    if len(feature_cols) == 0:
        # As a last resort, use all non-ID columns
        feature_cols = [c for c in df.columns if c not in ID_COLS.union({LABEL_COL, SPLIT_COL})]

    # compute anomaly_probability if missing
    if "anomaly_probability" not in df.columns:
        probs = predict_proba_safe(clf, df[feature_cols])
        df["anomaly_probability"] = probs

    # compute confidence (max class probability) if missing
    if "confidence" not in df.columns:
        if hasattr(clf, "predict_proba"):
            probs_all = clf.predict_proba(df[feature_cols])
            df["confidence"] = np.max(probs_all, axis=1)
        else:
            df["confidence"] = df["anomaly_probability"].abs()

    # compute binary prediction using threshold if missing
    if "prediction" not in df.columns:
        df["prediction"] = (df["anomaly_probability"] >= threshold).astype(int)

    return df


def local_surrogate_explanations(
    sample_df: pd.DataFrame,
    train_df: pd.DataFrame,
    global_importance_df: pd.DataFrame,
    out_dir: Path,
    base_model,
    calibrator,
    calibration_result,
) -> Tuple[pd.DataFrame, Path]:
    """
    Compute local surrogate explanations for samples using global importance as weights.
    Returns a DataFrame of top features per sample and the CSV path.
    """
    # ensure model outputs exist
    threshold = (
        calibration_result.get("best_threshold", 0.5) if calibration_result is not None else 0.5
    )
    sample_df = ensure_model_outputs(sample_df, base_model, calibrator, threshold)
    train_df = ensure_model_outputs(train_df, base_model, calibrator, threshold)

    # compute feature columns excluding ID and model-output columns (robust)
    exclude_cols = set(ID_COLS) | {
        LABEL_COL,
        SPLIT_COL,
        "anomaly_probability",
        "confidence",
        "prediction",
    }
    # prefer sample_df columns, but fall back to train_df if needed
    feature_cols = [c for c in sample_df.columns if c not in exclude_cols]
    if not feature_cols:
        feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    # keep only numeric columns (needed for median/std and zscore)
    feature_cols = [
        c for c in feature_cols if pd.api.types.is_numeric_dtype(sample_df.get(c, train_df[c]))
    ]
    if not feature_cols:
        raise ValueError(
            "No numeric feature columns found for local surrogate explanations. "
            "Check ID_COLS, LABEL_COL, SPLIT_COL and input DataFrame columns."
        )

    # medians/stds for normalization
    train_medians = train_df[feature_cols].median(numeric_only=True)
    train_stds = train_df[feature_cols].std(numeric_only=True).replace(0, np.nan).fillna(1.0)

    importance_map = global_importance_df.set_index("feature")["importance"].abs().to_dict()

    rows = []
    for _, sample in sample_df.iterrows():
        sid = sample.get("sample_id", None)
        prob = float(sample.get("anomaly_probability", 0.0))
        pred = int(sample.get("prediction", 0))
        label = sample.get(LABEL_COL, np.nan)

        # ensure x and z align with feature_cols and fill missing values
        x = sample.reindex(feature_cols).astype(float).fillna(0.0)

        # compute z-scores and clip to ±10 for numerical stability
        z = (
            ((x - train_medians.reindex(feature_cols)) / train_stds.reindex(feature_cols))
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        z = np.clip(z, -10, 10)

        scores = []
        for feat in feature_cols:
            if feat not in x.index or feat not in z.index:
                continue
            imp = float(importance_map.get(feat, 0.0))
            zval = float(z[feat])
            weighted = abs(zval) * imp
            scores.append(
                {
                    "sample_id": sid,
                    "label": label,
                    "prediction": pred,
                    "anomaly_probability": prob,
                    "feature": feat,
                    "value": float(x[feat]),
                    "train_median": float(train_medians.get(feat, 0.0)),
                    "train_std": float(train_stds.get(feat, 1.0)),
                    "zscore": zval,
                    "global_importance": imp,
                    "weighted_score": weighted,
                }
            )

        # build DataFrame only if we have scores
        df_scores = pd.DataFrame(scores)
        if df_scores.empty:
            # nothing to score for this sample — skip
            continue
        if "weighted_score" not in df_scores.columns:
            df_scores["weighted_score"] = 0.0

        top = df_scores.sort_values("weighted_score", ascending=False).head(10)
        rows.append(top)

    # after the loop, construct local_df defensively
    local_df = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(
            columns=[
                "sample_id",
                "label",
                "prediction",
                "anomaly_probability",
                "feature",
                "value",
                "train_median",
                "train_std",
                "zscore",
                "global_importance",
                "weighted_score",
            ]
        )
    )
    out_csv = out_dir / "local_explanations.csv"
    local_df.to_csv(out_csv, index=False)
    return local_df, out_csv


def plot_local_explanation(local_df: pd.DataFrame, sample_id: str, out_path: Path) -> None:
    sub = local_df[local_df["sample_id"] == sample_id].sort_values("weighted_score", ascending=True)
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(sub["feature"], sub["weighted_score"])
    ax.set_title(f"Local surrogate explanation: {sample_id}")
    ax.set_xlabel("Weighted contribution score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_json(path: Path, payload: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def cmd_calibrate(paths: Paths, calibration_method: str, seed: int) -> Dict:
    df = load_split_df(paths)
    train_df, val_df, test_df, _ = get_split_frames(df)
    feature_cols = get_feature_columns(df)

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("Train/val/test internal split is empty. Recreate the split first.")

    base_model, base_model_path = load_base_model(paths)
    X_train = train_df[feature_cols]
    y_train = train_df[LABEL_COL].astype(int).values
    X_val = val_df[feature_cols]
    y_val = val_df[LABEL_COL].astype(int).values
    X_test = test_df[feature_cols]
    y_test = test_df[LABEL_COL].astype(int).values

    if not hasattr(base_model, "predict_proba"):
        raise AttributeError(
            f"Base model loaded from {base_model_path} does not support predict_proba()."
        )

    # Calibrate using validation set only.

    calibrator = CalibratedClassifierCV(
        estimator=base_model,
        method=calibration_method,
        cv=3,  # use 3-fold CV
    )
    calibrator.fit(X_train, y_train)

    # Save calibrated artifact
    paths.phase5_model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, paths.phase5_model_dir / "rf_calibrated.joblib")

    val_prob = predict_proba_safe(calibrator, X_val)
    test_prob = predict_proba_safe(calibrator, X_test)

    best_thr, thr_stats = tune_threshold(y_val, val_prob)
    val_metrics = evaluate_at_threshold(y_val, val_prob, best_thr)
    test_metrics = evaluate_at_threshold(y_test, test_prob, best_thr)

    train_prob = predict_proba_safe(calibrator, X_train)
    calibration_payload = {
        "base_model_path": str(base_model_path),
        "calibration_method": calibration_method,
        "best_threshold": best_thr,
        "validation_threshold_stats": thr_stats,
        "validation": val_metrics,
        "internal_test": test_metrics,
        "train_probability_summary": {
            "min": float(np.min(train_prob)),
            "max": float(np.max(train_prob)),
            "mean": float(np.mean(train_prob)),
        },
    }

    save_json(paths.calibration_json, calibration_payload)
    save_json(paths.metrics_json, calibration_payload)

    # Prediction CSVs
    val_out = val_df[["sample_id", "split", "label"]].copy()
    val_out["anomaly_probability"] = val_prob
    val_out["confidence"] = np.maximum(val_prob, 1 - val_prob)
    val_out["prediction"] = (val_prob >= best_thr).astype(int)
    val_out.to_csv(paths.reports_dir / "validation_predictions_calibrated.csv", index=False)

    test_out = test_df[["sample_id", "split", "label"]].copy()
    test_out["anomaly_probability"] = test_prob
    test_out["confidence"] = np.maximum(test_prob, 1 - test_prob)
    test_out["prediction"] = (test_prob >= best_thr).astype(int)
    test_out.to_csv(paths.internal_predictions_csv, index=False)

    # Uncertainty metrics and plots
    uncertainty = compute_uncertainty_metrics(y_test, test_prob, best_thr, confidence_cutoff=0.8)
    save_json(paths.uncertainty_json, uncertainty)

    test_conf = np.maximum(test_prob, 1 - test_prob)
    plot_confidence_hist(
        test_conf, paths.phase5_plots / "confidence_hist.png", "Internal test confidence histogram"
    )
    plot_reliability_curve(
        y_test,
        test_prob,
        paths.phase5_plots / "reliability_curve.png",
        "Internal test reliability curve",
    )
    plot_probability_hist(
        y_test,
        test_prob,
        paths.phase5_plots / "internal_test_probability_hist.png",
        "Internal test anomaly probabilities",
    )

    # reliability details
    frac_pos, mean_pred = calibration_curve(
        np.clip(y_test, 0, 1), np.clip(test_prob, 1e-6, 1 - 1e-6), n_bins=10, strategy="uniform"
    )
    reliability_df = pd.DataFrame({"mean_predicted": mean_pred, "fraction_positive": frac_pos})
    reliability_df.to_csv(paths.reports_dir / "reliability_curve.csv", index=False)

    print("[OK] Calibration complete.")
    print(json.dumps(calibration_payload, indent=2))
    print(json.dumps(uncertainty, indent=2))
    return {
        "calibrator": calibrator,
        "base_model": base_model,
        "base_model_path": base_model_path,
        "feature_cols": feature_cols,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "y_test": y_test,
        "test_prob": test_prob,
        "best_thr": best_thr,
    }


def cmd_explain(paths: Paths, calibration_result: Dict, seed: int) -> Dict:
    calibrator = calibration_result["calibrator"]
    base_model = calibration_result["base_model"]
    train_df = calibration_result["train_df"]
    val_df = calibration_result["val_df"]
    test_df = calibration_result["test_df"]
    feature_cols = calibration_result["feature_cols"]
    y_test = calibration_result["y_test"]
    test_prob = calibration_result["test_prob"]
    best_thr = calibration_result["best_thr"]

    # Global importance from calibrated model
    val_features = val_df[feature_cols]
    val_labels = val_df[LABEL_COL].astype(int).values
    global_imp = compute_global_feature_importance(calibrator, val_features, val_labels, seed=seed)
    global_imp.to_csv(paths.phase5_explanations / "global_feature_importance.csv", index=False)
    plot_feature_importance(
        global_imp,
        paths.phase5_plots / "global_feature_importance.png",
        "Global feature importance",
    )

    # Prepare scored test frame (include feature columns for local explanations)
    test_scored = test_df[["sample_id", "split", "label"] + feature_cols].copy()
    test_scored["anomaly_probability"] = test_prob
    test_scored["confidence"] = np.maximum(test_prob, 1 - test_prob)
    test_scored["prediction"] = (test_prob >= best_thr).astype(int)

    # Choose a few examples for local explanation
    top_anom = test_scored.sort_values("anomaly_probability", ascending=False).head(3)
    top_norm = test_scored.sort_values("anomaly_probability", ascending=True).head(3)
    sample_df = pd.concat([top_anom, top_norm], ignore_index=True).drop_duplicates("sample_id")

    # Call local surrogate explanations and pass model, calibrator, and calibration_result
    local_df, local_csv = local_surrogate_explanations(
        sample_df=sample_df,
        train_df=train_df,
        global_importance_df=global_imp,
        out_dir=paths.phase5_explanations,
        base_model=base_model,
        calibrator=calibrator,
        calibration_result=(
            {"best_threshold": best_thr}
            if not isinstance(calibration_result, dict)
            else calibration_result
        ),
    )

    # plot one representative local explanation
    if not local_df.empty:
        top_sample_id = sample_df.sort_values("anomaly_probability", ascending=False).iloc[0][
            "sample_id"
        ]
        plot_local_explanation(
            local_df, top_sample_id, paths.phase5_plots / "local_explanation_top.png"
        )

    # Save an explainability summary
    summary = {
        "global_importance_csv": str(paths.phase5_explanations / "global_feature_importance.csv"),
        "local_explanations_csv": str(local_csv),
        "top_samples": sample_df[["sample_id", "anomaly_probability", "prediction"]].to_dict(
            orient="records"
        ),
    }
    save_json(paths.reports_dir / "explainability_summary.json", summary)

    print("[OK] Explainability complete.")
    print(json.dumps(summary, indent=2))
    return {"global_imp": global_imp, "local_df": local_df, "sample_df": sample_df}


def cmd_official_infer(paths: Paths, calibration_result: Dict) -> None:
    calibrator = calibration_result["calibrator"]
    df = load_split_df(paths)
    feature_cols = calibration_result["feature_cols"]
    official_df = df[df["split"] == "test"].copy()

    if len(official_df) == 0:
        raise ValueError("Official test split is empty.")

    official_prob = predict_proba_safe(calibrator, official_df[feature_cols])
    official_out = official_df[["sample_id", "split", "mile"]].copy()
    official_out["anomaly_probability"] = official_prob
    official_out["confidence"] = np.maximum(official_prob, 1 - official_prob)
    official_out["prediction"] = (official_prob >= calibration_result["best_thr"]).astype(int)
    official_out["rank"] = (
        official_out["anomaly_probability"].rank(ascending=False, method="first").astype(int)
    )
    official_out = official_out.sort_values("anomaly_probability", ascending=False)
    official_out.to_csv(paths.official_predictions_csv, index=False)

    plot_probability_hist(
        np.zeros(len(official_prob), dtype=int),
        official_prob,
        paths.phase5_plots / "official_test_probability_hist.png",
        "Official test anomaly probabilities",
    )

    print(f"[OK] Official test inference saved to {paths.official_predictions_csv}")
    print(official_out.head(10).to_string(index=False))


def cmd_run_all(paths: Paths, calibration_method: str, seed: int) -> None:
    calibration_result = cmd_calibrate(paths, calibration_method=calibration_method, seed=seed)
    explain_result = cmd_explain(paths, calibration_result, seed=seed)
    cmd_official_infer(paths, calibration_result)

    summary = {
        "base_model_path": str(calibration_result["base_model_path"]),
        "best_threshold": calibration_result["best_thr"],
        "internal_test_f1": (
            calibration_result["calibration_result"]["internal_test"]["f1"]
            if "calibration_result" in calibration_result
            else None
        ),
    }
    save_json(paths.model_selection_json, summary)

    with paths.summary_md.open("w", encoding="utf-8") as f:
        f.write("# Phase 5 Summary\n\n")
        f.write(f"- Base model: `{calibration_result['base_model_path']}`\n")
        f.write(f"- Calibration method: `{calibration_method}`\n")
        f.write(f"- Best threshold: `{calibration_result['best_thr']:.4f}`\n")
        f.write(f"- Explainability CSV: `{paths.phase5_explanations / 'local_explanations.csv'}`\n")
        f.write(f"- Official predictions: `{paths.official_predictions_csv}`\n")

    print("[SUCCESS] Phase 5 completed.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Explainability and uncertainty for Mendeley anomaly detector."
    )
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--phase4-model-dir", type=Path, required=True)
    parser.add_argument("--phase5-model-dir", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-method", choices=["isotonic", "sigmoid"], default="isotonic")

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("calibrate")
    sub.add_parser("explain")
    sub.add_parser("infer-official")
    sub.add_parser("run-all")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = Paths(
        processed_dir=args.processed_dir,
        phase4_model_dir=args.phase4_model_dir,
        phase5_model_dir=args.phase5_model_dir,
        reports_dir=args.reports_dir,
    )
    ensure_dirs(paths)

    if not paths.split_csv.exists():
        raise FileNotFoundError(
            f"Missing {paths.split_csv}. Run the phase-4 supervised split first."
        )

    if args.command == "calibrate":
        cmd_calibrate(paths, calibration_method=args.calibration_method, seed=args.seed)
    elif args.command == "explain":
        calibration_result = cmd_calibrate(
            paths, calibration_method=args.calibration_method, seed=args.seed
        )
        cmd_explain(paths, calibration_result, seed=args.seed)
    elif args.command == "infer-official":
        calibration_result = cmd_calibrate(
            paths, calibration_method=args.calibration_method, seed=args.seed
        )
        cmd_official_infer(paths, calibration_result)
    elif args.command == "run-all":
        cmd_run_all(paths, calibration_method=args.calibration_method, seed=args.seed)
    else:
        raise ValueError(f"Unknown command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
