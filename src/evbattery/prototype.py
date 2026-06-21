from __future__ import annotations
from pathlib import Path
import os 
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------
# Feature extraction
# ---------------------------

def _extract_cycles_from_mat(mat_path: Path):
    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    top_key = [k for k in data.keys() if not k.startswith("__")][0]
    battery = data[top_key]
    cycles = battery.cycle
    if not isinstance(cycles, (list, np.ndarray)):
        cycles = [cycles]

    records = []
    for idx, cyc in enumerate(cycles):
        rec = {
            "battery_id": mat_path.stem,
            "cycle_idx": idx,
            "cycle_type": str(getattr(cyc, "type", "unknown")),
            "ambient_temp": getattr(cyc, "ambient_temperature", np.nan),
            "capacity": np.nan,
            "voltage_mean": np.nan,
            "voltage_max": np.nan,
            "voltage_min": np.nan,
            "current_mean": np.nan,
            "current_max": np.nan,
            "current_min": np.nan,
            "temp_mean": np.nan,
            "temp_max": np.nan,
            "temp_min": np.nan,
        }
        cdata = getattr(cyc, "data", None)
        if cdata is not None:
            rec["capacity"] = getattr(cdata, "Capacity", np.nan)
            v = getattr(cdata, "Voltage_measured", [])
            c = getattr(cdata, "Current_measured", [])
            t = getattr(cdata, "Temperature_measured", [])

            if len(v):
                rec["voltage_mean"] = float(np.nanmean(v))
                rec["voltage_max"] = float(np.nanmax(v))
                rec["voltage_min"] = float(np.nanmin(v))
            if len(c):
                rec["current_mean"] = float(np.nanmean(c))
                rec["current_max"] = float(np.nanmax(c))
                rec["current_min"] = float(np.nanmin(c))
            if len(t):
                rec["temp_mean"] = float(np.nanmean(t))
                rec["temp_max"] = float(np.nanmax(t))
                rec["temp_min"] = float(np.nanmin(t))

        records.append(rec)
    return records

def build_nasa_feature_table(raw_dir: Path) -> pd.DataFrame:
    mat_files = sorted(raw_dir.rglob("*.mat"))
    rows = []
    for mat_path in mat_files:
        rows.extend(_extract_cycles_from_mat(mat_path))
    return pd.DataFrame(rows)

# ---------------------------
# Risk scoring
# ---------------------------

def add_proxy_risk_and_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    filled = df.fillna(df.median(numeric_only=True))
    scaler = StandardScaler()
    z = scaler.fit_transform(filled[["temp_mean","voltage_mean","current_mean"]].fillna(0.0))
    risk_score = 0.5*z[:,0] + 0.3*z[:,1] + 0.2*z[:,2]
    df["risk_score"] = risk_score
    threshold = np.quantile(risk_score, 0.90)
    df["proxy_label"] = (df["risk_score"] >= threshold).astype(int)
    return df

# ---------------------------
# Train/test split + metrics
# ---------------------------

def split_by_battery(df: pd.DataFrame, test_size=0.2, val_size=0.1, seed=42):
    ids = df["battery_id"].dropna().unique().tolist()
    train_ids, test_ids = train_test_split(ids, test_size=test_size, random_state=seed)
    train_ids, val_ids = train_test_split(train_ids, test_size=val_size/(1-test_size), random_state=seed)
    return train_ids, val_ids, test_ids

def metrics_dict(y_true, y_score, y_pred):
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true))>1 else None,
        "pr_auc": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true))>1 else None,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

# ---------------------------
# Isolation Forest
# ---------------------------

def train_isolation_forest(train_df, val_df, test_df, model_dir: Path, report_dir: Path):
    X_train = train_df[["temp_mean","voltage_mean","current_mean"]].fillna(0.0)
    X_val = val_df[["temp_mean","voltage_mean","current_mean"]].fillna(0.0)
    X_test = test_df[["temp_mean","voltage_mean","current_mean"]].fillna(0.0)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    clf = IsolationForest(n_estimators=200, contamination=0.1, random_state=42)
    clf.fit(X_train_s)
    train_score = -clf.decision_function(X_train_s)
    val_score = -clf.decision_function(X_val_s)
    test_score = -clf.decision_function(X_test_s)
    threshold = np.quantile(train_score, 0.90)
    val_pred = (val_score >= threshold).astype(int)
    test_pred = (test_score >= threshold).astype(int)
    val_metrics = metrics_dict(val_df["proxy_label"], val_score, val_pred)
    test_metrics = metrics_dict(test_df["proxy_label"], test_score, test_pred)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "scaler": scaler}, model_dir/"isoforest.joblib")
    with open(model_dir/"isoforest_metrics.json","w") as f:
        json.dump({"val": val_metrics, "test": test_metrics}, f, indent=2)
    return {"val": val_metrics, "test": test_metrics, "score": test_score, "threshold": threshold}

# ---------------------------
# Autoencoder
# ---------------------------

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    def forward(self, x):
        # 🔑 This is the missing piece
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

   
def train_autoencoder(train_df, val_df, test_df, model_dir: Path, report_dir: Path):
    all_features = [
        "capacity_ah", "ambient_temp_c", "re", "rct",
        "temp_range", "voltage_range", "current_range"
    ]

    # Only keep features that exist in the dataframe
    features = [f for f in all_features if f in train_df.columns]

    
    if not features:
        features = train_df.select_dtypes(include=[np.number]).columns.tolist()
        print("Fallback: using numeric columns:", features)
    # Clean feature columns: replace string "[]" with NaN and coerce to numeric
    for col in features:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
        val_df[col] = pd.to_numeric(val_df[col], errors="coerce")
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce")

    X_train = train_df[features].fillna(0.0).values.astype(np.float32)
    X_val = val_df[features].fillna(0.0).values.astype(np.float32)
    X_test = test_df[features].fillna(0.0).values.astype(np.float32)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train)), batch_size=32, shuffle=True)
    model = Autoencoder(input_dim=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    for epoch in range(200):
        model.train()
        for batch in train_loader:
            x = batch[0]
            optimizer.zero_grad()
            recon = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_recon = model(torch.tensor(X_val))
        test_recon = model(torch.tensor(X_test))
        val_score = np.mean((X_val - val_recon.numpy())**2, axis=1)
        test_score = np.mean((X_test - test_recon.numpy())**2, axis=1)

   
    # Sweep thresholds to maximize F1 on validation
    candidate_thresholds = np.linspace(0.1, 0.99, 50)  # 50 evenly spaced thresholds
    best_f1 = 0.0
    best_thresh = 0.5

    for th in candidate_thresholds:
        val_pred = (val_score >= th).astype(int)
        f1 = f1_score(val_df["proxy_label"], val_pred, zero_division=0)
        print(f"Threshold {th:.2f} → F1 {f1:.3f}")  # 👈 log each F1
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = th

    threshold = best_thresh
    val_pred = (val_score >= threshold).astype(int)
    test_pred = (test_score >= threshold).astype(int)

    val_metrics = metrics_dict(val_df["proxy_label"], val_score, val_pred)
    test_metrics = metrics_dict(test_df["proxy_label"], test_score, test_pred)

    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "autoencoder.pt")
    with open(model_dir / "autoencoder_metrics.json", "w") as f:
        json.dump({"val": val_metrics, "test": test_metrics}, f, indent=2)

    return {
        "val": val_metrics,
        "test": test_metrics,
        "score": test_score,
        "threshold": threshold
    }

# ---------------------------
# Build + Save feature table
# ---------------------------
def build_and_save(raw_dir: Path, processed_path: Path, report_dir: Path):
    """
    Build the NASA feature table from .mat files,
    add engineered features, risk scores and labels,
    then save it to CSV.
    """
    # Step 1: Extract raw cycle features from .mat files
    df = build_nasa_feature_table(raw_dir)

    # Step 2: Add engineered features
    df["temp_range"] = df["temp_max"] - df["temp_min"]
    df["voltage_range"] = df["voltage_max"] - df["voltage_min"]
    df["current_range"] = df["current_max"] - df["current_min"]

    df["capacity_ah"] = df["capacity"] if "capacity" in df.columns else np.nan
    df["ambient_temp_c"] = df["ambient_temp"] if "ambient_temp" in df.columns else np.nan

    df["re"] = df["impedance_re"] if "impedance_re" in df.columns else np.nan
    df["rct"] = df["impedance_rct"] if "impedance_rct" in df.columns else np.nan

    # Step 3: Add risk scores and proxy labels
    df = add_proxy_risk_and_labels(df)

    # Step 4: Save processed dataset
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)

    # Step 5: Quick plot of risk distribution
    report_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    plt.hist(df["risk_score"], bins=50)
    plt.xlabel("Risk score")
    plt.ylabel("Count")
    plt.title("Cycle risk distribution")
    plt.savefig(report_dir / "sample_cycle_risk.png")
    plt.close()

    return df

# ---------------------------
# Load processed feature table
# ---------------------------

def load_processed(processed_path: Path) -> pd.DataFrame:
    """
    Load the processed NASA cycle feature table from CSV.
    """
    return pd.read_csv(processed_path)
# ---------------------------
# Save predictions helper
# ---------------------------

def save_predictions(df: pd.DataFrame, scores: np.ndarray, preds: np.ndarray,
                     out_path: Path, model_name: str):
    """
    Save anomaly scores and predictions alongside the feature table.
    """
    out_df = df.copy()
    out_df[f"{model_name}_score"] = scores
    out_df[f"{model_name}_pred"] = preds
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")
def prepare_model_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    feature_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    X = df[feature_cols].copy()

    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols, index=df.index)
    return X_imp, feature_cols
