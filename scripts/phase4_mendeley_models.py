#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def train_autoencoder(X_train, X_val, input_dim):
    class Autoencoder(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 16), nn.ReLU(),
                nn.Linear(16, 8), nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(8, 16), nn.ReLU(),
                nn.Linear(16, input_dim)
            )
        def forward(self, x):
            return self.decoder(self.encoder(x))

    model = Autoencoder(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32)), batch_size=32, shuffle=True)

    for epoch in range(50):
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
        val_recon = model(torch.tensor(X_val, dtype=torch.float32))
        val_score = np.mean((X_val - val_recon.numpy())**2, axis=1)
    threshold = np.quantile(val_score, 0.90)
    return model, threshold

def main():
    parser = argparse.ArgumentParser(description="Phase 4 - Model comparison")

    parser.add_argument(
        "--features-csv",
        type=Path,
        required=True,
        help="CSV file with features and labels"
    )

    parser.add_argument(
        "--metrics-out",
        type=Path,
        required=True,
        help="Where to save metrics JSON"
    )

    args = parser.parse_args()

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv(args.features_csv)

    X = df.drop(columns=["label"]).values
    y = df["label"].astype(int).values

    # -----------------------------
    # Feature Scaling
    # -----------------------------
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    results = {}

    # =====================================================
    # Isolation Forest
    # =====================================================
    iso = IsolationForest(
        contamination=0.134,
        random_state=42
    )

    iso.fit(X)

    preds_iso = np.where(
        iso.predict(X) == -1,
        10,
        0
    )

    results["IsolationForest"] = classification_report(
        y,
        preds_iso,
        output_dict=True
    )

    # =====================================================
    # One-Class SVM
    # =====================================================
    ocsvm = OneClassSVM(
        gamma="scale",
        nu=0.134
    )

    ocsvm.fit(X)

    preds_svm = np.where(
        ocsvm.predict(X) == -1,
        10,
        0
    )

    results["OneClassSVM"] = classification_report(
        y,
        preds_svm,
        output_dict=True
    )

    # =====================================================
    # Autoencoder
    # =====================================================
    split = int(0.8 * len(X))

    X_train = X[:split]
    X_val = X[split:]

    model, threshold = train_autoencoder(
        X_train,
        X_val,
        X.shape[1]
    )

    with torch.no_grad():
        recon = model(
            torch.tensor(
                X,
                dtype=torch.float32
            )
        )

        score = np.mean(
            (X - recon.numpy()) ** 2,
            axis=1
        )

    preds_ae = np.where(
        score >= threshold,
        10,
        0
    )

    results["Autoencoder"] = classification_report(
        y,
        preds_ae,
        output_dict=True
    )

    # =====================================================
    # Save metrics
    # =====================================================
    args.metrics_out.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with args.metrics_out.open("w") as f:
        json.dump(
            results,
            f,
            indent=2
        )

    print(f"[SUCCESS] Metrics saved to {args.metrics_out}")
if __name__ == "__main__":
    main()
