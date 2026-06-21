from __future__ import annotations
import argparse
from pathlib import Path
import json

from src.evbattery.prototype import (
    build_and_save,
    load_processed,
    split_by_battery,
    train_autoencoder,
    train_isolation_forest,
    save_predictions,
)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "nasa"
PROCESSED_PATH = ROOT / "data" / "processed" / "nasa_cycle_features.csv"
MODEL_DIR = ROOT / "artifacts" / "models"
REPORT_DIR = ROOT / "reports" / "figures" / "phase2"
ARTIFACT_DIR = ROOT / "artifacts"


def cmd_build(_args):
    df = build_and_save(RAW_DIR, PROCESSED_PATH, REPORT_DIR)
    print(f"Built feature table: {PROCESSED_PATH}")
    print(df.head(3).to_string())


def cmd_train_baseline(_args):
    df = load_processed(PROCESSED_PATH)
    train_ids, val_ids, test_ids = split_by_battery(df)
    train_df = df[df["battery_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["battery_id"].isin(val_ids)].reset_index(drop=True)
    test_df = df[df["battery_id"].isin(test_ids)].reset_index(drop=True)

    result = train_isolation_forest(train_df, val_df, test_df, MODEL_DIR, REPORT_DIR)
    save_predictions(
        test_df,
        result["score"],
        (result["score"] >= result["threshold"]).astype(int),
        ARTIFACT_DIR / "predictions_isoforest.csv",
        "isoforest",
    )
    print(json.dumps(result["test"], indent=2))


def cmd_train_ae(_args):
    df = load_processed(PROCESSED_PATH)
    train_ids, val_ids, test_ids = split_by_battery(df)
    train_df = df[df["battery_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["battery_id"].isin(val_ids)].reset_index(drop=True)
    test_df = df[df["battery_id"].isin(test_ids)].reset_index(drop=True)

    result = train_autoencoder(train_df, val_df, test_df, MODEL_DIR, REPORT_DIR)
    save_predictions(
        test_df,
        result["score"],
        (result["score"] >= result["threshold"]).astype(int),
        ARTIFACT_DIR / "predictions_autoencoder.csv",
        "autoencoder",
    )
    print(json.dumps(result["test"], indent=2))


def main():
    parser = argparse.ArgumentParser(description="Run the EV battery thermal anomaly prototype.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    sub.add_parser("train-baseline")
    sub.add_parser("train-ae")
    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    elif args.command == "train-baseline":
        cmd_train_baseline(args)
    elif args.command == "train-ae":
        cmd_train_ae(args)
    else:
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
