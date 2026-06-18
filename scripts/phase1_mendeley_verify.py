#!/usr/bin/env python3
from __future__ import annotations
import argparse, pickle, zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

EXPECTED_FEATURES = [
    "volt","current","soc","max_single_volt","min_single_volt",
    "max_temp","min_temp","timestamp"
]

def extract_zip(zip_path: Path, extract_root: Path) -> Path:
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    target_dir = extract_root / zip_path.stem
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)
    return target_dir

def find_pickles(base_dir: Path) -> List[Path]:
    return sorted(base_dir.rglob("*.pkl")) if base_dir.exists() else []

def load_pickle(pkl_path: Path) -> Any:
    with pkl_path.open("rb") as f:
        return pickle.load(f)

def describe_sample(sample: Any) -> Dict[str, Any]:
    if not isinstance(sample, tuple) or len(sample)!=2:
        raise ValueError("Expected tuple (data, metadata)")
    data, metadata = sample
    info = {"data_type": type(data).__name__, "metadata_type": type(metadata).__name__}
    info["data_shape"] = tuple(data.shape) if hasattr(data,"shape") else None
    info["metadata_keys"] = list(metadata.keys()) if isinstance(metadata, dict) else []
    return info

def try_extract_label(metadata: Any) -> Tuple[str,str]:
    label_val, mile_val = "NA","NA"
    if isinstance(metadata, dict):
        if "label" in metadata: label_val=str(metadata["label"])
        if "mile" in metadata: mile_val=str(metadata["mile"])
    return label_val, mile_val

def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Mendeley anomaly dataset")
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("data/raw/mendeley"))
    args = parser.parse_args()

    extract_root = args.work_dir / "extracted"
    extract_root.mkdir(parents=True, exist_ok=True)

    zips = sorted(args.download_dir.rglob("*.zip"))
    print(f"[INFO] Found {len(zips)} zip(s)")
    if not zips: return 1

    extracted_dirs=[extract_zip(z,extract_root) for z in zips]
    pkl_files=[]
    for d in extracted_dirs: pkl_files.extend(find_pickles(d))
    print(f"[INFO] Found {len(pkl_files)} .pkl file(s)")
    if not pkl_files: return 1

    sample=load_pickle(pkl_files[0])
    info=describe_sample(sample)
    print("[VERIFY] Sample description:",info)
    data,metadata=sample
    label_val,mile_val=try_extract_label(metadata)
    print(f"[VERIFY] label={label_val}, mile={mile_val}")
    if hasattr(data,"shape") and tuple(data.shape)==(256,8):
        print("[OK] Data shape matches (256,8)")
    print("[EXPECTED FEATURES]",", ".join(EXPECTED_FEATURES))
    print("[SUCCESS] Verification complete")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
