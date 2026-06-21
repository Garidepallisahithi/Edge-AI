from pathlib import Path
from scipy.io import loadmat
import pandas as pd

raw_dir = Path("../data/raw/nasa")
mat_files = sorted(raw_dir.rglob("*.mat"))

rows = []
for mat_path in mat_files:
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    top_key = [k for k in mat.keys() if not k.startswith("__")][0]
    battery = mat[top_key]
    cycles = battery.cycle

    if isinstance(cycles, (list, tuple)):
        cycles = cycles
    else:
        cycles = [cycles]

    for idx, c in enumerate(cycles):
        rec = {
            "battery_id": mat_path.stem,
            "cycle_idx": idx,
            "cycle_type": str(getattr(c, "type", "unknown")),
            "ambient_temp": getattr(c, "ambient_temperature", None),
        }
        data = getattr(c, "data", None)
        if data is not None:
            rec["capacity"] = getattr(data, "Capacity", None)
        rows.append(rec)

df = pd.DataFrame(rows)
df.to_csv("../data/processed/nasa_cycle_features.csv", index=False)
print(df.head())
