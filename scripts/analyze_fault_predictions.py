import pickle
import pandas as pd
from pathlib import Path

pred = pd.read_csv("reports/final_predictions.csv")

files = sorted(
    Path("data/raw/mendeley/extracted/Test").rglob("*.pkl")
)

rows = []

for sid in pred[pred.prediction == 1].sample_id:

    with open(files[sid], "rb") as f:
        data, meta = pickle.load(f)

    rows.append({
        "sample_id": sid,
        "mileage": float(meta.get("mileage", 0)),
        "temp_max": float(data[:,5].max()),
        "temp_min": float(data[:,6].min())
    })

df = pd.DataFrame(rows)

print(df.describe())

df.to_csv(
    "reports/fault_prediction_analysis.csv",
    index=False
)

print()
print("Saved -> reports/fault_prediction_analysis.csv")
