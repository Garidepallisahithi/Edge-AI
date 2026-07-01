import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Load trained model
with open("models/final_xgb.pkl", "rb") as f:
    model = pickle.load(f)

# Test data directory
test_dir = Path("data/raw/mendeley/extracted/Test")

rows = []
sample_ids = []

for idx, pkl_file in enumerate(sorted(test_dir.rglob("*.pkl"))):

    try:
        with open(pkl_file, "rb") as f:
            data, meta = pickle.load(f)

        feat = {
            "volt_mean": np.mean(data[:, 0]),
            "volt_std": np.std(data[:, 0]),
            "volt_max": np.max(data[:, 0]),
            "volt_min": np.min(data[:, 0]),
            "volt_range": np.max(data[:, 0]) - np.min(data[:, 0]),

            "current_mean": np.mean(data[:, 1]),
            "current_std": np.std(data[:, 1]),
            "current_max": np.max(data[:, 1]),
            "current_min": np.min(data[:, 1]),
            "current_range": np.max(data[:, 1]) - np.min(data[:, 1]),

            "temp_max": np.max(data[:, 5]),
            "temp_min": np.min(data[:, 6]),
            "temp_mean": np.mean(data[:, 5]),
            "temp_std": np.std(data[:, 5]),
            "temp_range": np.max(data[:, 5]) - np.min(data[:, 6]),

            "soc_mean": np.mean(data[:, 2]),
            "soc_std": np.std(data[:, 2]),

            "mileage": float(meta.get("mileage", 0))
        }

        rows.append(feat)
        sample_ids.append(idx)

    except Exception as e:
        print(f"Skipped {pkl_file}: {e}")

# Create DataFrame
X_test = pd.DataFrame(rows)

# Predict
pred = model.predict(X_test)

# Save predictions
out = pd.DataFrame({
    "sample_id": sample_ids,
    "prediction": pred
})

Path("reports").mkdir(exist_ok=True)

out.to_csv(
    "reports/final_predictions.csv",
    index=False
)

print(out.head())
print()
print("Total predictions:", len(out))
print("Saved -> reports/final_predictions.csv")
