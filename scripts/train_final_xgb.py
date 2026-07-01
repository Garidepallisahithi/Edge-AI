import pandas as pd
import pickle
from pathlib import Path

from xgboost import XGBClassifier

df = pd.read_csv("data/processed/mendeley_features.csv")

X = df.drop(columns=["label"])
y = (df["label"] == 10).astype(int)

normal = (y == 0).sum()
fault = (y == 1).sum()

scale_pos_weight = normal / fault

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X, y)

Path("models").mkdir(exist_ok=True)

with open("models/final_xgb.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved -> models/final_xgb.pkl")
