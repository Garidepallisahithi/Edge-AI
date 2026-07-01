import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("data/processed/mendeley_features.csv")

# Features and labels
X = df.drop(columns=["label"])
y = (df["label"] == 10).astype(int)

# Handle class imbalance
scale_pos_weight = (y == 0).sum() / (y == 1).sum()

print(f"Samples: {len(df)}")
print(f"Normal: {(y == 0).sum()}")
print(f"Fault: {(y == 1).sum()}")
print(f"scale_pos_weight = {scale_pos_weight:.2f}")

# 5-Fold Stratified Cross Validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

accs = []
precs = []
recalls = []
f1s = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    print("\n" + "=" * 40)
    print(f"FOLD {fold}")
    print("=" * 40)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1       : {f1:.4f}")

    accs.append(acc)
    precs.append(prec)
    recalls.append(rec)
    f1s.append(f1)

print("\n" + "=" * 50)
print("CROSS VALIDATION SUMMARY")
print("=" * 50)

print(f"Mean Accuracy : {np.mean(accs):.4f}")
print(f"Mean Precision: {np.mean(precs):.4f}")
print(f"Mean Recall   : {np.mean(recalls):.4f}")
print(f"Mean F1       : {np.mean(f1s):.4f}")
