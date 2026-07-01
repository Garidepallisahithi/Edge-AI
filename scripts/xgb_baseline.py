import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

df = pd.read_csv("data/processed/mendeley_features.csv")

X = df.drop(columns=["label"])
y = (df["label"] == 10).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print(classification_report(y_test, pred))

print("\nFeature Importance\n")

feature_names = X.columns

for name, imp in sorted(
    zip(feature_names, model.feature_importances_),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{name}: {imp:.4f}")
