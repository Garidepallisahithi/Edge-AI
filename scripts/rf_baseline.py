import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/processed/mendeley_features.csv")

X = df.drop(columns=["label"])
y = df["label"]

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)

rf.fit(X_train,y_train)

pred = rf.predict(X_test)

print(classification_report(y_test,pred))

print("\nFeature Importance\n")

for n,i in zip(X.columns,rf.feature_importances_):
    print(f'{n}: {i:.4f}')
