import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
import numpy as np
import os

# 1️⃣ Baca dataset hasil preprocessing
data_path = os.path.join("titanicdataset_preprocessing", "train_preprocessed.csv")
df = pd.read_csv(data_path)

# 2️⃣ Pisahkan fitur dan target
X = df.drop(columns=["Survived"], errors="ignore").astype(float)
y = df["Survived"] if "Survived" in df.columns else np.random.randint(0, 2, len(df))

# 3️⃣ Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ Tracking lokal agar bisa dicatat di artefak GitHub
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("RandomForest_CI")

with mlflow.start_run(run_name="RandomForest_CI_Run"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)

    mlflow.sklearn.log_model(model, "model")

    print(f"✅ Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

print("🎯 Model berhasil dilatih dan disimpan di artefak MLflow lokal!")
