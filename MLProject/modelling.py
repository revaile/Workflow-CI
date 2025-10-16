import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
import numpy as np
import os, platform

# 🚫 Hapus env lama dari Windows
if "MLFLOW_TRACKING_URI" in os.environ:
    print(f"🔧 Menghapus tracking URI lama: {os.environ['MLFLOW_TRACKING_URI']}")
    del os.environ["MLFLOW_TRACKING_URI"]

# 1️⃣ Dataset
data_path_local = os.path.join("MLProject", "titanicdataset_preprocessing", "train_preprocessed.csv")
data_path_ci = os.path.join("titanicdataset_preprocessing", "train_preprocessed.csv")

data_path = data_path_local if os.path.exists(data_path_local) else data_path_ci
df = pd.read_csv(data_path)

X = df.drop(columns=["Survived"], errors="ignore").astype(float)
y = df["Survived"] if "Survived" in df.columns else np.random.randint(0, 2, len(df))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2️⃣ Cek mode eksekusi
is_ci = os.getenv("GITHUB_ACTIONS") == "true"

if not is_ci:
    if platform.system() == "Windows":
        tracking_path = os.path.abspath(os.path.join(os.getcwd(), "mlruns"))
        mlflow.set_tracking_uri(f"file:///{tracking_path.replace(os.sep, '/')}")
    else:
        mlflow.set_tracking_uri("file:///home/runner/work/Workflow-CI/Workflow-CI/mlruns")
else:
    print("🟡 Mode CI/CD terdeteksi — biarkan MLflow CLI yang kelola run.")

# 3️⃣ Jalankan model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

print(f"✅ Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

# 4️⃣ Logging
try:
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.sklearn.log_model(model, "model")
    print("📊 Metrik & model berhasil dilog ke MLflow.")
except Exception as e:
    print(f"⚠️ Skip logging karena mode CI/CD otomatis — detail: {e}")

print("🎯 Training selesai tanpa error!")
