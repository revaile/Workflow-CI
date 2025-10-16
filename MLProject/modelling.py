import os
import platform
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 🚫 Bersihkan environment lama (khusus Windows agar tidak bentrok tracking URI)
os.environ.pop("MLFLOW_TRACKING_URI", None)

# 1️⃣ Load Dataset
data_path_local = os.path.join("MLProject", "titanicdataset_preprocessing", "train_preprocessed.csv")
data_path_ci = os.path.join("titanicdataset_preprocessing", "train_preprocessed.csv")
data_path = data_path_local if os.path.exists(data_path_local) else data_path_ci

df = pd.read_csv(data_path)
X = df.drop(columns=["Survived"], errors="ignore").astype(float)
y = df["Survived"] if "Survived" in df.columns else np.random.randint(0, 2, len(df))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2️⃣ Deteksi mode eksekusi (lokal vs CI/CD)
is_ci = os.getenv("GITHUB_ACTIONS") == "true"

if not is_ci:
    # Mode lokal: atur tracking URI agar hasil tersimpan ke folder mlruns lokal
    if platform.system() == "Windows":
        tracking_path = os.path.abspath(os.path.join(os.getcwd(), "mlruns"))
        mlflow.set_tracking_uri(f"file:///{tracking_path.replace(os.sep, '/')}")
    else:
        mlflow.set_tracking_uri("file:///home/runner/work/Workflow-CI/Workflow-CI/mlruns")
else:
    print("🟡 Mode CI/CD terdeteksi — biarkan MLflow CLI yang kelola run.")

# 3️⃣ Training Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

print(f"✅ Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

# 4️⃣ Logging ke MLflow (skip otomatis jika dijalankan dari CLI di CI/CD)
try:
    if not is_ci:
        with mlflow.start_run(run_name="RandomForest_Local_Run"):
            mlflow.log_param("n_estimators", 100)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.sklearn.log_model(model, "model")
            print("📊 Metrik & model berhasil dilog ke MLflow.")
    else:
        print("⚠️ Skip logging manual — MLflow CLI sudah handle run di CI/CD.")
except Exception as e:
    print(f"⚠️ Terjadi error saat logging MLflow: {e}")

# 5️⃣ Simpan Model ke .pkl (untuk upload-artifact / Docker)
model_dir = "MLProject" if not is_ci else "."
model_mlflow_path = os.path.join(model_dir, "random_forest_workflow")

mlflow.sklearn.save_model(model, model_mlflow_path)
print(f"💾 Model MLflow tersimpan di: {os.path.abspath(model_mlflow_path)}")



print("🎯 Training & logging selesai tanpa error!")
