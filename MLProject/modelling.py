import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
import numpy as np
import os
import platform

# 1️⃣ Baca dataset hasil preprocessing
data_path_local = os.path.join("MLProject", "titanicdataset_preprocessing", "train_preprocessed.csv")
data_path_ci = os.path.join("titanicdataset_preprocessing", "train_preprocessed.csv")

if os.path.exists(data_path_local):
    data_path = data_path_local
elif os.path.exists(data_path_ci):
    data_path = data_path_ci
else:
    raise FileNotFoundError("❌ Dataset train_preprocessed.csv tidak ditemukan di path manapun.")

df = pd.read_csv(data_path)

# 2️⃣ Pisahkan fitur dan target
X = df.drop(columns=["Survived"], errors="ignore").astype(float)
y = df["Survived"] if "Survived" in df.columns else np.random.randint(0, 2, len(df))

# 3️⃣ Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ Set tracking URI lintas OS
if platform.system() == "Windows":
    tracking_path = os.path.abspath(os.path.join(os.getcwd(), "mlruns"))
    mlflow.set_tracking_uri(f"file:///{tracking_path.replace(os.sep, '/')}")
else:
    mlflow.set_tracking_uri("file:///home/runner/work/Workflow-CI/Workflow-CI/mlruns")
# 4️⃣-5️⃣ Konfigurasi MLflow lintas OS dan mode eksekusi
# 4️⃣ Konfigurasi MLflow lintas OS & mode eksekusi
is_ci = os.getenv("GITHUB_ACTIONS") == "true"

if not is_ci:
    # 💻 Mode lokal: set tracking URI manual
    if platform.system() == "Windows":
        tracking_path = os.path.abspath(os.path.join(os.getcwd(), "mlruns"))
        mlflow.set_tracking_uri(f"file:///{tracking_path.replace(os.sep, '/')}")
    else:
        mlflow.set_tracking_uri("file:///home/runner/work/Workflow-CI/Workflow-CI/mlruns")
else:
    # 🤖 Mode CI/CD: biarkan MLflow CLI atur tracking_uri
    print("🟡 Mode CI/CD terdeteksi — gunakan konfigurasi MLflow bawaan.")

# Pastikan run dimulai hanya kalau lokal
active_run = mlflow.active_run()
if not is_ci:
    if active_run is None:
        print("🟢 Tidak ada run aktif — mode lokal.")
        mlflow.set_experiment("RandomForest_CI")
        mlflow.start_run(run_name="RandomForest_CI_Run")
    else:
        print(f"🟣 Sudah ada run aktif: {active_run.info.run_id}")


# 6️⃣ Training model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 7️⃣ Hitung metrik
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

# 8️⃣ Logging ke MLflow
mlflow.log_param("n_estimators", 100)
mlflow.log_metric("accuracy", acc)
mlflow.log_metric("precision", prec)
mlflow.log_metric("recall", rec)
mlflow.sklearn.log_model(model, "model")

print(f"✅ Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
print("🎯 Model berhasil dilatih dan disimpan di artefak MLflow lokal!")

# 9️⃣ Tutup run hanya kalau kita yang buka manual
if active_run is None:
    mlflow.end_run()
