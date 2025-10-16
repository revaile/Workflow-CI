import dagshub
import mlflow
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
import numpy as np
import os
import joblib

# ================================
# 🧩 Argument Parser
# ================================
parser = argparse.ArgumentParser()
parser.add_argument("--test_size", type=float, default=0.2, help="Proporsi data test")
parser.add_argument("--n_estimators", type=int, default=100, help="Jumlah pohon dalam Random Forest")
parser.add_argument("--data_path", type=str, default="titanicdataset_preprocessing/train_preprocessed.csv", help="Path dataset hasil preprocessing")
args = parser.parse_args()

# ================================
# 🔐 Inisialisasi DagsHub & MLflow
# ================================
dagshub.init(
    repo_owner='revaile',  
    repo_name='Eksperimen_SML_Ade-Ripaldi-Nuralim',  
    mlflow=True
)

# ================================
# 📂 Load Dataset
# ================================
data_path = args.data_path
print(f"📂 Membaca dataset dari: {data_path}")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ File dataset tidak ditemukan di {data_path}")

df = pd.read_csv(data_path)

# ================================
# 🧠 Pisahkan fitur dan target
# ================================
if "Survived" not in df.columns:
    raise ValueError("❌ Kolom 'Survived' tidak ditemukan pada dataset!")

X = df.drop(columns=["Survived"]).astype(float)
y = df["Survived"]

# ================================
# ✂️ Split data
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=42
)

# ================================
# 🚀 Training + Logging MLflow
# ================================
mlflow.set_experiment("Titanic_Retrain_Workflow")

with mlflow.start_run(run_name="RandomForest_Workflow_Run"):
    print("🧠 Training model RandomForest...")
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Prediksi dan evaluasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    # Log parameter, metric, dan artifact
    mlflow.log_params({
        "n_estimators": args.n_estimators,
        "test_size": args.test_size,
        "data_path": data_path
    })
    mlflow.log_metrics({
        "accuracy": acc,
        "precision": prec,
        "recall": rec
    })

    model_path = "random_forest_workflow.pkl"
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)

    print(f"✅ Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
    print("🎯 Model dan metric berhasil dicatat di DagsHub MLflow!")

print("🚀 Selesai retraining!")
