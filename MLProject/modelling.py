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

# 🔧 Argument parser (parameter dari MLProject)
parser = argparse.ArgumentParser()
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--n_estimators", type=int, default=100)
args = parser.parse_args()

# 1️⃣ Setup DagsHub MLflow
dagshub.init(
    repo_owner='revaile',
    repo_name='Workflow-CI',
    mlflow=True
)

# 2️⃣ Baca dataset hasil preprocessing
data_path = os.path.join("titanicdataset_preprocessing", "train_preprocessed.csv")
df = pd.read_csv(data_path)

# 3️⃣ Pisahkan fitur dan target
X = df.drop(columns=["Survived"], errors="ignore").astype(float)
y = df["Survived"] if "Survived" in df.columns else np.random.randint(0, 2, len(df))

# 4️⃣ Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=42
)

# 5️⃣ Mulai MLflow run
mlflow.set_experiment("Titanic_Retrain_Workflow")
with mlflow.start_run(run_name="RandomForest_Workflow_Run"):
    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    # Log metrics dan parameters
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)

    # Simpan model dan log artifact
    model_path = "random_forest_workflow.pkl"
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)

    print(f"✅ Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
    print("🎯 Model dan metric berhasil dicatat di DagsHub MLflow!")

print("🚀 Selesai retraining!")
