"""
Modelling for Workflow CI - Wine Quality Classification
Author: Muhammad Arfy Rafa Fakhrezie (arfyrafa27)

Script ini dijalankan di dalam MLflow Project sebagai bagian dari
Workflow CI. Melatih model RandomForestClassifier dan logging ke MLflow.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import json

warnings.filterwarnings("ignore")


def main():
    print("=" * 60)
    print("WORKFLOW CI - WINE QUALITY TRAINING")
    print("Author: Muhammad Arfy Rafa Fakhrezie (arfyrafa27)")
    print("=" * 60)

    # Load data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "dataset_preprocessing", "wine_quality_preprocessed.csv")
    df = pd.read_csv(data_path)
    print(f"[INFO] Dataset loaded: {df.shape}")

    # Split data
    X = df.drop(columns=["quality_label"])
    y = df["quality_label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")

    # MLflow experiment
    mlflow.set_experiment("Wine-Quality-CI")

    with mlflow.start_run(run_name="CI-RandomForest") as run:
        # Parameters
        params = {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
        }

        # Log parameters
        for key, value in params.items():
            mlflow.log_param(key, value)

        # Train model
        model = RandomForestClassifier(**params, n_jobs=-1)
        model.fit(X_train, y_train)

        # Predict & Evaluate
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        print(f"\n[RESULT] Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")

        # Create artifacts directory
        artifact_dir = os.path.join(base_dir, "artifacts")
        os.makedirs(artifact_dir, exist_ok=True)

        # Confusion Matrix
        labels = ["high", "low", "medium"]
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.title("Confusion Matrix - Wine Quality CI")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        cm_path = os.path.join(artifact_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=150)
        plt.close()
        mlflow.log_artifact(cm_path)

        # Classification Report
        report = classification_report(y_test, y_pred, target_names=labels)
        report_path = os.path.join(artifact_dir, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        # Feature Importance
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(12, 8))
        plt.bar(range(len(X.columns)), importances[indices])
        plt.xticks(range(len(X.columns)),
                   [X.columns[i] for i in indices], rotation=45, ha="right")
        plt.title("Feature Importance")
        plt.tight_layout()
        fi_path = os.path.join(artifact_dir, "feature_importance.png")
        plt.savefig(fi_path, dpi=150)
        plt.close()
        mlflow.log_artifact(fi_path)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print(f"\n[INFO] Run ID: {run.info.run_id}")
        print(f"[INFO] Artifacts logged: confusion_matrix, classification_report, feature_importance")

    print("\n" + "=" * 60)
    print("CI TRAINING SELESAI!")
    print("=" * 60)


if __name__ == "__main__":
    main()
