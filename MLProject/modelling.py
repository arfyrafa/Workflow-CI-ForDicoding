"""
Modelling Script for MLProject - Product Positioning Dataset
Author: Muhammad Arfy Rafa Fakhrezie (arfyrafa27)

Script ini dirancang untuk dijalankan otomatis di pipeline CI
menggunakan MLflow Project CLI.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import argparse
import os
import warnings

warnings.filterwarnings("ignore")


def train(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    # Load dataset
    data_path = "dataset_preprocessing/product_positioning_preprocessed.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File {data_path} tidak ditemukan!")

    df = pd.read_csv(data_path)
    
    # Split features & target
    X = df.drop(columns=["Product Category"])
    y = df["Product Category"]

    # Split train & test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Inisialisasi model dengan params dari argument parser
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )

    # Fit model
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    # Evaluasi
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Log parameters ke MLflow
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("min_samples_split", min_samples_split)
    mlflow.log_param("min_samples_leaf", min_samples_leaf)

    # Log metrics ke MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Log model ke MLflow
    mlflow.sklearn.log_model(model, "model")

    print(f"[CI RUN] n_estimators: {n_estimators}, max_depth: {max_depth}")
    print(f"[CI RUN] Metrics - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=5)
    parser.add_argument("--min_samples_leaf", type=int, default=2)
    args = parser.parse_args()

    train(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf
    )
