"""Линейные модели: регрессия и классификация."""

import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationResult,
    ClassificationMetrics,
    RegressionInput,
    RegressionResult,
)


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучите собственную линейную регрессию на train и верните прогноз для test."""
    
    X_train = pd.DataFrame(data.train_features)
    y_train = np.array(data.train_target, dtype=float)
    X_test = pd.DataFrame(data.test_features)
    
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)
    
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr = data.learning_rate
    epochs = data.epochs
    
    n = X_train_scaled.shape[0]
    m = X_train_scaled.shape[1]
    
    weights = np.zeros(m)
    bias = 0.0
    loss_history = []
    
    for epoch in range(epochs):
        y_pred = X_train_scaled @ weights + bias
        error = y_pred - y_train
        mse = np.mean(error ** 2)
        loss_history.append(float(mse))
        
        grad_w = (2.0 / n) * (X_train_scaled.T @ error)
        grad_b = (2.0 / n) * np.sum(error)
        
        weights = weights - lr * grad_w
        bias = bias - lr * grad_b
    
    predictions = X_test_scaled @ weights + bias
    
    return RegressionResult(
        weights=weights,
        bias=float(bias),
        loss_history=loss_history,
        predictions=predictions,
    )


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучите собственный логистический классификатор на train и верните прогноз для test."""
    
    X_train = pd.DataFrame(data.train_features)
    y_train = np.array(data.train_target, dtype=float)
    X_test = pd.DataFrame(data.test_features)
    
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr = data.learning_rate
    epochs = data.epochs
    
    n = X_train_scaled.shape[0]
    m = X_train_scaled.shape[1]
    
    weights = np.zeros(m)
    bias = 0.0
    loss_history = []
    
    for epoch in range(epochs):
        z = X_train_scaled @ weights + bias
        p = sigmoid(z)
        
        eps = 1e-15
        p_clipped = np.clip(p, eps, 1 - eps)
        loss = -np.mean(y_train * np.log(p_clipped) + (1 - y_train) * np.log(1 - p_clipped))
        loss_history.append(float(loss))
        
        error = p - y_train
        grad_w = (1.0 / n) * (X_train_scaled.T @ error)
        grad_b = (1.0 / n) * np.sum(error)
        
        weights = weights - lr * grad_w
        bias = bias - lr * grad_b
    
    z_test = X_test_scaled @ weights + bias
    probabilities = sigmoid(z_test)
    predictions = (probabilities >= 0.5).astype(int)
    
    return ClassificationResult(
        weights=weights,
        bias=float(bias),
        loss_history=loss_history,
        probabilities=probabilities,
        predictions=predictions,
    )


def compute_metrics(y_true, y_pred, probabilities):
    roc_auc = float(roc_auc_score(y_true, probabilities))
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    return ClassificationMetrics(
        roc_auc=roc_auc,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравните готовые Logistic Regression, Decision Tree и KNN на одном split."""
    
    X_train = pd.DataFrame(data.train_features)
    y_train = np.array(data.train_target, dtype=float)
    X_test = pd.DataFrame(data.test_features)
    y_test = np.array(data.test_target, dtype=float)
    
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]
    lr_preds = lr_model.predict(X_test_scaled)
    lr_metrics = compute_metrics(y_test, lr_preds, lr_probs)
    
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train_scaled, y_train)
    dt_probs = dt_model.predict_proba(X_test_scaled)[:, 1]
    dt_preds = dt_model.predict(X_test_scaled)
    dt_metrics = compute_metrics(y_test, dt_preds, dt_probs)
    
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_scaled, y_train)
    knn_probs = knn_model.predict_proba(X_test_scaled)[:, 1]
    knn_preds = knn_model.predict(X_test_scaled)
    knn_metrics = compute_metrics(y_test, knn_preds, knn_probs)
    
    result = BaselineComparisonResult(
        logistic_regression=lr_metrics,
        decision_tree=dt_metrics,
        knn=knn_metrics,
    )
    
    experiment_results = {
        "classification": {
            "logistic_regression": {
                "roc_auc": lr_metrics.roc_auc,
                "accuracy": lr_metrics.accuracy,
                "precision": lr_metrics.precision,
                "recall": lr_metrics.recall,
                "f1": lr_metrics.f1,
            },
            "decision_tree": {
                "roc_auc": dt_metrics.roc_auc,
                "accuracy": dt_metrics.accuracy,
                "precision": dt_metrics.precision,
                "recall": dt_metrics.recall,
                "f1": dt_metrics.f1,
            },
            "knn": {
                "roc_auc": knn_metrics.roc_auc,
                "accuracy": knn_metrics.accuracy,
                "precision": knn_metrics.precision,
                "recall": knn_metrics.recall,
                "f1": knn_metrics.f1,
            },
        }
    }
    
    try:
        with open("experiment_results.json", "w", encoding="utf-8") as f:
            json.dump(experiment_results, f, indent=2, ensure_ascii=False)
    except Exception:
        pass
    
    return result
