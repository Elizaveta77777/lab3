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


def _to_numpy_features(features):
    if isinstance(features, pd.DataFrame):
        features = pd.get_dummies(features, drop_first=True)
        return features.values.astype(float)
    return np.array(features, dtype=float)


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    X_train = _to_numpy_features(data.train_features)
    y_train = np.array(data.train_target, dtype=float)
    X_test = _to_numpy_features(data.test_features)
    y_test = np.array(data.test_target, dtype=float)

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
        mse = float(np.mean(error ** 2))
        loss_history.append(mse)

        grad_w = (2.0 / n) * (X_train_scaled.T @ error)
        grad_b = (2.0 / n) * np.sum(error)

        weights = weights - lr * grad_w
        bias = bias - lr * grad_b

    predictions = X_test_scaled @ weights + bias

    test_error = predictions - y_test
    test_mse = float(np.mean(test_error ** 2))

    _save_to_json("linear_regression", {"test_mse": test_mse})

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
    X_train = _to_numpy_features(data.train_features)
    y_train = np.array(data.train_target, dtype=float)
    X_test = _to_numpy_features(data.test_features)

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
        loss = float(-np.mean(y_train * np.log(p_clipped) + (1 - y_train) * np.log(1 - p_clipped)))
        loss_history.append(loss)

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


def _compute_metrics(y_true, y_pred, probabilities):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    probabilities = np.array(probabilities)

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


def _metrics_to_dict(metrics):
    return {
        "roc_auc": metrics.roc_auc,
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
    }


def _save_to_json(key, value):
    try:
        try:
            with open("experiment_results.json", "r", encoding="utf-8") as f:
                results = json.load(f)
        except FileNotFoundError:
            results = {}

        results[key] = value

        with open("experiment_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    y_train = np.array(data.train_target, dtype=float)
    y_test = np.array(data.test_target, dtype=float)

    own_result = train_logistic_classifier(data)
    own_metrics = _compute_metrics(y_test, own_result.predictions, own_result.probabilities)

    X_train = _to_numpy_features(data.train_features)
    X_test = _to_numpy_features(data.test_features)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    sklearn_lr = LogisticRegression(max_iter=1000, random_state=42)
    sklearn_lr.fit(X_train_scaled, y_train)
    sklearn_lr_probs = sklearn_lr.predict_proba(X_test_scaled)[:, 1]
    sklearn_lr_preds = sklearn_lr.predict(X_test_scaled)
    sklearn_lr_metrics = _compute_metrics(y_test, sklearn_lr_preds, sklearn_lr_probs)

    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train_scaled, y_train)
    dt_probs = dt_model.predict_proba(X_test_scaled)[:, 1]
    dt_preds = dt_model.predict(X_test_scaled)
    dt_metrics = _compute_metrics(y_test, dt_preds, dt_probs)

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_scaled, y_train)
    knn_probs = knn_model.predict_proba(X_test_scaled)[:, 1]
    knn_preds = knn_model.predict(X_test_scaled)
    knn_metrics = _compute_metrics(y_test, knn_preds, knn_probs)

    result = BaselineComparisonResult(
        logistic_regression=own_metrics,
        decision_tree=dt_metrics,
        knn=knn_metrics,
    )

    _save_to_json("classification", {
        "own_logistic": _metrics_to_dict(own_metrics),
        "logistic_regression": _metrics_to_dict(sklearn_lr_metrics),
        "decision_tree": _metrics_to_dict(dt_metrics),
        "knn": _metrics_to_dict(knn_metrics),
    })

    return result