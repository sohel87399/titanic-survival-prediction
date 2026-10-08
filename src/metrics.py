"""
src/metrics.py
--------------
Metric computation helpers for classification models.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import LearningCurveDisplay, ValidationCurveDisplay
from sklearn.pipeline import Pipeline


def compute_all_metrics(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Compute a full suite of classification metrics on the test set.

    Parameters
    ----------
    pipeline:
        A **fitted** sklearn Pipeline with a classifier as the last step.
    X_test:
        Test features.
    y_test:
        True labels.

    Returns
    -------
    dict[str, float]
        Keys: Accuracy, Precision, Recall, F1, ROC-AUC.
    """
    y_pred = pipeline.predict(X_test)
    y_prob = _get_proba(pipeline, X_test)

    metrics: dict[str, float] = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "F1": float(f1_score(y_test, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_test, y_prob)) if y_prob is not None else float("nan"),
    }
    return metrics


def _get_proba(pipeline: Pipeline, X: pd.DataFrame) -> Optional[np.ndarray]:
    """Return positive-class probability if the pipeline supports it."""
    if hasattr(pipeline, "predict_proba"):
        try:
            return pipeline.predict_proba(X)[:, 1]
        except Exception:
            pass
    if hasattr(pipeline, "decision_function"):
        try:
            return pipeline.decision_function(X)
        except Exception:
            pass
    return None


def get_confusion_matrix(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> np.ndarray:
    """Return the 2×2 confusion matrix.

    Parameters
    ----------
    pipeline:
        Fitted pipeline.
    X_test, y_test:
        Test split.

    Returns
    -------
    np.ndarray
        Shape (2, 2).
    """
    y_pred = pipeline.predict(X_test)
    return confusion_matrix(y_test, y_pred)


def get_roc_curve(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute ROC curve data.

    Returns
    -------
    Tuple
        (fpr, tpr, auc_score)
    """
    y_prob = _get_proba(pipeline, X_test)
    if y_prob is None:
        raise ValueError("Model does not support probability estimates.")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = float(roc_auc_score(y_test, y_prob))
    return fpr, tpr, auc


def get_pr_curve(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute Precision-Recall curve data.

    Returns
    -------
    Tuple
        (precision, recall, average_precision)
    """
    y_prob = _get_proba(pipeline, X_test)
    if y_prob is None:
        raise ValueError("Model does not support probability estimates.")
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap = float(average_precision_score(y_test, y_prob))
    return precision, recall, ap


def get_classification_report(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_names: list[str] | None = None,
) -> str:
    """Return classification report as a string.

    Parameters
    ----------
    pipeline:
        Fitted pipeline.
    X_test, y_test:
        Test split.
    target_names:
        Optional class label strings.

    Returns
    -------
    str
    """
    y_pred = pipeline.predict(X_test)
    return classification_report(y_test, y_pred, target_names=target_names, zero_division=0)


def get_learning_curve_data(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    n_points: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute learning curve data (accuracy vs training set size).

    Parameters
    ----------
    pipeline:
        Unfitted pipeline (cloned internally).
    X:
        Full feature matrix.
    y:
        Full target.
    cv:
        Cross-validation folds.
    n_points:
        Number of training-size points to evaluate.

    Returns
    -------
    Tuple
        (train_sizes, train_scores_mean, test_scores_mean)
        Each ``_scores_mean`` is shape (n_points,).
    """
    from sklearn.model_selection import learning_curve

    train_sizes, train_scores, test_scores = learning_curve(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, n_points),
        n_jobs=-1,
        shuffle=True,
        random_state=42,
    )
    return (
        train_sizes,
        np.mean(train_scores, axis=1),
        np.mean(test_scores, axis=1),
    )


def get_validation_curve_data(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    param_name: str,
    param_range: list,
    cv: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute validation curve (accuracy vs a single hyperparameter).

    Parameters
    ----------
    pipeline:
        Unfitted pipeline.
    X, y:
        Full dataset.
    param_name:
        Full sklearn param path (e.g. ``'classifier__C'``).
    param_range:
        Values to sweep.
    cv:
        Cross-validation folds.

    Returns
    -------
    Tuple
        (train_scores_mean, test_scores_mean)  — shape (len(param_range),)
    """
    from sklearn.model_selection import validation_curve

    train_scores, test_scores = validation_curve(
        pipeline,
        X,
        y,
        param_name=param_name,
        param_range=param_range,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )
    return np.mean(train_scores, axis=1), np.mean(test_scores, axis=1)


def build_leaderboard(results: dict[str, dict]) -> pd.DataFrame:
    """Build a sortable leaderboard from training results.

    Parameters
    ----------
    results:
        Mapping of model name → metric dict (output of ``compute_all_metrics``
        optionally merged with cross-validation keys and ``Training Time``).

    Returns
    -------
    pd.DataFrame
        Leaderboard with one row per model.
    """
    rows = []
    for name, metrics in results.items():
        row = {"Model": name}
        row.update(metrics)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Sort by F1 descending by default
    if "F1" in df.columns:
        df = df.sort_values("F1", ascending=False).reset_index(drop=True)
    return df
