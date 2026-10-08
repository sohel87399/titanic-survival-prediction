"""
tests/test_metrics.py
---------------------
Unit tests for src/metrics.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics import (
    build_leaderboard,
    compute_all_metrics,
    get_classification_report,
    get_confusion_matrix,
    get_learning_curve_data,
    get_pr_curve,
    get_roc_curve,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def binary_pipeline():
    """A simple fitted pipeline for a synthetic binary problem."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5, random_state=0
    )
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y_s = pd.Series(y)
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_s, random_state=0)

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500))])
    pipe.fit(X_train, y_train)
    return pipe, X_test, y_test


# ── compute_all_metrics ───────────────────────────────────────────────────────


class TestComputeAllMetrics:
    def test_returns_five_metrics(self, binary_pipeline):
        pipe, X_test, y_test = binary_pipeline
        metrics = compute_all_metrics(pipe, X_test, y_test)
        for key in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
            assert key in metrics

    def test_all_values_in_range(self, binary_pipeline):
        pipe, X_test, y_test = binary_pipeline
        metrics = compute_all_metrics(pipe, X_test, y_test)
        for key, val in metrics.items():
            assert 0.0 <= val <= 1.0, f"{key} = {val} out of range [0, 1]"

    def test_accuracy_consistent(self, binary_pipeline):
        from sklearn.metrics import accuracy_score
        pipe, X_test, y_test = binary_pipeline
        metrics = compute_all_metrics(pipe, X_test, y_test)
        expected = accuracy_score(y_test, pipe.predict(X_test))
        assert abs(metrics["Accuracy"] - expected) < 1e-8


# ── get_confusion_matrix ──────────────────────────────────────────────────────


class TestGetConfusionMatrix:
    def test_shape(self, binary_pipeline):
        pipe, X_test, y_test = binary_pipeline
        cm = get_confusion_matrix(pipe, X_test, y_test)
        assert cm.shape == (2, 2)

    def test_sum_equals_n_samples(self, binary_pipeline):
        pipe, X_test, y_test = binary_pipeline
        cm = get_confusion_matrix(pipe, X_test, y_test)
        assert cm.sum() == len(y_test)

    def test_non_negative(self, binary_pipeline):
        pipe, X_test, y_test = binary_pipeline
        cm = get_confusion_matrix(pipe, X_test, y_test)
        assert (cm >= 0).all()


# ── get_roc_curve ─────────────────────────────────────────────────────────────


class TestGetROCCurve:
    def test_returns_tuple(self, binary_pipeline):
        pipe, X_test, y_test = binary_pipeline
        fpr, tpr, auc = get_roc_curve(pipe, X_test, y_test)
        assert len(fpr) == len(tpr)
        assert 0.0 <= auc <= 1.0

    def test_fpr_sorted(self, binary_pipeline):
        pipe, X_test, y_test = binary_pipeline
        fpr, tpr, auc = get_roc_curve(pipe, X_test, y_test)
        assert (np.diff(fpr) >= 0).all()


# ── get_pr_curve ──────────────────────────────────────────────────────────────


class TestGetPRCurve:
    def test_returns_tuple(self, binary_pipeline):
        pipe, X_test, y_test = binary_pipeline
        prec, rec, ap = get_pr_curve(pipe, X_test, y_test)
        assert len(prec) == len(rec)
        assert 0.0 <= ap <= 1.0


# ── get_classification_report ─────────────────────────────────────────────────


class TestGetClassificationReport:
    def test_returns_string(self, binary_pipeline):
        pipe, X_test, y_test = binary_pipeline
        report = get_classification_report(pipe, X_test, y_test)
        assert isinstance(report, str)
        assert "precision" in report.lower()

    def test_with_target_names(self, binary_pipeline):
        pipe, X_test, y_test = binary_pipeline
        report = get_classification_report(pipe, X_test, y_test, target_names=["Neg", "Pos"])
        assert "Neg" in report or "Pos" in report


# ── get_learning_curve_data ───────────────────────────────────────────────────


class TestGetLearningCurveData:
    def test_shapes_match(self, binary_pipeline):
        pipe, X_test, y_test = binary_pipeline
        X, y = make_classification(n_samples=200, n_features=10, random_state=0)
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        y_s = pd.Series(y)

        fresh_pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=200))])
        sizes, tr_s, te_s = get_learning_curve_data(fresh_pipe, X_df, y_s, cv=3, n_points=5)
        assert len(sizes) == len(tr_s) == len(te_s)
        assert len(sizes) == 5


# ── build_leaderboard ─────────────────────────────────────────────────────────


class TestBuildLeaderboard:
    def test_returns_dataframe(self):
        results = {
            "Model A": {"Accuracy": 0.85, "F1": 0.82, "ROC-AUC": 0.88},
            "Model B": {"Accuracy": 0.80, "F1": 0.79, "ROC-AUC": 0.84},
        }
        lb = build_leaderboard(results)
        assert isinstance(lb, pd.DataFrame)
        assert len(lb) == 2

    def test_sorted_by_f1(self):
        results = {
            "Good": {"Accuracy": 0.90, "F1": 0.89, "ROC-AUC": 0.92},
            "Bad": {"Accuracy": 0.60, "F1": 0.55, "ROC-AUC": 0.61},
        }
        lb = build_leaderboard(results)
        assert lb.iloc[0]["Model"] == "Good"

    def test_empty_input(self):
        lb = build_leaderboard({})
        assert lb.empty

    def test_model_column_present(self):
        results = {"M": {"Accuracy": 0.9, "F1": 0.88}}
        lb = build_leaderboard(results)
        assert "Model" in lb.columns
