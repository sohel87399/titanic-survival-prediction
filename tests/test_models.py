"""
tests/test_models.py
--------------------
Unit tests for src/models.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import HP_DEFAULTS
from src.models import build_estimator, build_pipeline, cross_validate_model, train_model


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def simple_data():
    """Synthetic binary classification dataset."""
    X, y = make_classification(n_samples=200, n_features=5, n_informative=3, random_state=42)
    cols = ["Age", "Fare", "SibSp", "Parch", "FamilySize"]
    X_df = pd.DataFrame(X, columns=cols)
    y_s = pd.Series(y, name="Survived")
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_s, random_state=42)
    return X_train, X_test, y_train, y_test


MODELS = [
    "Logistic Regression",
    "K-Nearest Neighbors",
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting",
    "XGBoost",
    "SVM",
    "Neural Network",
]

SELECTED_FEATURES = ["Age", "Fare", "SibSp", "Parch", "FamilySize"]


# ── build_estimator ───────────────────────────────────────────────────────────


class TestBuildEstimator:
    @pytest.mark.parametrize("model_name", MODELS)
    def test_returns_estimator(self, model_name):
        estimator = build_estimator(model_name, HP_DEFAULTS[model_name])
        assert hasattr(estimator, "fit")
        assert hasattr(estimator, "predict")

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model name"):
            build_estimator("Flux Capacitor", {})

    def test_logreg_penalty_solver_compat(self):
        """l1 penalty should use saga solver."""
        est = build_estimator("Logistic Regression", {"penalty": "l1", "C": 1.0, "max_iter": 200})
        assert est.solver == "saga"
        # sklearn ≥1.8 dropped penalty param; l1_ratio=1 encodes l1
        assert est.l1_ratio == 1.0

    def test_logreg_l2_uses_lbfgs(self):
        est = build_estimator("Logistic Regression", {"penalty": "l2", "C": 1.0, "max_iter": 200})
        assert est.solver == "lbfgs"


# ── build_pipeline ────────────────────────────────────────────────────────────


class TestBuildPipeline:
    @pytest.mark.parametrize("model_name", MODELS)
    def test_returns_pipeline(self, model_name):
        pipe = build_pipeline(model_name, HP_DEFAULTS[model_name], SELECTED_FEATURES)
        assert isinstance(pipe, Pipeline)
        assert "preprocessor" in pipe.named_steps
        assert "classifier" in pipe.named_steps

    def test_pipeline_can_fit_predict(self, simple_data):
        X_train, X_test, y_train, y_test = simple_data
        pipe = build_pipeline("Logistic Regression", HP_DEFAULTS["Logistic Regression"], SELECTED_FEATURES)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        assert len(preds) == len(y_test)
        assert set(preds).issubset({0, 1})


# ── train_model ───────────────────────────────────────────────────────────────


class TestTrainModel:
    def test_returns_fitted_pipeline_and_time(self, simple_data):
        X_train, X_test, y_train, y_test = simple_data
        pipe = build_pipeline("Decision Tree", HP_DEFAULTS["Decision Tree"], SELECTED_FEATURES)
        fitted, elapsed = train_model(pipe, X_train, y_train)
        assert isinstance(fitted, Pipeline)
        assert isinstance(elapsed, float)
        assert elapsed >= 0.0

    def test_fitted_pipeline_scores_above_chance(self, simple_data):
        X_train, X_test, y_train, y_test = simple_data
        pipe = build_pipeline("Random Forest", HP_DEFAULTS["Random Forest"], SELECTED_FEATURES)
        fitted, _ = train_model(pipe, X_train, y_train)
        accuracy = fitted.score(X_test, y_test)
        assert accuracy > 0.5, f"Expected above-chance accuracy, got {accuracy}"


# ── cross_validate_model ──────────────────────────────────────────────────────


class TestCrossValidateModel:
    def test_returns_mean_and_std(self, simple_data):
        X_train, X_test, y_train, y_test = simple_data
        all_X = pd.concat([X_train, X_test])
        all_y = pd.concat([y_train, y_test])
        pipe = build_pipeline("Logistic Regression", HP_DEFAULTS["Logistic Regression"], SELECTED_FEATURES)
        results = cross_validate_model(pipe, all_X, all_y, cv=3)
        for metric in ["accuracy", "f1", "roc_auc"]:
            assert f"cv_{metric}_mean" in results
            assert f"cv_{metric}_std" in results

    def test_cv_accuracy_in_range(self, simple_data):
        X_train, X_test, y_train, y_test = simple_data
        all_X = pd.concat([X_train, X_test])
        all_y = pd.concat([y_train, y_test])
        pipe = build_pipeline("Decision Tree", HP_DEFAULTS["Decision Tree"], SELECTED_FEATURES)
        results = cross_validate_model(pipe, all_X, all_y, cv=3)
        assert 0.0 <= results["cv_accuracy_mean"] <= 1.0
