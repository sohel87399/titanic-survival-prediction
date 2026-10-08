"""
src/models.py
-------------
Model factory and training utilities.

Each model is wrapped in a full sklearn ``Pipeline`` (preprocessor + estimator)
so that hyperparameter changes never cause data leakage.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from config import DEFAULT_CV_FOLDS, DEFAULT_RANDOM_STATE
from src.features import build_preprocessor, filter_feature_lists

# Suppress convergence warnings for cleaner UI
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ── Estimator factory ─────────────────────────────────────────────────────────


def build_estimator(model_name: str, params: dict[str, Any]) -> Any:
    """Instantiate a bare estimator (no pipeline) from *model_name* and *params*.

    Parameters
    ----------
    model_name:
        One of the 8 supported model names.
    params:
        Hyperparameter dictionary (validated by the caller).

    Returns
    -------
    sklearn-compatible estimator
    """
    if model_name == "Logistic Regression":
        # sklearn ≥1.8 deprecates the 'penalty' param in favour of l1_ratio / C.
        # We translate for backwards compatibility while future-proofing.
        penalty = params.get("penalty", "l2")
        C_val = params.get("C", 1.0)
        max_iter = params.get("max_iter", 200)

        if penalty == "l1":
            return LogisticRegression(
                C=C_val, max_iter=max_iter, solver="saga",
                l1_ratio=1.0, random_state=DEFAULT_RANDOM_STATE,
            )
        if penalty == "elasticnet":
            return LogisticRegression(
                C=C_val, max_iter=max_iter, solver="saga",
                l1_ratio=0.5, random_state=DEFAULT_RANDOM_STATE,
            )
        if penalty == "none":
            return LogisticRegression(
                C=float("inf"), max_iter=max_iter, solver="lbfgs",
                random_state=DEFAULT_RANDOM_STATE,
            )
        # Default: l2
        return LogisticRegression(
            C=C_val, max_iter=max_iter, solver="lbfgs",
            random_state=DEFAULT_RANDOM_STATE,
        )

    if model_name == "K-Nearest Neighbors":
        return KNeighborsClassifier(
            n_neighbors=params.get("n_neighbors", 5),
            weights=params.get("weights", "uniform"),
            metric=params.get("metric", "minkowski"),
        )

    if model_name == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=params.get("max_depth", 5),
            min_samples_split=params.get("min_samples_split", 2),
            criterion=params.get("criterion", "gini"),
            random_state=DEFAULT_RANDOM_STATE,
        )

    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 5),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            max_features=params.get("max_features", "sqrt"),
            random_state=DEFAULT_RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "Gradient Boosting":
        return GradientBoostingClassifier(
            n_estimators=params.get("n_estimators", 100),
            learning_rate=params.get("learning_rate", 0.1),
            max_depth=params.get("max_depth", 3),
            random_state=DEFAULT_RANDOM_STATE,
        )

    if model_name == "XGBoost":
        return XGBClassifier(
            n_estimators=params.get("n_estimators", 100),
            learning_rate=params.get("learning_rate", 0.1),
            max_depth=params.get("max_depth", 3),
            subsample=params.get("subsample", 0.8),
            eval_metric="logloss",
            random_state=DEFAULT_RANDOM_STATE,
            verbosity=0,
        )

    if model_name == "SVM":
        return SVC(
            C=params.get("C", 1.0),
            kernel=params.get("kernel", "rbf"),
            gamma=params.get("gamma", "scale"),
            probability=True,
            random_state=DEFAULT_RANDOM_STATE,
        )

    if model_name == "Neural Network":
        hidden = params.get("hidden_layer_sizes", (100,))
        if isinstance(hidden, str):
            hidden = tuple(int(x) for x in hidden.split(",") if x.strip())
        return MLPClassifier(
            hidden_layer_sizes=hidden,
            activation=params.get("activation", "relu"),
            learning_rate_init=params.get("learning_rate_init", 0.001),
            max_iter=params.get("max_iter", 200),
            random_state=DEFAULT_RANDOM_STATE,
            early_stopping=True,
            validation_fraction=0.1,
        )

    raise ValueError(f"Unknown model name: {model_name!r}")


def build_pipeline(
    model_name: str,
    params: dict[str, Any],
    selected_features: list[str],
) -> Pipeline:
    """Build a full sklearn ``Pipeline`` (preprocessor + estimator).

    Parameters
    ----------
    model_name:
        Model identifier string.
    params:
        Hyperparameter dict for the estimator.
    selected_features:
        Feature names chosen by the user (used to build the preprocessor).

    Returns
    -------
    Pipeline
        Unfitted pipeline.
    """
    num_feats, cat_feats = filter_feature_lists(selected_features)
    preprocessor = build_preprocessor(
        numeric_features=num_feats,
        categorical_features=cat_feats,
    )
    estimator = build_estimator(model_name, params)
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", estimator)])


# ── Training ──────────────────────────────────────────────────────────────────


def train_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[Pipeline, float]:
    """Fit *pipeline* on training data and return it with the wall-clock time.

    Parameters
    ----------
    pipeline:
        Unfitted pipeline.
    X_train:
        Training features.
    y_train:
        Training labels.

    Returns
    -------
    Tuple[Pipeline, float]
        (fitted_pipeline, training_time_seconds)
    """
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    return pipeline, elapsed


def cross_validate_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = DEFAULT_CV_FOLDS,
    stratified: bool = True,
) -> dict[str, float]:
    """Run stratified k-fold cross-validation and return mean ± std metrics.

    Parameters
    ----------
    pipeline:
        Unfitted pipeline (will be cloned internally by sklearn).
    X:
        Full feature matrix.
    y:
        Full target vector.
    cv:
        Number of folds.
    stratified:
        Whether to use stratified splits.

    Returns
    -------
    dict[str, float]
        Keys: ``cv_accuracy_mean``, ``cv_accuracy_std``, ``cv_f1_mean``, etc.
    """
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=DEFAULT_RANDOM_STATE) if stratified else cv
    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=splitter,
        scoring=["accuracy", "f1", "roc_auc", "precision", "recall"],
        return_train_score=False,
        n_jobs=-1,
    )
    result: dict[str, float] = {}
    for metric in ["accuracy", "f1", "roc_auc", "precision", "recall"]:
        vals = scores[f"test_{metric}"]
        result[f"cv_{metric}_mean"] = float(np.mean(vals))
        result[f"cv_{metric}_std"] = float(np.std(vals))
    return result


# ── Ensemble helpers ──────────────────────────────────────────────────────────


def build_voting_ensemble(
    fitted_pipelines: dict[str, Pipeline],
    voting: str = "soft",
) -> VotingClassifier:
    """Wrap already-fitted pipelines in a ``VotingClassifier``.

    Note: sklearn's ``VotingClassifier`` refits estimators internally unless
    we pass ``estimators`` with already-fitted ones. We work around this by
    using ``voting='soft'`` and delegating predict_proba calls.

    Parameters
    ----------
    fitted_pipelines:
        Mapping of model name → fitted pipeline.
    voting:
        ``'soft'`` (probability averaging) or ``'hard'``.

    Returns
    -------
    VotingClassifier
    """
    estimators = [(name, pipe) for name, pipe in fitted_pipelines.items()]
    return VotingClassifier(estimators=estimators, voting=voting, n_jobs=-1)


def build_stacking_ensemble(
    fitted_pipelines: dict[str, Pipeline],
    meta_estimator: Any | None = None,
) -> StackingClassifier:
    """Build a ``StackingClassifier`` from base pipelines.

    Parameters
    ----------
    fitted_pipelines:
        Base learners (name → fitted pipeline).
    meta_estimator:
        Meta-learner.  Defaults to ``LogisticRegression``.

    Returns
    -------
    StackingClassifier
    """
    if meta_estimator is None:
        meta_estimator = LogisticRegression(max_iter=500, random_state=DEFAULT_RANDOM_STATE)
    estimators = [(name, pipe) for name, pipe in fitted_pipelines.items()]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=meta_estimator,
        cv=5,
        n_jobs=-1,
    )


# ── Staged/sweep utilities (for Training Insights tab) ───────────────────────


def rf_accuracy_sweep(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict[str, Any],
    selected_features: list[str],
    max_estimators: int = 200,
    step: int = 10,
) -> Tuple[list[int], list[float], list[float]]:
    """Compute train/test accuracy of a Random Forest as n_estimators grows.

    Returns
    -------
    Tuple[list[int], list[float], list[float]]
        (n_estimators_values, train_accuracies, test_accuracies)
    """
    n_vals, train_accs, test_accs = [], [], []
    for n in range(step, max_estimators + 1, step):
        p = {**params, "n_estimators": n}
        pipe = build_pipeline("Random Forest", p, selected_features)
        pipe.fit(X_train, y_train)
        train_accs.append(float(pipe.score(X_train, y_train)))
        test_accs.append(float(pipe.score(X_test, y_test)))
        n_vals.append(n)
    return n_vals, train_accs, test_accs


def gb_staged_scores(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict[str, Any],
    selected_features: list[str],
) -> Tuple[list[int], list[float], list[float]]:
    """Compute Gradient Boosting staged train/test accuracy per round.

    Returns
    -------
    Tuple
        (rounds, train_accs, test_accs)
    """
    from sklearn.metrics import accuracy_score

    num_feats, cat_feats = filter_feature_lists(selected_features)
    preprocessor = build_preprocessor(num_feats, cat_feats)

    gb = GradientBoostingClassifier(
        n_estimators=params.get("n_estimators", 100),
        learning_rate=params.get("learning_rate", 0.1),
        max_depth=params.get("max_depth", 3),
        random_state=DEFAULT_RANDOM_STATE,
    )
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", gb)])
    pipe.fit(X_train, y_train)

    X_train_t = pipe.named_steps["preprocessor"].transform(X_train)
    X_test_t = pipe.named_steps["preprocessor"].transform(X_test)

    train_accs, test_accs, rounds = [], [], []
    for i, y_pred_train in enumerate(gb.staged_predict(X_train_t)):
        y_pred_test = list(gb.staged_predict(X_test_t))[i]
        train_accs.append(accuracy_score(y_train, y_pred_train))
        test_accs.append(accuracy_score(y_test, y_pred_test))
        rounds.append(i + 1)

    return rounds, train_accs, test_accs


def xgb_eval_results(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict[str, Any],
    selected_features: list[str],
) -> Tuple[list[int], list[float], list[float]]:
    """Compute XGBoost train/test logloss per boosting round.

    Returns
    -------
    Tuple
        (rounds, train_logloss, test_logloss)
    """
    num_feats, cat_feats = filter_feature_lists(selected_features)
    preprocessor = build_preprocessor(num_feats, cat_feats)

    X_train_t = preprocessor.fit_transform(X_train, y_train)
    X_test_t = preprocessor.transform(X_test)

    xgb = XGBClassifier(
        n_estimators=params.get("n_estimators", 100),
        learning_rate=params.get("learning_rate", 0.1),
        max_depth=params.get("max_depth", 3),
        subsample=params.get("subsample", 0.8),
        eval_metric="logloss",
        random_state=DEFAULT_RANDOM_STATE,
        verbosity=0,
    )
    eval_set = [(X_train_t, y_train), (X_test_t, y_test)]
    xgb.fit(X_train_t, y_train, eval_set=eval_set, verbose=False)

    results = xgb.evals_result()
    train_loss = results["validation_0"]["logloss"]
    test_loss = results["validation_1"]["logloss"]
    rounds = list(range(1, len(train_loss) + 1))
    return rounds, train_loss, test_loss
