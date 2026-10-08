"""
src/features.py
---------------
Feature engineering and sklearn preprocessing pipeline.

Design principles
~~~~~~~~~~~~~~~~~
* All imputation and scaling is fitted **only on training data** – transformers
  are passed through ``Pipeline`` so they are never fit on test rows.
* Engineered columns are added before the ``ColumnTransformer`` so that the
  same transformer sees them at predict time.
* Each public function is side-effect free; it returns new DataFrames.
"""

from __future__ import annotations

import re
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    AGE_BINS,
    AGE_LABELS,
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COL,
    TITLE_MAP,
)


# ── Low-level feature engineering ────────────────────────────────────────────


def extract_title(name: str) -> str:
    """Extract the social title from a passenger name string.

    Parameters
    ----------
    name:
        Raw Name field (e.g. ``'Braund, Mr. Owen Harris'``).

    Returns
    -------
    str
        Normalised title: one of ``Mr / Miss / Mrs / Master / Rare``.
    """
    match = re.search(r",\s*([^.]+)\.", name)
    if match:
        raw_title = match.group(1).strip()
        return TITLE_MAP.get(raw_title, "Rare")
    return "Rare"


def engineer_features(df: pd.DataFrame, selected: list[str] | None = None) -> pd.DataFrame:
    """Add engineered columns to *df* in-place (returns a copy).

    Parameters
    ----------
    df:
        Raw (or semi-processed) DataFrame that still contains ``Name`` and
        ``Cabin``.
    selected:
        Which engineered features to include.  ``None`` means all.

    Returns
    -------
    pd.DataFrame
        DataFrame augmented with engineered columns.
    """
    df = df.copy()
    all_eng = ["Title", "FamilySize", "IsAlone", "FarePerPerson", "AgeBin", "HasCabin"]
    if selected is None:
        selected = all_eng

    if "Title" in selected and "Name" in df.columns:
        df["Title"] = df["Name"].apply(extract_title)

    if "FamilySize" in selected:
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    if "IsAlone" in selected:
        family_col = df["FamilySize"] if "FamilySize" in df.columns else df["SibSp"] + df["Parch"] + 1
        df["IsAlone"] = (family_col == 1).astype(int)

    if "FarePerPerson" in selected:
        family_col = df["FamilySize"] if "FamilySize" in df.columns else df["SibSp"] + df["Parch"] + 1
        df["FarePerPerson"] = df["Fare"] / family_col.replace(0, 1)

    if "AgeBin" in selected:
        df["AgeBin"] = pd.cut(
            df["Age"],
            bins=AGE_BINS,
            labels=AGE_LABELS,
            right=False,
        ).astype(str)
        df["AgeBin"] = df["AgeBin"].replace("nan", "Unknown")

    if "HasCabin" in selected:
        df["HasCabin"] = df["Cabin"].notna().astype(int) if "Cabin" in df.columns else 0

    return df


# ── Pipeline builder ──────────────────────────────────────────────────────────


def build_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """Build a ``ColumnTransformer`` for numeric + categorical features.

    Numeric pipeline:
        median imputation → standard scaling.

    Categorical pipeline:
        most-frequent imputation → one-hot encoding (``handle_unknown='ignore'``).

    Parameters
    ----------
    numeric_features:
        List of numeric column names.  Defaults to ``config.NUMERIC_FEATURES``.
    categorical_features:
        List of categorical column names.  Defaults to ``config.CATEGORICAL_FEATURES``.

    Returns
    -------
    ColumnTransformer
        Unfitted transformer.
    """
    if numeric_features is None:
        numeric_features = list(NUMERIC_FEATURES)
    if categorical_features is None:
        categorical_features = list(CATEGORICAL_FEATURES)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    # Only include columns that are present in both lists
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names_out(preprocessor: ColumnTransformer) -> list[str]:
    """Return human-readable feature names after fitting the preprocessor.

    Parameters
    ----------
    preprocessor:
        A **fitted** ``ColumnTransformer``.

    Returns
    -------
    list[str]
        Feature names corresponding to the transformed columns.
    """
    names: list[str] = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder" or transformer == "drop":
            continue
        if hasattr(transformer, "get_feature_names_out"):
            names.extend(transformer.get_feature_names_out())
        elif hasattr(transformer, "named_steps"):
            last_step = list(transformer.named_steps.values())[-1]
            if hasattr(last_step, "get_feature_names_out"):
                try:
                    names.extend(last_step.get_feature_names_out(cols))
                except TypeError:
                    names.extend(last_step.get_feature_names_out())
            else:
                names.extend(cols)
        else:
            names.extend(cols if isinstance(cols, list) else [cols])
    return names


def prepare_xy(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = TARGET_COL,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split *df* into feature matrix X and target vector y.

    Parameters
    ----------
    df:
        Processed DataFrame (with engineered features).
    feature_cols:
        Columns to use as features.  Defaults to ``config.ALL_FEATURES``.
    target_col:
        Target column name.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        (X, y)
    """
    if feature_cols is None:
        feature_cols = list(ALL_FEATURES)

    # Only keep columns that exist in the dataframe
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].copy()
    y = df[target_col].copy()
    return X, y


def filter_feature_lists(
    selected_features: list[str],
) -> Tuple[list[str], list[str]]:
    """Split *selected_features* into numeric and categorical sub-lists.

    Parameters
    ----------
    selected_features:
        The user-chosen feature names (subset of ALL_FEATURES).

    Returns
    -------
    Tuple[list[str], list[str]]
        (numeric_cols, categorical_cols)
    """
    num = [f for f in selected_features if f in NUMERIC_FEATURES]
    cat = [f for f in selected_features if f in CATEGORICAL_FEATURES]
    return num, cat
