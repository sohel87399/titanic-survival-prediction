"""
tests/test_features.py
----------------------
Unit tests for src/features.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

# Ensure project root is on path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.features import (
    build_preprocessor,
    engineer_features,
    extract_title,
    filter_feature_lists,
    get_feature_names_out,
    prepare_xy,
)
from config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COL


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Minimal Titanic-like DataFrame for testing."""
    return pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4, 5],
            "Survived": [0, 1, 1, 0, 1],
            "Pclass": [3, 1, 3, 1, 3],
            "Name": [
                "Braund, Mr. Owen Harris",
                "Cumings, Mrs. John Bradley",
                "Heikkinen, Miss. Laina",
                "Futrelle, Mrs. Jacques Heath",
                "Allen, Mr. William Henry",
            ],
            "Sex": ["male", "female", "female", "female", "male"],
            "Age": [22.0, 38.0, np.nan, 35.0, 35.0],
            "SibSp": [1, 1, 0, 1, 0],
            "Parch": [0, 0, 0, 0, 0],
            "Ticket": ["A/5 21171", "PC 17599", "STON/O2. 3101282", "113803", "373450"],
            "Fare": [7.25, 71.28, 7.92, 53.10, 8.05],
            "Cabin": [np.nan, "C85", np.nan, "C123", np.nan],
            "Embarked": ["S", "C", "S", "S", "S"],
        }
    )


# ── extract_title ──────────────────────────────────────────────────────────────


class TestExtractTitle:
    def test_mr(self):
        assert extract_title("Braund, Mr. Owen Harris") == "Mr"

    def test_mrs(self):
        assert extract_title("Cumings, Mrs. John Bradley") == "Mrs"

    def test_miss(self):
        assert extract_title("Heikkinen, Miss. Laina") == "Miss"

    def test_master(self):
        assert extract_title("Moller, Master. Mansouer") == "Master"

    def test_rare(self):
        assert extract_title("Somebody, Dr. Smith") == "Rare"

    def test_no_title(self):
        # No match → "Rare"
        assert extract_title("NoTitle") == "Rare"

    def test_mlle_mapped_to_miss(self):
        assert extract_title("Somebody, Mlle. Name") == "Miss"


# ── engineer_features ─────────────────────────────────────────────────────────


class TestEngineerFeatures:
    def test_all_features_added(self, sample_df):
        result = engineer_features(sample_df)
        for col in ["Title", "FamilySize", "IsAlone", "FarePerPerson", "AgeBin", "HasCabin"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_family_size(self, sample_df):
        result = engineer_features(sample_df)
        # Row 0: SibSp=1, Parch=0 → FamilySize=2
        assert result.loc[0, "FamilySize"] == 2

    def test_is_alone(self, sample_df):
        result = engineer_features(sample_df)
        # Passenger with SibSp=0, Parch=0 → IsAlone=1
        solo_mask = (result["SibSp"] == 0) & (result["Parch"] == 0)
        assert (result.loc[solo_mask, "IsAlone"] == 1).all()

    def test_fare_per_person(self, sample_df):
        result = engineer_features(sample_df)
        # FarePerPerson = Fare / FamilySize
        for idx in result.index:
            expected = result.loc[idx, "Fare"] / result.loc[idx, "FamilySize"]
            assert abs(result.loc[idx, "FarePerPerson"] - expected) < 1e-6

    def test_has_cabin(self, sample_df):
        result = engineer_features(sample_df)
        # Row 1 has Cabin "C85" → HasCabin=1
        assert result.loc[1, "HasCabin"] == 1
        # Row 0 has no Cabin → HasCabin=0
        assert result.loc[0, "HasCabin"] == 0

    def test_selective_features(self, sample_df):
        result = engineer_features(sample_df, selected=["Title", "FamilySize"])
        assert "Title" in result.columns
        assert "FamilySize" in result.columns
        assert "IsAlone" not in result.columns

    def test_does_not_modify_original(self, sample_df):
        original_cols = set(sample_df.columns)
        _ = engineer_features(sample_df)
        assert set(sample_df.columns) == original_cols


# ── build_preprocessor ────────────────────────────────────────────────────────


class TestBuildPreprocessor:
    def test_returns_column_transformer(self, sample_df):
        from sklearn.compose import ColumnTransformer
        preprocessor = build_preprocessor(["Age", "Fare"], ["Sex", "Pclass"])
        assert isinstance(preprocessor, ColumnTransformer)

    def test_fit_transform_shape(self, sample_df):
        df = engineer_features(sample_df)
        preprocessor = build_preprocessor(
            numeric_features=["Age", "Fare", "SibSp", "Parch"],
            categorical_features=["Sex", "Pclass"],
        )
        X = df[["Age", "Fare", "SibSp", "Parch", "Sex", "Pclass"]]
        result = preprocessor.fit_transform(X)
        # Should have numeric + one-hot encoded columns
        assert result.shape[0] == len(df)
        assert result.ndim == 2

    def test_no_leakage(self, sample_df):
        """Preprocessor fitted on train must not see test data."""
        df = engineer_features(sample_df)
        train = df.iloc[:3]
        test = df.iloc[3:]
        preprocessor = build_preprocessor(["Age", "Fare"], ["Sex"])
        preprocessor.fit(train[["Age", "Fare", "Sex"]])
        # Transform should work even with missing levels in test
        result = preprocessor.transform(test[["Age", "Fare", "Sex"]])
        assert result.shape[0] == len(test)


# ── filter_feature_lists ──────────────────────────────────────────────────────


class TestFilterFeatureLists:
    def test_splits_correctly(self):
        feats = ["Age", "Fare", "Sex", "Pclass"]
        num, cat = filter_feature_lists(feats)
        assert "Age" in num
        assert "Fare" in num
        assert "Sex" in cat
        assert "Pclass" in cat

    def test_empty_input(self):
        num, cat = filter_feature_lists([])
        assert num == []
        assert cat == []


# ── prepare_xy ────────────────────────────────────────────────────────────────


class TestPrepareXY:
    def test_splits_correctly(self, sample_df):
        df = engineer_features(sample_df)
        feature_cols = ["Age", "Fare", "Sex"]
        # Only available cols
        X, y = prepare_xy(df, feature_cols=feature_cols)
        assert list(y) == list(df[TARGET_COL])
        assert all(c in X.columns for c in feature_cols if c in df.columns)

    def test_target_not_in_X(self, sample_df):
        df = engineer_features(sample_df)
        X, y = prepare_xy(df, feature_cols=["Age", "Fare"])
        assert TARGET_COL not in X.columns

    def test_handles_missing_cols(self, sample_df):
        df = engineer_features(sample_df)
        X, y = prepare_xy(df, feature_cols=["Age", "NonExistent"])
        assert "NonExistent" not in X.columns
        assert "Age" in X.columns
