"""
tests/test_data.py
------------------
Unit tests for src/data.py
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data import get_missing_summary, load_raw_data, sample_data


# ── get_missing_summary ───────────────────────────────────────────────────────


class TestGetMissingSummary:
    def test_returns_dataframe(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, None], "c": [1, 2, 3]})
        result = get_missing_summary(df)
        assert isinstance(result, pd.DataFrame)

    def test_columns_present(self):
        df = pd.DataFrame({"a": [1, None], "b": [1, 2]})
        result = get_missing_summary(df)
        assert "Column" in result.columns
        assert "Missing" in result.columns
        assert "Pct Missing" in result.columns

    def test_only_missing_cols_included(self):
        df = pd.DataFrame({"a": [1, None], "b": [1, 2]})
        result = get_missing_summary(df)
        assert "a" in result["Column"].values
        assert "b" not in result["Column"].values

    def test_no_missing_returns_empty(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = get_missing_summary(df)
        assert len(result) == 0

    def test_percentages_correct(self):
        df = pd.DataFrame({"a": [None, 1, None, 1]})  # 50% missing
        result = get_missing_summary(df)
        assert result.loc[result["Column"] == "a", "Pct Missing"].values[0] == 50.0


# ── sample_data ───────────────────────────────────────────────────────────────


class TestSampleData:
    def test_returns_n_rows(self):
        df = pd.DataFrame({"x": range(100)})
        result = sample_data(df, n=20, random_state=42)
        assert len(result) == 20

    def test_returns_full_when_n_ge_len(self):
        df = pd.DataFrame({"x": range(50)})
        result = sample_data(df, n=100, random_state=42)
        assert len(result) == 50

    def test_index_reset(self):
        df = pd.DataFrame({"x": range(100)})
        result = sample_data(df, n=10, random_state=42)
        assert list(result.index) == list(range(10))

    def test_reproducible(self):
        df = pd.DataFrame({"x": range(200)})
        r1 = sample_data(df, n=50, random_state=0)
        r2 = sample_data(df, n=50, random_state=0)
        pd.testing.assert_frame_equal(r1, r2)

    def test_different_seeds_different_samples(self):
        df = pd.DataFrame({"x": range(200)})
        r1 = sample_data(df, n=50, random_state=0)
        r2 = sample_data(df, n=50, random_state=1)
        # Very unlikely to be identical
        assert not r1["x"].equals(r2["x"])


# ── load_raw_data (uses cache / local file) ───────────────────────────────────


class TestLoadRawData:
    def test_loads_dataframe_from_local(self, tmp_path):
        """Test that load_raw_data reads a locally cached CSV correctly."""
        # Create a minimal CSV
        csv_content = "PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked\n"
        csv_content += "1,0,3,Braund Mr. Owen,male,22,1,0,A,7.25,,S\n"
        local = tmp_path / "titanic.csv"
        local.write_text(csv_content)

        df = load_raw_data(url="http://unused", local_path=local)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "Survived" in df.columns
