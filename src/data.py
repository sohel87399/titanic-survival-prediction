"""
src/data.py
-----------
Handles downloading, caching and loading the Titanic dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from config import TITANIC_LOCAL, TITANIC_URL

logger = logging.getLogger(__name__)


def download_titanic(url: str = TITANIC_URL, dest: Path = TITANIC_LOCAL) -> Path:
    """Download the Titanic CSV to *dest* if it is not already cached.

    Parameters
    ----------
    url:
        Remote URL of the dataset.
    dest:
        Local path to save the file.

    Returns
    -------
    Path
        The local path of the cached file.

    Raises
    ------
    RuntimeError
        If the download fails and no local cache exists.
    """
    if dest.exists():
        logger.info("Dataset already cached at %s", dest)
        return dest

    logger.info("Downloading dataset from %s …", url)
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(response.content)
        logger.info("Dataset saved to %s", dest)
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Failed to download dataset: {exc}. "
            "Make sure you have an internet connection for the first run."
        ) from exc

    return dest


def load_raw_data(
    url: str = TITANIC_URL,
    local_path: Path = TITANIC_LOCAL,
) -> pd.DataFrame:
    """Load the raw Titanic DataFrame, downloading it first if necessary.

    Parameters
    ----------
    url:
        Remote URL of the CSV.
    local_path:
        Local cache path.

    Returns
    -------
    pd.DataFrame
        Raw Titanic dataset.
    """
    path = download_titanic(url=url, dest=local_path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows × %d columns from %s", *df.shape, path)
    return df


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary of missing values per column.

    Parameters
    ----------
    df:
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: ['Column', 'Missing', 'Pct Missing'].
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    pct = (missing / len(df) * 100).round(2)
    return pd.DataFrame(
        {"Column": missing.index, "Missing": missing.values, "Pct Missing": pct.values}
    )


def sample_data(
    df: pd.DataFrame, n: int, random_state: int = 42
) -> pd.DataFrame:
    """Return a random sample of *n* rows (or the full DataFrame if n ≥ len).

    Parameters
    ----------
    df:
        Input DataFrame.
    n:
        Number of rows to sample.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Sampled (or full) DataFrame, index reset.
    """
    if n >= len(df):
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=random_state).reset_index(drop=True)
