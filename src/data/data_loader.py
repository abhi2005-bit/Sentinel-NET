"""ULB credit card fraud dataset loading and validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)

EXPECTED_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]
FEATURE_COLUMNS = [column for column in EXPECTED_COLUMNS if column != "Class"]


@dataclass(frozen=True)
class DatasetReport:
    path: str
    rows: int
    columns: int
    missing_values: int
    class_distribution: dict


def validate_creditcard_dataset(path: str | Path) -> DatasetReport:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. Download the ULB Credit Card Fraud Detection "
            "dataset and place creditcard.csv in datasets/."
        )
    df = pd.read_csv(dataset_path, nrows=5)
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")

    full_df = pd.read_csv(dataset_path)
    missing_values = int(full_df[EXPECTED_COLUMNS].isna().sum().sum())
    class_distribution = full_df["Class"].value_counts().sort_index().to_dict()
    LOGGER.info("Validated dataset %s with distribution %s", dataset_path, class_distribution)
    return DatasetReport(
        path=str(dataset_path),
        rows=int(len(full_df)),
        columns=int(len(full_df.columns)),
        missing_values=missing_values,
        class_distribution={int(k): int(v) for k, v in class_distribution.items()},
    )


def load_creditcard_dataset(path: str | Path) -> Tuple[pd.DataFrame, pd.Series, DatasetReport]:
    dataset_path = Path(path)
    report = validate_creditcard_dataset(dataset_path)
    df = pd.read_csv(dataset_path)
    X = df[FEATURE_COLUMNS]
    y = df["Class"].astype(int)
    return X, y, report


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def make_demo_creditcard_frame(n_rows: int = 2000, fraud_rate: float = 0.02, random_state: int = 42) -> pd.DataFrame:
    """Create a small schema-compatible frame for tests and dashboard fallback.

    This is not a substitute for the ULB dataset and is never used by the training
    script unless a test explicitly passes it.
    """
    import numpy as np

    rng = np.random.default_rng(random_state)
    y = rng.binomial(1, fraud_rate, n_rows)
    data = {
        "Time": rng.uniform(0, 172800, n_rows),
        "Amount": rng.gamma(shape=2.0 + y * 1.5, scale=45.0 + y * 25.0),
        "Class": y,
    }
    for i in range(1, 29):
        shift = y * (0.25 if i <= 6 else 0.05)
        data[f"V{i}"] = rng.normal(loc=shift, scale=1.0 + y * 0.2, size=n_rows)
    return pd.DataFrame(data)[EXPECTED_COLUMNS]
