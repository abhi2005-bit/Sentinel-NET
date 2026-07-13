"""Isolation Forest detector for monitoring behaviour."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)

MONITORING_FEATURES = [
    "fraud_probability",
    "confidence",
    "latency_ms",
    "cpu_percent",
    "memory_percent",
    "request_rate",
]


class IsolationForestDetector:
    """Isolation Forest with percentile-based anomaly score normalization.

    The raw Isolation Forest decision score is higher for normal observations and
    lower for abnormal observations. During fit we store the 5th and 95th
    percentiles of baseline scores. New scores are mapped as:

        severity = (q95 - score) / (q95 - q05)

    then clipped to [0, 1]. Observations below the low baseline tail approach 1.
    """

    def __init__(self, contamination: float = 0.005, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=200)
        self.score_q05 = 0.0
        self.score_q95 = 1.0
        self.baseline_median: Dict[str, float] = {}
        self.baseline_iqr: Dict[str, float] = {}
        self.is_fitted = False

    def fit(self, X: pd.DataFrame) -> "IsolationForestDetector":
        frame = self._frame(X)
        scaled = self.scaler.fit_transform(frame)
        self.model.fit(scaled)
        scores = self.model.decision_function(scaled)
        self.score_q05 = float(np.quantile(scores, 0.05))
        self.score_q95 = float(np.quantile(scores, 0.95))
        self.baseline_median = frame.median().to_dict()
        q75 = frame.quantile(0.75)
        q25 = frame.quantile(0.25)
        self.baseline_iqr = (q75 - q25).replace(0, 1e-9).to_dict()
        self.is_fitted = True
        LOGGER.info("Fitted Isolation Forest on %s baseline monitoring rows", len(frame))
        return self

    def predict(self, X: pd.DataFrame | Dict | Iterable[Dict]) -> np.ndarray:
        scaled = self.scaler.transform(self._frame(X))
        return self.model.predict(scaled)

    def raw_score(self, X: pd.DataFrame | Dict | Iterable[Dict]) -> np.ndarray:
        scaled = self.scaler.transform(self._frame(X))
        return self.model.decision_function(scaled)

    def score(self, X: pd.DataFrame | Dict | Iterable[Dict]) -> np.ndarray:
        raw = self.raw_score(X)
        denom = max(self.score_q95 - self.score_q05, 1e-9)
        return np.clip((self.score_q95 - raw) / denom, 0.0, 1.0)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "IsolationForestDetector":
        return joblib.load(path)

    @classmethod
    def load_or_demo(cls, path: str | Path) -> "IsolationForestDetector":
        path = Path(path)
        if path.exists():
            return cls.load(path)
        rng = np.random.default_rng(42)
        baseline = pd.DataFrame(
            {
                "fraud_probability": rng.beta(1, 20, 1000),
                "confidence": rng.uniform(0.85, 1.0, 1000),
                "latency_ms": rng.normal(15, 4, 1000).clip(1),
                "cpu_percent": rng.normal(20, 5, 1000).clip(0, 100),
                "memory_percent": rng.normal(45, 4, 1000).clip(0, 100),
                "request_rate": rng.normal(20, 3, 1000).clip(1),
            }
        )
        return cls().fit(baseline)

    def explain_deviation(self, event: Dict, top_n: int = 5) -> List[Dict]:
        rows = []
        for feature in MONITORING_FEATURES:
            value = float(event.get(feature, 0.0))
            median = float(self.baseline_median.get(feature, 0.0))
            iqr = float(self.baseline_iqr.get(feature, 1.0))
            rows.append(
                {
                    "feature": feature,
                    "value": value,
                    "baseline_median": median,
                    "robust_deviation": abs(value - median) / max(iqr, 1e-9),
                }
            )
        return sorted(rows, key=lambda item: item["robust_deviation"], reverse=True)[:top_n]

    def _frame(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
        elif isinstance(X, dict):
            frame = pd.DataFrame([X])
        else:
            frame = pd.DataFrame(list(X))
        missing = set(MONITORING_FEATURES) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing monitoring features: {sorted(missing)}")
        return frame[MONITORING_FEATURES]
