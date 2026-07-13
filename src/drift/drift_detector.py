"""KS-test based drift detection with Benjamini-Hochberg correction."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


@dataclass
class DriftResult:
    timestamp: str
    overall_drift_detected: bool
    overall_drift_score: float
    drifted_feature_count: int
    total_features: int
    feature_results: Dict[str, Dict]
    prediction_drift: Dict
    confidence_drift: Dict

    def to_dict(self) -> Dict:
        payload = asdict(self)
        payload["drift_score"] = payload["overall_drift_score"]
        payload["details_json"] = json.dumps(
            {
                "feature_results": self.feature_results,
                "prediction_drift": self.prediction_drift,
                "confidence_drift": self.confidence_drift,
            }
        )
        return payload


class DriftDetector:
    def __init__(self, reference_data: pd.DataFrame, p_value_threshold: float = 0.05):
        self.reference_data = reference_data.copy()
        self.p_value_threshold = p_value_threshold
        self.reference_predictions: list[float] = []
        self.reference_confidence: list[float] = []

    def set_reference_outputs(self, predictions, confidence) -> None:
        self.reference_predictions = list(predictions)
        self.reference_confidence = list(confidence)

    def detect(
        self,
        current_data: pd.DataFrame,
        current_predictions=None,
        current_confidence=None,
    ) -> DriftResult:
        common_columns = [column for column in self.reference_data.columns if column in current_data.columns]
        raw_results = {}
        p_values = []
        for column in common_columns:
            stat, p_value = ks_2samp(self.reference_data[column].dropna(), current_data[column].dropna())
            raw_results[column] = {"statistic": float(stat), "p_value": float(p_value), "drift_detected": False}
            p_values.append(float(p_value))

        rejected = self._benjamini_hochberg(p_values, self.p_value_threshold)
        for column, is_rejected in zip(common_columns, rejected):
            raw_results[column]["drift_detected"] = bool(is_rejected)

        prediction_drift = self._distribution_drift(self.reference_predictions, current_predictions)
        confidence_drift = self._distribution_drift(self.reference_confidence, current_confidence)
        drifted_count = sum(1 for result in raw_results.values() if result["drift_detected"])
        total = max(len(raw_results), 1)
        score = drifted_count / total
        overall = drifted_count > 0 or prediction_drift["drift_detected"] or confidence_drift["drift_detected"]
        return DriftResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            overall_drift_detected=bool(overall),
            overall_drift_score=float(score),
            drifted_feature_count=int(drifted_count),
            total_features=int(len(raw_results)),
            feature_results=raw_results,
            prediction_drift=prediction_drift,
            confidence_drift=confidence_drift,
        )

    def _distribution_drift(self, reference, current) -> Dict:
        if reference is None or current is None or len(reference) < 2 or len(current) < 2:
            return {"statistic": None, "p_value": None, "drift_detected": False}
        stat, p_value = ks_2samp(reference, current)
        return {"statistic": float(stat), "p_value": float(p_value), "drift_detected": bool(p_value < self.p_value_threshold)}

    @staticmethod
    def _benjamini_hochberg(p_values, alpha):
        if not p_values:
            return []
        p_values = np.asarray(p_values)
        order = np.argsort(p_values)
        ranked = p_values[order]
        thresholds = alpha * (np.arange(1, len(p_values) + 1) / len(p_values))
        passed = ranked <= thresholds
        rejected = np.zeros(len(p_values), dtype=bool)
        if np.any(passed):
            max_idx = np.max(np.where(passed))
            rejected[order[: max_idx + 1]] = True
        return rejected.tolist()
