"""Random Forest production fraud model wrapper."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from src.data.data_loader import FEATURE_COLUMNS, make_demo_creditcard_frame

LOGGER = logging.getLogger(__name__)


class FraudModel:
    feature_columns = FEATURE_COLUMNS

    def __init__(self, estimator: RandomForestClassifier | None = None):
        self.estimator = estimator if estimator is not None else RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.is_trained = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FraudModel":
        LOGGER.info("Training Random Forest fraud model on %s rows", len(X))
        self.estimator.fit(X[self.feature_columns], y)
        self.is_trained = True
        return self

    def predict(self, features: pd.DataFrame | Dict | Iterable[Dict]) -> np.ndarray:
        frame = self._to_frame(features)
        return self.estimator.predict(frame[self.feature_columns])

    def predict_proba(self, features: pd.DataFrame | Dict | Iterable[Dict]) -> np.ndarray:
        frame = self._to_frame(features)
        return self.estimator.predict_proba(frame[self.feature_columns])

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        probabilities = self.predict_proba(X)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        cm = confusion_matrix(y, predictions, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        metrics = {
            "precision": precision_score(y, predictions, zero_division=0),
            "recall": recall_score(y, predictions, zero_division=0),
            "f1": f1_score(y, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y, probabilities),
            "pr_auc": average_precision_score(y, probabilities),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        }
        return metrics

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"estimator": self.estimator, "feature_columns": self.feature_columns}, path)
        LOGGER.info("Saved fraud model to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "FraudModel":
        payload = joblib.load(path)
        model = cls(payload["estimator"])
        model.feature_columns = payload.get("feature_columns", FEATURE_COLUMNS)
        model.is_trained = True
        return model

    @classmethod
    def load_or_demo(cls, path: str | Path) -> "FraudModel":
        path = Path(path)
        if path.exists():
            return cls.load(path)
        demo = make_demo_creditcard_frame(1500)
        model = cls(RandomForestClassifier(n_estimators=50, class_weight="balanced", random_state=42, n_jobs=-1))
        model.fit(demo[FEATURE_COLUMNS], demo["Class"])
        return model

    def demo_reference_frame(self) -> pd.DataFrame:
        return make_demo_creditcard_frame(1000, random_state=7)[FEATURE_COLUMNS]

    @staticmethod
    def save_metrics(metrics: Dict, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2))

    def _to_frame(self, features) -> pd.DataFrame:
        if isinstance(features, pd.DataFrame):
            frame = features.copy()
        elif isinstance(features, dict):
            frame = pd.DataFrame([features])
        else:
            frame = pd.DataFrame(list(features))
        missing = set(self.feature_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing fraud model features: {sorted(missing)}")
        return frame
