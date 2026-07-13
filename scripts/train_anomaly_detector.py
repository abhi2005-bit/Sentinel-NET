"""Train Isolation Forest on baseline normal monitoring behaviour."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import psutil

from src.anomaly.isolation_forest_detector import IsolationForestDetector
from src.config import load_settings
from src.data.data_loader import FEATURE_COLUMNS, make_demo_creditcard_frame
from src.models.fraud_model import FraudModel


def build_model_baseline(model: FraudModel, n_rows: int = 2000) -> pd.DataFrame:
    frame = make_demo_creditcard_frame(n_rows, fraud_rate=0.005, random_state=101)[FEATURE_COLUMNS]
    rows = []
    for _, row in frame.iterrows():
        start = time.perf_counter()
        probabilities = model.predict_proba(pd.DataFrame([row]))[0]
        latency_ms = (time.perf_counter() - start) * 1000
        rows.append(
            {
                "fraud_probability": float(probabilities[1]),
                "confidence": float(max(probabilities)),
                "latency_ms": float(latency_ms),
                "cpu_percent": float(psutil.cpu_percent(interval=None)),
                "memory_percent": float(psutil.virtual_memory().percent),
                "request_rate": 20.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    settings = load_settings()
    model = FraudModel.load_or_demo(settings.model_path)
    baseline = build_model_baseline(model)
    detector = IsolationForestDetector(settings.isolation_contamination, settings.isolation_random_state).fit(baseline)
    detector.save(settings.anomaly_model_path)
    print(f"Saved anomaly detector to {settings.anomaly_model_path}")


if __name__ == "__main__":
    main()
