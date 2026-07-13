"""Real-time inference monitoring pipeline."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict

import pandas as pd
import psutil

from src.anomaly.isolation_forest_detector import MONITORING_FEATURES, IsolationForestDetector
from src.models.fraud_model import FraudModel


@dataclass
class MonitoringEvent:
    request_id: str
    timestamp: str
    model_version: str
    prediction: int
    fraud_probability: float
    confidence: float
    latency_ms: float
    cpu_percent: float
    memory_percent: float
    request_rate: float
    anomaly_raw_score: float
    anomaly_score: float
    is_anomaly: bool
    drift_status: str | None = None

    def to_dict(self) -> Dict:
        return asdict(self)


class MonitoringPipeline:
    def __init__(
        self,
        model: FraudModel,
        anomaly_detector: IsolationForestDetector,
        model_version: str = "rf-fraud-v1",
        anomaly_threshold: float = 0.75,
    ):
        self.model = model
        self.anomaly_detector = anomaly_detector
        self.model_version = model_version
        self.anomaly_threshold = anomaly_threshold

    def process(self, transaction: pd.Series | Dict, request_rate: float = 1.0, scenario_overrides: Dict | None = None) -> MonitoringEvent:
        request_id = str(uuid.uuid4())
        frame = pd.DataFrame([transaction])
        start = time.perf_counter()
        probabilities = self.model.predict_proba(frame)[0]
        prediction = int(probabilities[1] >= 0.5)
        latency_ms = (time.perf_counter() - start) * 1000
        fraud_probability = float(probabilities[1])
        confidence = float(max(probabilities))
        cpu_percent = float(psutil.cpu_percent(interval=None))
        memory_percent = float(psutil.virtual_memory().percent)

        overrides = scenario_overrides or {}
        latency_ms = float(overrides.get("latency_ms", latency_ms))
        cpu_percent = float(overrides.get("cpu_percent", cpu_percent))
        memory_percent = float(overrides.get("memory_percent", memory_percent))
        confidence = float(overrides.get("confidence", confidence))
        request_rate = float(overrides.get("request_rate", request_rate))

        monitoring_vector = {
            "fraud_probability": fraud_probability,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "request_rate": request_rate,
        }
        raw_score = float(self.anomaly_detector.raw_score(monitoring_vector)[0])
        anomaly_score = float(self.anomaly_detector.score(monitoring_vector)[0])
        # Isolation Forest's own convention drives classification:
        # 1 = inlier, -1 = anomaly. The normalized score is display severity.
        is_anomaly = bool(self.anomaly_detector.predict(monitoring_vector)[0] == -1)

        return MonitoringEvent(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_version=self.model_version,
            prediction=prediction,
            fraud_probability=fraud_probability,
            confidence=confidence,
            latency_ms=latency_ms,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            request_rate=request_rate,
            anomaly_raw_score=raw_score,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
        )

    @staticmethod
    def feature_frame(events: list[MonitoringEvent]) -> pd.DataFrame:
        return pd.DataFrame([{feature: getattr(event, feature) for feature in MONITORING_FEATURES} for event in events])
