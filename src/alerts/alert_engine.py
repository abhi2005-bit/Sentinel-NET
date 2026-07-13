"""Alert generation for monitoring and drift events."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List

from src.config import Settings
from src.drift.drift_detector import DriftResult
from src.monitoring.monitor import MonitoringEvent


@dataclass
class Alert:
    id: str
    timestamp: str
    alert_type: str
    severity: str
    message: str
    status: str = "OPEN"

    def to_dict(self) -> Dict:
        return asdict(self)


class AlertEngine:
    def __init__(
        self,
        anomaly_threshold: float = 0.75,
        repeated_anomaly_count: int = 3,
        latency_warning_ms: float = 100.0,
        latency_critical_ms: float = 250.0,
    ):
        self.anomaly_threshold = anomaly_threshold
        self.repeated_anomaly_count = repeated_anomaly_count
        self.latency_warning_ms = latency_warning_ms
        self.latency_critical_ms = latency_critical_ms
        self._recent_anomalies: list[bool] = []

    @classmethod
    def from_settings(cls, settings: Settings) -> "AlertEngine":
        return cls(
            anomaly_threshold=settings.anomaly_threshold,
            latency_warning_ms=settings.latency_warning_ms,
            latency_critical_ms=settings.latency_critical_ms,
        )

    def evaluate_event(self, event: MonitoringEvent) -> List[Alert]:
        alerts: List[Alert] = []
        self._recent_anomalies.append(event.is_anomaly)
        self._recent_anomalies = self._recent_anomalies[-self.repeated_anomaly_count :]
        if event.is_anomaly and event.anomaly_score >= self.anomaly_threshold:
            alerts.append(self._alert("ANOMALY_SCORE", "CRITICAL", f"Anomaly score {event.anomaly_score:.3f} exceeded threshold."))
        if len(self._recent_anomalies) == self.repeated_anomaly_count and all(self._recent_anomalies):
            alerts.append(self._alert("REPEATED_ANOMALIES", "CRITICAL", "Repeated anomalous monitoring events detected."))
        if event.latency_ms >= self.latency_critical_ms:
            alerts.append(self._alert("LATENCY", "CRITICAL", f"Inference latency {event.latency_ms:.2f} ms is critical."))
        elif event.latency_ms >= self.latency_warning_ms:
            alerts.append(self._alert("LATENCY", "WARNING", f"Inference latency {event.latency_ms:.2f} ms is elevated."))
        if event.confidence < 0.6:
            alerts.append(self._alert("CONFIDENCE_DEGRADATION", "WARNING", f"Prediction confidence dropped to {event.confidence:.3f}."))
        return alerts

    def evaluate_drift(self, drift: DriftResult) -> List[Alert]:
        alerts: List[Alert] = []
        if drift.drifted_feature_count:
            alerts.append(self._alert("FEATURE_DRIFT", "WARNING", f"{drift.drifted_feature_count} features drifted."))
        if drift.prediction_drift.get("drift_detected"):
            alerts.append(self._alert("PREDICTION_DRIFT", "WARNING", "Prediction distribution drift detected."))
        if drift.confidence_drift.get("drift_detected"):
            alerts.append(self._alert("CONFIDENCE_DRIFT", "WARNING", "Confidence distribution drift detected."))
        return alerts

    @staticmethod
    def _alert(alert_type: str, severity: str, message: str) -> Alert:
        return Alert(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            alert_type=alert_type,
            severity=severity,
            message=message,
        )
