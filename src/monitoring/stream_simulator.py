"""Transaction stream simulator with controlled anomaly and drift injection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List

import pandas as pd

from src.alerts.alert_engine import Alert, AlertEngine
from src.config import Settings
from src.data.data_loader import FEATURE_COLUMNS, make_demo_creditcard_frame
from src.drift.drift_detector import DriftResult, DriftDetector
from src.models.fraud_model import FraudModel
from src.monitoring.monitor import MonitoringEvent, MonitoringPipeline
from src.storage.database import EventDatabase


class StreamScenario(str, Enum):
    NORMAL = "Normal Traffic"
    FEATURE_DRIFT = "Inject Feature Drift"
    LATENCY_ANOMALY = "Inject Latency Anomaly"
    RESOURCE_ANOMALY = "Inject Resource Anomaly"
    REQUEST_SPIKE = "Inject Request Spike"
    CONFIDENCE_DEGRADATION = "Inject Confidence Degradation"


@dataclass
class BatchResult:
    events: List[MonitoringEvent]
    drift_events: List[DriftResult]
    alerts: List[Alert]
    next_index: int


class TransactionStreamSimulator:
    def __init__(
        self,
        model: FraudModel,
        monitor: MonitoringPipeline,
        drift_detector: DriftDetector,
        database: EventDatabase,
        alert_engine: AlertEngine,
        settings: Settings,
    ):
        self.model = model
        self.monitor = monitor
        self.drift_detector = drift_detector
        self.database = database
        self.alert_engine = alert_engine
        self.settings = settings
        self.transactions = self._load_transactions()
        self.current_window: list[pd.Series] = []
        self.current_predictions: list[int] = []
        self.current_confidence: list[float] = []

    def process_batch(self, start_index: int, batch_size: int, scenario: StreamScenario = StreamScenario.NORMAL) -> BatchResult:
        events: List[MonitoringEvent] = []
        drift_events: List[DriftResult] = []
        alerts: List[Alert] = []
        index = start_index
        for offset in range(batch_size):
            row = self.transactions.iloc[(index + offset) % len(self.transactions)].copy()
            row = self._apply_feature_scenario(row, scenario)
            event = self.monitor.process(row, request_rate=20.0, scenario_overrides=self._scenario_overrides(scenario))
            self.database.save_monitoring_event(event)
            event_alerts = self.alert_engine.evaluate_event(event)
            self.database.save_alerts(event_alerts)
            events.append(event)
            alerts.extend(event_alerts)
            self.current_window.append(row[FEATURE_COLUMNS])
            self.current_predictions.append(event.prediction)
            self.current_confidence.append(event.confidence)

        if len(self.current_window) >= self.settings.drift_window_size and len(events) >= 1:
            current_frame = pd.DataFrame(self.current_window[-self.settings.drift_window_size :])
            drift = self.drift_detector.detect(
                current_frame,
                self.current_predictions[-self.settings.drift_window_size :],
                self.current_confidence[-self.settings.drift_window_size :],
            )
            self.database.save_drift_event(drift)
            drift_alerts = self.alert_engine.evaluate_drift(drift)
            self.database.save_alerts(drift_alerts)
            drift_events.append(drift)
            alerts.extend(drift_alerts)
        return BatchResult(events=events, drift_events=drift_events, alerts=alerts, next_index=index + batch_size)

    def _load_transactions(self) -> pd.DataFrame:
        if self.settings.dataset_path.exists():
            return pd.read_csv(self.settings.dataset_path)[FEATURE_COLUMNS]
        return make_demo_creditcard_frame(3000)[FEATURE_COLUMNS]

    @staticmethod
    def _apply_feature_scenario(row: pd.Series, scenario: StreamScenario) -> pd.Series:
        if scenario == StreamScenario.FEATURE_DRIFT:
            row["Amount"] = row["Amount"] * 4 + 150
            row["V1"] = row["V1"] + 2.5
            row["V2"] = row["V2"] - 2.0
            row["V3"] = row["V3"] + 1.5
        return row

    @staticmethod
    def _scenario_overrides(scenario: StreamScenario):
        if scenario == StreamScenario.LATENCY_ANOMALY:
            return {"latency_ms": 350.0}
        if scenario == StreamScenario.RESOURCE_ANOMALY:
            return {"cpu_percent": 95.0, "memory_percent": 92.0}
        if scenario == StreamScenario.REQUEST_SPIKE:
            return {"request_rate": 250.0}
        if scenario == StreamScenario.CONFIDENCE_DEGRADATION:
            return {"confidence": 0.52}
        return {}
