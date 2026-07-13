"""SQLite event storage using SQLAlchemy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from sqlalchemy import Boolean, Column, Float, Integer, MetaData, String, Table, Text, create_engine, delete, insert, select

from src.alerts.alert_engine import Alert
from src.drift.drift_detector import DriftResult
from src.monitoring.monitor import MonitoringEvent

metadata = MetaData()

monitoring_events = Table(
    "monitoring_events",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("request_id", String, unique=True),
    Column("timestamp", String),
    Column("prediction", Integer),
    Column("fraud_probability", Float),
    Column("confidence", Float),
    Column("latency_ms", Float),
    Column("cpu_percent", Float),
    Column("memory_percent", Float),
    Column("request_rate", Float),
    Column("anomaly_score", Float),
    Column("anomaly_raw_score", Float),
    Column("is_anomaly", Boolean),
)

drift_events = Table(
    "drift_events",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("timestamp", String),
    Column("drift_score", Float),
    Column("drifted_feature_count", Integer),
    Column("prediction_drift", Boolean),
    Column("confidence_drift", Boolean),
    Column("details_json", Text),
)

alerts = Table(
    "alerts",
    metadata,
    Column("id", String, primary_key=True),
    Column("timestamp", String),
    Column("alert_type", String),
    Column("severity", String),
    Column("message", Text),
    Column("status", String),
)


class EventDatabase:
    def __init__(self, database_url: str = "sqlite:///artifacts/sentinel_net.db"):
        if database_url.startswith("sqlite:///"):
            path = Path(database_url.replace("sqlite:///", "", 1))
            if str(path) != ":memory:":
                path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(database_url, future=True)
        metadata.create_all(self.engine)

    def save_monitoring_event(self, event: MonitoringEvent) -> None:
        payload = event.to_dict()
        payload.pop("model_version", None)
        payload.pop("drift_status", None)
        with self.engine.begin() as conn:
            conn.execute(insert(monitoring_events), payload)

    def save_drift_event(self, result: DriftResult) -> None:
        payload = result.to_dict()
        with self.engine.begin() as conn:
            conn.execute(
                insert(drift_events),
                {
                    "timestamp": payload["timestamp"],
                    "drift_score": payload["overall_drift_score"],
                    "drifted_feature_count": payload["drifted_feature_count"],
                    "prediction_drift": bool(payload["prediction_drift"].get("drift_detected")),
                    "confidence_drift": bool(payload["confidence_drift"].get("drift_detected")),
                    "details_json": payload["details_json"],
                },
            )

    def save_alerts(self, new_alerts: Iterable[Alert]) -> None:
        rows = [alert.to_dict() for alert in new_alerts]
        if rows:
            with self.engine.begin() as conn:
                conn.execute(insert(alerts), rows)

    def fetch_monitoring_events(self, limit: int = 500):
        with self.engine.begin() as conn:
            return [dict(row._mapping) for row in conn.execute(select(monitoring_events).order_by(monitoring_events.c.id.desc()).limit(limit))]

    def reset(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(delete(monitoring_events))
            conn.execute(delete(drift_events))
            conn.execute(delete(alerts))
