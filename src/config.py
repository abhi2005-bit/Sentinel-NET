"""Central configuration loading for Sentinel-NET."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class Settings:
    dataset_path: Path
    model_path: Path
    anomaly_model_path: Path
    metrics_path: Path
    anomaly_config_path: Path
    model_version: str
    isolation_contamination: float
    isolation_random_state: int
    anomaly_threshold: float
    drift_p_value: float
    drift_window_size: int
    drift_check_interval: int
    latency_warning_ms: float
    latency_critical_ms: float
    simulation_delay_seconds: float
    database_url: str


def _get(config: Dict[str, Any], dotted_key: str, default: Any) -> Any:
    current: Any = config
    for key in dotted_key.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def load_settings(path: str | Path = "config/settings.yaml") -> Settings:
    config_path = Path(path)
    config: Dict[str, Any] = {}
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text()) or {}

    def env_or_config(env_name: str, dotted_key: str, default: Any) -> Any:
        return os.getenv(env_name, _get(config, dotted_key, default))

    return Settings(
        dataset_path=Path(env_or_config("SENTINEL_DATASET_PATH", "paths.dataset", "datasets/creditcard.csv")),
        model_path=Path(env_or_config("SENTINEL_MODEL_PATH", "paths.model", "artifacts/random_forest.pkl")),
        anomaly_model_path=Path(env_or_config("SENTINEL_ANOMALY_MODEL_PATH", "paths.anomaly_model", "artifacts/isolation_forest.pkl")),
        metrics_path=Path(env_or_config("SENTINEL_METRICS_PATH", "paths.metrics", "artifacts/model_metrics.json")),
        anomaly_config_path=Path(env_or_config("SENTINEL_ANOMALY_CONFIG_PATH", "paths.anomaly_config", "config/anomaly_config.json")),
        model_version=str(env_or_config("SENTINEL_MODEL_VERSION", "model.version", "rf-fraud-v1")),
        isolation_contamination=float(env_or_config("SENTINEL_ISOLATION_CONTAMINATION", "anomaly.contamination", 0.005)),
        isolation_random_state=int(env_or_config("SENTINEL_ISOLATION_RANDOM_STATE", "anomaly.random_state", 42)),
        anomaly_threshold=float(env_or_config("SENTINEL_ANOMALY_THRESHOLD", "anomaly.threshold", 0.75)),
        drift_p_value=float(env_or_config("SENTINEL_DRIFT_P_VALUE", "drift.p_value", 0.05)),
        drift_window_size=int(env_or_config("SENTINEL_DRIFT_WINDOW_SIZE", "drift.window_size", 500)),
        drift_check_interval=int(env_or_config("SENTINEL_DRIFT_CHECK_INTERVAL", "drift.check_interval", 100)),
        latency_warning_ms=float(env_or_config("SENTINEL_LATENCY_WARNING_MS", "alerts.latency_warning_ms", 100.0)),
        latency_critical_ms=float(env_or_config("SENTINEL_LATENCY_CRITICAL_MS", "alerts.latency_critical_ms", 250.0)),
        simulation_delay_seconds=float(env_or_config("SENTINEL_SIMULATION_DELAY", "simulation.delay_seconds", 0.0)),
        database_url=str(env_or_config("SENTINEL_DATABASE_URL", "storage.database_url", "sqlite:///artifacts/sentinel_net.db")),
    )
