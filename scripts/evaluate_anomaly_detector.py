"""Tune and evaluate the Isolation Forest detector."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_settings
from src.evaluation.anomaly_evaluator import make_baseline_monitoring_events, save_anomaly_config, tune_contamination


def main() -> None:
    settings = load_settings()
    baseline = make_baseline_monitoring_events()
    config = tune_contamination(baseline, random_state=settings.isolation_random_state)
    save_anomaly_config(config, settings.anomaly_config_path)
    print(config)


if __name__ == "__main__":
    main()
