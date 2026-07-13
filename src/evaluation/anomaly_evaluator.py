"""Controlled anomaly detector evaluation and contamination tuning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.anomaly.isolation_forest_detector import MONITORING_FEATURES, IsolationForestDetector


def make_baseline_monitoring_events(n_rows: int = 2000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    return pd.DataFrame(
        {
            "fraud_probability": rng.beta(1, 20, n_rows),
            "confidence": rng.uniform(0.82, 1.0, n_rows),
            "latency_ms": rng.normal(18, 5, n_rows).clip(1),
            "cpu_percent": rng.normal(25, 7, n_rows).clip(0, 100),
            "memory_percent": rng.normal(48, 6, n_rows).clip(0, 100),
            "request_rate": rng.normal(25, 4, n_rows).clip(1),
        }
    )


def make_controlled_anomaly_set(n_normal: int = 1000, n_each: int = 80, random_state: int = 43):
    normal = make_baseline_monitoring_events(n_normal, random_state)
    rng = np.random.default_rng(random_state)
    scenarios = []
    labels = [0] * len(normal)
    for name in ["latency", "confidence", "cpu", "memory", "request_rate"]:
        frame = make_baseline_monitoring_events(n_each, random_state + len(scenarios) + 1)
        if name == "latency":
            frame["latency_ms"] = rng.normal(320, 35, n_each)
        elif name == "confidence":
            frame["confidence"] = rng.uniform(0.40, 0.58, n_each)
        elif name == "cpu":
            frame["cpu_percent"] = rng.uniform(88, 99, n_each)
        elif name == "memory":
            frame["memory_percent"] = rng.uniform(86, 98, n_each)
        else:
            frame["request_rate"] = rng.uniform(180, 280, n_each)
        scenarios.append(frame)
        labels.extend([1] * n_each)
    return pd.concat([normal, *scenarios], ignore_index=True), np.asarray(labels)


def evaluate_detector(detector: IsolationForestDetector, X: pd.DataFrame, y_true: Iterable[int]) -> Dict:
    scores = detector.score(X)
    y_pred = (detector.predict(X) == -1).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def tune_contamination(
    baseline: pd.DataFrame,
    contamination_values=(0.001, 0.002, 0.005, 0.01, 0.02),
    random_state: int = 42,
    max_validation_fpr: float = 0.01,
) -> Dict:
    X_eval, y_eval = make_controlled_anomaly_set(random_state=random_state + 10)
    X_val, X_test, y_val, y_test = train_test_split(X_eval, y_eval, test_size=0.5, stratify=y_eval, random_state=random_state)
    results = {}
    best_value = None
    best_key = (-1.0, -1.0)
    fallback_value = None
    fallback_key = (-1.0, -1.0)
    for contamination in contamination_values:
        detector = IsolationForestDetector(contamination=contamination, random_state=random_state).fit(baseline)
        metrics = evaluate_detector(detector, X_val, y_val)
        results[str(contamination)] = metrics
        constrained_key = (metrics["f1"], -metrics["false_positive_rate"])
        fallback_key_candidate = (-metrics["false_positive_rate"], metrics["f1"])
        if metrics["false_positive_rate"] <= max_validation_fpr and constrained_key > best_key:
            best_key = constrained_key
            best_value = contamination
        if fallback_key_candidate > fallback_key:
            fallback_key = fallback_key_candidate
            fallback_value = contamination
    if best_value is None:
        best_value = fallback_value
    final_detector = IsolationForestDetector(contamination=best_value, random_state=random_state).fit(baseline)
    final_metrics = evaluate_detector(final_detector, X_test, y_test)
    return {
        "selected_contamination": best_value,
        "selection_rule": f"maximize validation F1 subject to false_positive_rate <= {max_validation_fpr}; otherwise choose lowest FPR",
        "validation_results": results,
        "test_metrics": final_metrics,
        "feature_columns": MONITORING_FEATURES,
    }


def save_anomaly_config(config: Dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2))
