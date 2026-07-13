import pandas as pd
import pytest

from src.alerts.alert_engine import AlertEngine
from src.anomaly.isolation_forest_detector import IsolationForestDetector
from src.data.data_loader import EXPECTED_COLUMNS, FEATURE_COLUMNS, make_demo_creditcard_frame, validate_creditcard_dataset
from src.drift.drift_detector import DriftDetector
from src.evaluation.anomaly_evaluator import make_baseline_monitoring_events
from src.models.fraud_model import FraudModel
from src.monitoring.monitor import MonitoringPipeline
from src.storage.database import EventDatabase


def test_dataset_validation(tmp_path):
    frame = make_demo_creditcard_frame(100)
    path = tmp_path / "creditcard.csv"
    frame.to_csv(path, index=False)
    report = validate_creditcard_dataset(path)
    assert report.rows == 100
    assert set(EXPECTED_COLUMNS).issubset(frame.columns)


def test_dataset_validation_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"x": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        validate_creditcard_dataset(path)


def trained_fraud_model():
    frame = make_demo_creditcard_frame(500)
    model = FraudModel()
    model.fit(frame[FEATURE_COLUMNS], frame["Class"])
    return model, frame


def test_random_forest_model_loading_and_prediction(tmp_path):
    model, frame = trained_fraud_model()
    path = tmp_path / "rf.pkl"
    model.save(path)
    loaded = FraudModel.load(path)
    probabilities = loaded.predict_proba(frame[FEATURE_COLUMNS].head(5))
    predictions = loaded.predict(frame[FEATURE_COLUMNS].head(5))
    assert probabilities.shape == (5, 2)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert set(predictions).issubset({0, 1})


def test_monitoring_event_creation():
    model, frame = trained_fraud_model()
    detector = IsolationForestDetector().fit(make_baseline_monitoring_events(300))
    monitor = MonitoringPipeline(model, detector)
    event = monitor.process(frame[FEATURE_COLUMNS].iloc[0], request_rate=20)
    assert event.request_id
    assert 0 <= event.fraud_probability <= 1
    assert 0 <= event.anomaly_score <= 1


def test_isolation_forest_prediction_and_score_increase():
    baseline = make_baseline_monitoring_events(500)
    detector = IsolationForestDetector(contamination=0.01).fit(baseline)
    normal_score = detector.score(baseline.head(1))[0]
    abnormal = baseline.head(1).copy()
    abnormal["latency_ms"] = 500
    abnormal["cpu_percent"] = 99
    abnormal["request_rate"] = 250
    abnormal_score = detector.score(abnormal)[0]
    assert 0 <= normal_score <= 1
    assert 0 <= abnormal_score <= 1
    assert abnormal_score >= normal_score
    assert detector.predict(abnormal)[0] in (-1, 1)


def test_drift_detector_identical_and_shifted():
    reference = make_demo_creditcard_frame(600, random_state=1)[FEATURE_COLUMNS]
    detector = DriftDetector(reference)
    same = reference.sample(300, random_state=2)
    same_result = detector.detect(same)
    shifted = same.copy()
    shifted["Amount"] = shifted["Amount"] * 8 + 500
    shifted["V1"] = shifted["V1"] + 4
    drift_result = detector.detect(shifted)
    assert same_result.overall_drift_score <= drift_result.overall_drift_score
    assert drift_result.overall_drift_detected


def test_alert_generation():
    engine = AlertEngine(anomaly_threshold=0.5, latency_warning_ms=10, latency_critical_ms=50)
    model, frame = trained_fraud_model()
    detector = IsolationForestDetector().fit(make_baseline_monitoring_events(300))
    monitor = MonitoringPipeline(model, detector)
    event = monitor.process(frame[FEATURE_COLUMNS].iloc[0], scenario_overrides={"latency_ms": 99, "confidence": 0.5})
    event.anomaly_score = 0.9
    alerts = engine.evaluate_event(event)
    assert {alert.alert_type for alert in alerts} >= {"ANOMALY_SCORE", "LATENCY", "CONFIDENCE_DEGRADATION"}


def test_database_event_persistence():
    db = EventDatabase("sqlite:///:memory:")
    model, frame = trained_fraud_model()
    detector = IsolationForestDetector().fit(make_baseline_monitoring_events(300))
    event = MonitoringPipeline(model, detector).process(frame[FEATURE_COLUMNS].iloc[0])
    db.save_monitoring_event(event)
    rows = db.fetch_monitoring_events()
    assert len(rows) == 1
    assert rows[0]["request_id"] == event.request_id
