# Code Walkthrough

1. A Streamlit button in `app.py` calls `process_batch`.
2. `TransactionStreamSimulator.process_batch` selects transaction rows.
3. Scenario functions optionally inject feature drift or operational anomalies.
4. `MonitoringPipeline.process` calls `FraudModel.predict_proba`.
5. The pipeline measures latency, CPU, memory, confidence, and fraud probability.
6. `IsolationForestDetector.raw_score` and `score` produce raw and normalized anomaly scores.
7. `EventDatabase.save_monitoring_event` stores the monitoring event.
8. `AlertEngine.evaluate_event` creates anomaly, latency, or confidence alerts.
9. Once the rolling window is large enough, `DriftDetector.detect` compares reference and current windows.
10. Drift events and alerts are stored in SQLite.
11. `app.py` renders tables and charts from session/database-backed events.
