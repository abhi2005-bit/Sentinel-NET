# Anomaly Detection

`src/anomaly/isolation_forest_detector.py` uses `sklearn.ensemble.IsolationForest`
on monitoring features:

- fraud probability
- prediction confidence
- latency in milliseconds
- CPU percent
- memory percent
- request rate

The normalized anomaly score is derived from the baseline decision-score
distribution. Run `python scripts/evaluate_anomaly_detector.py` to tune
contamination and measure false-positive rate on controlled scenarios.
