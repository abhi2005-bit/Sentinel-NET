# Interview Guide

## What problem does Sentinel-NET solve?

It monitors a deployed fraud model so abnormal behaviour, latency spikes,
confidence degradation, and data drift can be detected after deployment.

## Why Random Forest?

Random Forest is strong for tabular data, handles nonlinear patterns, and gives
a reliable baseline for imbalanced fraud classification when class weights are used.

## Why is accuracy misleading?

Fraud data is highly imbalanced. A model can predict every transaction as
legitimate and still get high accuracy while missing fraud.

## Why Isolation Forest?

Isolation Forest works well for unsupervised anomaly detection. It isolates
unusual points with fewer tree splits than normal points.

## What features are sent to Isolation Forest?

Fraud probability, confidence, latency, CPU percent, memory percent, and request rate.

## How is the anomaly score calculated?

The raw Isolation Forest decision score is normalized against baseline score
quantiles so lower-than-normal raw scores become higher severity values.

## How did you reduce false positives?

The evaluator tunes contamination on a validation split and reports false
positive rate on a held-out controlled test split.

## What is data drift?

Data drift is a change in input feature distributions between training/reference
data and current production traffic.

## What is concept drift?

Concept drift means the relationship between features and labels changes. This
project detects distribution drift, prediction drift, and confidence drift, not
confirmed label-based concept drift.

## How does the KS test work?

It compares two empirical distributions and returns a statistic and p-value for
whether they likely come from the same distribution.

## Why reference and current windows?

The reference window represents expected behaviour. The current rolling window
represents recent traffic. Comparing them catches distribution shifts over time.

## How does real-time monitoring work?

The simulator feeds transactions through the Random Forest model, records
metrics, scores the monitoring vector, stores the event, and updates the dashboard.

## How is latency measured?

`time.perf_counter()` measures elapsed prediction time in milliseconds.

## How are CPU and memory collected?

`psutil.cpu_percent()` and `psutil.virtual_memory().percent` collect host metrics.

## What happens on anomaly or drift?

The alert engine creates structured alerts and the database persists them for the dashboard.

## Limitations and scaling

The current version uses SQLite and local Streamlit. Production scaling would
replace SQLite with a service database or event store, add model registry
integration, schedule drift jobs, and deploy monitoring workers separately.
