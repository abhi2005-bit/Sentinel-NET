# Drift Detection

`src/drift/drift_detector.py` compares reference and current windows with the
two-sample Kolmogorov-Smirnov test. Feature p-values are corrected with
Benjamini-Hochberg. The drift score is:

```text
drifted_features / total_features
```

Prediction and confidence drift are tested separately.
