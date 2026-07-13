"""Train the Random Forest fraud model on datasets/creditcard.csv."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_settings
from src.data.data_loader import load_creditcard_dataset, stratified_split
from src.models.fraud_model import FraudModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    settings = load_settings()
    X, y, report = load_creditcard_dataset(settings.dataset_path)
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    model = FraudModel().fit(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    metrics["dataset_report"] = report.__dict__
    model.save(settings.model_path)
    FraudModel.save_metrics(metrics, settings.metrics_path)
    print(metrics)


if __name__ == "__main__":
    main()
