"""Sentinel-NET Streamlit dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.alerts.alert_engine import AlertEngine
from src.anomaly.isolation_forest_detector import IsolationForestDetector
from src.config import load_settings
from src.drift.drift_detector import DriftDetector
from src.models.fraud_model import FraudModel
from src.monitoring.monitor import MonitoringPipeline
from src.monitoring.stream_simulator import StreamScenario, TransactionStreamSimulator
from src.storage.database import EventDatabase


st.set_page_config(
    page_title="Sentinel-NET",
    page_icon="SN",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _metric_value(value, fallback="N/A"):
    if value is None:
        return fallback
    if isinstance(value, float):
        return f"{value:.4f}"
    return value


@st.cache_resource
def get_settings():
    return load_settings()


@st.cache_resource
def get_components():
    settings = get_settings()
    model = FraudModel.load_or_demo(settings.model_path)
    detector = IsolationForestDetector.load_or_demo(settings.anomaly_model_path)
    drift = DriftDetector(reference_data=model.demo_reference_frame())
    database = EventDatabase(settings.database_url)
    alerts = AlertEngine.from_settings(settings)
    monitor = MonitoringPipeline(
        model=model,
        anomaly_detector=detector,
        model_version=settings.model_version,
        anomaly_threshold=settings.anomaly_threshold,
    )
    simulator = TransactionStreamSimulator(
        model=model,
        monitor=monitor,
        drift_detector=drift,
        database=database,
        alert_engine=alerts,
        settings=settings,
    )
    return settings, model, detector, drift, database, alerts, simulator


def ensure_state():
    if "events" not in st.session_state:
        st.session_state.events = []
    if "drift_events" not in st.session_state:
        st.session_state.drift_events = []
    if "alerts" not in st.session_state:
        st.session_state.alerts = []
    if "stream_index" not in st.session_state:
        st.session_state.stream_index = 0


def process_batch(batch_size: int, scenario: str):
    _, _, _, _, _, _, simulator = get_components()
    result = simulator.process_batch(
        start_index=st.session_state.stream_index,
        batch_size=batch_size,
        scenario=StreamScenario(scenario),
    )
    st.session_state.stream_index = result.next_index
    st.session_state.events.extend([event.to_dict() for event in result.events])
    st.session_state.drift_events.extend([event.to_dict() for event in result.drift_events])
    st.session_state.alerts.extend([alert.to_dict() for alert in result.alerts])


def reset_session():
    _, _, _, _, database, _, _ = get_components()
    database.reset()
    st.session_state.events = []
    st.session_state.drift_events = []
    st.session_state.alerts = []
    st.session_state.stream_index = 0


ensure_state()
settings, model, detector, drift, database, alerts, simulator = get_components()

st.title("Sentinel-NET")
st.caption("AI Anomaly Detection & Model Monitoring System")

with st.sidebar:
    st.header("Simulation Control")
    scenario = st.selectbox(
        "Traffic Scenario",
        [item.value for item in StreamScenario],
        index=0,
    )
    batch_size = st.slider("Batch size", 1, 250, 50)
    if st.button("Process Batch", type="primary"):
        process_batch(batch_size, scenario)
    if st.button("Reset Monitoring Session"):
        reset_session()
    st.divider()
    st.write("Model path")
    st.code(str(settings.model_path))
    st.write("Dataset path")
    st.code(str(settings.dataset_path))

events_df = pd.DataFrame(st.session_state.events)
drift_df = pd.DataFrame(st.session_state.drift_events)
alerts_df = pd.DataFrame(st.session_state.alerts)

latest = events_df.iloc[-1].to_dict() if not events_df.empty else {}
latest_drift = drift_df.iloc[-1].to_dict() if not drift_df.empty else {}
anomaly_count = int(events_df["is_anomaly"].sum()) if "is_anomaly" in events_df else 0
avg_latency = float(events_df["latency_ms"].mean()) if "latency_ms" in events_df else None
current_score = latest.get("anomaly_score")
current_drift_score = latest_drift.get("drift_score")

health = "HEALTHY"
if current_score is not None and current_score >= settings.anomaly_threshold:
    health = "CRITICAL"
elif current_drift_score is not None and current_drift_score > 0:
    health = "WARNING"
elif avg_latency is not None and avg_latency >= settings.latency_warning_ms:
    health = "WARNING"

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Model Health", health)
c2.metric("Total Predictions", len(events_df))
c3.metric("Anomalies Detected", anomaly_count)
c4.metric("Current Anomaly Score", _metric_value(current_score))
c5.metric("Current Drift Score", _metric_value(current_drift_score))
c6.metric("Average Latency", f"{avg_latency:.2f} ms" if avg_latency is not None else "N/A")

tabs = st.tabs(
    [
        "Overview",
        "Live Monitoring",
        "Anomaly Detection",
        "Drift Monitoring",
        "Alerts",
        "Model Performance",
    ]
)

with tabs[0]:
    st.subheader("Monitoring Events")
    if events_df.empty:
        st.info("Process a batch to generate real monitoring events.")
    else:
        st.dataframe(events_df.tail(25), use_container_width=True)

with tabs[1]:
    if events_df.empty:
        st.info("No events yet.")
    else:
        chart_cols = ["anomaly_score", "latency_ms", "confidence", "cpu_percent", "memory_percent", "fraud_probability"]
        selected = st.multiselect("Signals", chart_cols, default=["anomaly_score", "latency_ms", "confidence"])
        if selected:
            st.plotly_chart(px.line(events_df.reset_index(), x="index", y=selected), use_container_width=True)
        st.plotly_chart(px.histogram(events_df, x="prediction", title="Prediction Distribution"), use_container_width=True)

with tabs[2]:
    st.metric("Latest Anomaly Score", _metric_value(current_score))
    st.metric("Anomaly Threshold", f"{settings.anomaly_threshold:.2f}")
    st.metric("Status", "ANOMALY" if latest.get("is_anomaly") else "NORMAL")
    if latest:
        explanation = detector.explain_deviation(latest)
        st.write("Why was this event anomalous?")
        st.dataframe(pd.DataFrame(explanation), use_container_width=True)
    if not events_df.empty:
        st.dataframe(events_df[events_df["is_anomaly"]].tail(20), use_container_width=True)

with tabs[3]:
    if drift_df.empty:
        st.info("Drift checks run once the configured rolling window is available.")
    else:
        st.metric("Overall Drift Score", _metric_value(current_drift_score))
        st.metric("Drift Status", "DRIFT" if latest_drift.get("overall_drift_detected") else "NO DRIFT")
        st.metric("Drifted Features", latest_drift.get("drifted_feature_count", 0))
        details = json.loads(latest_drift.get("details_json", "{}"))
        feature_results = details.get("feature_results", {})
        st.dataframe(pd.DataFrame(feature_results).T.sort_values("p_value").head(20), use_container_width=True)

with tabs[4]:
    severity = st.multiselect("Severity", ["INFO", "WARNING", "CRITICAL"], default=["INFO", "WARNING", "CRITICAL"])
    if alerts_df.empty:
        st.info("No alerts generated.")
    else:
        st.dataframe(alerts_df[alerts_df["severity"].isin(severity)].tail(50), use_container_width=True)

with tabs[5]:
    metrics_path = Path(settings.metrics_path)
    if metrics_path.exists():
        st.json(json.loads(metrics_path.read_text()))
    else:
        st.info("Train the Random Forest model to create artifacts/model_metrics.json.")
