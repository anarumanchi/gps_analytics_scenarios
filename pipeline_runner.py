# pipeline_runner.py
# Orchestrates the full anomaly detection pipeline end-to-end

import os
import json
import pandas as pd
from datetime import datetime

from config import PipelineConfig
from data_loader import load_batch_data, filter_insufficient_machines
from feature_engineer import engineer_features
from anomaly_detector import train_and_score, get_anomaly_summary


def run_pipeline(data_filepath: str, config: PipelineConfig) -> dict:
    """Execute the full GPS batch anomaly detection pipeline."""

    print("\n=== GPS Batch Anomaly Detection Pipeline ===\n")

    print("[pipeline] Stage 1: Loading data...")
    df_raw = load_batch_data(data_filepath)
    df_raw = filter_insufficient_machines(df_raw)

    print("[pipeline] Stage 2: Engineering features...")
    df_features = engineer_features(df_raw)

    print("[pipeline] Stage 3: Running anomaly detection...")
    df_scored, model, scaler = train_and_score(df_features, config)

    print("[pipeline] Stage 4: Generating report...")
    report = generate_report(df_scored, config)

    return report


def generate_report(df: pd.DataFrame, config: PipelineConfig) -> dict:
    """Build and save a JSON summary report of the pipeline run."""

    summary = get_anomaly_summary(df)

    report = {
        "run_timestamp": datetime.now().isoformat(),
        "total_batches_analyzed": len(df),
        "total_anomalies_detected": int(df["is_anomaly"].sum()),
        "anomaly_rate": round(df["is_anomaly"].mean(), 4),
        "machines_with_anomalies": int(summary["machine_id"].nunique()),
        "top_anomalies": summary.head(5).to_dict(orient="records")
    }

    output_path = os.path.join(config.output_dir, "anomaly_report.json")

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[pipeline] Report saved to: {output_path}")
    return report


def validate_report(report: dict) -> bool:
    """
    Sanity check: anomaly rate should be close to contamination rate.
    Returns False and warns if the anomaly rate looks wrong.
    """

    if report["anomaly_rate"] == 0:
        print(
            "[WARNING] Anomaly rate is 0.0 — no anomalies detected. "
            "Check ANOMALY_THRESHOLD in config.py."
        )
        return True

    print(f"[pipeline] Validation passed. Anomaly rate: {report['anomaly_rate']}")
    return True


if __name__ == "__main__":

    config = PipelineConfig()

    report = run_pipeline(
        data_filepath="data/batch_sensor_data.csv",
        config=config
    )

    is_valid = validate_report(report)

    if is_valid:
        print("\n✅ Pipeline completed successfully.")
    else:
        print("\n❌ Pipeline completed with warnings. Review report.")
