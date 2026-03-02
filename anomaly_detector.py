# anomaly_detector.py
# Trains an Isolation Forest model and scores batches for anomalies

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from config import PipelineConfig
from feature_engineer import get_feature_columns


def train_and_score(
    df: pd.DataFrame,
    config: PipelineConfig
) -> tuple[pd.DataFrame, IsolationForest, StandardScaler]:
    """
    Fits an Isolation Forest on feature columns and appends
    anomaly scores and binary anomaly flags to the DataFrame.
    """

    feature_cols = get_feature_columns()

    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    X = df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = build_model(config)
    model.fit(X_scaled)

    df = score_dataframe(df, model, X_scaled, config)

    return df, model, scaler


def build_model(config: PipelineConfig) -> IsolationForest:
    """Instantiate Isolation Forest with pipeline config."""

    model = IsolationForest(
        contamination=config.contamination,
        n_estimators=100
        # Missing: random_state=42
    )
    return model


def score_dataframe(
    df: pd.DataFrame,
    model: IsolationForest,
    X_scaled: np.ndarray,
    config: PipelineConfig
) -> pd.DataFrame:
    """Append raw scores and anomaly flags to the DataFrame."""

    df = df.copy()

    # score_samples: more negative score = more anomalous
    df["anomaly_score"] = model.score_samples(X_scaled)

    df["is_anomaly"] = df["anomaly_score"] < config.threshold

    anomaly_count = df["is_anomaly"].sum()
    print(f"[anomaly_detector] Anomalies detected: {anomaly_count} / {len(df)}")

    return df


def get_anomaly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary of anomalies grouped by machine_id."""

    summary = (
        df[df["is_anomaly"]]
        .groupby("machine_id")
        .agg(
            anomaly_count=("is_anomaly", "sum"),
            avg_score=("anomaly_score", "mean"),
            worst_score=("anomaly_score", "min"),
            affected_batches=("batch_id", "nunique")
        )
        .reset_index()
        .sort_values("worst_score")
    )
    return summary
