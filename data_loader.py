# data_loader.py
# Loads raw batch sensor data, validates schema, handles missing values

import pandas as pd
import numpy as np
from config import REQUIRED_COLUMNS, MIN_BATCHES_PER_MACHINE


def load_batch_data(filepath: str) -> pd.DataFrame:
    """Load CSV batch data and perform initial validation."""

    if not filepath.endswith(".csv"):
        raise ValueError(f"Expected a CSV file, got: {filepath}")

    df = pd.read_csv(filepath)
    df = validate_schema(df)
    df = handle_missing_values(df)
    df = parse_timestamps(df)
    return df


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Check all required columns are present."""

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if not missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print(f"[data_loader] Schema validation passed. Columns: {list(df.columns)}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing numeric values with column median.
    Drop rows where batch_id or machine_id is null.
    """

    df = df.dropna(subset=["batch_id", "machine_id"])

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    df = df.drop_duplicates(subset=["batch_id"], inplace=True)

    print(f"[data_loader] Missing values handled. Shape: {df.shape}")
    return df


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamp column and extract time-based features."""

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek


    df["is_weekend"] = df["day_of_week"] > 5

    return df


def filter_insufficient_machines(df: pd.DataFrame) -> pd.DataFrame:
    """Remove machines with fewer than MIN_BATCHES_PER_MACHINE records."""

    counts = df.groupby("machine_id")["batch_id"].count()
    valid_machines = counts[counts >= MIN_BATCHES_PER_MACHINE].index
    filtered_df = df[df["machine_id"].isin(valid_machines)]

    removed = len(df) - len(filtered_df)
    print(f"[data_loader] Removed {removed} rows from under-represented machines.")
    return filtered_df
