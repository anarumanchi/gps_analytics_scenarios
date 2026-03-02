# feature_engineer.py
# Computes rolling statistics and derived features per machine

import pandas as pd
import numpy as np
from config import ROLLING_WINDOW_SIZE


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main feature engineering function.
    Computes rolling means, standard deviations, and
    derived ratio features per machine group.
    """

    df = df.copy()
    df = compute_rolling_features(df)
    df = compute_derived_features(df)
    df = drop_warmup_rows(df)
    return df


def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-machine rolling mean and std for key sensor columns."""

    sensor_cols = ["temperature", "pressure", "ph_level", "batch_yield"]

    for col in sensor_cols:
        df[f"{col}_rolling_mean"] = (
            df.groupby("machine_id")[col]
            .transform(
                lambda x: x.rolling(ROLLING_WINDOW_SIZE, min_periods=1).mean()
            )
        )
        df[f"{col}_rolling_std"] = (
            df.groupby("machine_id")[col]
            .transform(
                lambda x: x.rolling(ROLLING_WINDOW_SIZE, min_periods=1)
                           .std()
                           .fillna(0)
            )
        )

    return df


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ratio and interaction features."""

    df["temp_pressure_ratio"] = (
        df["temperature"] / df["pressure"].replace(0, np.nan)
    )
    df["yield_ph_interaction"] = df["batch_yield"] * df["ph_level"]
    df["temp_deviation"] = (
        df["temperature"] - df["temperature_rolling_mean"]
    )

    return df


def drop_warmup_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove the first N rows per machine where rolling
    features are unreliable (warmup period = ROLLING_WINDOW_SIZE).
    """

    df["_row_num"] = df.groupby("machine_id").cumcount()
    df = df[df["_row_num"] > ROLLING_WINDOW_SIZE]
    df = df.drop(columns=["_row_num"])

    return df


def get_feature_columns() -> list:
    """Return the list of feature column names used for modeling."""

    sensor_cols = ["temperature", "pressure", "ph_level", "batch_yield"]
    rolling_features = (
        [f"{c}_rolling_mean" for c in sensor_cols] +
        [f"{c}_rolling_std" for c in sensor_cols]
    )
    derived = ["temp_pressure_ratio", "yield_ph_interaction", "temp_deviation"]
    time_features = ["hour_of_day", "day_of_week", "is_weekend"]

    return rolling_features + derived + time_features
