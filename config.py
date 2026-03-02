# config.py
# Global configuration for the GPS Batch Anomaly Detection Pipeline

import os
from dataclasses import dataclass
from typing import ClassVar

# -------------------------------------------------------------------
ANOMALY_THRESHOLD = 0.1


REQUIRED_COLUMNS = [
    "batch_id",
    "timestamp",
    "temperature",
    "pressure",
    "ph_level",
    "batch_yeild",
    "machine_id"
]

ROLLING_WINDOW_SIZE = 3
CONTAMINATION_RATE = 0.05
REPORT_OUTPUT_DIR = "reports"
MIN_BATCHES_PER_MACHINE = 5


@dataclass
class PipelineConfig:
    threshold: float = ANOMALY_THRESHOLD
    window: int = ROLLING_WINDOW_SIZE
    contamination: float = CONTAMINATION_RATE
    output_dir: str = REPORT_OUTPUT_DIR
    required_cols: ClassVar[list] = REQUIRED_COLUMNS
