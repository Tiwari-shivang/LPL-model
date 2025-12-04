"""
Position anomaly model using XGBoost.

Trains on position_analysis_training_data.csv and persists to models/position_model.pkl.
Features are drawn from the guidance in positionAnalysis.md.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

FEATURE_COLUMNS: List[str] = [
    "model_target_percent",
    "executed_quantity",
    "executed_percent",
    "allocation_after_trade",
    "exists_in_model",
    "drift_percent",
    "overweight_percent",
    "underweight_percent",
    "order_mismatch_flag",
    "timing_mismatch_flag",
    "allocation_excess_flag",
    "time_delta_seconds",
]


def _load_training(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "order_execution_time" in df.columns and "model_change_time" in df.columns:
        df["order_execution_time"] = pd.to_datetime(df["order_execution_time"])
        df["model_change_time"] = pd.to_datetime(df["model_change_time"])
        df["time_delta_seconds"] = (
            (df["order_execution_time"] - df["model_change_time"]).dt.total_seconds()
        )
    else:
        df["time_delta_seconds"] = 0.0

    # Ensure all features exist
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    df = df.dropna(subset=["label"]).reset_index(drop=True)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0.0)
    return df


def train_position_model(data_path: Path, model_path: Path) -> Optional[XGBClassifier]:
    df = _load_training(data_path)
    if df.empty:
        logger.warning("Position training data is empty at %s", data_path)
        return None

    X = df[FEATURE_COLUMNS]
    y = df["label"].astype(int)

    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary:logistic",
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X, y)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info("Position model trained and saved to %s (samples=%s)", model_path, len(df))
    return model


def load_position_model(model_path: Path) -> Optional[XGBClassifier]:
    if not model_path.exists():
        return None
    try:
        return joblib.load(model_path)
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to load position model: %s", exc)
        return None
