"""
Trainer for the Isolation Forest using combined trade logs + positions datasets.

Usage:
    python model_train.py --logs synthetic_training_logs.csv \
        --positions synthetic_positions_dataset.csv \
        --output models/iso_model.pkl
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

from feature_extractor import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def load_training_data(logs_path: Path, positions_path: Path) -> pd.DataFrame:
    logger.info("Loading training logs from %s", logs_path)
    logs = pd.read_csv(logs_path)
    logger.info("Loading positions from %s", positions_path)
    positions = pd.read_csv(positions_path)

    merged = logs.merge(positions, on="account_id", how="left", suffixes=("", "_pos"))
    missing = [col for col in FEATURE_COLUMNS if col not in merged.columns]
    if missing:
        raise ValueError(f"Training data missing required columns: {missing}")

    feature_df = merged[FEATURE_COLUMNS].copy()
    feature_df = feature_df.fillna(0.0)
    return feature_df


def fit_isolation_forest(
    df: pd.DataFrame,
    n_estimators: int,
    contamination: float,
    random_state: int,
) -> IsolationForest:
    logger.info(
        "Training Isolation Forest (n_estimators=%s, contamination=%s, samples=%s)",
        n_estimators,
        contamination,
        len(df),
    )
    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        verbose=0,
    )
    clf.fit(df[FEATURE_COLUMNS])
    return clf


def save_model(model: IsolationForest, path: Path) -> None:
    logger.info("Persisting trained model to %s", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Isolation Forest to detect portfolio anomalies."
    )
    parser.add_argument(
        "--logs",
        type=Path,
        default=Path("synthetic_training_logs.csv"),
        help="CSV containing trade logs",
    )
    parser.add_argument(
        "--positions",
        type=Path,
        default=Path("synthetic_positions_dataset.csv"),
        help="CSV containing positions snapshots",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/iso_model.pkl"),
        help="Path to write the serialized model",
    )
    parser.add_argument(
        "--n-estimators", type=int, default=400, help="Isolation Forest tree count"
    )
    parser.add_argument(
        "--contamination", type=float, default=0.05, help="Expected anomaly share"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Seed for reproducibility"
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    df = load_training_data(args.logs, args.positions)
    model = fit_isolation_forest(
        df,
        n_estimators=args.n_estimators,
        contamination=args.contamination,
        random_state=args.random_state,
    )
    save_model(model, args.output)
    logger.info("Training complete. Samples seen: %s", len(df))


if __name__ == "__main__":
    main()

