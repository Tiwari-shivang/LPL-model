"""
Console entry point to fit an Isolation Forest on historic portfolio snapshots.

Usage:
    python model_train.py --data data/training_data.csv --output models/iso_model.pkl
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

from feature_extractor import extract_feature_matrix

DEFAULT_FEATURE_COLUMNS: List[str] = [
    "total_trade_amount",
    "cash_trade_amount",
    "cash_trade_ratio",
    "cash_model_percent",
    "cash_vs_model_diff",
    "total_securities",
    "num_sleeves",
]

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def load_training_data(path: Path) -> pd.DataFrame:
    logger.info("Loading training data from %s", path)
    df = pd.read_csv(path)
    rename_map = {}
    if "cash_ratio" in df.columns and "cash_trade_ratio" not in df.columns:
        rename_map["cash_ratio"] = "cash_trade_ratio"

    if rename_map:
        df = df.rename(columns=rename_map)

    missing = set(DEFAULT_FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Training data is missing required columns: {missing}")

    return df[DEFAULT_FEATURE_COLUMNS]


def fit_isolation_forest(
    df: pd.DataFrame,
    features: List[str],
    n_estimators: int,
    contamination: float,
    random_state: int,
) -> IsolationForest:
    logger.info(
        "Training Isolation Forest (n_estimators=%s, contamination=%s)",
        n_estimators,
        contamination,
    )
    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        verbose=0,
    )
    clf.fit(df[features])
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
        "--data",
        type=Path,
        default=Path("data/training_data.csv"),
        help="CSV containing historic feature snapshots",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/iso_model.pkl"),
        help="Path to write the serialized model",
    )
    parser.add_argument(
        "--n-estimators", type=int, default=200, help="Isolation Forest tree count"
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
    df = load_training_data(args.data)
    model = fit_isolation_forest(
        df,
        features=DEFAULT_FEATURE_COLUMNS,
        n_estimators=args.n_estimators,
        contamination=args.contamination,
        random_state=args.random_state,
    )
    save_model(model, args.output)
    logger.info("Training complete. Samples seen: %s", len(df))


if __name__ == "__main__":
    main()

