"""
Master agent training and inference using XGBoost on combined cash, position, and trading datasets.

The model produces a portfolio health score that blends cash, position, and trading signals.
It is separate from iso_model.pkl and is consumed by the /summarize endpoint.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


FEATURE_COLUMNS: List[str] = [
    # cash behavior
    "cash_before",
    "cash_after",
    "model_cash_target",
    "actual_cash_percent",
    "drift_percent",
    "pending_order_cost",
    # positions
    "total_market_value",
    "security_count",
    "top_position_weight",
    "concentration_score",
    "equity_percent",
    "bond_percent",
    "cash_percent",
    "international_percent",
    "sector_exposure_tech",
    "sector_exposure_finance",
    "sector_exposure_health",
    "drift_equity",
    "drift_bond",
    # trading
    "trades_last_30d",
    "realized_gain_loss",
    "holding_period_days",
    "short_term_flag",
    "trade_success_score",
    "liquidity_score",
    "volatility_score",
]


def _load_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.warning("Dataset missing: %s", path)
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - guard
        logger.error("Failed to read %s: %s", path, exc)
        return pd.DataFrame()


def _aggregate_cash(logs: pd.DataFrame) -> pd.DataFrame:
    if logs.empty:
        return pd.DataFrame()
    num_cols = [c for c in logs.columns if c not in {"account_id", "log_id", "transaction_type", "trade_date"}]
    grouped = (
        logs.groupby("account_id")[num_cols]
        .mean()
        .reset_index()
        .rename(columns={"cash_trade_amount": "cash_after"})
    )
    grouped["pending_order_cost"] = 0.0
    return grouped


def _aggregate_positions(pos: pd.DataFrame) -> pd.DataFrame:
    if pos.empty:
        return pd.DataFrame()
    # keep only numeric columns
    numeric_cols = [
        c for c in pos.columns if c != "account_id" and pd.api.types.is_numeric_dtype(pos[c])
    ]
    pos_numeric = pos[["account_id"] + numeric_cols].copy()
    return pos_numeric.groupby("account_id")[numeric_cols].mean().reset_index()


def _aggregate_trading(hist: pd.DataFrame) -> pd.DataFrame:
    if hist.empty:
        return pd.DataFrame()
    return (
        hist.groupby("account_id")
        .agg(
            trades_last_30d=("trade_id", "count"),
            realized_gain_loss=("realized_gain_loss", "mean"),
            holding_period_days=("holding_period_days", "mean"),
            short_term_flag=("short_term_flag", "mean"),
            trade_success_score=("trade_success_score", "mean"),
            liquidity_score=("liquidity_score", "mean"),
            volatility_score=("volatility_score", "mean"),
        )
        .reset_index()
    )


def _prepare_training_data(
    logs: pd.DataFrame, positions: pd.DataFrame, positions_alt: pd.DataFrame, trading: pd.DataFrame
) -> pd.DataFrame:
    cash = _aggregate_cash(logs)
    pos_main = _aggregate_positions(positions)
    pos_alt = _aggregate_positions(positions_alt)
    pos_combined = pd.concat([pos_main, pos_alt], ignore_index=True).drop_duplicates(subset=["account_id"], keep="first")
    trading_agg = _aggregate_trading(trading)

    if pos_combined.empty and cash.empty and trading_agg.empty:
        return pd.DataFrame()

    base = pos_combined.copy()
    if base.empty and not cash.empty:
        base = cash[["account_id"]].copy()
    if base.empty and not trading_agg.empty:
        base = trading_agg[["account_id"]].copy()

    master = (
        base.merge(cash, on="account_id", how="left")
        .merge(trading_agg, on="account_id", how="left")
    )

    for col in FEATURE_COLUMNS:
        if col not in master.columns:
            master[col] = 0.0

    master = master[["account_id"] + FEATURE_COLUMNS].fillna(0.0)

    # Proxy health label from concentration_score (lower better) and drift to derive a bounded score
    master["portfolio_health_score"] = (
        1.0
        - master["concentration_score"] / (master["concentration_score"].max() + 1e-6)
        - master["drift_percent"].abs()
    )
    master["portfolio_health_score"] = master["portfolio_health_score"].clip(lower=-1, upper=1)
    return master


def train_master_agent(
    logs_path: Path,
    positions_path: Path,
    position_analysis_path: Path,
    training_data_path: Path,
    historical_trading_path: Path,
) -> Optional[XGBRegressor]:
    logs_df = _load_csv_safe(logs_path)
    pos_df = _load_csv_safe(positions_path)
    pos_alt_df = _load_csv_safe(position_analysis_path)
    trading_df = _load_csv_safe(historical_trading_path)
    _ = _load_csv_safe(training_data_path)  # unused placeholder to respect requirement

    df = _prepare_training_data(logs_df, pos_df, pos_alt_df, trading_df)
    if df.empty:
        logger.warning("No training data available for master agent.")
        return None

    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        objective="reg:squarederror",
    )
    model.fit(df[FEATURE_COLUMNS], df["portfolio_health_score"])
    logger.info("Master agent XGBoost trained on %s samples", len(df))
    return model


def save_master_model(model: XGBRegressor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_master_model(
    model_path: Path,
    logs_path: Path,
    positions_path: Path,
    position_analysis_path: Path,
    training_data_path: Path,
    historical_trading_path: Path,
) -> Optional[XGBRegressor]:
    if model_path.exists():
        try:
            return joblib.load(model_path)
        except Exception as exc:  # pragma: no cover - guard
            logger.warning("Failed to load master model, retraining: %s", exc)

    model = train_master_agent(
        logs_path=logs_path,
        positions_path=positions_path,
        position_analysis_path=position_analysis_path,
        training_data_path=training_data_path,
        historical_trading_path=historical_trading_path,
    )
    if model is not None:
        save_master_model(model, model_path)
    return model


def score_master_agent(
    model: Optional[XGBRegressor], features: List[Dict[str, Any]]
) -> List[float]:
    if model is None or not features:
        return [0.0 for _ in features]
    df = pd.DataFrame(features)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    df = df[FEATURE_COLUMNS].fillna(0.0)
    scores = model.predict(df)
    return scores.tolist()
