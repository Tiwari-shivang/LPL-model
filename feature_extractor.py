"""
Feature extraction for the Isolation Forest and downstream intelligence layers.

The live service builds per-account snapshots (cash, trades, positions), then we
normalize them into the feature vector the model expects. Defaults are chosen to
be conservative so the model remains stable even when some DB fields are missing.
"""

from typing import Any, Dict, Iterable, List

FEATURE_COLUMNS: List[str] = [
    "cash_before",
    "cash_after",
    "model_cash_target",
    "actual_cash_percent",
    "quantity",
    "avg_price",
    "market_price",
    "market_value",
    "model_percent_target",
    "current_percent",
    "drift_percent",
]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_percent(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return max(0.0, min(1.0, numerator / denominator))


def extract_features_from_account(snapshot: Dict[str, Any]) -> Dict[str, float]:
    """
    Flatten one enriched account snapshot into the model feature vector.
    Expected keys (best-effort): cash_before, cash_after, model_cash_target,
    actual_cash_percent, quantity, avg_price, market_price, market_value,
    model_percent_target, current_percent, drift_percent.
    """

    cash_before = _to_float(
        snapshot.get("cash_before"),
        snapshot.get("cash_balance", snapshot.get("total_cash_available", 0.0)),
    )
    cash_after = _to_float(
        snapshot.get("cash_after"),
        snapshot.get("available_cash", snapshot.get("total_cash_available", cash_before)),
    )
    market_value = _to_float(
        snapshot.get("market_value"),
        snapshot.get("total_market_value", 0.0),
    )

    # Model targets come either from DB model metadata or defaults to 5%
    model_cash_target = _to_float(snapshot.get("model_cash_target", 0.05))
    model_percent_target = _to_float(
        snapshot.get("model_percent_target", model_cash_target)
    )

    actual_cash_percent = _to_float(
        snapshot.get("actual_cash_percent"),
        _safe_percent(cash_after, market_value + cash_after),
    )
    current_percent = _to_float(
        snapshot.get("current_percent"),
        actual_cash_percent,
    )
    drift_percent = _to_float(
        snapshot.get("drift_percent"),
        current_percent - model_percent_target,
    )

    return {
        "cash_before": cash_before,
        "cash_after": cash_after,
        "model_cash_target": model_cash_target,
        "actual_cash_percent": actual_cash_percent,
        "quantity": _to_float(snapshot.get("quantity", 0.0)),
        "avg_price": _to_float(snapshot.get("avg_price", 0.0)),
        "market_price": _to_float(snapshot.get("market_price", 0.0)),
        "market_value": market_value,
        "model_percent_target": model_percent_target,
        "current_percent": current_percent,
        "drift_percent": drift_percent,
    }


def extract_feature_matrix(accounts: Iterable[Dict[str, Any]]) -> List[Dict[str, float]]:
    """
    Process a batch of accounts into a feature matrix in model column order.
    """
    rows: List[Dict[str, float]] = []
    for acc in accounts:
        features = extract_features_from_account(acc)
        # Ensure stable ordering / completeness
        rows.append({col: features.get(col, 0.0) for col in FEATURE_COLUMNS})
    return rows

