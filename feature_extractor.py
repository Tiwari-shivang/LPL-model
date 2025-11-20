"""
Utility functions to transform nested account JSON into numeric features for the ML model.

The extractor is deliberately defensive to handle missing keys, new structure, or malformed entries.
"""

from typing import Any, Dict, Iterable, List


def extract_features_from_account(account: Dict[str, Any]) -> Dict[str, float]:
    """
    Flatten one account snapshot into the feature set expected by the model.

    Args:
        account: Nested JSON data from the backend representing one portfolio snapshot.

    Returns:
        Dictionary with seven numeric features.
    """

    total_trade_amount = float(account.get("TradeAmount", 0.0)) or 0.0
    sleeves = account.get("RebalancingModels") or []

    cash_trade_amount = 0.0
    cash_model_percent = 0.0
    total_securities = 0

    for sleeve in sleeves:
        for sec in sleeve.get("RebalancingModels") or []:
            total_securities += 1
            sec_type = sec.get("SecurityType", "")
            sec_trade = float(sec.get("TradeAmount", 0.0)) or 0.0
            sec_model_percent = float(sec.get("ModelPercent", 0.0)) or 0.0

            if "Cash" in sec_type:
                cash_trade_amount += sec_trade
                cash_model_percent += sec_model_percent

    cash_trade_ratio = (
        cash_trade_amount / total_trade_amount if total_trade_amount > 0 else 0.0
    )
    cash_vs_model_diff = cash_trade_ratio * 100 - cash_model_percent

    return {
        "total_trade_amount": total_trade_amount,
        "cash_trade_amount": cash_trade_amount,
        "cash_trade_ratio": cash_trade_ratio,
        "cash_model_percent": cash_model_percent,
        "cash_vs_model_diff": cash_vs_model_diff,
        "total_securities": float(total_securities),
        "num_sleeves": float(len(sleeves)),
    }


def extract_feature_matrix(accounts: Iterable[Dict[str, Any]]) -> List[Dict[str, float]]:
    """
    Process a batch of accounts into a feature matrix.
    """
    return [extract_features_from_account(acc) for acc in accounts]

