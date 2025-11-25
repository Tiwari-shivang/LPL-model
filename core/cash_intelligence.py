"""
Cash Intelligence Engine: rule-based reasoning layered on top of anomaly signals.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List


def _severity_from_drift(drift: float) -> str:
    drift_abs = abs(drift)
    if drift_abs >= 0.1:
        return "high"
    if drift_abs >= 0.05:
        return "medium"
    return "low"


def _estimate_cash_drag(market_value: float, drift: float) -> float:
    # Rough monthly drag assuming 6% annual return equivalent
    annual_return = 0.06
    return max(0.0, market_value * drift * (annual_return / 12))


def build_cash_intelligence(
    snapshot: Dict[str, Any],
    features: Dict[str, float],
    is_anomaly: bool,
) -> Dict[str, Any]:
    """
    Produces root causes, risk, and action hints for a portfolio snapshot.
    """
    root_causes: List[str] = []
    actions: List[str] = []

    drift = float(features.get("drift_percent", 0.0))
    drift_severity = _severity_from_drift(drift)
    market_value = float(features.get("market_value", 0.0))
    cash_after = float(features.get("cash_after", 0.0))
    model_target = float(features.get("model_cash_target", 0.0))

    if drift > 0:
        root_causes.append(
            f"Cash is {drift * 100:.1f}% above model target; holdings under-deployed."
        )
        actions.append("Deploy excess cash toward target sleeves with highest expected return.")
    elif drift < 0:
        root_causes.append(
            f"Cash is {abs(drift) * 100:.1f}% below model target; liquidity may be constrained."
        )
        actions.append("Raise cash by trimming overweight positions or pausing new buys.")

    pending_est_cost = float(snapshot.get("pending_est_cost") or 0.0)
    if pending_est_cost > 0:
        root_causes.append(
            f"Pending orders worth {pending_est_cost:,.0f} could shift cash after settlement."
        )
        actions.append("Confirm settlement calendar and ensure cash covers pending orders.")

    dividends = float(snapshot.get("recent_dividends") or 0.0)
    if dividends > 0:
        root_causes.append(f"Recent dividends of {dividends:,.0f} increased cash.")

    deposits = float(snapshot.get("recent_deposits") or 0.0)
    withdrawals = float(snapshot.get("recent_withdrawals") or 0.0)
    if deposits > withdrawals and deposits > 0:
        root_causes.append(f"Net deposits of {deposits - withdrawals:,.0f} in lookback window.")
    elif withdrawals > deposits and withdrawals > 0:
        root_causes.append(f"Net withdrawals of {withdrawals - deposits:,.0f} reduced cash.")

    impact = _estimate_cash_drag(market_value, drift)
    if impact > 0:
        actions.append(f"Potential cash drag ~{impact:,.0f}/month; rebalance to reduce drift.")

    if not root_causes:
        root_causes.append("No clear cash driver identified; monitor upcoming trades and flows.")

    intelligence = {
        "root_causes": root_causes,
        "actions": actions,
        "risk_score": round(min(1.0, 0.3 + abs(drift) * 2), 3),
        "drift_percent": drift,
        "severity": drift_severity,
        "impact_estimate": impact,
        "model_cash_target": model_target,
        "cash_after": cash_after,
        "market_value": market_value,
        "is_anomaly": is_anomaly,
    }

    return intelligence