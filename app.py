"""
FastAPI + Socket.IO service that wraps Isolation Forest scoring for live anomaly recommendations.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import joblib
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from core.cash_intelligence import build_cash_intelligence
from core.natural_language import NaturalLanguageHelper
from core.master_agent import load_master_model, score_master_agent
from core.position_model import load_position_model
from db.database_client import DatabaseClient
from feature_extractor import FEATURE_COLUMNS, extract_feature_matrix

MODEL_PATH = Path("models/iso_model.pkl")

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


load_dotenv()
db_client: Optional[DatabaseClient]
try:
    db_client = DatabaseClient()
except ImportError:
    db_client = None
    logger.warning("python-tds not installed; database connectivity is unavailable.")

nl_helper = NaturalLanguageHelper(api_key=os.getenv("OPENAI_API_KEY"))


class CashAnalysisContext(BaseModel):
    accounts: List[Dict[str, Any]]
    summary: Optional[str] = None


class CashAnalysisRequest(BaseModel):
    question: str


class MasterFollowUpRequest(BaseModel):
    question: str
    class Config:
        extra = "allow"


class CashFollowUpRequest(BaseModel):
    question: str
    class Config:
        extra = "allow"


def _load_position_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "order_execution_time" in df.columns and "model_change_time" in df.columns:
        df["order_execution_time"] = pd.to_datetime(df["order_execution_time"])
        df["model_change_time"] = pd.to_datetime(df["model_change_time"])
        df["time_delta_seconds"] = (
            (df["order_execution_time"] - df["model_change_time"]).dt.total_seconds()
        )
    else:
        df["time_delta_seconds"] = 0.0
    return df


def _to_iso(dt: Any) -> Any:
    try:
        import pandas as pd  # type: ignore
    except Exception:  # pragma: no cover
        pd = None
    if isinstance(dt, datetime):
        return dt.isoformat()
    if pd is not None and isinstance(dt, pd.Timestamp):
        return dt.to_pydatetime().isoformat()
    return dt


def load_model() -> Any:
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found: {MODEL_PATH}")
    logger.info("Loading model from %s", MODEL_PATH)
    return joblib.load(MODEL_PATH)


configure_logging()
app = FastAPI(title="Cash Portfolio Anomaly Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

model = load_model()
MASTER_MODEL_PATH = Path("models/xgb_master_model.pkl")
MASTER_MODEL = load_master_model(
    model_path=MASTER_MODEL_PATH,
    logs_path=Path("synthetic_training_logs.csv"),
    positions_path=Path("synthetic_positions_dataset.csv"),
    position_analysis_path=Path("position_analysis_dataset.csv"),
    training_data_path=Path("training_data.csv"),
    historical_trading_path=Path("historical_trading_dataset.csv"),
)
POSITION_MODEL_PATH = Path("models/position_model.pkl")
POSITION_MODEL = load_position_model(POSITION_MODEL_PATH)

# In-memory storage for the last analysis context
analysis_memory: Dict[str, Any] = {"recommendations": [], "summary": ""}


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok"}

@app.get("/analyze")
async def analyze() -> List[Dict[str, Any]]:
    """
    Pull accounts directly from the DB, score anomalies, and enrich with cash intelligence.
    Clients call this endpoint with an empty body.
    """
    if db_client is None:
        raise HTTPException(
            status_code=500,
            detail="Database client unavailable. Ensure python-tds is installed.",
        )

    try:
        accounts = await run_in_threadpool(db_client.fetch_enriched_accounts, 500)
    except Exception as exc:  # pragma: no cover - runtime guard
        logger.exception("Failed to fetch accounts from DB: %s", exc)
        raise HTTPException(status_code=502, detail="Failed to fetch accounts from database.")

    if not isinstance(accounts, list) or not accounts:
        raise HTTPException(
            status_code=400,
            detail="Database returned no accounts to analyze.",
        )

    feature_rows = extract_feature_matrix(accounts)
    df = pd.DataFrame(feature_rows)

    try:
        preds = model.predict(df[FEATURE_COLUMNS])
        scores = model.decision_function(df[FEATURE_COLUMNS])
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {exc}")

    recommendations: List[Dict[str, Any]] = []
    for idx, (feat, pred, score, snapshot) in enumerate(
        zip(feature_rows, preds.tolist(), scores.tolist(), accounts)
    ):
        is_anomaly = pred == -1
        intelligence = build_cash_intelligence(snapshot, feat, is_anomaly=is_anomaly)

        reasons: List[str] = []
        reasons.extend(intelligence.get("root_causes", []))
        reasons.extend(intelligence.get("actions", []))

        idle_days = snapshot.get("cash_idle_days")
        last_trade_date = snapshot.get("last_trade_date")
        days_since_last_trade = snapshot.get("days_since_last_trade")
        if idle_days is not None:
            reasons.append(f"Cash has been idle for ~{idle_days} days since last trade.")
        if days_since_last_trade is not None and last_trade_date:
            reasons.append(f"Last trade date: {str(last_trade_date)} ({days_since_last_trade} days ago).")

        recommendations.append(
            {
                "accountId": snapshot.get("account_id")
                or snapshot.get("AccountId")
                or f"unknown-{idx}",
                "cash_idle_days": idle_days,
                "days_since_last_trade": days_since_last_trade,
                "last_trade_date": last_trade_date,
                "features": {
                    **feat,
                    "risk_score": intelligence.get("risk_score"),
                    "severity": intelligence.get("severity"),
                    "portfolio": snapshot.get("portfolio", {}),
                },
                "is_anomaly": is_anomaly,
                "score": float(score),
                "reasons": reasons,
                "recommendationViewed": False,
            }
        )

    logger.info("Generated %s recommendations", len(recommendations))

    # Generate a single LLM summary across all accounts (batch prompt)
    summary_text = await nl_helper.summarize_batch(recommendations)

    # Store in memory for follow-up questions
    anomaly_count = sum(1 for r in recommendations if r.get("is_anomaly"))
    analysis_memory["recommendations"] = recommendations
    analysis_memory["summary"] = summary_text or (
        f"Analyzed {len(recommendations)} accounts, {anomaly_count} flagged as anomalies."
    )

    # Include batch summary as the first element in the response array
    response_payload: List[Dict[str, Any]] = [{"batch_summary": analysis_memory["summary"]}]
    response_payload.extend(recommendations)
    return response_payload


@app.get("/summarize")
async def summarize() -> Dict[str, Any]:
    """
    Master agent summary across cash, positions, and historical trading.
    Uses XGBoost scores plus DB signals; response grouped into three categories.
    """
    if db_client is None:
        raise HTTPException(
            status_code=500,
            detail="Database client unavailable. Ensure python-tds is installed.",
        )

    try:
        accounts = await run_in_threadpool(db_client.fetch_enriched_accounts, 500)
    except Exception as exc:
        logger.exception("Failed to fetch accounts for summarize: %s", exc)
        raise HTTPException(status_code=502, detail="Failed to fetch accounts from database.")

    if not accounts:
        raise HTTPException(status_code=400, detail="Database returned no accounts to summarize.")

    feature_rows = extract_feature_matrix(accounts)
    master_scores = score_master_agent(MASTER_MODEL, feature_rows)

    # Build category-wise findings
    cash_findings: List[str] = []
    position_findings: List[str] = []
    trading_findings: List[str] = []

    high_idle = [
        (acc, score)
        for acc, score in zip(accounts, master_scores)
        if acc.get("cash_idle_days") and acc.get("cash_idle_days") > 14
    ]
    if high_idle:
        cash_findings.append(
            f"{len(high_idle)} accounts show cash idle >14 days; prioritize deployment."
        )

    high_scores = [
        (acc, s) for acc, s in zip(accounts, master_scores) if s >= 0.6
    ]
    if high_scores:
        cash_findings.append(
            f"{len(high_scores)} accounts flagged with elevated cash anomaly likelihood via XGBoost."
        )

    # Position proxy: drift signals
    drift_issues = [
        acc for acc in accounts if abs(acc.get("drift_percent", 0.0)) > 0.1
    ]
    if drift_issues:
        position_findings.append(
            f"{len(drift_issues)} accounts exceed 10% drift vs model targets; rebalance suggested."
        )
    else:
        position_findings.append("Position alignment generally within 10% drift tolerance.")

    # Historical trading
    long_no_trade = [
        acc for acc in accounts if acc.get("days_since_last_trade") and acc["days_since_last_trade"] > 30
    ]
    if long_no_trade:
        trading_findings.append(
            f"{len(long_no_trade)} accounts show no trades in 30+ days; review inactivity vs strategy."
        )
    else:
        trading_findings.append("Trading cadence within expected ranges across accounts.")

    response = {
        "Cash_analysis": {
            "summary": cash_findings or ["Cash activity appears stable."],
            "sample_accounts": [
                {
                    "account_id": acc.get("account_id"),
                    "cash_idle_days": acc.get("cash_idle_days"),
                    "anomaly_score": score,
                    "cash_after": acc.get("cash_after"),
                    "total_cash_available": acc.get("total_cash_available"),
                }
                for acc, score in high_scores[:5]
            ],
        },
        "Position_analysis": {
            "summary": position_findings,
            "sample_accounts": [
                {
                    "account_id": acc.get("account_id"),
                    "drift_percent": acc.get("drift_percent"),
                    "model_percent_target": acc.get("model_percent_target"),
                }
                for acc in drift_issues[:5]
            ],
        },
        "Historical_Trading": {
            "summary": trading_findings,
            "sample_accounts": [
                {
                    "account_id": acc.get("account_id"),
                    "days_since_last_trade": acc.get("days_since_last_trade"),
                    "last_trade_date": acc.get("last_trade_date"),
                }
                for acc in long_no_trade[:5]
            ],
        },
    }

    # LLM master summary
    response["master_summary"] = await nl_helper.summarize_master(response)

    return response


@app.post("/follow-up-master")
async def follow_up_master(payload: MasterFollowUpRequest) -> Dict[str, Any]:
    if not payload.question:
        raise HTTPException(status_code=400, detail="question is required")

    # Pass the entire payload (including master outputs) as context
    context = payload.dict()
    answer = await nl_helper.answer_master_follow_up(payload.question, context)
    return {"answer": answer, "model": nl_helper.model}


@app.post("/follow-up-cash")
async def follow_up_cash(payload: CashFollowUpRequest) -> Dict[str, Any]:
    if not payload.question:
        raise HTTPException(status_code=400, detail="question is required")

    context = payload.dict()
    answer = await nl_helper.answer_cash_follow_up(payload.question, context)
    return {"answer": answer, "model": nl_helper.model}


@app.get("/position-summrize")
async def position_summrize(limit: int = 200, page: int = 1, page_size: int = 5) -> Dict[str, Any]:
    """
    Run position anomaly scoring via position_model.pkl and return per-security rows.
    Fetches data from DB, builds feature rows, scores anomalies.
    """
    if POSITION_MODEL is None:
        raise HTTPException(
            status_code=500,
            detail="Position model not available. Train position_model.pkl first.",
        )

    if db_client is None:
        raise HTTPException(
            status_code=500,
            detail="Database client unavailable. Ensure python-tds is installed.",
        )

    try:
        rows = await run_in_threadpool(db_client.fetch_position_feature_rows, limit)
    except Exception as exc:
        logger.exception("Failed to fetch position data: %s", exc)
        raise HTTPException(status_code=502, detail="Failed to fetch position data from database.")

    if not rows:
        return {"rows": [], "total_rows": 0, "anomalies": 0}

    df = pd.DataFrame(rows)
    feature_cols = [
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
        # compute time delta if timestamps exist
    ]
    if "order_execution_time" in df.columns and "model_change_time" in df.columns:
        df["order_execution_time"] = pd.to_datetime(df["order_execution_time"])
        df["model_change_time"] = pd.to_datetime(df["model_change_time"])
        df["time_delta_seconds"] = (
            (df["order_execution_time"] - df["model_change_time"]).dt.total_seconds()
        )
    else:
        df["time_delta_seconds"] = 0.0
    feature_cols.append("time_delta_seconds")

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    df[feature_cols] = df[feature_cols].fillna(0.0)

    scores = POSITION_MODEL.predict_proba(df[feature_cols])[:, 1]
    df["score"] = scores
    df["is_anomaly"] = (df["score"] > 0.5).astype(bool)

    # Assemble output matching requested structure
    response_rows = []
    for _, row in df.iterrows():
        response_rows.append(
            {
                "account_id": row.get("account_id"),
                "account_short_name": row.get("account_short_name"),
                "model_symbol": row.get("model_symbol"),
                "model_target_percent": row.get("model_target_percent"),
                "executed_symbol": row.get("executed_symbol"),
                "executed_quantity": row.get("executed_quantity"),
                "executed_percent": row.get("executed_percent"),
                "allocation_after_trade": row.get("allocation_after_trade"),
                "model_change_time": _to_iso(row.get("model_change_time")),
                "order_execution_time": _to_iso(row.get("order_execution_time")),
                "exists_in_model": row.get("exists_in_model"),
                "drift_percent": row.get("drift_percent"),
                "overweight_percent": row.get("overweight_percent"),
                "underweight_percent": row.get("underweight_percent"),
                "order_mismatch_flag": row.get("order_mismatch_flag"),
                "timing_mismatch_flag": row.get("timing_mismatch_flag"),
                "allocation_excess_flag": row.get("allocation_excess_flag"),
                "score": float(row.get("score")),
                "is_anomaly": bool(row.get("is_anomaly")),
            }
        )

    anomalies = [r for r in response_rows if r["is_anomaly"]]

    # pagination
    if page_size <= 0:
        page_size = 5
    if page <= 0:
        page = 1
    start = (page - 1) * page_size
    end = start + page_size
    paged_rows = response_rows[start:end]

    # Attach LLM reason per row
    enriched_rows: List[Dict[str, Any]] = []
    for row in paged_rows:
        reason = await nl_helper.position_reason(row)
        new_row = dict(row)
        new_row["reason"] = reason
        enriched_rows.append(new_row)

    return {
        "total_rows": len(response_rows),
        "anomalies": len(anomalies),
        "page": page,
        "page_size": page_size,
        "rows": enriched_rows,
    }


@app.post("/cash-analysis")
async def cash_analysis(payload: CashAnalysisRequest) -> Dict[str, Any]:
    # Check if we have stored analysis context
    if not analysis_memory.get("recommendations"):
        raise HTTPException(
            status_code=400,
            detail="No analysis context found. Please run /analyze first."
        )
    
    try:
        answer_text = await nl_helper.answer_follow_up(
            payload.question,
            recommendations=analysis_memory["recommendations"],
            summary=analysis_memory["summary"],
        )
        logger.info("OpenAI analysis completed for question: %s", payload.question[:50])
        return {"answer": answer_text, "model": nl_helper.model}
    except Exception as exc:  # pragma: no cover - safety net
        logger.exception("Cash_analysis failed: %s", exc)
        raise HTTPException(status_code=500, detail="Cash_analysis failed. Please try again.")
