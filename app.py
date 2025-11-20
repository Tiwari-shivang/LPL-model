"""
FastAPI + Socket.IO service that wraps Isolation Forest scoring for live anomaly recommendations.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from google import genai
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from feature_extractor import extract_feature_matrix

MODEL_PATH = Path("models/iso_model.pkl")

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


load_dotenv()
GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY", "AIzaSyDC2sWF_hoNYvz3-Xm23I9_Wm3lExUf_P0"
)
# Set API key for google-genai client
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
gemini_client = genai.Client(api_key=GEMINI_API_KEY)


class AccountBatch(BaseModel):
    data: List[Dict[str, Any]]


class CashAnalysisContext(BaseModel):
    accounts: List[Dict[str, Any]]
    summary: Optional[str] = None


class CashAnalysisRequest(BaseModel):
    question: str


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

# In-memory storage for the last analysis context
analysis_memory: Dict[str, Any] = {"recommendations": [], "summary": ""}


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok"}


def build_recommendations(
    feature_rows: List[Dict[str, Any]],
    predictions: List[int],
    scores: List[float],
    accounts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    recommendations = []
    for idx, (feat, pred, score, metadata) in enumerate(
        zip(feature_rows, predictions, scores, accounts)
    ):
        is_anomaly = pred == -1
        reasons = []

        diff = feat.get("cash_vs_model_diff", 0.0)
        if diff > 5:
            reasons.append("Cash allocation significantly above model")
        elif diff < -5:
            reasons.append("Cash allocation significantly below model")

        recommendations.append(
            {
                "accountId": metadata.get("AccountId")
                or metadata.get("accountId")
                or f"unknown-{idx}",
                "features": feat,
                "is_anomaly": is_anomaly,
                "score": float(score),
                "reasons": reasons,
                "recommendationViewed": False,
            }
        )
    return recommendations


def build_gemini_prompt(question: str, context: Dict[str, Any]) -> str:
    accounts = context.get("accounts", [])
    summary = context.get("summary", "No summary provided.")

    return f"""
You are an experienced financial advisor specializing in portfolio management, cash allocation, and rebalancing.
You are talking to a professional investment advisor, not the end client.
Your job is to explain clearly, be conservative, and avoid overly aggressive risk-taking.

Summary:
{summary}

Accounts:
{accounts}

User question:
{question}

Instructions:
- Use simple, professional language.
- Refer to accounts using their accountId where helpful.
- Explain why the cash situation is normal or abnormal.
- Propose specific, actionable next steps (e.g., rebalance, invest excess cash, review client objectives).
- Do not give any tax or legal advice.
- If information is incomplete, say what else you would like to know.
"""


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> List[Dict[str, Any]]:
    if file.content_type not in {"application/json", "text/json"}:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type, please upload a JSON account array.",
        )

    payload = await file.read()
    try:
        accounts = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}")

    if not isinstance(accounts, list) or not accounts:
        raise HTTPException(status_code=400, detail="JSON must be a non-empty array of accounts")

    rows = extract_feature_matrix(accounts)
    df = pd.DataFrame(rows)
    try:
        preds = model.predict(df)
        scores = model.decision_function(df)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {exc}")

    recommendations = build_recommendations(rows, preds.tolist(), scores.tolist(), accounts)
    logger.info("Generated %s recommendations", len(recommendations))
    
    # Store in memory for follow-up questions
    anomaly_count = sum(1 for r in recommendations if r.get("is_anomaly"))
    analysis_memory["recommendations"] = recommendations
    analysis_memory["summary"] = (
        f"Analyzed {len(recommendations)} accounts, "
        f"{anomaly_count} flagged as anomalies."
    )
    
    return recommendations


@app.post("/cash-analysis")
async def cash_analysis(payload: CashAnalysisRequest) -> Dict[str, Any]:
    # Check if we have stored analysis context
    if not analysis_memory.get("recommendations"):
        raise HTTPException(
            status_code=400,
            detail="No analysis context found. Please run /analyze first."
        )
    
    try:
        # Build context from memory
        context = {
            "accounts": analysis_memory["recommendations"],
            "summary": analysis_memory["summary"]
        }
        
        prompt = build_gemini_prompt(question=payload.question, context=context)
        
        # Use new google-genai client API
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt
        )
        
        # Extract text from response
        answer_text = response.text if hasattr(response, "text") else str(response)
        
        logger.info("Gemini analysis completed for question: %s", payload.question[:50])
        return {"answer": answer_text, "model": "gemini-2.0-flash-exp"}
    except Exception as exc:
        error_msg = str(exc)
        logger.exception("Gemini cash analysis failed: %s", exc)
        
        # Handle RESOURCE_EXHAUSTED (429) errors with generic message
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            raise HTTPException(
                status_code=429,
                detail="Resource exhausted. API quota exceeded. Please try again later."
            )
        
        # Handle other errors
        raise HTTPException(status_code=500, detail="Cash analysis failed. Please try again.")
