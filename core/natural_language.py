"""
Natural language generation for broker-style summaries and Q&A using OpenAI.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List

try:
    from openai import AsyncOpenAI, OpenAIError
except ImportError:  # pragma: no cover - dependency guard
    AsyncOpenAI = None

    class OpenAIError(Exception):
        ...

DEFAULT_OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")


class NaturalLanguageHelper:
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini") -> None:
        self.model = model
        if AsyncOpenAI is None:
            self.client = None
        else:
            self.client = AsyncOpenAI(api_key=api_key or DEFAULT_OPENAI_KEY)

    async def summarize_account(
        self, account: Dict[str, Any], intelligence: Dict[str, Any]
    ) -> str:
        prompt = (
            "You are a senior portfolio analyst. Create a crisp broker-style summary "
            "about the account's cash position. Keep it under 120 words and focus on "
            "root causes, risk, and recommended action. Avoid tax/legal advice."
        )
        content = {
            "account_id": account.get("account_id"),
            "account_name": account.get("account_name"),
            "intelligence": intelligence,
        }
        if self.client is None:
            return (
                "Cash positioning analyzed. Follow recommendations and monitor drift; "
                "no LLM summary available right now."
            )
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": json.dumps(content, default=str),
                    },
                ],
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except (OpenAIError, asyncio.TimeoutError, Exception):
            return (
                "Cash positioning analyzed. Follow recommendations and monitor drift; "
                "no LLM summary available right now."
            )

    async def answer_follow_up(
        self, question: str, recommendations: List[Dict[str, Any]], summary: str
    ) -> str:
        prompt = (
            "You are a helpful cash management assistant for financial advisors. "
            "Use the provided recommendations to answer the follow-up question. "
            "Stay concise and avoid speculative statements."
        )
        payload = {"summary": summary, "recommendations": recommendations, "question": question}
        if self.client is None:
            return (
                "Stored analysis reviewed. Please rerun after configuring OPENAI_API_KEY "
                "and installing openai package to enable follow-up answers."
            )
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(payload, default=str)},
                ],
                temperature=0.4,
            )
            return resp.choices[0].message.content.strip()
        except (OpenAIError, asyncio.TimeoutError, Exception):
            return "Unable to generate an answer right now. Please try again shortly."

    async def summarize_batch(self, recommendations: List[Dict[str, Any]]) -> str:
        """
        Generate a single Warren Buffett-style assessment across all accounts.
        """
        if self.client is None:
            return (
                "LLM summary unavailable (OPENAI_API_KEY or openai package missing). "
                "Review anomalies and actions above."
            )

        prompt = (
            "Act as Warren Buffett — a world-renowned investor, financial advisor, and former broker with decades of "
            "experience in evaluating portfolio construction, cash efficiency, liquidity behavior, and risk. Analyze "
            "all provided account snapshots, intelligence signals, and the anomaly classifications (is_anomaly).\n\n"
            "Your task is to provide a formal, deeply reasoned financial assessment that includes the following sections:\n\n"
            "1. Detailed Expert Summary (Warren Buffett Style)\n"
            "Explain the cash condition, portfolio behavior, and overall financial posture of the accounts with the calm, "
            "long-term, principle-driven tone Warren Buffett is known for. Avoid unnecessary technical jargon. Focus on "
            "rational analysis, capital allocation discipline, and client-first judgement.\n\n"
            "2. Why Each is_anomaly: true Portfolio Is Anomalous\n"
            "For every account flagged as an anomaly, clearly explain:\n"
            "- What is unusual about its cash behavior\n"
            "- What operational or behavioral signals are causing the anomaly\n"
            "- Whether these issues stem from trades, deposits/withdrawals, dividends, drift, liquidity imbalance, or model deviation\n"
            "- Any underlying structural issues (e.g., lack of reinvestment, outsized cash accumulation, unexplained drawdowns)\n"
            "Write these explanations as if mentoring a junior advisor at Berkshire Hathaway.\n\n"
            "3. Pattern Detection Across All Anomalies\n"
            "Identify common threads among all is_anomaly: true accounts, such as repeated high drift behaviors, consistently elevated idle cash, sudden unexplained deposits, cash spikes after trade settlements, under-funded accounts, significant deviations from model cash targets, and unusual timing patterns. Explain why these patterns matter and what long-term implications they carry.\n\n"
            "4. Compare Anomalous vs. Non-Anomalous Portfolios\n"
            "Describe the differences in behavior between accounts marked is_anomaly: true vs false. Highlight what stable (normal) accounts have in common, what abnormal accounts consistently lack, and how the behaviors diverge in terms of cash efficiency and allocation discipline.\n\n"
            "5. Key Insights (Bullet Points)\n"
            "Provide short, essential observations Warren Buffett would highlight, focusing on cash utilization, drift and allocation alignment, liquidity patterns, trade-driven movements, advisory risks, and capital efficiency.\n\n"
            "6. Recommendations (Warren-style Guidance)\n"
            "Present clear, conservative, client-protective recommendations (deploy cash, trim positions, rebalance, wait due to upcoming flows, deeper review), with long-term, principle-centered advice.\n\n"
            "Tone & Style: Formal, authoritative, calm, rational; long-term focus; avoid hype or technical noise; speak as a seasoned financial mentor; simple but deeply insightful explanations."
        )

        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(recommendations, default=str)},
                ],
                temperature=0.4,
            )
            return resp.choices[0].message.content.strip()
        except (OpenAIError, asyncio.TimeoutError, Exception):
            return "Unable to generate portfolio-wide summary right now. Please try again shortly."
