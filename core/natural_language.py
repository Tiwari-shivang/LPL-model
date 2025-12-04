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
            "Act as a highly experienced senior financial analyst with decades of expertise in portfolio construction, "
    "cash efficiency, liquidity behavior, risk evaluation, and trading patterns. Analyze all provided account "
    "snapshots, intelligence signals, and anomaly classifications (is_anomaly) and produce a clean, structured "
    "HTML-formatted summary.\n\n"

    "Your response MUST be valid HTML using <h3>, <ul>, <li>, <strong>, and <br/> tags. Keep the tone professional, "
    "calm, rational, and insight-driven — no storytelling, no long paragraphs, and no personal names.\n\n"

    "The response must include the following sections, exactly in this structure:\n\n"

    "<h2>Expert Summary</h2>\n"
    "<ul>\n"
    "  <li><strong>Overview:</strong>\n"
    "    <ul>\n"
    "      <li>Short bullets explaining overall portfolio condition</li>\n"
    "      <li>Use numbers and percentages when relevant</li>\n"
    "    </ul>\n"
    "  </li>\n"
    "</ul>\n\n"

    "<h3>Anomalous Accounts Explanation</h3>\n"
    "<ul>\n"
    "  <li><strong>Why accounts are flagged:</strong>\n"
    "    <ul>\n"
    "      <li>Explain unusual cash behavior</li>\n"
    "      <li>Highlight operational or behavioral signals</li>\n"
    "      <li>Indicate if caused by deposits, withdrawals, dividends, drift, liquidity imbalance, or model deviation</li>\n"
    "      <li>Note structural issues such as idle cash, oversized build-up, or unusual drawdowns</li>\n"
    "    </ul>\n"
    "  </li>\n"
    "</ul>\n\n"

    "<h3>Pattern Detection</h3>\n"
    "<ul>\n"
    "  <li><strong>Common patterns across anomalies:</strong>\n"
    "    <ul>\n"
    "      <li>Short bullets summarizing repeated behaviors (e.g., persistent drift, high idle cash, timing patterns)</li>\n"
    "      <li>Explain why these patterns matter, using clear financial terms</li>\n"
    "    </ul>\n"
    "  </li>\n"
    "</ul>\n\n"

    "<h3>Anomalous vs. Normal Accounts</h3>\n"
    "<ul>\n"
    "  <li><strong>Comparison:</strong>\n"
    "    <ul>\n"
    "      <li>Show how stable accounts behave differently from anomalous accounts</li>\n"
    "      <li>Highlight differences in cash efficiency, drift behavior, and trading consistency</li>\n"
    "    </ul>\n"
    "  </li>\n"
    "</ul>\n\n"

    "<h3>Key Insights</h3>\n"
    "<ul>\n"
    "  <li><strong>Important observations:</strong>\n"
    "    <ul>\n"
    "      <li>Short, punchy bullets using advisor-friendly keywords</li>\n"
    "      <li>Must be easy to scan and understand quickly</li>\n"
    "    </ul>\n"
    "  </li>\n"
    "</ul>\n\n"

    "<h3>AI Recommendations</h3>\n"
    "<ul>\n"
    "  <li><strong>Action items:</strong>\n"
    "    <ul>\n"
    "      <li>Provide 3–5 practical steps aligned with risk control, cash deployment, rebalance actions, or review needs</li>\n"
    "      <li>Keep each item one line and actionable</li>\n"
    "    </ul>\n"
    "  </li>\n"
    "</ul>\n\n"

    "Rules:\n"
    "• Output must be HTML only — no markdown.\n"
    "• Each bullet must be short and high-value.\n"
    "• Use digits and percentages whenever relevant.\n"
    "• Avoid fluff and long paragraphs — focus on actionable financial insights.\n"
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

    async def summarize_master(self, analysis_payload: Dict[str, Any]) -> str:
        """
        Summarize master-agent outputs across cash, positions, and historical trading.
        """
        if self.client is None:
            return (
                "LLM summary unavailable (OPENAI_API_KEY or openai package missing). "
                "Review cash, position, and trading sections above."
            )

        prompt = (
"Act as a senior portfolio analyst. Using the provided model outputs, generate a clean HTML-formatted summary "
    "for the three sections: Cash analysis, Position analysis, and Historical Trading.\n\n"
    "Your response MUST be valid HTML and follow this exact nested structure:\n\n"
    "<h3>Cash analysis</h3>\n"
    "<ul>\n"
    "  <li><strong>What is happening:</strong>\n"
    "    <ul>\n"
    "      <li>short pointer using financial keywords</li>\n"
    "      <li>short pointer with relevant digits</li>\n"
    "    </ul>\n"
    "  </li>\n"
    "  <li><strong>Why this is happening:</strong>\n"
    "    <ul>\n"
    "      <li>root cause pointer with numeric context</li>\n"
    "    </ul>\n"
    "  </li>\n"
    "  <li><strong>AI recommendations:</strong>\n"
    "    <ul>\n"
    "      <li>short, actionable pointer</li>\n"
    "    </ul>\n"
    "  </li>\n"
    "</ul>\n\n"
    "<h3>Position analysis</h3>\n"
    "<ul>\n"
    "  <li><strong>What is happening:</strong>\n"
    "    <ul>\n"
    "      <li>short pointer</li>\n"
    "    </ul>\n"
    "  </li>\n"
    "  <li><strong>Why this is happening:</strong>\n"
    "    <ul>\n"
    "      <li>short root cause pointer</li>\n"
    "    </ul>\n"
    "  </li>\n"
    "  <li><strong>AI recommendations:</strong>\n"
    "    <ul>\n"
    "      <li>short action pointer</li>\n"
    "    </ul>\n"
    "  </li>\n"
    "</ul>\n\n"
    "<h3>Historical trading</h3>\n"
    "<ul>\n"
    "  <li><strong>What is happening:</strong>\n"
    "    <ul>\n"
    "      <li>short pointer</li>\n"
    "    </ul>\n"
    "  </li>\n"
    "  <li><strong>Why this is happening:</strong>\n"
    "    <ul>\n"
    "      <li>behavior or pattern cause pointer</li>\n"
    "    </ul>\n"
    "  </li>\n"
    "  <li><strong>AI recommendations:</strong>\n"
    "    <ul>\n"
    "      <li>one-line corrective action</li>\n"
    "    </ul>\n"
    "  </li>\n"
    "</ul>\n\n"
    "Rules:\n"
    "• Output MUST be a valid HTML string only — no markdown.\n"
    "• Each list item must be short and high-value.\n"
    "• Use financial keywords advisors understand (e.g., drift, cash drag, turnover, deviation, overweight).\n"
    "• Include exact digits/percentages from the model output.\n"
    "• Do NOT write long paragraphs — only structured HTML lists.\n"
        )

        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(analysis_payload, default=str)},
                ],
                temperature=0.35,
            )
            return resp.choices[0].message.content.strip()
        except (OpenAIError, asyncio.TimeoutError, Exception):
            return "Unable to generate master summary right now. Please try again shortly."

    async def answer_master_follow_up(
        self, question: str, analysis_payload: Dict[str, Any]
    ) -> str:
        """
        Answer a follow-up question grounded in master agent output.
        """
        if self.client is None:
            return (
                "LLM unavailable. Please configure OPENAI_API_KEY and install openai package."
            )

        prompt = (
            "You are a senior trader with 50+ years of experience. "
            "Using the provided Master Agent output (cash analysis, position drift, trading activity), "
            "answer my follow-up question with a short, high-clarity insight.\n\n"
            "Your answer should:\n"
            "- Stay tightly grounded in the data\n"
            "- Highlight the key trading implication\n"
            "- Provide a simple next step a professional would take\n"
            "- Remain brief and avoid long explanations"
        )

        payload = {"question": question, "analysis": analysis_payload}

        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(payload, default=str)},
                ],
                temperature=0.35,
            )
            return resp.choices[0].message.content.strip()
        except (OpenAIError, asyncio.TimeoutError, Exception):
            return "Unable to answer follow-up right now. Please try again shortly."

    async def answer_cash_follow_up(
        self, question: str, cash_payload: Dict[str, Any]
    ) -> str:
        """
        Answer a cash-only follow-up using Buffett-style concise guidance.
        """
        if self.client is None:
            return (
                "LLM unavailable. Please configure OPENAI_API_KEY and install openai package."
            )

        prompt = (
            "Act as 50 years experienced trader/broker, applying calm, long-term, principle-driven judgment. "
            "Use ONLY the provided cash-analysis data to answer the follow-up question.\n\n"
            "Your response MUST:\n"
            "• Be short, high-value, and no more than 12 lines\n"
            "• Include precise digits and percentages directly from the data\n"
            "• Provide simple, investor-friendly interpretation of what the numbers mean\n"
            "• Highlight the most important implication for capital efficiency\n"
            "• Give exactly ONE clear next step a trader should take\n"
            "• Use HTML-friendly formatting such as \\n, **bold**, and structured short lines\n"
            "• Do NOT restate the entire payload; summarize only the insight\n\n"
            "Write with Buffett’s clarity, calmness, and long-term focus."
        )

        payload = {"question": question, "cash_analysis": cash_payload}
        user_message = (
            f"Follow-up question: {question}\n\n"
            f"Cash analysis data:\n{json.dumps(cash_payload, default=str)}"
        )

        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.25,
            )
            return resp.choices[0].message.content.strip()
        except (OpenAIError, asyncio.TimeoutError, Exception):
            return "Unable to answer cash follow-up right now. Please try again shortly."
