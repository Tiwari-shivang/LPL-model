# 💼 Cash Analysis Q&A with Gemini — Endpoint `/cash-analysis`

This document explains how to add a new endpoint `/cash-analysis` to your existing FastAPI + Isolation Forest service.

The new endpoint will:

- Take a **user question** (follow-up) from the frontend  
- Take the **current analyzed data** (recommendations/summary from `/analyze`)  
- Call **Google Gemini** with a **“best finance advisor” persona**  
- Return a **natural-language answer** for the broker

---

## 1. High-Level Flow

1. **User runs analysis**  
   - Frontend calls `POST /analyze` with all account data  
   - Backend runs Isolation Forest  
   - Recommendations are emitted (e.g., via Socket.IO) and shown in UI

2. **User has a follow-up question**  
   - User clicks **“I have a follow up question”**  
   - UI shows a text input and a **Send** button

3. **Frontend calls `/cash-analysis`**  
   - Payload contains:
     - `question` → user’s text  
     - `context` → relevant analyzed data from `/analyze` (e.g. selected accounts, anomalies)

4. **Backend `/cash-analysis`**  
   - Builds a prompt with:
     - Persona: *expert finance advisor*  
     - User question  
     - Context: anomaly data, cash drift, etc.  
   - Sends it to **Gemini API**  
   - Returns Gemini’s answer in JSON

5. **Frontend shows the answer**  
   - Display answer as a chat bubble or explanation panel under the analysis.

---

## 2. Prerequisites

### 2.1. You already have

- FastAPI app with `/analyze` endpoint  
- Isolation Forest model working and generating recommendations  
- Frontend that:
  - Calls `/analyze`
  - Shows list/table of recommendations or analyzed cash data

### 2.2. You need additionally

- A **Gemini API key** (you already have, free tier is fine)
- Python library: `google-generativeai`
- Optional: `python-dotenv` to load env variables from `.env`

Install:

```bash
pip install google-generativeai python-dotenv
```

---

## 3. Designing the `/cash-analysis` Endpoint

We will create a **new POST endpoint** in your FastAPI app:

```http
POST /cash-analysis
Content-Type: application/json
```

### 3.1. Request Body Shape

Recommended schema:

```json
{
  "question": "How should I manage cash anomalies in these accounts?",
  "context": {
    "accounts": [
      {
        "accountId": "293",
        "features": {
          "total_trade_amount": 12000.0,
          "cash_trade_amount": 5000.0,
          "cash_trade_ratio": 0.4167,
          "cash_model_percent": 20.0,
          "cash_vs_model_diff": 21.6667
        },
        "is_anomaly": true,
        "score": -0.0368,
        "reasons": ["Cash allocation significantly above model"],
        "recommendationViewed": false
      },
      {
        "accountId": "501",
        "features": {
          "total_trade_amount": 250000.0,
          "cash_trade_amount": 15000.0,
          "cash_trade_ratio": 0.06,
          "cash_model_percent": 10.0,
          "cash_vs_model_diff": -4.0
        },
        "is_anomaly": false,
        "score": 0.08,
        "reasons": [],
        "recommendationViewed": false
      }
    ],
    "summary": "2 accounts analyzed, 1 flagged for high cash above target."
  }
}
```

You can adjust the `context` structure, but it should at least contain:

- A list of accounts with:
  - `accountId`
  - relevant features (cash %, diff from model)
  - `is_anomaly`
  - `reasons`

### 3.2. Response Body

Example response from `/cash-analysis`:

```json
{
  "answer": "Account 293 is holding significantly more cash than the model target (around 21% above). You should consider reallocating part of that cash into the model's recommended securities, starting with underweight sleeves...",
  "model": "gemini-1.5-flash"
}
```

You can also include:

- `suggestedActions`: array of bullet-point actions  
- `tone`: "professional"  

---

## 4. Backend Setup for Gemini

### 4.1. Configure API Key

Create a `.env` file at your project root:

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE
```

Install `python-dotenv` if not already:

```bash
pip install python-dotenv
```

In `app.py` (or a separate module like `gemini_client.py`), configure Gemini:

```python
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
```

---

## 5. Adding the `/cash-analysis` Endpoint

Below is a minimal but complete implementation for your existing `app.py`.

### 5.1. Define Request Model

Add near your other Pydantic models:

```python
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

class CashAnalysisContext(BaseModel):
    accounts: List[Dict[str, Any]]
    summary: Optional[str] = None

class CashAnalysisRequest(BaseModel):
    question: str
    context: CashAnalysisContext
```

### 5.2. Build the Prompt for Gemini

We will give Gemini a persona:

> “You are the best, most trusted, conservative financial advisor helping a professional broker…”

Add helper in `app.py`:

```python
def build_gemini_prompt(question: str, context: Dict[str, Any]) -> str:
    # You can control how much detail to send
    accounts = context.get("accounts", [])
    summary = context.get("summary", "")

    # We keep this simple and structured
    return f'''
You are an experienced financial advisor specializing in portfolio management, cash allocation, and rebalancing.
You are talking to a professional investment advisor (the user), not the end client.
Your job is to explain clearly, be conservative, and avoid overly aggressive risk-taking.

Here is the current cash anomaly analysis context from the ML model:

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
'''
```

> In a real system, you may want to truncate the `accounts` list or only send top N anomalous accounts to keep prompts small.

### 5.3. Implement `/cash-analysis`

Add this endpoint to `app.py`:

```python
@app.post("/cash-analysis")
async def cash_analysis(payload: CashAnalysisRequest) -> Dict[str, Any]:
    try:
        prompt = build_gemini_prompt(
            question=payload.question,
            context=payload.context.dict()
        )

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)

        answer_text = response.text if hasattr(response, "text") else str(response)

        return {
            "answer": answer_text,
            "model": "gemini-1.5-flash"
        }
    except Exception as exc:
        logger.exception("Gemini cash analysis failed: %s", exc)
        raise HTTPException(status_code=500, detail="Cash analysis AI failed")
```

---

## 6. Frontend Integration Flow

### 6.1. After `/analyze` call

When the user runs analysis:

1. You call `POST /analyze`  
2. You receive the recommendations (via WebSocket or REST)  
3. You store them in frontend state, e.g.:

```ts
const [analysisContext, setAnalysisContext] = useState({
  accounts: recommendations,
  summary: `Analyzed ${recommendations.length} accounts, ${
    recommendations.filter(r => r.is_anomaly).length
  } flagged as anomalies.`
});
```

### 6.2. “I have a follow up question” UI

- Show a button:
  ```html
  <button>I have a follow up question</button>
  ```
- On click → open a small panel with:
  - Textarea `question`
  - “Ask AI” button

### 6.3. Call `/cash-analysis` from frontend

Example in TypeScript / Axios:

```ts
import axios from "axios";

async function askCashAnalysis(question: string, context: any) {
  const res = await axios.post("http://localhost:5000/cash-analysis", {
    question,
    context
  });

  return res.data; // { answer, model }
}
```

Usage:

```ts
const onAsk = async () => {
  const result = await askCashAnalysis(userQuestion, analysisContext);
  setAiAnswer(result.answer);
};
```

### 6.4. Display AI Answer

Just show `aiAnswer` below the analysis table, e.g.:

> **AI Advisor:**  
> “Account 293 is holding significantly more cash than its model target…”

---

## 7. Safety & Best Practices

1. **Never let Gemini make final trading decisions**  
   - It should **suggest**, not execute.

2. **Always show it as “AI Assistant”**  
   - Broker is responsible for final judgment.

3. **Limit context size**  
   - Send only relevant accounts (e.g., selected rows or top N anomalies).

4. **Log prompts & responses** (without sensitive info)  
   - Helpful for debugging and improving prompts.

---

## 8. Summary of Steps

1. ✅ Install `google-generativeai` and `python-dotenv`  
2. ✅ Configure `GEMINI_API_KEY` in `.env` and load it  
3. ✅ Design `CashAnalysisRequest` with `question` + `context`  
4. ✅ Build `build_gemini_prompt()` with clear advisor persona  
5. ✅ Implement `POST /cash-analysis` in FastAPI  
6. ✅ Hook frontend:
   - Store analysis results from `/analyze`
   - Add “I have a follow up question” button
   - Call `/cash-analysis` with question + context
   - Display Gemini’s answer in UI

Once this is done, your app becomes a **full AI co-pilot for brokers**:  
- Isolation Forest → finds anomalies  
- Gemini → explains them and suggests actions in simple language.
