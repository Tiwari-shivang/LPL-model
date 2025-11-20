
# 📘 ML Cash / Portfolio Anomaly Detection — Complete Setup Guide

This document provides a full end‑to‑end setup guide for building a Python‑based **Real‑Time Portfolio Anomaly Detection System** using **Isolation Forest**, connected to your backend and emitting live recommendations via **WebSockets**.

---

# 🧱 1. Overview

This ML service performs:

- Fetch data from backend (array of account JSON snapshots)
- Extract features from deeply nested portfolio JSON
- Run **Isolation Forest** to detect anomalies (e.g., cash above target)
- Generate human‑readable recommendations
- Emit recommendations to frontend using **WebSockets**
- Maintain `recommendationViewed: true/false` state

The system is designed for portfolio rebalancing, cash analysis, abnormal activity detection, and advisor insights.

---

# 🖥 2. Prerequisites

### Software Requirements
- Python **3.9+**
- pip (latest)
- Virtual environment support
- Node.js (for frontend socket tests)
- Backend capable of sending account JSON arrays

### Python Libraries
These will be included in `requirements.txt`:
```
pandas
scikit-learn
joblib
fastapi
uvicorn
python-socketio
```

### Understanding Required
- JSON data structures
- Basic Python scripting
- REST API calls
- WebSocket communication

---

# 📦 3. Project Structure

```
ml-service/
│
├── app.py                     # Main FastAPI + Socket server
├── model_train.py             # Isolation Forest training job
├── feature_extractor.py       # Extracts numeric features from JSON
├── requirements.txt
│
├── models/
│   └── iso_model.pkl          # Saved Isolation Forest model
│
└── data/
    └── training_data.csv      # Historical dataset for training
```

---

# 📥 4. Input Format (Backend → ML Service)

Backend sends:

```
POST /analyze
[
  {...account1_json...},
  {...account2_json...},
  ...
]
```

Each account JSON contains:
- Sleeves
- Securities
- Trade amounts
- Model percentages
- Market values  
(and more)

---

# 🧬 5. Feature Extraction

Raw JSON is large, nested, and unsuitable for ML.

We convert each account into flat numeric features:

### Key extracted features
- `total_trade_amount`
- `cash_trade_amount`
- `cash_trade_ratio`
- `cash_model_percent`
- `cash_vs_model_diff`
- `total_securities`
- `num_sleeves`

### Feature extractor file (`feature_extractor.py`):

```python
def extract_features_from_account(account):
    root = account

    total_trade_amount = root.get("TradeAmount", 0.0)
    sleeves = root.get("RebalancingModels", [])

    cash_trade_amount = 0.0
    cash_model_percent = 0.0
    total_securities = 0

    for sleeve in sleeves:
        for sec in sleeve.get("RebalancingModels", []):
            total_securities += 1
            sec_type = sec.get("SecurityType", "")
            sec_trade = sec.get("TradeAmount", 0.0)
            sec_model_percent = sec.get("ModelPercent", 0.0)

            if "Cash" in sec_type:
                cash_trade_amount += sec_trade
                cash_model_percent += sec_model_percent

    if total_trade_amount > 0:
        cash_trade_ratio = cash_trade_amount / total_trade_amount
    else:
        cash_trade_ratio = 0

    cash_vs_model_diff = cash_trade_ratio * 100 - cash_model_percent

    return {
        "total_trade_amount": total_trade_amount,
        "cash_trade_amount": cash_trade_amount,
        "cash_trade_ratio": cash_trade_ratio,
        "cash_model_percent": cash_model_percent,
        "cash_vs_model_diff": cash_vs_model_diff,
        "total_securities": total_securities,
        "num_sleeves": len(sleeves),
    }
```

---

# 🧪 6. Training the Isolation Forest Model

### Training dataset format (`training_data.csv`)
Each row represents one account snapshot:

```
account_id,total_trade_amount,cash_trade_amount,cash_trade_ratio,cash_model_percent,cash_vs_model_diff,total_securities,num_sleeves
```

### Training script (`model_train.py`):

```python
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

df = pd.read_csv("data/training_data.csv")

feature_cols = [
    "total_trade_amount",
    "cash_trade_amount",
    "cash_trade_ratio",
    "cash_model_percent",
    "cash_vs_model_diff",
    "total_securities",
    "num_sleeves",
]

X = df[feature_cols]

model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)

model.fit(X)
joblib.dump(model, "models/iso_model.pkl")

print("Model trained successfully.")
```

Run:

```bash
python model_train.py
```

---

# ⚙️ 7. Real‑Time Analysis Service (`app.py`)

This service:
- Accepts backend’s JSON array
- Runs feature extraction
- Predicts anomalies
- Creates recommendations
- Emits via WebSockets

```python
import pandas as pd
import joblib
import socketio
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

from feature_extractor import extract_features_from_account

app = FastAPI()
sio = socketio.AsyncServer(cors_allowed_origins="*")
model = joblib.load("models/iso_model.pkl")

class Payload(BaseModel):
    data: List[Dict[str, Any]]

@app.post("/analyze")
async def analyze(payload: Payload):
    accounts = payload.data
    rows = []
    meta = []

    for acc in accounts:
        feats = extract_features_from_account(acc)
        rows.append(feats)
        meta.append({"accountId": acc.get("AccountId", "Unknown")})

    df = pd.DataFrame(rows)
    preds = model.predict(df)
    scores = model.decision_function(df)

    recommendations = []

    for i, feats in enumerate(rows):
        is_anomaly = preds[i] == -1
        score = float(scores[i])

        reason = []
        if feats["cash_vs_model_diff"] > 5:
            reason.append("Cash allocation significantly above model")
        if feats["cash_vs_model_diff"] < -5:
            reason.append("Cash allocation significantly below model")

        recommendations.append({
            **meta[i],
            "features": feats,
            "is_anomaly": is_anomaly,
            "score": score,
            "reasons": reason,
            "recommendationViewed": False
        })

    await sio.emit("ml_recommendations", recommendations)
    return {"status": "ok", "count": len(recommendations)}
```

---

# 📡 8. WebSocket Configuration

In the same folder, create:

### `socket_server.py` (handled inside `app.py` using python-socketio)

Frontend listens:

```javascript
socket.on("ml_recommendations", (data) => {
   console.log("Received ML recommendations:", data);
});
```

---

# 👁 9. recommendationViewed Handling

Each recommendation includes:

```json
{
  "accountId": "123",
  "is_anomaly": true,
  "score": -0.42,
  "reasons": ["Cash above model"],
  "recommendationViewed": false
}
```

### Frontend will mark it viewed:

```javascript
recommendation.recommendationViewed = true;
```

No ML logic changes — only frontend UI logic.

---

# 🚀 10. How the Entire Workflow Runs

### **1. Backend**
- Gathers all accounts & rebalance JSON
- Sends array to ML service:  
  `POST /analyze`

### **2. Python ML Service**
- Extracts features  
- Runs Isolation Forest  
- Generates recommendations  
- Emits via socket:  
  `"ml_recommendations"`

### **3. Frontend**
- Displays recommendations  
- Marks `recommendationViewed = true` when user sees them

---

# 🧪 11. Running the Service

### Start backend:
(Your backend)

### Start ML service:

```bash
uvicorn app:app --reload --port 8001
```

### Socket & FastAPI now live:
- REST: `http://localhost:8001/analyze`
- WebSocket emitted on channel: `"ml_recommendations"`

---

# 📁 12. requirements.txt

```
pandas
scikit-learn
joblib
fastapi
uvicorn
python-socketio
```

Install:

```bash
pip install -r requirements.txt
```

---

# 🏁 13. Final Notes

- Isolation Forest is **unsupervised**, no labels needed
- Feature extraction is the MOST IMPORTANT step
- Recommendations can be fully customized
- Model can be retrained any time with more data
- This design supports:
  - Real‑time ML
  - Dashboard alerts
  - Historical analysis
  - Multi‑account processing

---

# 🎉 The system is now fully defined & ready to build!
