# 📘 Agentic ML Cash / Portfolio Analysis System — Updated Architecture (2025)

This updated document reflects the **new requirements** of your Python ML application, now upgraded from a simple Isolation Forest anomaly detector into a fully **agentic, DB‑powered cash‑analysis engine**.

Your Python service **now connects directly to your MSSQL database**, extracts rich portfolio data, performs multi‑layer analysis, and provides broker‑level explanations using your OpenAI key.

---

# 🧱 1. What This System Does (High Level)

Your updated ML system now performs:

- 🔄 Fetch data from backend **and directly from SQL DB**
- 🧬 Extract **advanced cash features** from Accounts, Models, Orders, Tax Lots, Trades, Dividends
- 🌲 Use Isolation Forest **only for anomaly detection**
- 🧠 Run a new **Cash Intelligence Engine** for:
  - Root‑cause analysis
  - Risk scoring
  - Impact projection
  - Action recommendations
- 🗣 Convert outputs into clean, broker‑style insights using **OpenAI**
- 📡 Emit live insights via **WebSockets**
- 💬 Support user follow‑up questions via your UI chat

---

# 🏗 2. Updated Project Structure

```
ml-service/
│
├── db/
│   └── database_client.py        # New MSSQL DB connector
│
├── core/
│   ├── feature_extractor.py      # Now extracts from DB + JSON
│   ├── cash_intelligence.py      # NEW intelligent reasoning engine
│   ├── anomaly_detector.py       # Isolation Forest wrapper
│   ├── recommendation_engine.py  # Rule + ML + LLM actions
│   └── natural_language.py       # Converts output → OpenAI readable
│
├── app.py                        # FastAPI + Socket server
├── model_train.py                # Isolation Forest trainer
├── requirements.txt
│
├── models/
│   └── iso_model.pkl
│
└── data/
    └── training_data.csv
```

---

# 🔌 3. Database Connectivity (New)

Your Python application now uses the following connection string:

```
Server=hvtoms-dev-sqlserver-1.cfemiu68wkqx.ap-south-1.rds.amazonaws.com;
Database=hvtoms-01;
User ID=hvtoms;
Password=Hvt0m$@To25;
Encrypt=True;
TrustServerCertificate=True;
Connection Timeout=60;
Command Timeout=120;
MultipleActiveResultSets=True;
ApplicationIntent=ReadOnly
```

### ✔ How Python connects to MSSQL (pyodbc example)
```python
import pyodbc

def get_connection():
    conn_str = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=hvtoms-dev-sqlserver-1.cfemiu68wkqx.ap-south-1.rds.amazonaws.com;"
        "DATABASE=hvtoms-01;"
        "UID=hvtoms;"
        "PWD=Hvt0m$@To25;"
        "Encrypt=yes;TrustServerCertificate=yes;"
    )
    return pyodbc.connect(conn_str)
```

---

# 🧬 4. New Data Sources (Direct From DB)

Your system can now access all key schemas:

### **Accounts**
- current cash
- market value
- model target cash
- pending trades

### **Orders**
- settlements
- upcoming buys/sells
- recent trade activity

### **Securities / Positions**
- overweight/underweight positions
- cash‑generating assets

### **Tax Lots**
- sells creating cash
- maturing positions

### **Dividends**
- future expected dividends

### **Deposits / Withdrawals**
- recurring investor behavior
- large recent movements

---

# 🤖 5. Isolation Forest (Unchanged Behavior)

Isolation Forest continues to provide ONE signal:

### **“Is this cash behavior anomalous?”**

That’s all. No explanations. No actions.

The real intelligence is added **after** this step.

---

# 🧠 6. The Cash Intelligence Engine (New)

This is the core of your agentic system.

After identifying an anomaly, the engine analyzes DB data to answer 3 questions:

---

## **A) What happened? (Root Cause Analysis)**

Examples:
- Cash increased due to **₹45,000 dividends posted yesterday**
- Cash dropped because of **two settlement trades worth ₹2,10,000**
- Cash drifted due to **model weight change last week**
- Cash is stuck because **pending buy orders are blocked**

---

## **B) Why does it matter? (Risk & Impact)**

Examples:
- Drift is crossing **8% threshold**
- Cash drag may reduce returns by **₹12,000/month**
- Insufficient cash to settle **upcoming orders**
- Portfolio may violate **client IPS**

---

## **C) What should we do? (Recommended Action)**

Examples:
- Deploy ₹1.2L into Model Growth (40% allocation)
- Raise ₹50,000 by trimming overweight positions
- Hold cash temporarily due to expected withdrawal
- Schedule rebalance for next market window

---

# 🏗 7. Recommendation Engine

Your action layer combines:

### ✔ Rule-Based Logic
- cash > model target → deploy
- cash < required settlement → raise cash
- drift > X% → rebalance

### ✔ ML-Based Predictors (Optional)
- cash trend prediction
- dividend forecast

### ✔ LLM (OpenAI) Layer
Converts the technical insights into:

- clean explanations
- risk summaries
- portfolio suggestions
- alternative actions
- follow‑up Q&A

---

# 🗣 8. Natural Language Layer (New)

You now use your **OpenAI key** to generate:

- human-readable broker‑style summaries
- insights like a senior portfolio analyst
- follow‑up question responses

### Example Output
```
Cash is ₹1.18L above model expectations, primarily driven by dividends and lack of reinvestment. Recommend deploying into Growth sleeve to reduce drift from 7.9% to 2.3%.
```

---

# 📡 9. Updated Real-Time Workflow

### **1. Backend**
- Sends account list
- Python service optionally enriches with DB data

### **2. Python Service**
- Extracts features
- Runs Isolation Forest
- If anomaly → Cash Intelligence Engine
- Generates action + explanation
- Calls OpenAI for natural language output
- Emits via WebSockets

### **3. Frontend**
- Shows actionable card:
  - root cause
  - impact
  - recommendation
  - severity
- Provides “Ask follow-up” chat

---

# 🛠 10. Updated requirements.txt

```
pandas
scikit-learn
joblib
fastapi
uvicorn
python-socketio
pyodbc
openai
```

---

# ✔ 11. Final Notes

- Isolation Forest is **only anomaly detection**, nothing else.
- Real agent intelligence comes from:
  - DB-driven root cause analysis
  - Risk & impact modeling
  - Action rules
  - Natural language explanations via OpenAI
- This converts your system into a true **agentic ML assistant** for financial advisors.

---

# 📚 12. Database Table Structures (Reference)

Below are the key tables from your MSSQL system, simplified for ML reference.

---

## **Accounts**
```
(Id, Description, short_name, account_status, accounting_method,
address_line_1, address_line_2, address_line_3,
City, State, zip_code, Country,
TotalMarketValue, CashBalance, AvailableCash, TotalCashAvailable,
model_id,
CreatedAt, UpdatedAt, CreatedBy, UpdatedBy,
IsDeleted)
```

## **Cash Transactions**
```
(Id, portfolio_account_id,
Amount, transaction_type, transaction_date, Comments,
created_at, updated_at, IsDeleted)
```

## **Models**
```
(Id, Name, Description, IsActive, IsDeleted,
CreatedDate, LastModifiedDate, CreatedBy, UpdatedBy)
```

## **ModelSleeves**
```
(Id, ModelId, SleeveId, AllocationPercentage,
CreatedDate, LastModifiedDate,
CreatedBy, UpdatedBy, IsDeleted)
```

## **OrderAllocations**
```
(Id, AllocationEstCost, CurrentQuantity, CurrPercent,
DeltaPercent, EndPercent, EndQuantity,
IsFractional, ModAppPercent, ModelPercent,
Quantity, Tolerance, OrderId, AccountId,
IsDeleted)
```

## **Orders**
```
(Id, AccountName, AccountingMethod,
ApprovalStatus, Comment,
CreateDate, EstCost, FilledPrice, FilledQuantity,
OrderId, OrderType, Quantity, RequestId,
SettleDate, State, TradeDate, Tran,
UpdatedAt, SecurityId, AccountId, IsDeleted)
```

## **Securities**
```
(Id, Name, Symbol, CUSIP,
SecurityTypeId, Currency, Price, LastPriceDate,
PreviousClosingPrice, Rate, Description,
IsActive, IsTradeable,
CreatedAt, UpdatedAt, CreatedBy, UpdatedBy,
IsDeleted)
```

## **SecurityTypes**
```
(Id, Name, SecurityTypeCode,
PricingMultiplier, ShareDecimal,
CFICode, SecurityTypeDescription,
IsActive, PriceDecimals, HoldingPeriod,
StaleDataCheck, StaleDataWindow,
CreatedAt, UpdatedAt, CreatedBy, UpdatedBy,
IsDeleted)
```

## **Sleeves**
```
(Id, Name, Description,
IsActive, IsDeleted,
CreatedDate, LastModifiedDate,
CreatedBy, UpdatedBy)
```

## **SleeveSecurities**
```
(Id, SleeveId, SecurityId, AllocationPercentage,
CreatedDate, LastModifiedDate,
IsDeleted)
```

## **TaxLots**
```
(Id, OriginalPrice, OriginalTradeDate,
Quantity, ReservedQuantity,
SellPrice, SoldQuantity, TaxLotType,
OrderAllocationId, AccountId, SecurityId,
IsDeleted)
```

---

# 📘 13. How to Use Both Training Datasets to Retrain the Model

Your retraining now uses **two datasets**:

1. **Trade Logs Dataset** (`synthetic_training_logs.csv`)
2. **Positions Dataset** (`synthetic_positions_dataset.csv`)

Both combine to give the model enough signals to detect:
- abnormal cash changes
- drift patterns
- liquidity anomalies
- cash-from-trades behavior

---

## **Step 1: Load Both Datasets**
```python
import pandas as pd

logs = pd.read_csv("data/synthetic_training_logs.csv")
pos = pd.read_csv("data/synthetic_positions_dataset.csv")
```

---

## **Step 2: Merge Datasets on account_id**
```python
merged = logs.merge(pos, on="account_id", how="left")
```

---

## **Step 3: Build the Feature Matrix**
Suggested combined features:
```
cash_before
cash_after
model_cash_target
actual_cash_percent
quantity
avg_price
market_price
market_value
model_percent_target
current_percent
drift_percent
```

```python
feature_cols = [
    "cash_before", "cash_after", "model_cash_target",
    "actual_cash_percent", "quantity", "avg_price",
    "market_price", "market_value", "model_percent_target",
    "current_percent", "drift_percent"
]

X = merged[feature_cols]
```

---

## **Step 4: Train Isolation Forest**
```python
from sklearn.ensemble import IsolationForest
import joblib

model = IsolationForest(
    n_estimators=400,
    contamination=0.05,
    random_state=42
)

model.fit(X)
joblib.dump(model, "models/iso_model.pkl")
```

---

# 🎉 System is now fully upgraded to Agentic + DB Powered + Broker-Level Intelligence

