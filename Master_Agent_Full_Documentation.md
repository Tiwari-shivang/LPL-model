
> **UPDATE — Agentic Master Model Requirement**
> - The final system objective is to build a **fully agentic Master Agent**.
> - A **new standalone XGBoost model must be created and saved** (e.g., `xgb_master_model.pkl`).
> - This Master Agent model is **separate from the existing `iso_model.pkl`**.
> - The Master XGBoost model will be **invoked exclusively via the `/summarize` API`**.
> - All original requirements in this document remain unchanged.



# 📘 Master Agent — XGBoost Portfolio Summary Model  
*(Cash Analysis + Position Analysis + Historical Trading)*  

This document explains how to create a **new XGBoost-based Master Agent** capable of providing a **combined portfolio summary**, using insights from:

- **Cash Analysis**
- **Position Analysis**
- **Historical Trading**

The Master Agent will work **alongside your existing Cash Agent**, but will run through a **separate model file** and **separate endpoint** inside the same repo.

---

# 🧱 1. What This New Model Does (High-Level)

The Master Agent model is designed to act like a senior portfolio analyst evaluating **overall account health**, combining three domains:

### ✔ Cash Behavior  
Liquidity, drift, cash movements, pending orders.

### ✔ Position Risk  
Concentration, diversification, volatility exposure, asset class balance.

### ✔ Historical Trading Behavior  
Turnover, short-term selling, gain/loss patterns, trade discipline.

---

# 🏛 2. Model Architecture Overview

```
master-agent/
│
├── data/
│   └── historical_trading_dataset.csv
│   └── position_analysis_dataset.csv
│   └── cash_dataset.csv
│
├── models/
│   └── xgb_master_model.json
│
├── core/
│   ├── master_feature_builder.py
│   ├── master_model.py
│   ├── master_explainer.py
│   └── natural_language.py
│
└── app.py
```

The Master Agent will be imported into the same FastAPI app and exposed through:

```
/analyze/master
```

---

# 🧬 3. Features Required for XGBoost Model

## 🟦 3.1 Cash Analysis Features
- cash_balance  
- cash_percent_of_portfolio  
- model_cash_target  
- cash_drift  
- deposits_30d  
- withdrawals_30d  
- pending_order_cost  

## 🟦 3.2 Position Risk Features
- security_count  
- top_position_weight  
- position_concentration_score  
- asset_class_equity_percent  
- asset_class_bond_percent  
- asset_class_cash_percent  
- international_percent  
- sector_exposures (tech, finance, health)  
- drift_equity  
- drift_bond  

## 🟦 3.3 Historical Trading Features
- trades_last_30d  
- buy_to_sell_ratio  
- turnover_rate  
- realized_gain_loss  
- holding_period_days_avg  
- short_term_flag_percent  
- liquidity_score  
- trade_success_score  

---

# 🏗 4. Building the Master Dataset

### Step 1 — Load datasets

```python
import pandas as pd

pos = pd.read_csv("data/position_analysis_dataset.csv")
hist = pd.read_csv("data/historical_trading_dataset.csv")
cash = pd.read_csv("data/cash_analysis_dataset.csv")

hist_agg = hist.groupby("account_id").agg({
    "trade_id": "count",
    "realized_gain_loss": "mean",
    "holding_period_days": "mean",
    "short_term_flag": "mean",
    "trade_success_score": "mean",
    "volatility_score": "mean"
}).reset_index()
```

### Step 2 — Merge into one master dataframe

```python
master = (
    pos
    .merge(hist_agg, on="account_id", how="left")
    .merge(cash, on="account_id", how="left")
)
```

### Step 3 — Create the target label

```python
master["portfolio_health_score"] = ...
```

---

# 🤖 5. Training the XGBoost Model

```python
from xgboost import XGBRegressor
import joblib

features = master.drop(columns=["portfolio_health_score", "account_id"])
labels = master["portfolio_health_score"]

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.7,
    random_state=42
)

model.fit(features, labels)
joblib.dump(model, "models/xgb_master_model.json")
```

---

# 🧠 6. The Master Agent Explanation Layer

The model output provides:

### ✔ Cash Summary  
### ✔ Position Summary  
### ✔ Trading Summary  
### ✔ Overall Portfolio Summary  

---

# 🗣 7. Natural Language Layer

```python
prompt = '''
Summarize the following portfolio analytics:

Cash:
- Drift: {drift}
- Pending Orders: {pending}

Positions:
- Top Security Weight: {top_weight}
- Concentration Score: {conc_score}

Trading:
- Trades Last 30d: {trades_30d}
- Short Term %: {short_term}

Provide:
- A professional summary
- Three key insights
- Three recommendations
'''
```

---

# 📡 8. FastAPI Endpoint

```python
@app.post("/analyze/master")
def analyze_master(request: MasterRequest):
    df = feature_builder.build(request.account_id)
    score = master_model.predict(df)
    summary = explainer.generate(score, df)
    return summary
```

---

# 🧩 9. Expected Final Output

```json
{
  "cash_analysis": {
    "summary": "Cash above target due to pending trades.",
    "risk": "Low",
    "suggestion": "Deploy excess during rebalance."
  },
  "position_analysis": {
    "summary": "Moderate concentration in top security.",
    "risk": "Medium",
    "suggestion": "Trim overweight exposure."
  },
  "historical_trading": {
    "summary": "Elevated short-term selling detected.",
    "risk": "Medium",
    "suggestion": "Reduce turnover to improve tax efficiency."
  },
  "overall_health_score": 74
}
```

---

# 🎉 10. Conclusion

You now have the full architecture, training pipeline, feature list, and expected output format to build a **Master XGBoost Portfolio Model** that generates summaries using cash, position, and historical trading insights.
