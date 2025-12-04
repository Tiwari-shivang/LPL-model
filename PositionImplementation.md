Here’s a clean **Markdown summary** you can drop straight into your docs.
It explains, step-by-step, **how to fetch data from your SQL tables and feed it into your XGBoost position-anomaly model**.

---

# 📘 Position Anomaly Model — Data Fetch & Feature Pipeline

This document explains, step by step, **how to pull data from your SQL Server tables** and **prepare features for the XGBoost position anomaly model**.

Goal:
For each **account + security**, tell the model:

> “What did the model want, what actually happened in trading, and how far did we drift?”

---

## 🧱 1. Tables & What They’re Used For

From your schema:

* **Accounts** – portfolio metadata and total market values

  * `accounts`
* **Models & Sleeves** – model allocation structure

  * `Models`, `ModelSleeves`, `Sleeves`, `SleeveSecurities`
* **Securities & SecurityTypes** – symbol, price, asset class

  * `Securities`, `SecurityTypes`
* **Orders & OrderAllocations** – what was actually traded

  * `Orders`, `OrderAllocations`
* **TaxLots** – historical lots & P/L (optional for position anomalies)

  * `TaxLots`
* **cash_transactions** – used mainly for cash agent, can be ignored for core position anomalies.

---

## 2️⃣ Overall Flow (High-Level)

For each account you want to analyze:

1. **Get account details & total market value**
2. **Get its model & target allocation per security**
3. **Get executed / pending orders and map to securities**
4. **Compute executed and current allocation per security**
5. **Derive drift, overweight, underweight, and flags**
6. **Build a flat feature row per (account, security)**
7. **Send this feature matrix to the XGBoost model**

---

## 3️⃣ Step 1 — Fetch Accounts to Analyze

From **accounts** table:

```sql
SELECT
    a.Id              AS account_id,
    a.Description     AS account_description,
    a.short_name      AS account_short_name,
    a.TotalMarketValue,
    a.CashBalance,
    a.AvailableCash,
    a.TotalCashAvailable,
    a.model_id
FROM [SharedDatabase-11].[dbo].[accounts] AS a
WHERE a.IsDeleted = 0;
```

This gives you the **universe of accounts** plus `model_id` and `TotalMarketValue` (used for percent calculations).

---

## 4️⃣ Step 2 — Fetch Model Allocations Per Account

### 4.1 Get the model → sleeves

```sql
SELECT
    ms.ModelId,
    ms.SleeveId,
    ms.AllocationPercentage AS sleeve_allocation_percent
FROM [SharedDatabase-11].[dbo].[ModelSleeves] AS ms
WHERE ms.IsDeleted = 0;
```

### 4.2 Get sleeves → securities

```sql
SELECT
    ss.SleeveId,
    ss.SecurityId,
    ss.AllocationPercentage AS security_allocation_percent
FROM [SharedDatabase-11].[dbo].[SleeveSecurities] AS ss
WHERE ss.IsDeleted = 0;
```

### 4.3 Get security metadata

```sql
SELECT
    s.Id         AS SecurityId,
    s.Symbol     AS SecuritySymbol,
    s.Name       AS SecurityName,
    s.Price      AS LastPrice,
    s.SecurityTypeId
FROM [SharedDatabase-11].[dbo].[Securities] AS s
WHERE s.IsDeleted = 0;
```

### 4.4 Join everything to compute model_target_percent per security

At the feature layer (Python), compute:

```python
model_target_percent = (sleeve_allocation_percent / 100.0) * (security_allocation_percent / 100.0)
```

This tells you, for each (ModelId, SecurityId):

> “What percent of the portfolio this security *should* be.”

You then map `ModelId` → `account.model_id` to know the **model expectations per account**.

---

## 5️⃣ Step 3 — Fetch Executed & Pending Orders

Use **Orders** + **OrderAllocations** + **Securities**.

### 5.1 Orders per account

```sql
SELECT
    o.Id          AS order_id,
    o.AccountId   AS account_id,
    o.SecurityId,
    o.Quantity,
    o.EstCost     AS est_cost,
    o.FilledQuantity,
    o.FilledPrice,
    o.TradeDate,
    o.SettleDate,
    o.State,
    o.Tran        AS trans_code  -- Buy/Sell
FROM [SharedDatabase-11].[dbo].[Orders] AS o
WHERE o.IsDeleted = 0;
```

### 5.2 Order allocations (per account/security)

```sql
SELECT
    oa.OrderId,
    oa.AccountId,
    oa.CurrentQuantity,
    oa.CurrPercent,
    oa.ModelPercent,
    oa.ModAppPercent,
    oa.DeltaPercent AS delta_percent,
    oa.AllocationEstCost
FROM [SharedDatabase-11].[dbo].[OrderAllocations] AS oa
WHERE oa.IsDeleted = 0;
```

You join `Orders` → `OrderAllocations` on `OrderId`, and `Orders.SecurityId` → `Securities.Id` to get `SecuritySymbol`.

This lets you derive **what was actually traded** and **planned allocation vs current allocation**.

---

## 6️⃣ Step 4 — Compute Per-Security Features

For each **(account_id, SecurityId/Symbol)**, you want to end up with something like:

* `account_id`
* `model_symbol` (from Securities.Symbol via model)
* `model_target_percent` (from ModelSleeves + SleeveSecurities)
* `executed_symbol` (from Orders.SecurityId → Securities.Symbol; or use model_symbol if same)
* `executed_quantity` (from Orders / OrderAllocations)
* `executed_percent` (TradeAmount / TotalMarketValue)
* `allocation_after_trade` (use `OrderAllocations.EndPercent` or `CurrPercent` if available)
* `model_change_time` (you can approximate from `Models.CreatedDate` or rebalance request time)
* `order_execution_time` (from Orders.TradeDate)

Example pseudo-code (Python-like):

```python
executed_percent = trade_amount / account_total_market_value
drift_percent = executed_percent - model_target_percent
overweight_percent = max(drift_percent, 0.0)
underweight_percent = max(-drift_percent, 0.0)

exists_in_model = 1 if executed_symbol in model_symbols_for_account else 0
order_mismatch_flag = 1 if exists_in_model == 0 else 0
timing_mismatch_flag = 1 if order_execution_time > model_change_time else 0
allocation_excess_flag = 1 if overweight_percent > 0.10 else 0
```

These become exactly the fields you used in your training dataset.

---

## 7️⃣ Step 5 — Assembling the Feature Rows

For each **(account, security)** that has either:

* a model allocation, or
* an order,

build a **single flat row** with:

```json
{
  "account_id": 1234,
  "model_symbol": "SWSXX",
  "model_target_percent": 0.0511,
  "executed_symbol": "SWSXX",
  "executed_quantity": 562.5,
  "executed_percent": 0.055,
  "allocation_after_trade": 0.055,
  "model_change_time": "2025-07-15T19:54:41Z",
  "order_execution_time": "2025-07-15T19:54:41Z",
  "exists_in_model": 1,
  "drift_percent": 0.0039,
  "overweight_percent": 0.0039,
  "underweight_percent": 0.0,
  "order_mismatch_flag": 0,
  "timing_mismatch_flag": 0,
  "allocation_excess_flag": 0
}
```

Collect many such rows into a **DataFrame** (or list of dicts) → pass to XGBoost:

```python
df = pd.DataFrame(feature_rows)
scores = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
```

---

## 8️⃣ Step 6 — From Scores to Position Anomaly Output

For each row:

* If `score` is high → anomaly
* Decide anomaly type from flags:

```python
if order_mismatch_flag:
    type = "Symbol mismatch"
elif timing_mismatch_flag:
    type = "Timing mismatch"
elif allocation_excess_flag and overweight_percent > 0:
    type = "Overweight execution"
elif underweight_percent > threshold:
    type = "Underweight execution"
else:
    type = "Normal"
```

Return a response to frontend like:

```json
{
  "account_id": 1234,
  "symbol": "SWSXX",
  "is_anomaly": true,
  "score": 0.92,
  "anomaly_type": "Overweight execution",
  "drift_percent": 0.21,
  "overweight_percent": 0.21
}
```

---

## 9️⃣ Practical Tips

* Start with **one account** and **one model** to validate joins and percent math.
* Log intermediate values: `model_target_percent`, `executed_percent`, `drift_percent`.
* Make sure all percents are on the **0–1 scale** (not 0–100) or stay consistent.
* Cache `Securities` and `SecurityTypes` in memory; they change less often.

---

## 🔚 Summary

Using your tables, the pipeline is:

1. `accounts` → which portfolios + total market value
2. `Models`, `ModelSleeves`, `Sleeves`, `SleeveSecurities`, `Securities` → model allocations per security
3. `Orders`, `OrderAllocations`, `Securities` → executed trades per security
4. Join model + orders per (account, security)
5. Compute drift, overweight, underweight, mismatch & timing flags
6. Build flat feature rows → send to **XGBoost** position anomaly model
7. Return readable anomaly summaries for advisors & traders.
