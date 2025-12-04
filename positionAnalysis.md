# 📘 XGBoost-Based Position Anomaly Detection Model

*A complete guide crafted with best practices from a senior ML architect (20+ years experience)*

---

## 🧠 Overview

This document describes how to build a **Position Anomaly Detection Model** using **XGBoost**, able to detect inconsistencies between *model allocations* and *executed trades*.
The model identifies:

* **Positive anomalies:** overweight execution, excess allocation
* **Negative anomalies:** underweight execution, missed allocations
* **Symbol mismatch anomalies**
* **Timing mismatch anomalies**
* **Allocation misalignment anomalies**

Once trained, the model outputs actionable insights formatted for traders:

* **Start of business anomalies** (positive & negative)
* **Last 7-day anomalies** (positive & negative)
* **Detailed anomaly table** with short descriptions

This ensures rapid decision-making across advisory, trading, and compliance teams.

---

# 1️⃣ Problem Definition

Financial trading systems often face issues like:

* Orders executed for a symbol **no longer in the model**
* Trades over-executed or under-executed vs model target
* Model allocation changed after order placement
* Drift between intended and actual allocation
* Poor timing alignment between model change & execution

These anomalies create **portfolio misalignment**, **compliance risks**, and **trading inefficiencies**.

XGBoost is ideal because it handles **tabular data**, learns **complex interactions**, provides **explainability**, and scales for production.

---

# 2️⃣ Dataset Schema (Training Data Format)

The training dataset must contain the following columns:

| Feature                | Description                          |
| ---------------------- | ------------------------------------ |
| account_id             | Account identifier                   |
| model_symbol           | Security in model                    |
| model_target_percent   | Target % (0–1 scale)                 |
| executed_symbol        | Actual traded symbol                 |
| executed_quantity      | Quantity traded                      |
| executed_percent       | Trade value % of account             |
| allocation_after_trade | Allocation after execution           |
| model_change_time      | Timestamp of model update            |
| order_execution_time   | Timestamp of order execution         |
| exists_in_model        | 1 if executed symbol exists in model |
| drift_percent          | executed % − model target %          |
| overweight_percent     | max(drift, 0)                        |
| underweight_percent    | max(-drift, 0)                       |
| order_mismatch_flag    | 1 if executed symbol not in model    |
| timing_mismatch_flag   | 1 if execution after symbol removal  |
| allocation_excess_flag | 1 if executed% exceeds threshold     |
| label                  | 1 = anomaly, 0 = normal              |

No field must be empty.

---

# 3️⃣ Label Engineering

Here is the simple yet powerful labeling strategy:

```python
label = (
    order_mismatch_flag |
    timing_mismatch_flag |
    allocation_excess_flag
).astype(int)
```

This allows the model to learn the *true structure* of anomaly behavior.

You may enhance labeling using additional business rules like:

* drift > 20% → anomaly
* allocation_after_trade outside model range → anomaly

---

# 4️⃣ Feature Engineering

Your model features should represent:

### ✔ Allocation Behavior

* model_target_percent
* executed_percent
* drift_percent
* overweight_percent
* underweight_percent

### ✔ Execution Behavior

* executed_quantity
* allocation_after_trade

### ✔ Timing Factors

* order_execution_time - model_change_time

### ✔ Compliance Flags

* exists_in_model
* order_mismatch_flag
* timing_mismatch_flag
* allocation_excess_flag

---

# 5️⃣ XGBoost Model Creation

### Recommended Hyperparameters

(Optimized for financial anomaly detection work)

```python
model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="binary:logistic",
    random_state=42
)
```

### Train the Model

```python
model.fit(X_train, y_train)
```

### Evaluate Using:

* **Precision** (critical for anomalies)
* **Recall** (catch all meaningful anomalies)
* **AUC / AUPRC**

---

# 6️⃣ Output Format for UI

The model must produce responses in this structure:

---

## **Start of Business — Anomalies**

### **Positive anomalies (+)**

* Account 1245: AAPL overweight by **+22%**
* Account 5311: TSLA executed after model removal

### **Negative anomalies (−)**

* Account 8842: MSFT underweight by **−18%**
* Account 2741: IBM expected but missing

---

## **Last 7 Days — Anomalies**

### **Positive anomalies (+)**

* Account 3391: Allocation exceeded target by **+12%**

### **Negative anomalies (−)**

* Account 7720: Wrong symbol execution flagged

---

## **More Details (Table)**

| Account | Short Description      |
| ------- | ---------------------- |
| 1245    | Overweight +22% (AAPL) |
| 5311    | Executed after removal |
| 7720    | Wrong symbol executed  |
| 8842    | Underweight –18%       |

---

# 7️⃣ Deployment Best Practices (20+ years expertise)

### ✔ Model Management

* Retrain quarterly or when feature drift detected
* Log predictions for audit trails
* Maintain model versioning & rollback

### ✔ Feature Monitoring

* Track drift_percent & overweight_percent distributions
* Watch for shifts in symbol frequency

### ✔ Explainability

Use SHAP to provide compliance-grade explanations.

### ✔ Inference

* Run nightly batch scoring
* Provide real-time scoring on trade submission

---

# 8️⃣ UI Integration Recommendations

### Display three main panels:

---

### 1️⃣ **Start of Business Alerts**

Shows +ve and −ve anomalies for quick morning checks.

---

### 2️⃣ **Last 7 Days Watchlist**

Helps identify *repeated bad behavior* and *newly emerging issues*.

---

### 3️⃣ **Detailed Table View**

Provide:

| Acc | Short Description |
| --- | ----------------- |

Fast actionable insights for traders.

---