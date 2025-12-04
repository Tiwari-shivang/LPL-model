"""
Database connectivity and portfolio snapshot assembly.

python-tds version (NO ODBC DRIVERS REQUIRED).
Clean, stable, AWS RDS–compatible.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    import pytds  # pure TDS driver
except ImportError:
    pytds = None

logger = logging.getLogger(__name__)

DEFAULT_CONN_STR = os.getenv(
    "DB_CONNECTION_STRING",
    (
        "SERVER=hvtoms-dev-sqlserver-1.cfemiu68wkqx.ap-south-1.rds.amazonaws.com;"
        "DATABASE=hvtoms-01;"
        "UID=hvtoms;"
        "PWD=Hvt0m$@To25;"
        "Connection Timeout=60;"
        "Command Timeout=120;"
        "MultipleActiveResultSets=True;"
        "ApplicationIntent=ReadOnly;"
    ),
)


class DatabaseClient:
    def __init__(self, conn_str: str | None = None) -> None:
        if pytds is None:
            raise ImportError(
                "python-tds is required. Install using: pip install python-tds"
            )
        self.conn_str = conn_str or DEFAULT_CONN_STR

    # -----------------------------------------------------------
    # 🔹 Parse Connection String (simple & safe)
    # -----------------------------------------------------------
    def _parse_conn_str(self, raw: str) -> Dict[str, Any]:
        parts = [p for p in raw.split(";") if p.strip()]
        parsed: Dict[str, str] = {}

        for part in parts:
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            parsed[k.strip().lower()] = v.strip()

        server = parsed.get("server")
        database = parsed.get("database")
        user = parsed.get("uid") or parsed.get("user id") or parsed.get("user")
        password = parsed.get("pwd") or parsed.get("password")
        port = int(parsed.get("port") or 1433)

        if not all([server, database, user, password]):
            raise RuntimeError("Missing required DB config: server/database/user/password")

        return {
            "server": server,
            "database": database,
            "user": user,
            "password": password,
            "port": port,
        }

    # -----------------------------------------------------------
    # 🔹 python-tds Connection (NO ODBC, NO encrypt flags)
    # -----------------------------------------------------------
    def _get_connection(self):
        params = self._parse_conn_str(self.conn_str)

        # python-tds negotiates TLS automatically with RDS
        return pytds.connect(
            server=params["server"],
            database=params["database"],
            user=params["user"],
            password=params["password"],
            port=params["port"],
            autocommit=True,
            timeout=60,
            login_timeout=30,
        )

    # -----------------------------------------------------------
    # 🔹 Generic SELECT helper
    # -----------------------------------------------------------
    def _fetchall(self, query: str, params: Tuple[Any, ...] = ()) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(query, params)
            columns = [c[0] for c in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

    # -----------------------------------------------------------
    # 🔹 FETCH ACCOUNTS  (TOP(?) FIXED)
    # -----------------------------------------------------------
    def fetch_accounts(self, limit: int = 200) -> List[Dict[str, Any]]:
        query = f"""
            SELECT TOP {limit}
                Id AS account_id,
                Description AS account_name,
                TotalMarketValue AS total_market_value,
                CashBalance AS cash_balance,
                AvailableCash AS available_cash,
                TotalCashAvailable AS total_cash_available,
                model_id
            FROM accounts
            WHERE IsDeleted = 0
            ORDER BY UpdatedAt DESC;
        """
        return self._fetchall(query)

    # -----------------------------------------------------------
    # 🔹 FETCH ORDER SUMMARIES
    # -----------------------------------------------------------
    def fetch_order_summaries(self) -> Dict[Any, Dict[str, float]]:
        query = """
            SELECT 
                AccountId AS account_id,
                SUM(COALESCE(EstCost, 0)) AS est_cost,
                SUM(COALESCE(FilledQuantity, 0)) AS filled_quantity,
                AVG(NULLIF(FilledPrice, 0)) AS avg_price,
                MAX(NULLIF(FilledPrice, 0)) AS last_filled_price
            , MAX(TradeDate) AS last_trade_date
            FROM Orders
            WHERE IsDeleted = 0
            GROUP BY AccountId;
        """
        rows = self._fetchall(query)
        return {
            r["account_id"]: {
                "est_cost": float(r.get("est_cost") or 0.0),
                "filled_quantity": float(r.get("filled_quantity") or 0.0),
                "avg_price": float(r.get("avg_price") or 0.0),
                "last_filled_price": float(r.get("last_filled_price") or 0.0),
                "last_trade_date": r.get("last_trade_date"),
            }
            for r in rows
        }

    # -----------------------------------------------------------
    # 🔹 FETCH CASH TRANSACTIONS
    # -----------------------------------------------------------
    def fetch_cash_transactions(self, lookback_days: int = 30) -> Dict[Any, Dict[str, float]]:
        query = f"""
            SELECT 
                portfolio_account_id AS account_id,
                SUM(CASE WHEN transaction_type = 'DEPOSIT' THEN Amount ELSE 0 END) AS deposits,
                SUM(CASE WHEN transaction_type = 'WITHDRAWAL' THEN Amount ELSE 0 END) AS withdrawals,
                SUM(CASE WHEN transaction_type = 'DIVIDEND' THEN Amount ELSE 0 END) AS dividends
            FROM cash_transactions
            WHERE IsDeleted = 0
              AND transaction_date >= DATEADD(day, -{lookback_days}, GETDATE())
            GROUP BY portfolio_account_id;
        """
        rows = self._fetchall(query)
        return {
            r["account_id"]: {
                "deposits": float(r.get("deposits") or 0.0),
                "withdrawals": float(r.get("withdrawals") or 0.0),
                "dividends": float(r.get("dividends") or 0.0),
            }
            for r in rows
        }

    # -----------------------------------------------------------
    # 🔹 FETCH MODEL CASH TARGETS
    # -----------------------------------------------------------
    def fetch_model_cash_targets(self) -> Dict[Any, float]:
        query = """
            SELECT 
                ms.ModelId AS model_id,
                AVG(COALESCE(ss.AllocationPercentage, 0)) AS allocation_pct
            FROM ModelSleeves ms
            JOIN SleeveSecurities ss ON ms.SleeveId = ss.SleeveId
            JOIN Securities s ON ss.SecurityId = s.Id
            JOIN SecurityTypes st ON s.SecurityTypeId = st.Id
            WHERE ss.IsDeleted = 0
              AND st.Name LIKE '%%Cash%%'
            GROUP BY ms.ModelId;
        """
        rows = self._fetchall(query)
        return {
            r["model_id"]: float(r.get("allocation_pct") or 0.0) for r in rows
        }

    # -----------------------------------------------------------
    # dY"1 FETCH MODEL / SLEEVE / SECURITY STRUCTURE
    # -----------------------------------------------------------
    def fetch_model_structures(self) -> Dict[Any, Dict[str, Any]]:
        model_rows = self._fetchall(
            """
            SELECT Id, Name
            FROM Models
            WHERE IsDeleted = 0
            """
        )
        models = {m["Id"]: {"id": m["Id"], "name": m.get("Name"), "sleeves": []} for m in model_rows}

        sleeve_rows = self._fetchall(
            """
            SELECT 
                ms.Id AS model_sleeve_id,
                ms.ModelId,
                ms.SleeveId,
                ms.AllocationPercentage,
                sl.Name AS SleeveName
            FROM ModelSleeves ms
            JOIN Sleeves sl ON sl.Id = ms.SleeveId
            WHERE ms.IsDeleted = 0 AND sl.IsDeleted = 0
            """
        )
        sleeve_lookup = {s["SleeveId"]: s for s in sleeve_rows}

        sleeve_securities = self._fetchall(
            """
            SELECT ss.SleeveId, ss.SecurityId, ss.AllocationPercentage, sec.Name, sec.Price
            FROM SleeveSecurities ss
            JOIN Securities sec ON ss.SecurityId = sec.Id
            WHERE ss.IsDeleted = 0 AND sec.IsDeleted = 0
            """
        )

        securities_by_sleeve: Dict[Any, List[Dict[str, Any]]] = {}
        for row in sleeve_securities:
            securities_by_sleeve.setdefault(row["SleeveId"], []).append(
                {
                    "id": row.get("SecurityId"),
                    "name": row.get("Name"),
                    "price": float(row.get("Price") or 0.0),
                    "allocation_percentage": float(row.get("AllocationPercentage") or 0.0),
                }
            )

        for sleeve_id, sleeve in sleeve_lookup.items():
            model_id = sleeve.get("ModelId")
            if model_id not in models:
                continue
            models[model_id]["sleeves"].append(
                {
                    "id": sleeve_id,
                    "name": sleeve.get("SleeveName"),
                    "allocation_percentage": float(sleeve.get("AllocationPercentage") or 0.0),
                    "securities": securities_by_sleeve.get(sleeve_id, []),
                }
            )

        return models

    # -----------------------------------------------------------
    # 🔹 FULL ENRICHED ACCOUNT SNAPSHOT
    # -----------------------------------------------------------
    def fetch_enriched_accounts(self, limit: int = 200) -> List[Dict[str, Any]]:
        accounts = self.fetch_accounts(limit=limit)
        if not accounts:
            return []

        orders = self.fetch_order_summaries()
        cash_txn = self.fetch_cash_transactions()
        model_targets = self.fetch_model_cash_targets()
        model_structures = self.fetch_model_structures()

        enriched = []

        for acc in accounts:
            acc_id = acc["account_id"]

            mv = float(acc.get("total_market_value") or 0.0)
            cash = float(acc.get("cash_balance") or 0.0)
            available = float(acc.get("total_cash_available") or cash)

            order = orders.get(acc_id, {})
            txn = cash_txn.get(acc_id, {})

            last_trade_date = order.get("last_trade_date")
            days_since_last_trade: Optional[int] = None
            if isinstance(last_trade_date, datetime):
                # Normalize to aware UTC for safe subtraction
                now_utc = datetime.now(timezone.utc)
                if last_trade_date.tzinfo is None:
                    last_trade_date = last_trade_date.replace(tzinfo=timezone.utc)
                days_since_last_trade = max(0, (now_utc - last_trade_date).days)

            # Cash idle days approximated by days since last trade when cash sits unused
            cash_idle_days = days_since_last_trade if (days_since_last_trade is not None and available > 0) else None

            model_target = model_targets.get(acc.get("model_id"), 0.05)
            actual_pct = (available / mv) if mv > 0 else 0.0
            portfolio: Dict[str, Any] = {}
            model_info = model_structures.get(acc.get("model_id"))
            if model_info:
                portfolio = {"model": model_info}

            enriched.append(
                {
                    "account_id": acc_id,
                    "account_name": acc.get("account_name"),
                    "market_value": mv,
                    "cash_balance": cash,
                    "available_cash": float(acc.get("available_cash") or available),
                    "total_cash_available": available,
                    "model_cash_target": model_target,

                    # % metrics
                    "model_percent_target": model_target,
                    "actual_cash_percent": actual_pct,
                    "current_percent": actual_pct,
                    "drift_percent": actual_pct - model_target,

                    # trade-driven signals
                    "quantity": order.get("filled_quantity", 0.0),
                    "avg_price": order.get("avg_price", 0.0),
                    "market_price": order.get("last_filled_price", 0.0),
                    "pending_est_cost": order.get("est_cost", 0.0),
                    "last_trade_date": last_trade_date,
                    "days_since_last_trade": days_since_last_trade,
                    "cash_idle_days": cash_idle_days,

                    # cash movement
                    "recent_deposits": txn.get("deposits", 0.0),
                    "recent_withdrawals": txn.get("withdrawals", 0.0),
                    "recent_dividends": txn.get("dividends", 0.0),

                    # training-aligned
                    "cash_before": cash,
                    "cash_after": available,
                    "portfolio": portfolio,
                }
            )

        return enriched
