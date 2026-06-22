"""SQLite storage — the 'active database' the tool stays linked to.

Three tables:
  * ohlcv     — raw candles, idempotent upserts keyed by (symbol, interval, open_time)
  * signals   — every signal snapshot we persist, for history / audit
  * backtests — stored backtest parameter sets and their metrics

Using SQLite keeps the tool zero-config and file-based; the schema/queries are
plain SQL, so pointing at Postgres later is a small change.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ohlcv (
    symbol      TEXT    NOT NULL,
    interval    TEXT    NOT NULL,
    open_time   INTEGER NOT NULL,
    open        REAL    NOT NULL,
    high        REAL    NOT NULL,
    low         REAL    NOT NULL,
    close       REAL    NOT NULL,
    volume      REAL    NOT NULL,
    close_time  INTEGER,
    num_trades  INTEGER,
    PRIMARY KEY (symbol, interval, open_time)
);

CREATE TABLE IF NOT EXISTS signals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT    NOT NULL,
    interval    TEXT    NOT NULL,
    ts          INTEGER NOT NULL,
    composite   REAL,
    confidence  REAL,
    flag        TEXT,
    rationale   TEXT,
    created_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol, interval, ts);

CREATE TABLE IF NOT EXISTS backtests (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT    NOT NULL,
    interval    TEXT    NOT NULL,
    params      TEXT,
    metrics     TEXT,
    created_at  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_account (
    name          TEXT PRIMARY KEY,
    created_at    INTEGER,
    starting_cash REAL,
    learn         INTEGER,
    interval      TEXT,
    params        TEXT,
    metrics       TEXT,
    weights       TEXT,
    positions     TEXT,
    last_ts       INTEGER
);

CREATE TABLE IF NOT EXISTS paper_equity (
    name      TEXT    NOT NULL,
    ts        INTEGER NOT NULL,
    equity    REAL,
    cash      REAL,
    invested  REAL,
    benchmark REAL,
    PRIMARY KEY (name, ts)
);

CREATE TABLE IF NOT EXISTS paper_trade (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    name     TEXT    NOT NULL,
    ts       INTEGER NOT NULL,
    symbol   TEXT,
    side     TEXT,
    price    REAL,
    units    REAL,
    notional REAL,
    fee      REAL
);
CREATE INDEX IF NOT EXISTS idx_paper_trade ON paper_trade(name, ts);

CREATE TABLE IF NOT EXISTS paper_weight (
    name    TEXT    NOT NULL,
    ts      INTEGER NOT NULL,
    weights TEXT,
    PRIMARY KEY (name, ts)
);
"""


def connect(db_path: str) -> sqlite3.Connection:
    """Open (creating parent dirs and schema as needed) a SQLite connection."""
    parent = os.path.dirname(os.path.abspath(db_path))
    os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()


# --------------------------------------------------------------------------- #
# OHLCV
# --------------------------------------------------------------------------- #
def upsert_ohlcv(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Insert-or-replace candles. ``df`` must contain the standard columns."""
    if df is None or df.empty:
        return 0
    cols = ["symbol", "interval", "open_time", "open", "high", "low",
            "close", "volume", "close_time", "num_trades"]
    records: Iterable = (tuple(row) for row in df[cols].itertuples(index=False, name=None))
    conn.executemany(
        """
        INSERT INTO ohlcv
            (symbol, interval, open_time, open, high, low, close, volume, close_time, num_trades)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, interval, open_time) DO UPDATE SET
            open=excluded.open, high=excluded.high, low=excluded.low,
            close=excluded.close, volume=excluded.volume,
            close_time=excluded.close_time, num_trades=excluded.num_trades
        """,
        records,
    )
    conn.commit()
    return len(df)


def load_ohlcv(
    conn: sqlite3.Connection,
    symbol: str,
    interval: str,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Load candles for a symbol/interval, ascending by time."""
    query = (
        "SELECT symbol, interval, open_time, open, high, low, close, volume, "
        "close_time, num_trades FROM ohlcv WHERE symbol=? AND interval=? "
        "ORDER BY open_time ASC"
    )
    df = pd.read_sql_query(query, conn, params=(symbol.upper(), interval))
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be >= 0")
        if len(df) > limit:
            df = df.tail(limit).reset_index(drop=True)
    return df


def delete_ohlcv(conn: sqlite3.Connection, symbol: str, interval: str) -> int:
    """Remove all stored candles for a symbol/interval. Returns rows deleted."""
    cur = conn.execute(
        "DELETE FROM ohlcv WHERE symbol=? AND interval=?", (symbol.upper(), interval)
    )
    conn.commit()
    return cur.rowcount


def list_symbols(conn: sqlite3.Connection, interval: Optional[str] = None) -> List[str]:
    if interval:
        rows = conn.execute(
            "SELECT DISTINCT symbol FROM ohlcv WHERE interval=? ORDER BY symbol",
            (interval,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT DISTINCT symbol FROM ohlcv ORDER BY symbol").fetchall()
    return [r[0] for r in rows]


def ohlcv_coverage(conn: sqlite3.Connection) -> pd.DataFrame:
    """Per symbol/interval: row count and first/last candle time (ms)."""
    return pd.read_sql_query(
        "SELECT symbol, interval, COUNT(*) AS bars, "
        "MIN(open_time) AS first_ms, MAX(open_time) AS last_ms "
        "FROM ohlcv GROUP BY symbol, interval ORDER BY symbol, interval",
        conn,
    )


# --------------------------------------------------------------------------- #
# Signals
# --------------------------------------------------------------------------- #
def save_signal(
    conn: sqlite3.Connection,
    symbol: str,
    interval: str,
    ts: int,
    composite: float,
    confidence: float,
    flag: str,
    rationale: Any,
) -> None:
    conn.execute(
        "INSERT INTO signals (symbol, interval, ts, composite, confidence, flag, rationale, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            symbol.upper(), interval, int(ts), float(composite), float(confidence),
            flag, json.dumps(rationale), int(time.time()),
        ),
    )
    conn.commit()


def latest_signals(conn: sqlite3.Connection, interval: Optional[str] = None) -> pd.DataFrame:
    """Most recent stored signal per symbol."""
    # Two fully-static SQL literals (no string interpolation) — the interval
    # value is always passed as a bound parameter.
    join_tail = ") m ON s.symbol=m.symbol AND s.interval=m.interval AND s.ts=m.mts"
    if interval:
        query = (
            "SELECT s.* FROM signals s JOIN ("
            "SELECT symbol, interval, MAX(ts) AS mts FROM signals "
            "WHERE interval=? GROUP BY symbol, interval" + join_tail
        )
        return pd.read_sql_query(query, conn, params=(interval,))
    query = (
        "SELECT s.* FROM signals s JOIN ("
        "SELECT symbol, interval, MAX(ts) AS mts FROM signals "
        "GROUP BY symbol, interval" + join_tail
    )
    return pd.read_sql_query(query, conn)


# --------------------------------------------------------------------------- #
# Backtests
# --------------------------------------------------------------------------- #
def save_backtest(
    conn: sqlite3.Connection,
    symbol: str,
    interval: str,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
) -> int:
    cur = conn.execute(
        "INSERT INTO backtests (symbol, interval, params, metrics, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (symbol.upper(), interval, json.dumps(params), json.dumps(metrics), int(time.time())),
    )
    conn.commit()
    return int(cur.lastrowid)


# --------------------------------------------------------------------------- #
# Paper-trading accounts
# --------------------------------------------------------------------------- #
def create_paper_account(
    conn: sqlite3.Connection,
    name: str,
    starting_cash: float,
    learn: bool,
    interval: str,
    params: Dict[str, Any],
) -> None:
    """Create or reset an account's configuration (clears any prior results)."""
    delete_paper_account(conn, name)
    conn.execute(
        "INSERT INTO paper_account (name, created_at, starting_cash, learn, interval, "
        "params, metrics, weights, positions, last_ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (name, int(time.time()), float(starting_cash), 1 if learn else 0, interval,
         json.dumps(params), None, None, None, None),
    )
    conn.commit()


def get_paper_account(conn: sqlite3.Connection, name: str) -> Optional[dict]:
    row = conn.execute("SELECT * FROM paper_account WHERE name=?", (name,)).fetchone()
    if row is None:
        return None
    acc = dict(row)
    for key in ("params", "metrics", "weights", "positions"):
        acc[key] = json.loads(acc[key]) if acc.get(key) else None
    acc["learn"] = bool(acc["learn"])
    return acc


def list_paper_accounts(conn: sqlite3.Connection) -> List[str]:
    return [r[0] for r in conn.execute("SELECT name FROM paper_account ORDER BY name").fetchall()]


def delete_paper_account(conn: sqlite3.Connection, name: str) -> None:
    for table in ("paper_equity", "paper_trade", "paper_weight", "paper_account"):
        conn.execute(f"DELETE FROM {table} WHERE name=?", (name,))
    conn.commit()


def save_paper_results(
    conn: sqlite3.Connection,
    name: str,
    equity_rows: list,
    trades: list,
    weight_rows: list,
    metrics: Dict[str, Any],
    weights: Dict[str, Any],
    positions: Dict[str, Any],
    last_ts: int,
) -> None:
    """Replace the stored run results for an account (equity, trades, weights)."""
    conn.execute("DELETE FROM paper_equity WHERE name=?", (name,))
    conn.execute("DELETE FROM paper_trade WHERE name=?", (name,))
    conn.execute("DELETE FROM paper_weight WHERE name=?", (name,))
    conn.executemany(
        "INSERT INTO paper_equity (name, ts, equity, cash, invested, benchmark) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [(name, int(ts), eq, c, inv, bm) for (ts, eq, c, inv, bm) in equity_rows],
    )
    conn.executemany(
        "INSERT INTO paper_trade (name, ts, symbol, side, price, units, notional, fee) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [(name, int(t["ts"]), t["symbol"], t["side"], t["price"], t["units"],
          t["notional"], t["fee"]) for t in trades],
    )
    conn.executemany(
        "INSERT INTO paper_weight (name, ts, weights) VALUES (?, ?, ?)",
        [(name, int(ts), json.dumps(w)) for (ts, w) in weight_rows],
    )
    conn.execute(
        "UPDATE paper_account SET metrics=?, weights=?, positions=?, last_ts=? WHERE name=?",
        (json.dumps(metrics), json.dumps(weights), json.dumps(positions), int(last_ts), name),
    )
    conn.commit()


def load_paper_equity(conn: sqlite3.Connection, name: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT ts, equity, cash, invested, benchmark FROM paper_equity "
        "WHERE name=? ORDER BY ts", conn, params=(name,))


def load_paper_trades(conn: sqlite3.Connection, name: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT ts, symbol, side, price, units, notional, fee FROM paper_trade "
        "WHERE name=? ORDER BY ts", conn, params=(name,))


def load_paper_weights(conn: sqlite3.Connection, name: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT ts, weights FROM paper_weight WHERE name=? ORDER BY ts", conn, params=(name,))
    if not df.empty:
        df["weights"] = df["weights"].apply(json.loads)
    return df
