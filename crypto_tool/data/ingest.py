"""Ingestion pipeline: pull candles (live or synthetic) and persist to SQLite."""
from __future__ import annotations

from typing import Any, Dict, List

from . import binance_client, database, synthetic


def ingest_symbol(
    conn,
    symbol: str,
    interval: str,
    limit: int,
    base_urls: List[str],
    timeout: int,
) -> Dict[str, Any]:
    """Fetch one symbol from Binance and upsert it. Returns a status dict."""
    try:
        df = binance_client.fetch_klines_history(
            symbol, interval=interval, total=limit, base_urls=base_urls, timeout=timeout
        )
        rows = database.upsert_ohlcv(conn, df)
        return {"symbol": symbol, "ok": True, "rows": rows, "error": None}
    except Exception as exc:  # noqa: BLE001 — contract: one bad symbol never aborts the batch
        return {"symbol": symbol, "ok": False, "rows": 0,
                "error": f"{type(exc).__name__}: {exc}"}


def ingest_all(conn, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Ingest every configured symbol. Never raises — collects per-symbol status."""
    d = config["data"]
    results = []
    for symbol in d["symbols"]:
        results.append(
            ingest_symbol(
                conn, symbol, d["interval"], d["history_limit"],
                d["base_urls"], d["request_timeout"],
            )
        )
    return results


def seed_demo(conn, config: Dict[str, Any], n: int | None = None) -> List[Dict[str, Any]]:
    """Populate the database with deterministic synthetic candles (offline mode).

    Clears any existing candles for each symbol/interval first, so demo data is
    self-consistent and never interleaves with previously-fetched live candles
    (which sit at a different time anchor).
    """
    d = config["data"]
    n = n or d["history_limit"]
    results = []
    for symbol in d["symbols"]:
        database.delete_ohlcv(conn, symbol, d["interval"])
        df = synthetic.generate_ohlcv(symbol, interval=d["interval"], n=n)
        rows = database.upsert_ohlcv(conn, df)
        results.append({"symbol": symbol, "ok": True, "rows": rows, "error": None})
    return results
