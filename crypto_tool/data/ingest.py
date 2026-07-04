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


def resolve_symbols(config: Dict[str, Any], *, pairs=None, tickers=None) -> List[str]:
    """The tracked universe, synchronized with the exchange.

    Always contains the explicit ``data.symbols`` watchlist. When
    ``data.symbols_auto`` > 0, the list is topped up with the most-traded real
    USDT pairs on Binance (same stablecoin/leveraged-token/volume filters as
    the scout) until it holds max(symbols_auto, len(explicit)) coins — so your
    hand-picked coins are never dropped, and the rest tracks what the market
    actually trades. Falls back to the explicit list when offline."""
    d = config["data"]
    explicit = [s.upper() for s in d["symbols"]]
    n_auto = int(d.get("symbols_auto", 0) or 0)
    if n_auto <= 0:
        return explicit
    from ..analysis import scout  # lazy: reuse its market filters, avoid an import cycle
    try:
        if pairs is None:
            pairs = binance_client.fetch_usdt_pairs(d["base_urls"], d["request_timeout"])
        if tickers is None:
            tickers = binance_client.fetch_all_24h_tickers(d["base_urls"], d["request_timeout"])
        ranked = scout.eligible_pairs(
            pairs, tickers,
            min_quote_volume=float(config["scout"]["min_quote_volume_usd"]),
            max_symbols=0)
    except Exception:  # noqa: BLE001 — sync is best-effort; the watchlist still works
        return explicit
    if not ranked:
        return explicit
    merged = list(dict.fromkeys(explicit + [r["symbol"] for r in ranked]))
    return merged[: max(n_auto, len(explicit))]


def ingest_all(conn, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Ingest the synchronized symbol universe. Never raises — collects
    per-symbol status, and records the tracked list for the page builder."""
    d = config["data"]
    symbols = resolve_symbols(config)
    results = []
    for symbol in symbols:
        results.append(
            ingest_symbol(
                conn, symbol, d["interval"], d["history_limit"],
                d["base_urls"], d["request_timeout"],
            )
        )
    ok = [r["symbol"] for r in results if r["ok"]]
    if ok:
        database.save_kv(conn, "tracked_symbols", ok)
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
