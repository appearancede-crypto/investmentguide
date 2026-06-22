"""Binance public market-data client.

Only the public ``/api/v3/klines`` (candlesticks) and ``/api/v3/ticker/24hr``
endpoints are used — these require **no API key** and grant **no trading
ability**. This tool reads market data only; it can never place an order.
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd
import requests

# Columns returned by Binance kline endpoint, in order.
_KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "num_trades",
    "taker_base", "taker_quote", "ignore",
]

# Valid Binance kline intervals (for early validation / nicer errors).
VALID_INTERVALS = {
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1M",
}


class BinanceError(RuntimeError):
    """Raised when market data could not be retrieved from any host."""


def fetch_klines(
    symbol: str,
    interval: str = "1h",
    limit: int = 1000,
    base_urls: Optional[List[str]] = None,
    timeout: int = 15,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
) -> pd.DataFrame:
    """Fetch up to 1000 OHLCV candles for ``symbol`` (one request).

    Returns a DataFrame sorted ascending by ``open_time`` with float OHLCV
    columns. ``start_time`` / ``end_time`` (ms) bound the window — pass
    ``end_time`` to page backwards into history. Tries each host in
    ``base_urls`` until one succeeds; raises :class:`BinanceError` if all fail.
    """
    if interval not in VALID_INTERVALS:
        raise BinanceError(
            f"Unsupported interval {interval!r}. Valid: {sorted(VALID_INTERVALS)}"
        )
    base_urls = base_urls or ["https://api.binance.com", "https://data-api.binance.vision"]
    limit = max(1, min(int(limit), 1000))
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    if start_time is not None:
        params["startTime"] = int(start_time)
    if end_time is not None:
        params["endTime"] = int(end_time)

    errors = []
    for base in base_urls:
        url = f"{base.rstrip('/')}/api/v3/klines"
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code != 200:
                errors.append(f"{base} -> HTTP {resp.status_code}: {resp.text[:160]}")
                continue
            rows = resp.json()
            if not rows:
                errors.append(f"{base} -> empty response for {symbol}")
                continue
            return _to_frame(rows, symbol, interval)
        except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
            # network / JSON decode / malformed-payload — fall through to next host
            errors.append(f"{base} -> {type(exc).__name__}: {exc}")
            continue

    raise BinanceError(
        f"Failed to fetch {symbol} [{interval}] from all hosts:\n  " + "\n  ".join(errors)
    )


def fetch_klines_history(
    symbol: str,
    interval: str = "1h",
    total: int = 1000,
    base_urls: Optional[List[str]] = None,
    timeout: int = 15,
    max_pages: int = 60,
) -> pd.DataFrame:
    """Fetch up to ``total`` candles by paging backwards through history.

    Binance serves at most 1000 candles per request, so to assemble the widest
    possible window we walk backwards using ``endTime`` until we have enough
    candles or the exchange runs out of history. The first page (no ``endTime``)
    anchors on the latest candle, so the series always ends at "now".
    """
    total = max(1, int(total))
    if total <= 1000:
        return fetch_klines(symbol, interval, total, base_urls, timeout)

    frames: List[pd.DataFrame] = []
    end_time: Optional[int] = None
    collected = 0
    for _ in range(max_pages):
        req = min(1000, total - collected)
        if req <= 0:
            break
        page = fetch_klines(symbol, interval, req, base_urls, timeout, end_time=end_time)
        if page.empty:
            break
        frames.append(page)
        collected += len(page)
        oldest = int(page["open_time"].iloc[0])
        end_time = oldest - 1            # next page ends just before the oldest we have
        if len(page) < req:             # exchange has no more history
            break

    if not frames:
        raise BinanceError(f"No history returned for {symbol} [{interval}].")
    out = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["open_time"])
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    return out.tail(total).reset_index(drop=True)


def _to_frame(rows: list, symbol: str, interval: str) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=_KLINE_COLS)
    numeric = ["open", "high", "low", "close", "volume", "quote_volume"]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    # Coerce integer fields defensively: a malformed/partial candle from a mirror
    # host can carry a JSON null, which a bare .astype("int64") would crash on.
    for col in ["open_time", "close_time", "num_trades"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open_time", "open", "high", "low", "close"]).copy()
    df["open_time"] = df["open_time"].astype("int64")
    df["close_time"] = df["close_time"].fillna(0).astype("int64")
    df["num_trades"] = df["num_trades"].fillna(0).astype("int64")
    df["symbol"] = symbol.upper()
    df["interval"] = interval
    keep = ["symbol", "interval", "open_time", "open", "high", "low",
            "close", "volume", "close_time", "num_trades"]
    return df[keep].sort_values("open_time").reset_index(drop=True)


def fetch_usdt_symbols(
    base_urls: Optional[List[str]] = None,
    timeout: int = 15,
) -> set:
    """Return the set of base assets with a TRADING USDT spot pair on Binance
    (e.g. {"BTC", "ETH", ...}). Best-effort: returns an empty set on failure."""
    base_urls = base_urls or ["https://api.binance.com", "https://data-api.binance.vision"]
    for base in base_urls:
        try:
            resp = requests.get(f"{base.rstrip('/')}/api/v3/exchangeInfo", timeout=timeout)
            if resp.status_code != 200:
                continue
            data = resp.json()
            return {
                s["baseAsset"].upper()
                for s in data.get("symbols", [])
                if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"
            }
        except (requests.RequestException, ValueError, KeyError):
            continue
    return set()


def fetch_24h_ticker(
    symbol: str,
    base_urls: Optional[List[str]] = None,
    timeout: int = 15,
) -> dict:
    """Fetch the 24h rolling ticker stats for a symbol (best-effort)."""
    base_urls = base_urls or ["https://api.binance.com", "https://data-api.binance.vision"]
    for base in base_urls:
        url = f"{base.rstrip('/')}/api/v3/ticker/24hr"
        try:
            resp = requests.get(url, params={"symbol": symbol.upper()}, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
        except requests.RequestException:
            continue
    return {}
