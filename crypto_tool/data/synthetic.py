"""Deterministic synthetic OHLCV generator.

Used for two things:
  * offline demos / first-run experience when Binance is unreachable
    (e.g. geo-blocked or no network), and
  * reproducible unit tests.

The generator layers a drifting geometric random walk with slow sinusoidal
cycles so the series exhibits real trends, reversals and *curvature* — exactly
the structure the signal engine is meant to read.
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

# Interval string -> milliseconds. '1M' is approximated as 30 days.
_UNIT_MS = {"m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000, "M": 2_592_000_000}

# Fixed anchor for the *last* candle: timestamps stay deterministic yet land in
# a recent, plausible window rather than years in the past.
_END_ANCHOR_MS = 1_748_736_000_000  # ~2025-06-01T00:00:00Z


def interval_to_ms(interval: str) -> int:
    """Convert a Binance interval string (e.g. '15m', '4h', '1d') to milliseconds."""
    interval = interval.strip()
    unit = interval[-1:]
    if unit not in _UNIT_MS:
        raise ValueError(
            f"Unrecognised interval unit in {interval!r}; expected one of {sorted(_UNIT_MS)}"
        )
    qty = interval[:-1]
    if not qty.isdigit():
        raise ValueError(
            f"Invalid interval {interval!r}: expected <integer><unit>, e.g. '15m', '4h', '1d'"
        )
    return int(qty) * _UNIT_MS[unit]


# Stable per-symbol starting prices so demo data looks plausible.
_BASE_PRICES = {
    "BTCUSDT": 42000.0, "ETHUSDT": 2300.0, "BNBUSDT": 320.0, "SOLUSDT": 95.0,
    "XRPUSDT": 0.52, "ADAUSDT": 0.45, "DOGEUSDT": 0.085, "AVAXUSDT": 28.0,
    "LINKUSDT": 14.0, "LTCUSDT": 72.0,
}


def _seed_from_symbol(symbol: str) -> int:
    # zlib.crc32 is stable across processes, unlike the builtin hash() which is
    # salted by PYTHONHASHSEED — so seed-demo data is reproducible run to run.
    return zlib.crc32(symbol.encode("utf-8"))


def generate_ohlcv(
    symbol: str,
    interval: str = "1h",
    n: int = 1000,
    seed: int | None = None,
    start_price: float | None = None,
) -> pd.DataFrame:
    """Generate ``n`` synthetic candles for ``symbol`` as a standard OHLCV frame."""
    if seed is None:
        seed = _seed_from_symbol(symbol)
    rng = np.random.default_rng(seed)
    start_price = start_price or _BASE_PRICES.get(symbol.upper(), 100.0)
    step = interval_to_ms(interval)

    # Per-bar drift and volatility (tuned to feel like hourly crypto).
    mu = 0.00005
    sigma = 0.012

    t = np.arange(n, dtype=np.int64)  # int64 so (n-1-t)*step never overflows
    # Two slow cycles create curving trends / regime changes.
    cycle = (
        0.0010 * np.sin(2 * np.pi * t / 220.0)
        + 0.0006 * np.sin(2 * np.pi * t / 70.0 + 1.3)
    )
    shocks = rng.normal(0.0, 1.0, size=n)
    log_ret = (mu - 0.5 * sigma**2) + sigma * shocks + cycle
    close = start_price * np.exp(np.cumsum(log_ret))

    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]

    body_hi = np.maximum(open_, close)
    body_lo = np.minimum(open_, close)
    wick_up = np.abs(rng.normal(0.0, 0.004, size=n))
    wick_dn = np.abs(rng.normal(0.0, 0.004, size=n))
    high = body_hi * (1.0 + wick_up)
    low = body_lo * (1.0 - wick_dn)

    # Volume rises with absolute returns (activity clusters around big moves).
    base_vol = max(start_price, 1.0) * 50.0
    volume = base_vol * (0.6 + 4.0 * np.abs(log_ret)) * (0.5 + rng.random(n))

    open_time = _END_ANCHOR_MS - (n - 1 - t) * step  # last candle at the anchor
    close_time = open_time + step - 1

    return pd.DataFrame(
        {
            "symbol": symbol.upper(),
            "interval": interval,
            "open_time": open_time.astype("int64"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "close_time": close_time.astype("int64"),
            "num_trades": (volume / 10.0).astype("int64"),
        }
    )
