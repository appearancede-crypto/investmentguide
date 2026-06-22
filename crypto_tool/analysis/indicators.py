"""Technical indicators, implemented in pure pandas/numpy (no native deps).

Every indicator here is **causal**: the value at bar *t* depends only on bars
<= *t*. Appending future bars never changes an earlier value. That property is
what makes the backtester's results free of look-ahead bias, and it is checked
directly in the test-suite.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI in [0, 100]."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss
    out = 100.0 - 100.0 / (1.0 + rs)
    # When average loss is zero, RSI is defined as 100.
    out = out.where(avg_loss != 0, 100.0)
    out = out.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return out


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist})


def bollinger(series: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = series.rolling(n, min_periods=n).mean()
    sd = series.rolling(n, min_periods=n).std(ddof=0)
    upper = mid + k * sd
    lower = mid - k * sd
    width = (upper - lower)
    pctb = (series - lower) / width.replace(0.0, np.nan)
    bandwidth = width / mid.replace(0.0, np.nan)
    return pd.DataFrame(
        {"bb_mid": mid, "bb_upper": upper, "bb_lower": lower,
         "bb_pctb": pctb, "bb_bw": bandwidth}
    )


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average True Range (Wilder)."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def roc(series: pd.Series, n: int = 10) -> pd.Series:
    """Rate of change in percent."""
    return (series / series.shift(n) - 1.0) * 100.0


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(df["close"].diff()).fillna(0.0)
    return (direction * df["volume"]).fillna(0.0).cumsum()


def realized_vol(series: pd.Series, n: int = 20) -> pd.Series:
    """Rolling standard deviation of log returns (per-bar volatility)."""
    log_ret = np.log(series / series.shift(1))
    return log_ret.rolling(n, min_periods=n).std(ddof=0)


def compute_indicators(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Return a copy of ``df`` with all indicator columns appended.

    ``df`` must contain open/high/low/close/volume and be ordered ascending by
    ``open_time``.
    """
    ic = cfg["indicators"]
    out = df.copy().reset_index(drop=True)
    close = out["close"]

    out["sma_short"] = sma(close, ic["sma_short"])
    out["sma_long"] = sma(close, ic["sma_long"])
    out["ema_fast"] = ema(close, ic["ema_fast"])
    out["ema_slow"] = ema(close, ic["ema_slow"])
    out["rsi"] = rsi(close, ic["rsi_period"])

    m = macd(close, ic["ema_fast"], ic["ema_slow"], ic["macd_signal"])
    out[["macd", "macd_signal", "macd_hist"]] = m

    b = bollinger(close, ic["bb_period"], ic["bb_std"])
    out[["bb_mid", "bb_upper", "bb_lower", "bb_pctb", "bb_bw"]] = b

    out["atr"] = atr(out, ic["atr_period"])
    out["roc"] = roc(close, ic["roc_period"])
    out["obv"] = obv(out)
    out["rvol"] = realized_vol(close, ic["vol_ma_period"])
    out["vol_ma"] = sma(out["volume"], ic["vol_ma_period"])
    return out
