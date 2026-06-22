"""Indicator correctness and the all-important causality (no look-ahead) check."""
import numpy as np
import pandas as pd

from crypto_tool.analysis import indicators


def _cols_equal_on_prefix(full: pd.DataFrame, prefix: pd.DataFrame, cols, k):
    """Assert every column matches on the first k rows (NaNs treated as equal)."""
    for c in cols:
        a = full[c].to_numpy()[:k]
        b = prefix[c].to_numpy()[:k]
        both_nan = np.isnan(a) & np.isnan(b)
        close = np.isclose(a, b, rtol=1e-9, atol=1e-9) | both_nan
        assert close.all(), f"column {c} differs on prefix — look-ahead leak!"


def test_causality_no_lookahead(cfg, ohlcv):
    """Appending future bars must not change any earlier indicator value."""
    k = 300
    full = indicators.compute_indicators(ohlcv, cfg)
    prefix = indicators.compute_indicators(ohlcv.head(k).copy(), cfg)
    cols = ["sma_short", "ema_fast", "ema_slow", "rsi", "macd", "macd_signal",
            "macd_hist", "bb_mid", "bb_upper", "bb_lower", "bb_pctb", "atr", "roc"]
    _cols_equal_on_prefix(full, prefix, cols, k)


def test_rsi_bounds(cfg, ohlcv):
    out = indicators.compute_indicators(ohlcv, cfg)
    rsi = out["rsi"].dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all()


def test_macd_hist_identity(cfg, ohlcv):
    out = indicators.compute_indicators(ohlcv, cfg)
    diff = (out["macd_hist"] - (out["macd"] - out["macd_signal"])).abs().max()
    assert diff < 1e-9


def test_bollinger_ordering(cfg, ohlcv):
    out = indicators.compute_indicators(ohlcv, cfg)
    sub = out[["bb_lower", "bb_mid", "bb_upper"]].dropna()
    assert (sub["bb_lower"] <= sub["bb_mid"] + 1e-9).all()
    assert (sub["bb_mid"] <= sub["bb_upper"] + 1e-9).all()


def test_sma_matches_manual(cfg, ohlcv):
    out = indicators.compute_indicators(ohlcv, cfg)
    n = cfg["indicators"]["sma_short"]
    manual = ohlcv["close"].rolling(n).mean()
    assert np.allclose(out["sma_short"].dropna(), manual.dropna(), rtol=1e-12)
