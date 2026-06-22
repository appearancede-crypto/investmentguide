"""Curvature features: sign sanity and causality."""
import numpy as np
import pandas as pd

from crypto_tool.analysis import curvature
from crypto_tool.data import synthetic


def _frame_from_close(close):
    n = len(close)
    return pd.DataFrame({
        "symbol": "X", "interval": "1h",
        "open_time": np.arange(n) * 3_600_000,
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": np.ones(n), "close_time": np.arange(n),
        "num_trades": np.ones(n),
    })


def test_curvature_positive_on_convex_up(cfg):
    # A convex-up accelerating series should end with positive curvature.
    t = np.arange(120, dtype=float)
    close = 100.0 + 0.02 * t**2          # accelerating upward => positive 2nd deriv
    out = curvature.compute_curvature(_frame_from_close(close), cfg)
    assert out["accel"].iloc[-1] > 0


def test_curvature_negative_on_concave(cfg):
    t = np.arange(120, dtype=float)
    close = 100.0 + 5.0 * t - 0.02 * t**2  # decelerating => negative 2nd deriv
    out = curvature.compute_curvature(_frame_from_close(close), cfg)
    assert out["accel"].iloc[-1] < 0


def test_curvature_causality(cfg):
    df = synthetic.generate_ohlcv("ETHUSDT", interval="1h", n=300, seed=3)
    k = 200
    full = curvature.compute_curvature(df, cfg)
    prefix = curvature.compute_curvature(df.head(k).copy(), cfg)
    for c in ["velocity", "accel", "vel_z", "acc_z"]:
        a, b = full[c].to_numpy()[:k], prefix[c].to_numpy()[:k]
        both_nan = np.isnan(a) & np.isnan(b)
        assert (np.isclose(a, b, rtol=1e-9, atol=1e-9) | both_nan).all(), c
