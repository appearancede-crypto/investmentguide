"""Exit-point analysis tests."""
import numpy as np

from crypto_tool.analysis import exits, signals
from crypto_tool.data import synthetic


def test_exit_summary_keys(cfg, ohlcv):
    e = signals.enrich(ohlcv, cfg)
    summ = exits.exit_summary(e, cfg)
    for k in ["recommendation", "trailing_stop", "take_profit", "atr_stop",
              "swing_low", "resistance", "risk_pct", "reward_pct", "reasons"]:
        assert k in summ
    assert summ["recommendation"] in {"EXIT", "TRIM / TIGHTEN STOP", "HOLD — trail the stop"}
    assert summ["take_profit"] > summ["price"]          # target is above price (long)
    assert summ["swing_low"] <= summ["price"]           # structural stop below price


def test_exit_levels_finite_and_ordered(cfg, ohlcv):
    df = exits.add_exit_levels(signals.enrich(ohlcv, cfg), cfg)
    tail = df.tail(50)
    assert np.isfinite(tail["trailing_stop"]).all()
    assert np.isfinite(tail["take_profit"]).all()
    # take-profit is always above the close; trailing stop is below the recent high
    assert (tail["take_profit"] > tail["close"]).all()
    assert (tail["trailing_stop"] <= tail["roll_high"]).all()


def test_exit_levels_causal(cfg):
    """Appending future bars must not change earlier exit levels (no look-ahead)."""
    df = synthetic.generate_ohlcv("BTCUSDT", interval="1h", n=400, seed=4)
    k = 300
    full = exits.add_exit_levels(signals.enrich(df, cfg), cfg)
    prefix = exits.add_exit_levels(signals.enrich(df.head(k).copy(), cfg), cfg)
    for c in ["trailing_stop", "take_profit", "roll_low"]:
        a, b = full[c].to_numpy()[:k], prefix[c].to_numpy()[:k]
        both_nan = np.isnan(a) & np.isnan(b)
        assert (np.isclose(a, b, rtol=1e-9, atol=1e-9) | both_nan).all(), c
