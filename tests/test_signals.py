"""Signal engine: ranges, causality and rationale shape."""
import numpy as np

from crypto_tool.analysis import signals
from crypto_tool.data import synthetic


def test_composite_and_confidence_ranges(cfg, ohlcv):
    e = signals.enrich(ohlcv, cfg)
    comp = e["composite"].dropna()
    conf = e["confidence"].dropna()
    assert (comp >= -100.0001).all() and (comp <= 100.0001).all()
    assert (conf >= 0).all() and (conf <= 1.0001).all()


def test_flags_consistent_with_thresholds(cfg, ohlcv):
    e = signals.enrich(ohlcv, cfg)
    s = cfg["signals"]
    buys = e[e["flag"].isin(["BUY", "STRONG BUY"])]
    sells = e[e["flag"].isin(["SELL", "STRONG SELL"])]
    assert (buys["composite"] >= s["buy_threshold"]).all()
    assert (sells["composite"] <= s["sell_threshold"]).all()


def test_signal_causality(cfg):
    df = synthetic.generate_ohlcv("SOLUSDT", interval="1h", n=350, seed=11)
    k = 250
    full = signals.enrich(df, cfg)
    prefix = signals.enrich(df.head(k).copy(), cfg)
    for c in ["composite", "confidence", "c_curvature", "c_trend", "c_rsi"]:
        a, b = full[c].to_numpy()[:k], prefix[c].to_numpy()[:k]
        both_nan = np.isnan(a) & np.isnan(b)
        assert (np.isclose(a, b, rtol=1e-9, atol=1e-9) | both_nan).all(), c


def test_latest_signal_shape(cfg, ohlcv):
    sig = signals.latest_signal(ohlcv, cfg)
    assert set(["composite", "confidence", "flag", "rationale"]).issubset(sig)
    assert len(sig["rationale"]) == len(signals.COMPONENT_NAMES)
    # contributions should sum (weighted) consistently with the composite sign-ish
    assert isinstance(sig["rationale"][0]["detail"], str)
