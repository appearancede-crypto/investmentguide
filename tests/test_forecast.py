"""Outlook engine tests: quantile sanity, causality with teeth, honesty."""
import json

import numpy as np
import pandas as pd

from crypto_tool.analysis import forecast, signals
from crypto_tool.data import synthetic


def _enriched(cfg, n=400, seed=7):
    df = synthetic.generate_ohlcv("BTCUSDT", interval="1h", n=n, seed=seed)
    return signals.enrich(df, cfg)


def _fcfg(cfg):
    return {**forecast._DEFAULTS, **(cfg.get("web", {}).get("forecast") or {})}


def test_project_shape_and_quantile_order(cfg):
    fc = forecast.project(_enriched(cfg), cfg)
    assert fc is not None
    H = fc["horizon"]
    bands = fc["bands"]
    assert all(len(bands[f"p{q}"]) == H for q in (10, 25, 50, 75, 90))
    for j in range(H):
        assert (bands["p10"][j] <= bands["p25"][j] <= bands["p50"][j]
                <= bands["p75"][j] <= bands["p90"][j])
    # anchored to the current price's scale
    price = fc["asofPrice"]
    assert 0.2 * price < bands["p50"][0] < 5 * price
    assert fc["analogues"] <= fc["candidates"]


def test_project_checkpoints_and_calibration_sane(cfg):
    fc = forecast.project(_enriched(cfg, n=3000), cfg)
    assert fc["checkpoints"], "expected checkpoint stats"
    for cp in fc["checkpoints"]:
        assert 0.0 <= cp["prob_up"] <= 100.0
        assert cp["p10_pct"] <= cp["median_pct"] <= cp["p90_pct"]
        assert cp["bars"] <= fc["horizon"]
    cal = fc["calibration"]
    assert cal is not None
    assert 0.0 <= cal["coverage_pct"] <= 100.0
    assert cal["samples"] >= 10
    assert fc["quality"] in ("LOW", "MEDIUM", "HIGH")
    assert "not a promise" in fc["summary"]


def test_project_is_json_safe(cfg):
    json.dumps(forecast.project(_enriched(cfg), cfg), allow_nan=False)


def test_project_is_deterministic(cfg):
    e = _enriched(cfg)
    assert forecast.project(e, cfg) == forecast.project(e, cfg)


def test_project_refuses_thin_history(cfg):
    # 120 bars leaves far fewer fully-resolved analogues than min_candidates:
    # the honest answer is "no outlook", not a cone built on nothing.
    assert forecast.project(_enriched(cfg, n=120), cfg) is None


def test_analogues_are_declustered(cfg):
    """Selected analogues must be distinct episodes, not consecutive bars."""
    e = _enriched(cfg, n=3000)
    fc = forecast.project(e, cfg)
    n, H = len(e), fc["horizon"]
    X = forecast._features(e)
    close = e["close"].to_numpy(dtype=float)
    ok = np.isfinite(close) & (close > 0)
    feat_ok = np.all(np.isfinite(X), axis=1) & ok
    cand_ok = feat_ok & forecast._forward_clean(ok, H)
    idx = np.arange(n)
    candidates = idx[cand_ok & (idx + H <= n - 1)]
    sep = forecast._min_sep(H)
    nearest = forecast._knn(X, X[n - 1], candidates, forecast._pick_k(len(candidates), _fcfg(cfg)), sep)
    assert len(nearest) == fc["analogues"]
    gaps = np.diff(np.sort(nearest))
    assert gaps.min() >= sep


def _decoy_frame(cfg, n=400, seed=7):
    """Minimal enriched-like frame with hand-controlled features."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({
        "open_time": np.arange(n, dtype="int64") * 3_600_000,
        "close": close,
        "composite": rng.uniform(-60, 60, n),
        "vel_z": rng.normal(0, 1, n),
        "acc_z": rng.normal(0, 1, n),
        "rsi": rng.uniform(20, 80, n),
        "bb_pctb": rng.uniform(0, 1, n),
    })


def test_forbidden_zone_bars_are_never_analogues(cfg):
    """Causality with teeth: bars whose forward windows are unresolved are
    planted as PERFECT feature matches of 'now'. A leaky candidate mask would
    rank them nearest; the output must be identical to the un-poisoned run."""
    fcfg = _fcfg(cfg)
    H = int(fcfg["bars"])
    base = _decoy_frame(cfg)
    n = len(base)
    poisoned = base.copy()
    for col in ["composite", "vel_z", "acc_z", "rsi", "bb_pctb"]:
        poisoned.loc[n - H: n - 2, col] = base[col].iloc[n - 1]
    assert forecast.project(poisoned, cfg) == forecast.project(base, cfg)


def test_band_at_ignores_the_future(cfg):
    """Poison everything strictly after the replay bar t; the calibration band
    built 'as of t' must not change by a hair."""
    fcfg = _fcfg(cfg)
    H = int(fcfg["bars"])
    h_star = int(fcfg["checkpoints"][-1])
    e = _enriched(cfg, n=1500)
    close = e["close"].to_numpy(dtype=float)
    X = forecast._features(e)
    ok = np.isfinite(close) & (close > 0)
    cand_ok = (np.all(np.isfinite(X), axis=1) & ok) & forecast._forward_clean(ok, H)
    t = len(e) - h_star - 1
    band = forecast._band_at(t, close, X, cand_ok, H, h_star, fcfg)
    assert band is not None
    close2, X2 = close.copy(), X.copy()
    close2[t + 1:] *= 1000.0            # finite but absurd — must be invisible at t
    X2[t + 1:] += 100.0
    ok2 = np.isfinite(close2) & (close2 > 0)
    cand_ok2 = (np.all(np.isfinite(X2), axis=1) & ok2) & forecast._forward_clean(ok2, H)
    assert forecast._band_at(t, close2, X2, cand_ok2, H, h_star, fcfg) == band


def test_calibration_windows_do_not_overlap(cfg):
    """The reality check must count independent trials: replay bars are spaced
    at least h_star apart, so samples <= history / h_star."""
    e = _enriched(cfg, n=1500)
    fc = forecast.project(e, cfg)
    cal = fc["calibration"]
    assert cal is not None
    assert cal["samples"] <= len(e) // cal["horizon_bars"] + 1


def test_quality_medium_when_uncalibrated():
    assert forecast._quality(40, 200, None) == "MEDIUM"
    assert forecast._quality(10, 200, None) == "LOW"
    good = {"samples": 60, "coverage_pct": 80.0, "target_pct": 80}
    assert forecast._quality(40, 200, good) == "HIGH"
    bad = {"samples": 60, "coverage_pct": 50.0, "target_pct": 80}
    assert forecast._quality(40, 200, bad) == "MEDIUM"


def test_pick_k_never_exceeds_pool():
    fcfg = dict(forecast._DEFAULTS)
    fcfg["k_min"] = 25
    assert forecast._pick_k(10, fcfg) == 10      # pool smaller than k_min
    assert forecast._pick_k(1000, fcfg) <= fcfg["k_max"]


def test_nan_close_mid_history_is_survivable(cfg):
    """A bad candle must not crash the page: result is JSON-safe or None."""
    e = _enriched(cfg, n=1000)
    e.loc[500, "close"] = float("nan")
    fc = forecast.project(e, cfg)
    json.dumps(fc, allow_nan=False)
