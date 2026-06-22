"""Signal engine: combine indicators + curvature into a transparent score.

Design principles
-----------------
* **Every rule is explainable.** Each component returns a vote in [-1, +1]
  (bearish .. bullish). The composite is their weighted average, rescaled to
  [-100, +100]. Nothing is a black box — the dashboard shows each vote.
* **Confidence is separate from direction.** A score can be strongly bullish
  but low-confidence (rules disagree, thin volume, little history). We surface
  both so you never act on a loud-but-flimsy signal.
* **Causal.** Components use only causal indicators, so the exact same function
  produces the per-bar series the backtester replays — no look-ahead.

This is a systematic *opinion*, not a prediction. It is wiser than a human only
in the narrow, real sense of being consistent, unemotional and complete across
every coin, every bar, 24/7. It cannot see the future. Not financial advice.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from . import curvature, indicators

COMPONENT_NAMES = ["trend", "macd", "rsi", "bollinger", "curvature", "momentum", "volume"]


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def enrich(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Full pipeline: indicators -> curvature -> components -> composite/flag."""
    out = indicators.compute_indicators(df, cfg)
    out = curvature.compute_curvature(out, cfg)
    out = compute_components(out, cfg)
    return out


def _clip(s: pd.Series) -> pd.Series:
    return s.clip(-1.0, 1.0)


def compute_components(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Compute each per-bar component vote, the composite score and confidence."""
    out = df.copy()
    eps = 1e-12

    # --- Trend (trend-following): EMA gap + price vs slow EMA + long-term cross
    slow = out["ema_slow"].replace(0.0, np.nan)
    align = np.tanh((out["ema_fast"] - out["ema_slow"]) / (slow.abs() * 0.01))
    pos = np.tanh((out["close"] - out["ema_slow"]) / (slow.abs() * 0.01))
    long_cross = np.sign(out["sma_short"] - out["sma_long"]).fillna(0.0)
    out["c_trend"] = _clip(0.4 * align + 0.3 * pos + 0.3 * long_cross)

    # --- MACD (trend-following): histogram magnitude + crossover sign
    hist_scale = out["macd_hist"].rolling(50, min_periods=10).std(ddof=0)
    hist_norm = out["macd_hist"] / hist_scale.replace(0.0, np.nan)
    cross = np.sign(out["macd"] - out["macd_signal"]).fillna(0.0)
    out["c_macd"] = _clip(0.7 * np.tanh(hist_norm.fillna(0.0)) + 0.3 * cross)

    # --- RSI (mean-reversion): oversold = bullish, overbought = bearish
    out["c_rsi"] = _clip((50.0 - out["rsi"]) / 30.0)

    # --- Bollinger (mean-reversion): below lower band = bullish
    out["c_bollinger"] = _clip((0.5 - out["bb_pctb"]) * 2.0)

    # --- Curvature (the headline): acceleration (bend) + velocity (change)
    # Left NaN through the z-score warm-up so it reads as not-yet-live for
    # confidence/sufficiency (the composite still treats warm-up as neutral).
    out["c_curvature"] = _clip(0.6 * np.tanh(out["acc_z"]) + 0.4 * np.tanh(out["vel_z"]))

    # --- Momentum (trend-following): normalised rate of change
    roc_scale = out["roc"].rolling(50, min_periods=10).std(ddof=0)
    out["c_momentum"] = _clip(np.tanh(out["roc"] / roc_scale.replace(0.0, np.nan)))

    # --- Volume (confirmation): smoothed OBV slope direction
    obv_slope = out["obv"].diff().rolling(cfg["indicators"]["obv_smooth"], min_periods=1).mean()
    obv_scale = obv_slope.abs().rolling(50, min_periods=10).mean()
    out["c_volume"] = _clip(np.tanh(obv_slope / obv_scale.replace(0.0, np.nan)))

    # Volume *factor* (0..1) feeds confidence, not direction: are we above the
    # typical volume? Left NaN during the volume-MA warm-up so its confidence
    # term is dropped rather than silently faked as average (0.5) volume.
    vol_ratio = out["volume"] / out["vol_ma"].replace(0.0, np.nan)
    out["vol_factor"] = (np.tanh(vol_ratio - 1.0) * 0.5 + 0.5).clip(0.0, 1.0)

    # --- Weighted composite ------------------------------------------------- #
    weights = cfg["signals"]["weights"]
    comp_cols = [f"c_{name}" for name in COMPONENT_NAMES]
    w = np.array([weights[name] for name in COMPONENT_NAMES], dtype=float)
    w_sum = w.sum()

    values = out[comp_cols].to_numpy()                 # (bars, n_components)
    valid = ~np.isnan(values)                          # which rules are live this bar
    filled = np.nan_to_num(values, nan=0.0)            # warm-up rules count as neutral

    num = filled @ w                                   # weighted signed sum
    abs_sum = np.abs(filled) @ w                       # weighted magnitude sum
    live_w = valid @ w                                 # weight of the live rules

    # Composite = weighted average over the rules that are actually live, so an
    # early bar is not diluted toward 0 by rules still warming up. Bounded to
    # [-100, 100] since |num| <= abs_sum <= live_w. trend & macd are live from
    # bar 0, so live_w > 0 in practice (guarded regardless).
    safe_w = np.where(live_w > eps, live_w, 1.0)
    composite = np.where(live_w > eps, num / safe_w, 0.0) * 100.0
    out["composite"] = composite

    # Directional agreement among the rules that have an opinion: 0 (cancel) .. 1.
    safe_abs = np.where(abs_sum > eps, abs_sum, 1.0)
    agreement = np.where(abs_sum > eps, np.abs(num) / safe_abs, 0.0)
    magnitude = np.abs(composite) / 100.0
    sufficiency = live_w / w_sum                       # fraction of weight live
    vol_factor = out["vol_factor"].to_numpy()
    vol_live = ~np.isnan(vol_factor)

    # Confidence blends rule agreement, signal strength, volume confirmation and
    # data depth. The volume term is dropped (weights renormalised) while volume
    # data is still warming up, rather than being faked as average.
    w_agr, w_mag, w_suf, w_vol = 0.45, 0.25, 0.15, 0.15
    conf_num = (w_agr * agreement + w_mag * magnitude + w_suf * sufficiency
                + np.where(vol_live, w_vol * np.nan_to_num(vol_factor), 0.0))
    conf_den = w_agr + w_mag + w_suf + np.where(vol_live, w_vol, 0.0)
    out["confidence"] = np.clip(conf_num / conf_den, 0.0, 1.0)
    out["agreement"] = agreement

    out["flag"] = _flags(out["composite"], cfg)
    return out


def _flags(composite: pd.Series, cfg: Dict[str, Any]) -> pd.Series:
    return pd.Series(classify(composite.to_numpy(), cfg), index=composite.index)


def classify(composite, cfg: Dict[str, Any]):
    """Vectorised flag labels for an array/Series of composite scores."""
    comp = np.asarray(composite, dtype=float)
    s = cfg["signals"]
    conditions = [
        comp >= s["strong_buy_threshold"],
        comp >= s["buy_threshold"],
        comp <= s["strong_sell_threshold"],
        comp <= s["sell_threshold"],
    ]
    choices = ["STRONG BUY", "BUY", "STRONG SELL", "SELL"]
    return np.select(conditions, choices, default="NEUTRAL")


# --------------------------------------------------------------------------- #
# Reweighting helpers — let the paper trader / learner recompute the composite
# from precomputed component votes with arbitrary weights, cheaply (no need to
# recompute indicators). The formula matches compute_components exactly.
# --------------------------------------------------------------------------- #
def weights_vector(weights: Dict[str, Any]) -> np.ndarray:
    return np.array([float(weights[name]) for name in COMPONENT_NAMES], dtype=float)


def component_matrix(df: pd.DataFrame) -> np.ndarray:
    """(bars, 7) array of the c_* component votes, in COMPONENT_NAMES order."""
    return df[[f"c_{name}" for name in COMPONENT_NAMES]].to_numpy(dtype=float)


def composite_from_components(values: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Recompute composite scores from a (bars, 7) vote matrix and a weight
    vector, using the same live-weight renormalisation as compute_components."""
    eps = 1e-12
    values = np.atleast_2d(values)
    valid = ~np.isnan(values)
    filled = np.nan_to_num(values, nan=0.0)
    num = filled @ w
    live_w = valid @ w
    safe = np.where(live_w > eps, live_w, 1.0)
    return np.where(live_w > eps, num / safe, 0.0) * 100.0


# --------------------------------------------------------------------------- #
# Human-readable rationale for the latest bar
# --------------------------------------------------------------------------- #
def _rationale_rows(row: pd.Series, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    weights = cfg["signals"]["weights"]
    rsi_v = row.get("rsi", float("nan"))
    pctb = row.get("bb_pctb", float("nan"))
    explain = {
        "trend": _trend_text(row),
        "macd": _macd_text(row),
        "rsi": f"RSI {rsi_v:.1f} — " + ("oversold (bullish)" if rsi_v < 30 else
               "overbought (bearish)" if rsi_v > 70 else "neutral"),
        "bollinger": f"%B {pctb:.2f} — " + ("near/under lower band (bullish)" if pctb < 0.2 else
                     "near/over upper band (bearish)" if pctb > 0.8 else "mid-band"),
        "curvature": _curvature_text(row),
        "momentum": f"ROC {row.get('roc', float('nan')):.2f}% over period",
        "volume": "OBV rising (accumulation)" if row.get("c_volume", 0) > 0 else
                  "OBV falling (distribution)" if row.get("c_volume", 0) < 0 else "flat volume flow",
    }
    rows = []
    for name in COMPONENT_NAMES:
        val = float(row.get(f"c_{name}", 0.0) or 0.0)
        rows.append({
            "component": name,
            "vote": round(val, 3),
            "weight": weights[name],
            "contribution": round(val * weights[name], 3),
            "detail": explain[name],
        })
    rows.sort(key=lambda r: abs(r["contribution"]), reverse=True)
    return rows


def _trend_text(row: pd.Series) -> str:
    regime = int(row.get("regime", 0) or 0)
    word = {1: "up-trend", -1: "down-trend", 0: "sideways"}[regime]
    cross = "golden (50>200)" if row.get("sma_short", 0) > row.get("sma_long", 0) else "death (50<200)"
    return f"{word}; MA cross {cross}"


def _macd_text(row: pd.Series) -> str:
    above = row.get("macd", 0) > row.get("macd_signal", 0)
    return ("MACD above signal (bullish)" if above else "MACD below signal (bearish)") + \
           f"; hist {row.get('macd_hist', float('nan')):.4g}"


def _curvature_text(row: pd.Series) -> str:
    vel_z = row.get("vel_z", float("nan"))
    acc_z = row.get("acc_z", float("nan"))
    bend = "curving up" if (acc_z or 0) > 0 else "curving down"
    slope = "rising" if (vel_z or 0) > 0 else "falling"
    note = ""
    if (vel_z or 0) < -0.3 and (acc_z or 0) > 0.3:
        note = " — decelerating decline (possible bottoming)"
    elif (vel_z or 0) > 0.3 and (acc_z or 0) < -0.3:
        note = " — fading rally (possible topping)"
    return f"slope {slope} (z={vel_z:.2f}), {bend} (z={acc_z:.2f}){note}"


def latest_signal(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Enrich and return a structured summary of the most recent bar."""
    enriched = enrich(df, cfg)
    row = enriched.iloc[-1]
    return {
        "symbol": row.get("symbol", "?"),
        "interval": row.get("interval", "?"),
        "open_time": int(row["open_time"]),
        "close": float(row["close"]),
        "composite": float(row["composite"]),
        "confidence": float(row["confidence"]),
        "flag": str(row["flag"]),
        "rationale": _rationale_rows(row, cfg),
        "enriched": enriched,
    }


def scan(conn, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Compute the latest signal for every symbol present in the DB.

    Returns a DataFrame ranked by composite score (most bullish first).
    """
    from ..data import database  # local import to avoid a cycle

    interval = cfg["data"]["interval"]
    min_bars = max(cfg["indicators"]["ema_slow"], cfg["indicators"]["bb_period"]) + 5
    rows = []
    for symbol in database.list_symbols(conn, interval):
        df = database.load_ohlcv(conn, symbol, interval)
        if len(df) < min_bars:
            continue
        sig = latest_signal(df, cfg)
        top = sig["rationale"][0]["detail"] if sig["rationale"] else ""
        rows.append({
            "symbol": symbol,
            "price": sig["close"],
            "composite": round(sig["composite"], 1),
            "confidence": round(sig["confidence"] * 100, 1),
            "flag": sig["flag"],
            "top_driver": top,
            "open_time": sig["open_time"],
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["composite", "confidence"], ascending=False).reset_index(drop=True)
    return out
