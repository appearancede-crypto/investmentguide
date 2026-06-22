"""Exit-point analysis — where to get out, not just where to get in.

The signal engine flags entries; this module answers the harder question every
holder faces: *"I'm in — when do I take profit or cut the loss?"* From the
latest bar it computes:

  * a **trailing stop** (Chandelier Exit: recent-high − ATR×k) that ratchets up
    as price rises and never moves down,
  * a **structural stop** (recent swing low),
  * an **ATR take-profit target** and the nearest overhead **resistance**,
  * the **risk/reward** those levels imply, and
  * an **exit signal** with plain-English reasons (momentum rolled over,
    overbought, trailing stop breached, rally fading, bearish MACD…).

These are disciplined, rule-based exits — precisely the unemotional part humans
are worst at (holding losers, bailing on winners). They are not predictions or
financial advice.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def add_exit_levels(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Append per-bar exit levels so they can be plotted as lines.

    Expects an *enriched* frame (with ``atr`` from the indicator stage).
    """
    e = cfg["exits"]
    n = e["lookback"]
    out = df.copy()
    atr = out["atr"]
    out["roll_high"] = out["high"].rolling(n, min_periods=1).max()
    out["roll_low"] = out["low"].rolling(n, min_periods=1).min()
    # Chandelier Exit (long): trails the *rolling* N-bar high by k ATRs. It rises
    # with new highs and eases down as old highs roll off the window — so a coin
    # far below its peak isn't stuck with an all-time-high stop.
    out["trailing_stop"] = out["roll_high"] - atr * e["atr_mult_stop"]
    out["atr_stop"] = out["close"] - atr * e["atr_mult_stop"]
    out["take_profit"] = out["close"] + atr * e["atr_mult_target"]
    return out


def exit_summary(enriched: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Exit guidance for the latest bar, assuming a long position is held."""
    e = cfg["exits"]
    s = cfg["signals"]
    df = add_exit_levels(enriched, cfg)
    row = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else row

    price = float(row["close"])
    trailing = float(row["trailing_stop"])
    # The stop in force entering this bar (no peeking at this bar's own high).
    trailing_in_force = float(prev["trailing_stop"]) if len(df) > 1 else trailing
    swing_low = float(row["roll_low"])
    target = float(row["take_profit"])
    resistance = float(row["roll_high"])

    reasons: List[str] = []
    hard_exit = False

    if price < trailing_in_force:
        reasons.append(f"Trailing stop breached (price {price:.4g} < stop {trailing_in_force:.4g})")
        hard_exit = True
    comp = float(row.get("composite", 0.0) or 0.0)
    if comp <= s["strong_sell_threshold"]:
        reasons.append(f"Strong bearish signal (score {comp:.0f})")
        hard_exit = True
    elif comp <= s["sell_threshold"]:
        reasons.append(f"Momentum turned bearish (score {comp:.0f})")

    rsi = float(row.get("rsi", float("nan")))
    if rsi == rsi and rsi > e["rsi_overbought"]:
        reasons.append(f"Overbought (RSI {rsi:.0f}) — consider trimming into strength")
    if "bb_upper" in row and price >= float(row["bb_upper"]):
        reasons.append("Stretched above the upper Bollinger band")
    vel_z = float(row.get("vel_z", float("nan")) or 0.0)
    acc_z = float(row.get("acc_z", float("nan")) or 0.0)
    if vel_z > 0.3 and acc_z < -0.3:
        reasons.append("Rally fading — curvature rolling over (possible top)")
    if float(row.get("macd", 0.0) or 0.0) < float(row.get("macd_signal", 0.0) or 0.0):
        reasons.append("MACD below its signal line (bearish)")

    soft = len([r for r in reasons if "Trailing stop" not in r and "Strong bearish" not in r])
    if hard_exit:
        recommendation = "EXIT"
    elif soft >= 2:
        recommendation = "TRIM / TIGHTEN STOP"
    else:
        recommendation = "HOLD — trail the stop"
        if not reasons:
            reasons.append("No exit trigger yet — keep trailing the stop upward")

    risk = price - trailing                          # < 0 means the stop is above price
    reward = max(target - price, 0.0)
    return {
        "symbol": row.get("symbol", "?"),
        "price": price,
        "recommendation": recommendation,
        "reasons": reasons,
        "trailing_stop": trailing,
        "atr_stop": float(row["atr_stop"]),
        "swing_low": swing_low,
        "take_profit": target,
        "resistance": resistance,
        "risk_pct": round((price - trailing) / price * 100, 2),
        "reward_pct": round((target - price) / price * 100, 2),
        # R:R only meaningful while the stop is still below price (not yet hit).
        "risk_reward": round(reward / risk, 2) if risk > 1e-9 else None,
    }


def exit_scan(conn, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Exit guidance for every symbol with enough history (ranked: act-now first)."""
    from . import signals
    from ..data import database

    interval = cfg["data"]["interval"]
    min_bars = max(cfg["indicators"]["ema_slow"], cfg["indicators"]["bb_period"],
                   cfg["exits"]["lookback"]) + 5
    order = {"EXIT": 0, "TRIM / TIGHTEN STOP": 1, "HOLD — trail the stop": 2}
    rows = []
    for symbol in database.list_symbols(conn, interval):
        df = database.load_ohlcv(conn, symbol, interval)
        if len(df) < min_bars:
            continue
        enriched = signals.enrich(df, cfg)
        summ = exit_summary(enriched, cfg)
        rows.append({
            "symbol": symbol,
            "price": summ["price"],
            "action": summ["recommendation"],
            "trailing_stop": round(summ["trailing_stop"], 6),
            "take_profit": round(summ["take_profit"], 6),
            "risk_pct": summ["risk_pct"],
            "reward_pct": summ["reward_pct"],
            "rr": summ["risk_reward"],
            "why": summ["reasons"][0] if summ["reasons"] else "",
            "_o": order.get(summ["recommendation"], 3),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["_o", "symbol"]).drop(columns="_o").reset_index(drop=True)
    return out
