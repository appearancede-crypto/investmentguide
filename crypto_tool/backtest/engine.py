"""Event-driven backtester — honest by construction.

Rules of the simulation (chosen to avoid the classic ways backtests lie):

  * **No look-ahead.** A decision uses the composite score known at a bar's
    *close*; the resulting trade fills at the *next* bar's *open*. You can never
    trade on information you would not yet have had.
  * **Costs are real.** Every fill pays a configurable fee and slippage on both
    entry and exit.
  * **Stops/targets fill intrabar** using the bar's high/low (stop checked
    first). A bar that gaps through the level fills at the bar's *open*, not the
    untraded level price, so gap risk is not silently assumed away.
  * **Long/flat only.** Spot-style: hold the coin or hold cash. No leverage,
    no shorting — matching the analysis-only, no-derivatives scope.

A benchmark buy-&-hold equity curve is produced alongside so results are always
judged against "just holding the coin", which is a brutally hard benchmark to
beat in a bull market.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..analysis import signals


def run_backtest(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Replay the strategy over ``df``. Returns equity curve, trades and signals."""
    p = {**cfg["backtest"], **(params or {})}
    e = signals.enrich(df, cfg).reset_index(drop=True)
    n = len(e)

    open_ = e["open"].to_numpy(dtype=float)
    high = e["high"].to_numpy(dtype=float)
    low = e["low"].to_numpy(dtype=float)
    close = e["close"].to_numpy(dtype=float)
    comp = e["composite"].to_numpy(dtype=float)
    otime = e["open_time"].to_numpy(dtype="int64")
    ctime = e["close_time"].to_numpy(dtype="int64")

    fee = p["fee_pct"] / 100.0
    slip = p["slippage_pct"] / 100.0
    entry_score = p["entry_score"]
    exit_score = p["exit_score"]
    sl = p["stop_loss_pct"] / 100.0
    tp = p["take_profit_pct"] / 100.0

    cash = 1.0          # normalised starting capital
    units = 0.0
    position = False
    entry_fill = 0.0
    entry_time = 0
    stop_price = -np.inf
    tp_price = np.inf

    equity = np.empty(n)
    in_market = np.zeros(n, dtype=bool)
    trades = []
    pending = None       # 'enter' | 'exit' — order placed at bar i, filled at i+1

    def _close_position(exit_price: float, exit_time: int, reason: str):
        nonlocal cash, units, position, entry_fill
        fill = exit_price * (1.0 - fee - slip)
        cash = units * fill
        ret = fill / entry_fill - 1.0
        trades.append({
            "entry_time": int(entry_time), "exit_time": int(exit_time),
            "entry_price": round(entry_fill, 8), "exit_price": round(fill, 8),
            "return_pct": ret * 100.0, "reason": reason,
        })
        units = 0.0
        position = False

    for i in range(n):
        # 1) Execute any order placed on the previous bar, at this bar's open.
        if pending == "enter" and not position:
            entry_fill = open_[i] * (1.0 + fee + slip)
            units = cash / entry_fill
            cash = 0.0
            position = True
            entry_time = otime[i]
            stop_price = open_[i] * (1.0 - sl) if sl > 0 else -np.inf
            tp_price = open_[i] * (1.0 + tp) if tp > 0 else np.inf
        elif pending == "exit" and position:
            _close_position(open_[i], otime[i], "signal")
        pending = None

        # 2) Intrabar protective exits (stop checked before target). Fills are
        #    clamped to this bar's open: a bar that gaps THROUGH the level fills
        #    at the open (the realistic worst case for a stop), not at the
        #    untraded level price. exit_time is the bar's close (intrabar, after
        #    the open) so it never collides with a same-bar entry timestamp.
        if position:
            if sl > 0 and low[i] <= stop_price:
                _close_position(min(stop_price, open_[i]), ctime[i], "stop_loss")
            elif tp > 0 and high[i] >= tp_price:
                _close_position(max(tp_price, open_[i]), ctime[i], "take_profit")

        # 3) Mark-to-market at the close.
        equity[i] = cash if not position else units * close[i]
        in_market[i] = position

        # 4) Decide an order for the *next* bar from this bar's (closed) signal.
        if i < n - 1 and not np.isnan(comp[i]):
            if not position and comp[i] >= entry_score:
                pending = "enter"
            elif position and comp[i] <= exit_score:
                pending = "exit"

    # Force-close any open position at the final close for clean accounting.
    if position:
        _close_position(close[-1], otime[-1], "end_of_data")
        equity[-1] = cash
        in_market[-1] = False

    result = pd.DataFrame({
        "open_time": otime,
        "close": close,
        "composite": comp,
        "equity": equity,
        "in_market": in_market,
    })
    bh = close / close[0]          # buy-&-hold equity, same starting capital
    result["buy_hold"] = bh

    return {
        "result": result,
        "trades": pd.DataFrame(trades),
        "enriched": e,
        "params": p,
    }
