"""Performance metrics for a backtest result.

Annualisation is derived from the actual median bar spacing in the data, so the
same code gives sensible Sharpe/CAGR for 1m, 1h or 1d candles. We report the
strategy *and* buy-&-hold side by side, plus the metrics that reveal whether an
edge is real or just a few lucky trades (profit factor, trade count, exposure).
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

_MS_PER_YEAR = 365.25 * 24 * 3600 * 1000


def _periods_per_year(open_time: np.ndarray) -> float:
    if len(open_time) < 3:
        return 365.0
    deltas = np.diff(open_time.astype("float64"))
    median_ms = float(np.median(deltas))
    if median_ms <= 0:
        return 365.0
    return _MS_PER_YEAR / median_ms


def _sharpe(returns: pd.Series, ppy: float) -> float:
    std = returns.std(ddof=0)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(returns.mean() / std * np.sqrt(ppy))


def _sortino(returns: pd.Series, ppy: float) -> float:
    downside = returns[returns < 0].std(ddof=0)
    if downside == 0 or np.isnan(downside):
        return float("nan")
    return float(returns.mean() / downside * np.sqrt(ppy))


def _max_drawdown(equity: pd.Series) -> float:
    cummax = equity.cummax()
    drawdown = equity / cummax - 1.0
    return float(drawdown.min())


def compute_metrics(bt: Dict[str, Any]) -> Dict[str, Any]:
    """Turn a :func:`run_backtest` result into a metrics dict."""
    result: pd.DataFrame = bt["result"]
    trades: pd.DataFrame = bt["trades"]
    equity = result["equity"]
    close = result["close"]
    open_time = result["open_time"].to_numpy()

    ppy = _periods_per_year(open_time)
    strat_ret = equity.pct_change().dropna()
    bh_ret = close.pct_change().dropna()

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    bh_return = float(close.iloc[-1] / close.iloc[0] - 1.0)

    span_years = (open_time[-1] - open_time[0]) / _MS_PER_YEAR if len(open_time) > 1 else 0.0
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / span_years) - 1.0) if span_years > 0 else 0.0
    bh_cagr = float((close.iloc[-1] / close.iloc[0]) ** (1.0 / span_years) - 1.0) if span_years > 0 else 0.0

    n_trades = int(len(trades))
    if n_trades:
        rets = trades["return_pct"]
        wins = rets[rets > 0]
        losses = rets[rets < 0]  # exactly-zero (break-even) trades are neither
        win_rate = float(len(wins) / n_trades)
        gross_win = float(wins.sum())
        gross_loss = float(losses.sum())
        profit_factor = float(gross_win / abs(gross_loss)) if gross_loss < 0 else float("inf")
        avg_trade = float(rets.mean())
        best_trade = float(rets.max())
        worst_trade = float(rets.min())
    else:
        win_rate = profit_factor = avg_trade = best_trade = worst_trade = 0.0

    return {
        "periods_per_year": round(ppy, 1),
        "span_years": round(span_years, 3),
        "bars": int(len(result)),
        "total_return_pct": round(total_return * 100, 2),
        "buy_hold_return_pct": round(bh_return * 100, 2),
        "excess_vs_hold_pct": round((total_return - bh_return) * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "buy_hold_cagr_pct": round(bh_cagr * 100, 2),
        "sharpe": round(_sharpe(strat_ret, ppy), 2),
        "buy_hold_sharpe": round(_sharpe(bh_ret, ppy), 2),
        "sortino": round(_sortino(strat_ret, ppy), 2),
        "max_drawdown_pct": round(_max_drawdown(equity) * 100, 2),
        "buy_hold_max_drawdown_pct": round(_max_drawdown(close / close.iloc[0]) * 100, 2),
        "exposure_pct": round(float(result["in_market"].mean()) * 100, 1),
        "num_trades": n_trades,
        "win_rate_pct": round(win_rate * 100, 1),
        "profit_factor": round(profit_factor, 2) if np.isfinite(profit_factor) else None,
        "avg_trade_pct": round(avg_trade, 2),
        "best_trade_pct": round(best_trade, 2),
        "worst_trade_pct": round(worst_trade, 2),
    }
