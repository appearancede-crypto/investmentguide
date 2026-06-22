"""Fictional-money portfolio simulation + the 'learn from money made/lost' loop.

WHAT THIS IS
------------
You put in virtual cash. Each bar, the engine scores every coin; the portfolio
allocates capital across the coins in proportion to how bullish (and confident)
the signals are, long-only, capped per coin, with the rest in cash. Trades pay a
fee and fill at the **next bar's open** (no look-ahead). The result is an equity
curve you can compare to simply holding an equal-weight basket of the coins.

THE LEARNING LOOP (the honest version of "learn from the money made or lost")
-----------------------------------------------------------------------------
Every ``retrain_every`` bars, for each signal component we measure its realised
profitability over a trailing window: the average of ``vote * next_bar_return``
(its information coefficient, in return units). Components whose votes have been
*paying off* score positive; ones that have been *losing money* score negative.
We tilt the weights toward the positive scorers (negatives are floored to zero),
renormalise to the original total weight, and EMA toward that target so it adapts
gradually. This is strictly walk-forward — the score at bar *t* uses only returns
realised up to *t*, and the updated weights drive trades from *t* onward.

This is a feedback loop, NOT foresight. Tilting toward what worked recently can
just as easily chase noise and underperform. Always run it against the static
book (``learn=False``) and judge by the realised dollars, not the premise.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..analysis import signals
from ..data import database

_MS_PER_YEAR = 365.25 * 24 * 3600 * 1000


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def _prepare(conn, cfg, min_history: int):
    """Enrich every symbol once and cache the arrays the sim needs."""
    interval = cfg["data"]["interval"]
    data: Dict[str, dict] = {}
    for sym in database.list_symbols(conn, interval):
        df = database.load_ohlcv(conn, sym, interval)
        if len(df) < min_history:
            continue
        e = signals.enrich(df, cfg).reset_index(drop=True)
        close = e["close"].to_numpy(dtype=float)
        fwd = np.empty(len(close))
        fwd[:-1] = close[1:] / close[:-1] - 1.0     # next-bar return (NaN on last)
        fwd[-1] = np.nan
        ts = e["open_time"].to_numpy(dtype="int64")
        data[sym] = {
            "ts": ts,
            "open": e["open"].to_numpy(dtype=float),
            "close": close,
            "conf": e["confidence"].to_numpy(dtype=float),
            "comps": signals.component_matrix(e),   # (bars, 7) votes
            "fwd": fwd,
            "pos": {int(t): i for i, t in enumerate(ts)},
        }
    return data, interval


# --------------------------------------------------------------------------- #
# Learner
# --------------------------------------------------------------------------- #
def _learn_weights(base_w, prev_w, comp_slices, fwd_slices, lr, min_obs):
    """EMA the weights toward each component's recent realised profitability.

    The score is a **volatility-normalised information coefficient**: per symbol
    the forward returns are divided by their own std before pooling, so a few
    high-volatility coins can't dominate the update. Each component is divided by
    the number of bars on which it was actually *live* (its warm-up NaNs don't
    count against it), so late-warming rules — including curvature — aren't
    structurally penalised. When the pooled sample is too thin, weights are left
    unchanged. (The reward is a close-to-close IC: a slightly idealised but highly
    correlated proxy for the open-to-open returns the book actually trades.)
    """
    n = len(base_w)
    acc = np.zeros(n)
    counts = np.zeros(n)
    for C, f in zip(comp_slices, fwd_slices):
        if len(f) == 0:
            continue
        fmask = ~np.isnan(f)
        if int(fmask.sum()) < 2:
            continue
        fv = f[fmask]
        sd = fv.std()
        if sd <= 1e-12:
            continue
        fv = fv / sd                                  # vol-normalised, per symbol
        Cm = C[fmask]
        live = ~np.isnan(Cm)
        C0 = np.nan_to_num(Cm, nan=0.0)
        acc += (C0 * fv[:, None]).sum(axis=0)         # sum of vote * normalised return
        counts += live.sum(axis=0)                    # per-component live-bar count
    if counts.max() < min_obs:                        # too little live data -> hold steady
        return prev_w
    skill = acc / np.where(counts > 0, counts, 1.0)   # per-component live-bar IC
    raw = np.clip(skill, 0.0, None)                   # ignore recently-unprofitable rules
    if raw.sum() <= 1e-12:
        target = base_w.copy()                        # nothing worked -> revert to baseline
    else:
        target = raw / raw.sum() * base_w.sum()       # keep total weight (composite scale)
    return (1.0 - lr) * prev_w + lr * target


def _waterfill(raw: Dict[str, float], cap: float) -> Dict[str, float]:
    """Cap each weight at ``cap`` and redistribute the clipped excess to the
    uncapped names, so capital isn't needlessly stranded in cash when a few
    coins would otherwise breach the per-coin cap."""
    w = dict(raw)
    for _ in range(len(w) + 1):
        over = [s for s, v in w.items() if v > cap + 1e-12]
        if not over:
            break
        excess = sum(w[s] - cap for s in over)
        for s in over:
            w[s] = cap
        uncapped = {s: v for s, v in w.items() if s not in over}
        usum = sum(uncapped.values())
        if usum <= 1e-12:
            break                                     # all at cap -> remainder stays cash
        for s in uncapped:
            w[s] += excess * (uncapped[s] / usum)
    return {s: min(v, cap) for s, v in w.items()}


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
def run_paper(
    conn,
    cfg: Dict[str, Any],
    *,
    starting_cash: Optional[float] = None,
    learn: bool = False,
    params: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Simulate the portfolio over all stored history. Returns a results dict
    (equity rows, trades, weight history, metrics, final positions/weights) or
    ``None`` if there is not enough data."""
    p = {**cfg["paper"], **(params or {})}
    lcfg = cfg["paper"]["learn"]
    starting_cash = float(starting_cash if starting_cash is not None else p["starting_cash"])
    min_history = max(cfg["indicators"]["ema_slow"], cfg["indicators"]["bb_period"]) + 5

    data, _ = _prepare(conn, cfg, min_history)
    if not data:
        return None
    symbols = sorted(data)
    base_w = signals.weights_vector(cfg["signals"]["weights"])
    w = base_w.copy()

    all_ts = sorted({int(t) for d in data.values() for t in d["ts"]})
    fee = p["fee_pct"] / 100.0
    band, min_score, cap = p["rebalance_band"], p["min_score"], p["max_weight_per_coin"]
    window, retrain_every = lcfg["window"], lcfg["retrain_every"]
    lr, min_obs = lcfg["lr"], lcfg["min_obs"]

    cash = starting_cash
    units = {s: 0.0 for s in symbols}
    pending: Optional[Dict[str, float]] = None
    equity_rows: List[tuple] = []
    trades: List[dict] = []
    weight_rows: List[tuple] = [(all_ts[0], _w_dict(base_w))]
    since_retrain = 0

    # Last-observation-carried-forward price: a held position with no bar at this
    # timestamp (data gap, or it lists/delists out of step with its peers) is
    # valued at its last known price — never silently marked to $0.
    def asof(sym, ts, field):
        d = data[sym]
        i = int(np.searchsorted(d["ts"], ts, side="right")) - 1
        return None if i < 0 else float(d[field][i])

    def mark_value(ts):
        return sum(units[s] * (asof(s, ts, "close") or 0.0) for s in symbols if units[s])

    # Equal-weight benchmark over the SAME evolving universe as the strategy:
    # each coin is staged in (cash-funded) at its own first appearance, so the
    # excess-return comparison is apples-to-apples.
    bench_cash = starting_cash
    bench_units: Dict[str, float] = {}
    bench_per = starting_cash / len(symbols)

    for ts in all_ts:
        present = [s for s in symbols
                   if (j := data[s]["pos"].get(ts)) is not None and j >= min_history - 1]

        # 1) Execute the rebalance decided on the previous bar, at this bar's open.
        if pending is not None:
            eq_open = cash + mark_value(ts)
            for s in symbols:
                jj = data[s]["pos"].get(ts)
                if jj is None:                        # only trade coins with a real bar
                    continue
                pr = float(data[s]["open"][jj])
                if pr <= 0:
                    continue
                target_val = pending.get(s, 0.0) * eq_open
                delta_val = target_val - units[s] * pr
                if abs(delta_val) < band * eq_open:   # no-trade band: don't churn fees
                    continue
                du = delta_val / pr
                cash -= du * pr                       # buy (du>0) spends cash
                cash -= abs(du * pr) * fee
                units[s] += du
                trades.append({"ts": ts, "symbol": s, "side": "BUY" if du > 0 else "SELL",
                               "price": pr, "units": du, "notional": du * pr,
                               "fee": abs(du * pr) * fee})
            pending = None

        # 2) Walk-forward learning: rescore weights from realised profitability.
        if learn and since_retrain >= retrain_every and present:
            comp_slices, fwd_slices = [], []
            for s in present:
                j = data[s]["pos"][ts]
                lo = max(0, j - window)
                comp_slices.append(data[s]["comps"][lo:j])   # votes up to bar j-1
                fwd_slices.append(data[s]["fwd"][lo:j])       # their realised next returns
            w = _learn_weights(base_w, w, comp_slices, fwd_slices, lr, min_obs)
            weight_rows.append((ts, _w_dict(w)))
            since_retrain = 0

        # 3) Mark equity at this bar's close (carry-forward prices across gaps).
        invested = mark_value(ts)
        equity = cash + invested
        for s in present:                             # stage each coin into the basket
            if s not in bench_units:
                op = float(data[s]["open"][data[s]["pos"][ts]])
                if op > 0:
                    bench_units[s] = bench_per / op
                    bench_cash -= bench_per
        bench = bench_cash + sum(u * (asof(s, ts, "close") or 0.0) for s, u in bench_units.items())
        equity_rows.append((ts, equity, cash, invested, bench))

        # 4) Decide target weights from this bar's (closed) signals -> next-bar order.
        scores = {}
        for s in present:
            j = data[s]["pos"][ts]
            comp = float(signals.composite_from_components(data[s]["comps"][j:j + 1], w)[0])
            conf = data[s]["conf"][j]
            conf = conf if conf == conf else 0.1      # NaN-safe
            sc = max(comp - min_score, 0.0) * max(conf, 0.1)
            if sc > 0:
                scores[s] = sc
        total = sum(scores.values())
        pending = _waterfill({s: sc / total for s, sc in scores.items()}, cap) if total > 0 else {}
        since_retrain += 1

    last_ts = all_ts[-1]
    positions = {s: units[s] for s in symbols if abs(units[s]) > 1e-12}
    final_alloc = {s: units[s] * (asof(s, last_ts, "close") or 0.0) for s in positions}
    if len(equity_rows) < 2:
        return None
    metrics = _metrics(equity_rows, trades, starting_cash)
    return {
        "equity_rows": equity_rows,
        "trades": trades,
        "weight_rows": weight_rows,
        "metrics": metrics,
        "weights": _w_dict(w),
        "positions": positions,
        "final_alloc": final_alloc,
        "final_cash": round(cash, 2),
        "last_ts": last_ts,
        "symbols": symbols,
        "learn": learn,
    }


def _w_dict(w: np.ndarray) -> Dict[str, float]:
    return {name: round(float(v), 4) for name, v in zip(signals.COMPONENT_NAMES, w)}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _metrics(equity_rows, trades, starting_cash) -> Dict[str, Any]:
    ts = np.array([r[0] for r in equity_rows], dtype="float64")
    eq = np.array([r[1] for r in equity_rows], dtype="float64")
    invested = np.array([r[3] for r in equity_rows], dtype="float64")
    bench = np.array([r[4] for r in equity_rows], dtype="float64")
    if len(eq) < 2:
        return {"bars": int(len(eq)), "final_equity": float(eq[-1]) if len(eq) else starting_cash}

    rets = eq[1:] / eq[:-1] - 1.0
    bench_rets = bench[1:] / np.where(bench[:-1] > 0, bench[:-1], np.nan) - 1.0
    median_ms = float(np.median(np.diff(ts))) or 1.0
    ppy = _MS_PER_YEAR / median_ms

    def sharpe(r):
        r = r[~np.isnan(r)]
        sd = r.std(ddof=0)
        return float(r.mean() / sd * np.sqrt(ppy)) if sd > 0 else 0.0

    def max_dd(series):
        peak = np.maximum.accumulate(series)
        return float((series / peak - 1.0).min())

    turnover = sum(abs(t["notional"]) for t in trades) / starting_cash
    return {
        "bars": int(len(eq)),
        "final_equity": round(float(eq[-1]), 2),
        "total_return_pct": round(float(eq[-1] / eq[0] - 1.0) * 100, 2),
        "benchmark_return_pct": round(float(bench[-1] / bench[0] - 1.0) * 100, 2),
        "excess_vs_benchmark_pct": round(float((eq[-1] / eq[0] - bench[-1] / bench[0])) * 100, 2),
        "sharpe": round(sharpe(rets), 2),
        "benchmark_sharpe": round(sharpe(bench_rets), 2),
        "max_drawdown_pct": round(max_dd(eq) * 100, 2),
        "benchmark_max_drawdown_pct": round(max_dd(bench) * 100, 2),
        "avg_exposure_pct": round(float(np.mean(invested / np.where(eq > 0, eq, np.nan))) * 100, 1),
        "num_trades": len(trades),
        "turnover_x": round(turnover, 2),
    }


# --------------------------------------------------------------------------- #
# Persisted-account convenience wrapper
# --------------------------------------------------------------------------- #
def run_and_save(conn, cfg, name: str) -> Optional[Dict[str, Any]]:
    """Run the simulation for a stored account and persist its results."""
    acc = database.get_paper_account(conn, name)
    if acc is None:
        raise ValueError(f"No paper account named {name!r}. Create it first.")
    res = run_paper(conn, cfg, starting_cash=acc["starting_cash"],
                    learn=acc["learn"], params=acc["params"])
    if res is None:
        return None
    database.save_paper_results(
        conn, name, res["equity_rows"], res["trades"], res["weight_rows"],
        res["metrics"], res["weights"], res["positions"], res["last_ts"],
    )
    return res
