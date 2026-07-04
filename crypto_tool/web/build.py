"""Compute the data payload for the Signal Engine page from the real engine.

Everything the front-end shows — the scanner ranking, each coin's indicator
series, the 7-rule rationale, the chart arrays — comes from
``signals.enrich`` / ``signals.latest_signal`` over real candles. The front-end
is a pure view layer; this module is the source of truth.
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, List

import pandas as pd

from ..analysis import discovery, forecast, signals
from ..data import binance_client, coingecko, database

# Display metadata mirroring the design.
FULL_NAMES = {
    "BTCUSDT": "Bitcoin", "ETHUSDT": "Ethereum", "BNBUSDT": "BNB", "SOLUSDT": "Solana",
    "XRPUSDT": "XRP", "ADAUSDT": "Cardano", "DOGEUSDT": "Dogecoin", "AVAXUSDT": "Avalanche",
    "LINKUSDT": "Chainlink", "LTCUSDT": "Litecoin", "MATICUSDT": "Polygon", "POLUSDT": "Polygon",
    "DOTUSDT": "Polkadot", "TRXUSDT": "TRON", "BCHUSDT": "Bitcoin Cash", "NEARUSDT": "NEAR Protocol",
    "UNIUSDT": "Uniswap", "ATOMUSDT": "Cosmos", "ETCUSDT": "Ethereum Classic", "XLMUSDT": "Stellar",
    "FILUSDT": "Filecoin", "APTUSDT": "Aptos", "ARBUSDT": "Arbitrum", "OPUSDT": "Optimism",
    "INJUSDT": "Injective", "SUIUSDT": "Sui", "TONUSDT": "Toncoin", "SHIBUSDT": "Shiba Inu",
}
LABELS = {"trend": "TREND", "macd": "MACD", "rsi": "RSI", "bollinger": "BOLLINGER",
          "curvature": "CURVATURE", "momentum": "MOMENTUM", "volume": "VOLUME"}
KIND = {"trend": "TREND-FOLLOW", "macd": "TREND-FOLLOW", "rsi": "MEAN-REVERT",
        "bollinger": "MEAN-REVERT", "curvature": "HEADLINE", "momentum": "TREND-FOLLOW",
        "volume": "CONFIRMATION"}
_COMPS = ["trend", "macd", "rsi", "bollinger", "curvature", "momentum", "volume"]


def _num(v) -> float | None:
    """JSON-safe number: NaN/inf -> None (the front-end treats null as 'no value')."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _arr(series: pd.Series) -> List[float | None]:
    return [_num(v) for v in series.tolist()]


def _latest(e: pd.DataFrame) -> Dict[str, Any]:
    row = e.iloc[-1]

    def g(key):
        return _num(row.get(key))

    sma_s, sma_l = row.get("sma_short"), row.get("sma_long")
    golden = bool(_num(sma_s) is not None and _num(sma_l) is not None and sma_s > sma_l)
    return {
        "price": float(row["close"]),
        "composite": float(row["composite"]),
        "confidence": float(row["confidence"]),
        "flag": str(row["flag"]),
        "rsi": g("rsi"), "pctb": g("bb_pctb"), "roc": g("roc"),
        "velZ": g("vel_z"), "accZ": g("acc_z"),
        "macd": g("macd"), "macdSig": g("macd_signal"), "hist": g("macd_hist"),
        "regime": int(row.get("regime", 0) or 0),
        "golden": golden,
        "cvol": g("c_volume"),
        "comps": {k: g(f"c_{k}") for k in _COMPS},
    }


def _rationale(sig: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for r in sig["rationale"]:               # already sorted by |contribution|
        k = r["component"]
        out.append({
            "key": k, "label": LABELS.get(k, k.upper()), "kind": KIND.get(k, ""),
            "vote": _num(r["vote"]), "weight": _num(r["weight"]),
            "contrib": _num(r["contribution"]), "detail": r["detail"],
        })
    return out


def signal_eval(tail: pd.DataFrame, horizon: int, band: float,
                conf_gate: float = 0.0, persist: int = 1,
                extra_horizons: List[int] | None = None) -> Dict[str, Any]:
    """Hindsight scorecard: at each bar where a *new* directional call fires
    (the flag flips into BUY/SELL territory), check whether price actually moved
    that way ``horizon`` bars later. A move must clear +/-``band`` % to count.

    Stability rules:
    * a call only counts once the flag has HELD for ``persist`` bars — a
      single-bar blip is noise, not a call. It is marked and graded from the
      bar that confirmed it (still strictly causal).
    * calls where confidence >= ``conf_gate`` AND the trend regime agreed are
      *additionally* graded as "confident" — the engine's own filter for which
      of its calls deserve weight.
    * ``extra_horizons`` adds summary-only gradings at longer horizons (the
      per-bar arrays stay on the primary horizon).

    Returns per-bar arrays (aligned to ``tail``) plus a summary. This is
    evaluation of PAST calls against what happened next — not a forecast.
    """
    close = [float(x) for x in tail["close"].tolist()]
    flag = [str(x) for x in tail["flag"].tolist()]
    n = len(close)
    has_conf = "confidence" in tail.columns
    confs = tail["confidence"].tolist() if has_conf else [None] * n
    regimes = tail["regime"].tolist() if "regime" in tail.columns else [0] * n
    persist = max(1, int(persist))
    extras = list(dict.fromkeys(int(h) for h in (extra_horizons or []) if int(h) != horizon))

    call: List[str | None] = [None] * n
    conf_call: List[bool] = [False] * n
    outcome: List[int | None] = [None] * n   # 1 right, -1 wrong, 0 unresolved
    fwd: List[float | None] = [None] * n
    BULL, BEAR = {"BUY", "STRONG BUY"}, {"SELL", "STRONG SELL"}

    def dirn(f):
        return 1 if f in BULL else (-1 if f in BEAR else 0)

    def grade(c: int, bull: bool, h: int):
        """(outcome, fwd%) for the call confirmed at bar c, at horizon h."""
        if c + h >= n:
            return 0, None
        r = (close[c + h] / close[c] - 1.0) * 100.0
        if bull:
            oc = 1 if r >= band else (-1 if r <= -band else 0)
        else:
            oc = 1 if r <= -band else (-1 if r >= band else 0)
        return oc, round(r, 2)

    def bucket():
        return {"calls": 0, "hits": 0, "misses": 0, "unresolved": 0}

    tally = bucket()
    conf_tally = bucket()
    alt_tallies = {h: (bucket(), bucket()) for h in extras}   # (all, confident)

    def count(b, oc):
        b["calls"] += 1
        b["hits"] += oc == 1
        b["misses"] += oc == -1
        b["unresolved"] += oc == 0

    for i in range(n):
        d = dirn(flag[i])
        if d == 0:
            continue
        if i > 0 and dirn(flag[i - 1]) == d:
            continue                          # not the start of this run
        c = i + persist - 1                   # confirmation bar
        if c >= n or any(dirn(flag[j]) != d for j in range(i, c + 1)):
            continue                          # blip died before confirming
        bull = d == 1
        call[c] = "buy" if bull else "sell"
        cv = confs[c]
        rg = regimes[c] if regimes[c] == regimes[c] else 0
        confident = (cv is not None and cv == cv and float(cv) >= conf_gate > 0
                     and (int(rg) >= 0 if bull else int(rg) <= 0))
        conf_call[c] = bool(confident)

        oc, r = grade(c, bull, horizon)
        outcome[c], fwd[c] = oc, r
        count(tally, oc)
        if confident:
            count(conf_tally, oc)
        for h in extras:
            aoc, _ = grade(c, bull, h)
            count(alt_tallies[h][0], aoc)
            if confident:
                count(alt_tallies[h][1], aoc)

    def acc(b):
        res = b["hits"] + b["misses"]
        return round(b["hits"] / res * 100, 1) if res else None

    return {
        "call": call, "outcome": outcome, "fwd": fwd, "confCall": conf_call,
        "summary": {
            **tally, "horizon": horizon, "band": band, "persist": persist,
            "accuracy": acc(tally),
            "conf": {**conf_tally, "gate": conf_gate, "accuracy": acc(conf_tally)},
            "alts": [{"horizon": h, "accuracy": acc(a), "resolved": a["hits"] + a["misses"],
                      "confAccuracy": acc(cb), "confResolved": cb["hits"] + cb["misses"]}
                     for h, (a, cb) in alt_tallies.items()],
        },
    }


_discover_cache: Dict[str, Any] = {}


def build_discover(cfg: Dict[str, Any], tracked: set, ttl: int = 60) -> Dict[str, Any]:
    """Risk-flagged movers block for the page (cached; degrades gracefully)."""
    now = time.time()
    hit = _discover_cache.get("d")
    if hit and now - hit[0] < ttl:
        return hit[1]
    d, disc = cfg["data"], cfg["discovery"]
    try:
        markets = coingecko.fetch_markets(pages=disc["pages"], timeout=d["request_timeout"])
        listed = binance_client.fetch_usdt_symbols(d["base_urls"], d["request_timeout"])
        rows = discovery.screen(markets, listed, tracked, top=50, sort=disc["sort"],
                                min_volume=disc["min_volume_usd"])
        block = {"rows": rows, "sort": disc["sort"], "count": len(markets),
                 "asOf": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "error": None}
    except Exception as exc:  # noqa: BLE001 — the page must still build without discovery
        block = {"rows": [], "sort": disc.get("sort", "momentum"), "count": 0,
                 "asOf": None, "error": str(exc)}
    _discover_cache["d"] = (now, block)
    return block


def _coin_dict(df: pd.DataFrame, cfg: Dict[str, Any], sym: str,
               history: int, horizon: int, band: float) -> Dict[str, Any]:
    """Assemble one coin's full chart+signal payload (shared by the page build
    and the on-demand deep-dive endpoint)."""
    sig = signals.latest_signal(df, cfg)
    e = sig["enriched"]
    tail = e.tail(history).reset_index(drop=True)
    web = cfg.get("web", {})
    ev = signal_eval(tail, horizon, band,
                     conf_gate=float(web.get("eval_conf_gate", 0.6)),
                     persist=int(web.get("eval_persist", 2)),
                     extra_horizons=[int(web.get("eval_horizon_short", 6)),
                                     int(web.get("eval_horizon_long", 72))])
    try:
        fc = forecast.project(e, cfg)   # full history -> best analogue pool
    except Exception:  # noqa: BLE001 — the page must never die on the outlook
        fc = None
    return {
        "forecast": fc,
        "name": FULL_NAMES.get(sym.upper(), sym.replace("USDT", "")),
        "t": [int(x) for x in tail["open_time"].tolist()],
        "o": _arr(tail["open"]), "h": _arr(tail["high"]), "l": _arr(tail["low"]),
        "c": _arr(tail["close"]), "v": _arr(tail["volume"]),
        "emaF": _arr(tail["ema_fast"]), "emaS": _arr(tail["ema_slow"]),
        "bbUp": _arr(tail["bb_upper"]), "bbLo": _arr(tail["bb_lower"]),
        "rsi": _arr(tail["rsi"]), "comp": _arr(tail["composite"]),
        "conf": _arr(tail["confidence"].round(3)),
        "regime": [int(x) for x in tail["regime"].fillna(0).tolist()],
        "velZ": _arr(tail["vel_z"]), "accZ": _arr(tail["acc_z"]),
        "flag": [str(x) for x in tail["flag"].tolist()],
        "call": ev["call"], "outcome": ev["outcome"], "fwd": ev["fwd"],
        "confCall": ev["confCall"],
        "eval": ev["summary"], "latest": _latest(e), "rationale": _rationale(sig),
    }


def build_coin_payload(conn, cfg: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    """Build one coin's full payload on demand — from the DB if we have it, else
    by fetching its candles live from Binance. Used by the Discover deep-dive."""
    symbol = symbol.upper()
    d = cfg["data"]
    interval = d["interval"]
    web = cfg.get("web", {})
    history = web.get("history", 1000)
    horizon, band = int(web.get("eval_horizon", 24)), float(web.get("eval_band", 1.0))
    min_bars = max(cfg["indicators"]["ema_slow"], cfg["indicators"]["bb_period"]) + 5

    df = database.load_ohlcv(conn, symbol, interval)
    if len(df) < min_bars:
        df = binance_client.fetch_klines_history(
            symbol, interval=interval, total=max(history + 250, 1250),
            base_urls=d["base_urls"], timeout=d["request_timeout"])
    if len(df) < min_bars:
        raise ValueError(f"not enough candle history for {symbol}")
    return _coin_dict(df, cfg, symbol, history, horizon, band)


def build_payload(conn, cfg: Dict[str, Any], history: int | None = None) -> Dict[str, Any]:
    """Assemble the full data payload for the page."""
    interval = cfg["data"]["interval"]
    web = cfg.get("web", {})
    history = history or web.get("history", 1000)
    horizon = int(web.get("eval_horizon", 24))
    band = float(web.get("eval_band", 1.0))
    min_bars = max(cfg["indicators"]["ema_slow"], cfg["indicators"]["bb_period"]) + 5

    # Prefer the tracked list recorded at ingest time (the synchronized
    # universe); fall back to whatever the DB holds. This keeps the page from
    # growing without bound as the auto-picked list rotates over the weeks.
    tracked = database.load_kv(conn, "tracked_symbols")
    available = database.list_symbols(conn, interval)
    symbols = [s for s in tracked if s in set(available)] if tracked else available

    coins: Dict[str, Any] = {}
    names: List[str] = []
    for sym in symbols:
        df = database.load_ohlcv(conn, sym, interval)
        if len(df) < min_bars:
            continue
        coins[sym] = _coin_dict(df, cfg, sym, history, horizon, band)
        names.append(sym)

    s = cfg["signals"]
    b = cfg["backtest"]
    return {
        "interval": interval,
        "markets": len(coins),
        "lastScan": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "evalHorizon": horizon,
        "evalBand": band,
        "evalConfGate": float(web.get("eval_conf_gate", 0.6)),
        "evalPersist": int(web.get("eval_persist", 2)),
        "evalHorizonLong": int(web.get("eval_horizon_long", 72)),
        "evalHorizonShort": int(web.get("eval_horizon_short", 6)),
        "names": names,
        "discover": build_discover(cfg, set(names)),
        "scout": database.load_scout(conn),
        "scoutMin": int(cfg.get("scout", {}).get("active_min") or 0),
        "thresholds": {
            "buy": s["buy_threshold"], "strongBuy": s["strong_buy_threshold"],
            "sell": s["sell_threshold"], "strongSell": s["strong_sell_threshold"],
        },
        "backtestDefaults": {
            "entry": b["entry_score"], "exit": b["exit_score"],
            "stop": b["stop_loss_pct"], "fee": b["fee_pct"],
            "gate": float(b.get("conf_gate", 0.6)),
            "confirm": int(b.get("confirm_bars", 2)),
        },
        "coins": coins,
    }


def _template_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "template.html")


def render_page(payload: Dict[str, Any]) -> str:
    """Inject the payload into the HTML template, returning a self-contained page."""
    with open(_template_path(), "r", encoding="utf-8") as fh:
        template = fh.read()
    data_json = json.dumps(payload, allow_nan=False, separators=(",", ":"))
    if "/*__SIGNAL_DATA__*/" not in template:
        raise RuntimeError("template.html is missing the /*__SIGNAL_DATA__*/ marker")
    return template.replace("/*__SIGNAL_DATA__*/", data_json)


def build_page(conn, cfg: Dict[str, Any], history: int | None = None) -> str:
    return render_page(build_payload(conn, cfg, history=history))
