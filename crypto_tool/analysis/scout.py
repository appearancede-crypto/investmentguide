"""Coin scout — run the full signal engine across the whole exchange.

WHAT IT IS
----------
The scanner watches your hand-picked watchlist; the scout asks a bigger
question: **of every liquid USDT pair on Binance, where do the engine's own
rules see the strongest setup right now?** One sweep:

  1. pulls the 24h ticker for every pair in a single request,
  2. keeps real spot markets (drops leveraged tokens and stablecoin pairs)
     with enough 24h turnover to plausibly get in AND out,
  3. fetches a few hundred candles per surviving pair and runs the exact same
     enrich -> composite/confidence pipeline the scanner uses,
  4. grades each coin's own hindsight track record and (where history allows)
     a compact outlook, and tags risk from CoinGecko market data,
  5. ranks everything and keeps the strongest rows.

WHAT IT IS NOT
--------------
Not a buy list. "Potentially profitable" here means *the rules that are shown,
applied evenly, score this setup highly* — the same rules that are sometimes
wrong, on coins where risk tags matter more than rank. Strong setups fail all
the time; small caps fail harder. Not financial advice.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Optional

from ..data import binance_client, coingecko, database
from . import forecast, signals

# Leveraged-token suffixes and stable/fiat bases that aren't real "coins to
# scout" — they'd only clutter the ranking.
_LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")


def _is_leveraged(base: str, pairs: set) -> bool:
    """BTCUP is a leveraged token only because BTC itself trades — a plain
    suffix match would wrongly drop real coins like JUP (Jupiter) or SYRUP."""
    for sfx in _LEVERAGED_SUFFIXES:
        if base.endswith(sfx) and len(base) > len(sfx):
            root = base[: -len(sfx)]
            if root + "USDT" in pairs:
                return True
    return False
_STABLE_BASES = {
    "USDC", "TUSD", "FDUSD", "USDP", "BUSD", "DAI", "EUR", "EURI", "AEUR",
    "GBP", "TRY", "BRL", "ARS", "COP", "JPY", "MXN", "PLN", "RON", "UAH",
    "ZAR", "CZK", "USTC", "FRAX", "GUSD", "LUSD", "SUSD", "EURS", "USDS",
    "USDD",
}


def _looks_stable(base: str, price: Optional[float], p24h: Optional[float]) -> bool:
    """Stablecoins aren't scoutable 'coins' — a signal on a peg is noise.
    Catch them by name (…USD/USD… bases are pegs by design) and by behaviour
    (glued to $1 with a dead-flat day)."""
    if base in _STABLE_BASES or base.startswith("USD") or base.endswith("USD"):
        return True
    if (price is not None and 0.95 <= price <= 1.05
            and p24h is not None and abs(p24h) < 0.25):
        return True
    return False


def _num(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def eligible_pairs(pairs: set, tickers: List[dict], *,
                   min_quote_volume: float, max_symbols: int) -> List[dict]:
    """Filter the exchange down to real, liquid spot pairs, most-traded first."""
    rows = []
    for t in tickers:
        sym = str(t.get("symbol", "")).upper()
        if not sym.endswith("USDT") or sym not in pairs:
            continue
        base = sym[:-4]
        if _is_leveraged(base, pairs):
            continue
        price = _num(t.get("lastPrice"))
        p24h = _num(t.get("priceChangePercent"))
        if _looks_stable(base, price, p24h):
            continue
        qv = _num(t.get("quoteVolume")) or 0.0
        if qv < min_quote_volume:
            continue
        rows.append({"symbol": sym, "base": base, "quoteVolume": qv, "p24h": p24h})
    rows.sort(key=lambda r: r["quoteVolume"], reverse=True)
    return rows[:max_symbols] if max_symbols else rows


def _cg_risk_map(cfg: Dict[str, Any], markets: Optional[List[dict]]) -> Dict[str, dict]:
    """base asset -> {risk tier, flags, mcap, name} from CoinGecko (best-effort)."""
    from . import discovery  # local import: reuse its risk model
    if markets is None:
        try:
            markets = coingecko.fetch_markets(
                pages=cfg["discovery"]["pages"], timeout=cfg["data"]["request_timeout"])
        except Exception:  # noqa: BLE001 — scout must run without CoinGecko too
            markets = []
    out: Dict[str, dict] = {}
    for c in markets:
        base = (c.get("symbol") or "").upper()
        if not base or base in out:
            continue
        tier, flags, _ = discovery._risk(c, on_binance=True)
        out[base] = {"risk": tier, "flags": flags[:2], "mcap": c.get("market_cap"),
                     "name": c.get("name")}
    return out


def _eval_accuracy(e, cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Hindsight scorecard over the swept window (same rules as the web page)."""
    from ..web.build import signal_eval  # local import: avoid a hard layer dependency
    web = cfg.get("web", {})
    ev = signal_eval(e.reset_index(drop=True),
                     int(web.get("eval_horizon", 24)), float(web.get("eval_band", 1.0)),
                     conf_gate=float(web.get("eval_conf_gate", 0.6)),
                     persist=int(web.get("eval_persist", 2)))
    s = ev["summary"]
    return {"accuracy": s["accuracy"], "hits": s["hits"],
            "misses": s["misses"], "calls": s["calls"],
            "confAccuracy": s["conf"]["accuracy"],
            "confHits": s["conf"]["hits"], "confMisses": s["conf"]["misses"]}


def run_scout(
    cfg: Dict[str, Any],
    *,
    pairs: Optional[set] = None,
    tickers: Optional[List[dict]] = None,
    markets: Optional[List[dict]] = None,
    fetch_klines: Optional[Callable[..., Any]] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """Sweep the exchange and return a compact, JSON-safe snapshot.

    The keyword hooks (``pairs``/``tickers``/``markets``/``fetch_klines``)
    exist so tests can run the whole pipeline offline.
    """
    scfg = cfg["scout"]
    d = cfg["data"]
    interval = d["interval"]
    bars = int(scfg["bars"])
    min_bars = max(cfg["indicators"]["ema_slow"], cfg["indicators"]["bb_period"]) + 5

    if pairs is None:
        pairs = binance_client.fetch_usdt_pairs(d["base_urls"], d["request_timeout"])
    if tickers is None:
        tickers = binance_client.fetch_all_24h_tickers(d["base_urls"], d["request_timeout"])
    # Both endpoints are best-effort and return empty on outage. An empty
    # universe is an ERROR, not a truthful "0 coins" sweep — raising here lets
    # the server loop retry and keeps the last good snapshot on the page.
    if not pairs or not tickers:
        raise RuntimeError("exchange unavailable: could not list pairs/tickers "
                           "(network or rate limit) — keeping the previous sweep")
    fetch = fetch_klines or (lambda sym: binance_client.fetch_klines(
        sym, interval=interval, limit=bars,
        base_urls=d["base_urls"], timeout=d["request_timeout"]))

    selected = eligible_pairs(
        pairs, tickers,
        min_quote_volume=float(scfg["min_quote_volume_usd"]),
        max_symbols=int(scfg["max_symbols"]))
    risk_map = _cg_risk_map(cfg, markets)
    tracked = {s.upper() for s in cfg["data"]["symbols"]}

    rows: List[Dict[str, Any]] = []
    errors = 0
    consecutive = 0
    aborted = False
    pause = float(scfg.get("pause_ms", 60)) / 1000.0
    live = fetch_klines is None            # injected fetchers (tests) never sleep

    def nap(mult: float = 1.0):
        if live and pause:
            time.sleep(min(pause * mult, 5.0))

    for i, cand in enumerate(selected):
        sym = cand["symbol"]
        if progress:
            progress(i + 1, len(selected), sym)
        try:
            df = fetch(sym)
        except Exception:  # noqa: BLE001 — one bad pair must not kill the sweep
            errors += 1
            consecutive += 1
            if consecutive >= 10:          # circuit breaker: the exchange is
                aborted = True             # refusing us — stop hammering it
                break
            nap(2.0 ** min(consecutive, 6))  # back OFF on errors, don't speed up
            continue
        consecutive = 0
        if df is None or len(df) < min_bars:
            errors += 1
            nap()
            continue
        e = signals.enrich(df, cfg)
        last = e.iloc[-1]
        composite = _num(last.get("composite"))
        conf = _num(last.get("confidence"))
        if composite is None:
            errors += 1
            nap()
            continue

        try:
            fc = forecast.project(e, cfg)
        except Exception:  # noqa: BLE001
            fc = None
        outlook = None
        if fc:
            cp = next((c for c in fc["checkpoints"] if c["bars"] == 24),
                      fc["checkpoints"][0] if fc["checkpoints"] else None)
            if cp:
                outlook = {"probUp": cp["prob_up"], "median": cp["median_pct"],
                           "hours": cp.get("hours"), "quality": fc["quality"]}

        meta = risk_map.get(cand["base"], {})
        rows.append({
            "symbol": sym,
            "name": meta.get("name") or cand["base"],
            "price": _num(last.get("close")),
            "p24h": cand["p24h"],
            "quoteVolume": round(cand["quoteVolume"], 0),
            "composite": round(composite, 1),
            "confidence": round((conf or 0.0) * 100, 1),
            "flag": str(last.get("flag", "NEUTRAL")),
            "driver": _top_driver(e, cfg),
            "outlook": outlook,
            "record": _eval_accuracy(e, cfg),
            "risk": meta.get("risk", "HIGH"),
            "riskFlags": meta.get("flags", ["not in CoinGecko top list"]),
            "mcap": _num(meta.get("mcap")),
            "tracked": sym in tracked,
            "bars": int(len(df)),
        })
        nap()                          # politeness toward the public API

    # Strongest setups first: composite is the engine's opinion, confidence
    # breaks ties. (The front-end offers other sort orders client-side.)
    rows.sort(key=lambda r: (r["composite"], r["confidence"]), reverse=True)
    top = int(scfg["top"])
    return {
        "asOf": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "asOfMs": int(time.time() * 1000),
        "interval": interval,
        "bars": bars,
        "exchangePairs": len(pairs),
        "eligible": len(selected),
        "scanned": len(rows) + errors,
        "scored": len(rows),
        "errors": errors,
        "aborted": aborted,
        "minQuoteVolume": float(scfg["min_quote_volume_usd"]),
        "rows": rows[:top],
    }


def _top_driver(e, cfg: Dict[str, Any]) -> str:
    """One-line 'why' for the latest bar (biggest absolute contribution)."""
    row = e.iloc[-1]
    weights = cfg["signals"]["weights"]
    best, best_c = "", 0.0
    for name in signals.COMPONENT_NAMES:
        v = row.get(f"c_{name}")
        v = float(v) if v == v and v is not None else 0.0
        c = abs(v * float(weights[name]))
        if c > best_c:
            best, best_c = name, c
    return best


def run_and_save(conn, cfg: Dict[str, Any],
                 progress: Optional[Callable[[int, int, str], None]] = None,
                 **hooks) -> Dict[str, Any]:
    snap = run_scout(cfg, progress=progress, **hooks)
    # Never clobber a good snapshot with a degenerate one: a sweep that scored
    # nothing is an outage, not fresher information.
    if snap["scored"] == 0 and database.load_scout(conn) is not None:
        raise RuntimeError("sweep scored 0 pairs — keeping the previous snapshot")
    database.save_scout(conn, snap)
    return snap
