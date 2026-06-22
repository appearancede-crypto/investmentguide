"""Market discovery / movers screener — risk-aware by design.

This screens the whole market for momentum and unusual volume so you can see
what's moving. But be clear-eyed about what these rows are: **high-risk
speculation**, not gems. Microcaps are where most pump-and-dumps, rug pulls and
honeypots live, and the large majority of low-cap coins trend to zero. Every row
carries explicit risk flags (market-cap tier, liquidity, volatility, whether it
even trades on a major exchange). Nothing here is a buy signal or a "sure flip",
and none of it is financial advice. Treat a high momentum score as "this is
moving and risky", never as "this is going to pay out".
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

_MICRO = 10_000_000
_SMALL = 50_000_000
_MID = 1_000_000_000


def _f(v) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _risk(coin: Dict[str, Any], on_binance: bool):
    """Return (tier, flags, turnover). Tier is MEDIUM/HIGH/EXTREME — never 'low':
    everything a movers screen surfaces is speculative."""
    flags: List[str] = []
    score = 0
    mcap = coin.get("market_cap") or 0
    vol = coin.get("total_volume") or 0
    rank = coin.get("market_cap_rank") or 99999
    p24 = _f(coin.get("price_change_percentage_24h_in_currency")) or 0.0
    athc = _f(coin.get("ath_change_percentage"))
    turnover = (vol / mcap) if mcap > 0 else 0.0

    if mcap < _MICRO:
        flags.append("microcap <$10M"); score += 3
    elif mcap < _SMALL:
        flags.append("small-cap <$50M"); score += 2
    elif mcap < _MID:
        flags.append("mid-cap"); score += 1

    if vol < 250_000:
        flags.append("very low liquidity"); score += 3
    elif vol < 2_000_000:
        flags.append("thin liquidity"); score += 1
    if turnover > 1.5:
        flags.append("volume > 1.5x mcap (froth/pump risk)"); score += 2

    if abs(p24) > 30:
        flags.append("extreme 24h move"); score += 2
    elif abs(p24) > 15:
        flags.append("very volatile"); score += 1

    if athc is not None and athc < -90:
        flags.append("down >90% from ATH"); score += 1
    if not on_binance:
        flags.append("not on Binance (harder to exit)"); score += 2
    if rank and rank > 300:
        flags.append(f"rank #{rank}"); score += 1

    tier = "EXTREME" if score >= 6 else "HIGH" if score >= 3 else "MEDIUM"
    return tier, flags, round(turnover, 3)


def momentum(coin: Dict[str, Any]) -> float:
    """Blended short-term momentum, weighted toward the 24h window."""
    p1 = _f(coin.get("price_change_percentage_1h_in_currency")) or 0.0
    p24 = _f(coin.get("price_change_percentage_24h_in_currency")) or 0.0
    p7 = _f(coin.get("price_change_percentage_7d_in_currency")) or 0.0
    return 0.5 * p24 + 0.3 * (p1 * 6.0) + 0.2 * (p7 / 3.0)


_SORTERS = {
    "gainers": lambda r: r["p24h"] if r["p24h"] is not None else -1e9,
    "volume": lambda r: r["turnover"] or 0.0,
    "momentum": lambda r: r["momentum"] if r["momentum"] is not None else -1e9,
}


def screen(
    markets: List[Dict[str, Any]],
    binance_listed: Set[str],
    tracked: Set[str],
    top: int = 30,
    sort: str = "momentum",
    min_volume: float = 250_000,
    max_rank: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Rank the market into a risk-flagged shortlist.

    ``binance_listed`` = base assets with a Binance USDT pair (liquidity signal).
    ``tracked`` = symbols already in our DB (deep-divable with the full engine).
    """
    listed = {s.upper() for s in binance_listed}
    tracked_bases = {s.replace("USDT", "").upper() for s in tracked}
    rows: List[Dict[str, Any]] = []
    for c in markets:
        sym = (c.get("symbol") or "").upper()
        if not sym:
            continue
        if min_volume and (c.get("total_volume") or 0) < min_volume:
            continue
        if max_rank and (c.get("market_cap_rank") or 99999) > max_rank:
            continue
        on_b = sym in listed
        # Any Binance-listed coin can be deep-dived on demand (its candles are
        # fetched live); already-tracked ones open instantly.
        deep = on_b
        tier, flags, turnover = _risk(c, on_b)
        rows.append({
            "symbol": sym, "name": c.get("name"),
            "price": _f(c.get("current_price")),
            "mcap": c.get("market_cap"), "rank": c.get("market_cap_rank"),
            "volume": c.get("total_volume"), "turnover": turnover,
            "p1h": _f(c.get("price_change_percentage_1h_in_currency")),
            "p24h": _f(c.get("price_change_percentage_24h_in_currency")),
            "p7d": _f(c.get("price_change_percentage_7d_in_currency")),
            "momentum": round(momentum(c), 2),
            "risk": tier, "flags": flags,
            "onBinance": on_b, "deepDive": deep, "tracked": sym in tracked_bases,
            "binanceSymbol": (sym + "USDT") if on_b else None,
        })
    rows.sort(key=_SORTERS.get(sort, _SORTERS["momentum"]), reverse=True)
    return rows[:top]
