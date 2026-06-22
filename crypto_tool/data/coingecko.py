"""CoinGecko market-data client — the broad, cross-exchange crypto database.

Used for *market discovery*: thousands of coins with market cap, volume and
multi-window price changes in a couple of calls (no API key). A small in-process
TTL cache keeps us well under the free rate limit when the page regenerates on
every load.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

import requests

_BASE = "https://api.coingecko.com/api/v3"
_cache: Dict[str, Any] = {}


class CoinGeckoError(RuntimeError):
    """Raised when market data could not be retrieved from CoinGecko."""


def fetch_markets(pages: int = 2, per_page: int = 250, vs: str = "usd",
                  timeout: int = 15, ttl: int = 60) -> List[Dict[str, Any]]:
    """Top ``pages*per_page`` coins by market cap, with 1h/24h/7d change.

    Cached for ``ttl`` seconds. Raises :class:`CoinGeckoError` only if the very
    first page fails; partial results from later-page failures are returned.
    """
    key = f"markets:{vs}:{pages}:{per_page}"
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]

    out: List[Dict[str, Any]] = []
    for page in range(1, max(1, pages) + 1):
        params = {
            "vs_currency": vs, "order": "market_cap_desc",
            "per_page": per_page, "page": page,
            "price_change_percentage": "1h,24h,7d", "sparkline": "false",
        }
        try:
            resp = requests.get(f"{_BASE}/coins/markets", params=params, timeout=timeout)
        except requests.RequestException as exc:
            if out:
                break
            raise CoinGeckoError(f"network error: {exc}")
        if resp.status_code == 429:
            if out:
                break
            raise CoinGeckoError("rate limited by CoinGecko — try again shortly")
        if resp.status_code != 200:
            if out:
                break
            raise CoinGeckoError(f"HTTP {resp.status_code}: {resp.text[:160]}")
        try:
            rows = resp.json()
        except ValueError as exc:
            if out:
                break
            raise CoinGeckoError(f"bad JSON: {exc}")
        if not rows:
            break
        out.extend(rows)
        if len(rows) < per_page:
            break

    _cache[key] = (now, out)
    return out
