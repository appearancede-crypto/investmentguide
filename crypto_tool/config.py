"""Configuration loading with sensible built-in defaults.

The YAML file is optional: if it is missing or only partially specified, the
defaults below fill the gaps via a deep merge. This keeps every other module
free of magic numbers.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Dict

import yaml

# --------------------------------------------------------------------------- #
# Defaults — kept in sync with config.yaml. config.yaml wins where it overlaps.
# --------------------------------------------------------------------------- #
DEFAULTS: Dict[str, Any] = {
    "data": {
        "base_urls": ["https://api.binance.com", "https://data-api.binance.vision"],
        "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"],
        "symbols_auto": 60,  # top up the watchlist to N coins from Binance's most-traded (0 = off)
        "interval": "1h",
        "history_limit": 5000,
        "db_path": "data/market.db",
        "request_timeout": 15,
    },
    "indicators": {
        "ema_fast": 12,
        "ema_slow": 26,
        "macd_signal": 9,
        "rsi_period": 14,
        "bb_period": 20,
        "bb_std": 2.0,
        "atr_period": 14,
        "sma_short": 50,
        "sma_long": 200,
        "roc_period": 10,
        "obv_smooth": 5,
        "vol_ma_period": 20,
        "curvature_smooth": 10,
    },
    "signals": {
        "weights": {
            "trend": 1.0,
            "macd": 1.0,
            "rsi": 0.8,
            "bollinger": 0.8,
            "curvature": 1.2,
            "momentum": 0.7,
            "volume": 0.5,
        },
        "buy_threshold": 35,
        "strong_buy_threshold": 65,
        "sell_threshold": -35,
        "strong_sell_threshold": -65,
    },
    "backtest": {
        "fee_pct": 0.10,
        "slippage_pct": 0.05,
        "entry_score": 40,
        "exit_score": 0,
        "stop_loss_pct": 8.0,
        "take_profit_pct": 0.0,
    },
    "paper": {
        "starting_cash": 10000,
        "fee_pct": 0.10,
        "min_score": 40,
        "max_weight_per_coin": 0.34,
        "rebalance_band": 0.05,
        "learn": {"window": 200, "retrain_every": 50, "lr": 0.25, "min_obs": 100},
    },
    "exits": {
        "atr_mult_stop": 3.0,
        "atr_mult_target": 6.0,
        "lookback": 22,
        "rsi_overbought": 70,
    },
    "web": {
        "history": 720,     # candles per coin embedded in the page (30 days of 1h;
                            # keeps a 60-coin page lean — forecasts still use full history)
        "theme": "carbon",  # carbon | bone | blueprint | oxide
        "port": 8787,
        "eval_horizon": 24,  # bars ahead used to score a directional call (24 x 1h = 1 day)
        "eval_band": 1.0,    # a move must clear +/- this % to count as right or wrong
        "eval_persist": 2,   # a flag must HOLD this many bars to count as a call (blip filter)
        "eval_conf_gate": 0.6,   # calls with confidence >= this AND regime agreement are graded separately
        "eval_horizon_long": 72,   # second, longer grading horizon (summary only)
        "eval_horizon_short": 6,   # quick-flip grading horizon (summary only)
        "forecast": {
            "bars": 48,             # how far forward the outlook cone is projected
            "checkpoints": [4, 12, 24, 48],  # stats + typical miss at these steps ahead
            "min_candidates": 60,   # refuse to project on thinner history than this
            "k_frac": 0.08,         # analogues = this fraction of candidate bars …
            "k_min": 25,            # … clamped between k_min and k_max
            "k_max": 150,
            "calib_samples": 60,    # past bars replayed for the honesty (coverage) check
        },
    },
    "discovery": {
        "pages": 2,             # CoinGecko pages of 250 -> top ~500 coins by market cap
        "top": 40,              # rows surfaced
        "sort": "momentum",     # momentum | gainers | volume
        "min_volume_usd": 250000,
    },
    "scout": {
        # Whole-exchange sweep: run the engine over every liquid Binance USDT
        # pair and keep the strongest setups.
        "bars": 400,                    # candles per pair (1 request each)
        "min_quote_volume_usd": 1_000_000,  # skip pairs with thinner 24h turnover
        "max_symbols": 300,             # sweep at most this many (most-traded first)
        "top": 60,                      # rows kept in the snapshot / shown on the page
        "pause_ms": 60,                 # politeness delay between kline requests
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def project_root() -> str:
    """Absolute path to the repository root (one level above this package)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_config_path() -> str:
    return os.path.join(project_root(), "config.yaml")


def load_config(path: str | None = None) -> Dict[str, Any]:
    """Load config from YAML, deep-merged onto built-in defaults.

    Missing file → defaults are used as-is (the tool still runs).
    """
    path = path or default_config_path()
    user: Dict[str, Any] = {}
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            user = yaml.safe_load(fh) or {}
    return _deep_merge(DEFAULTS, user)


def resolve_db_path(config: Dict[str, Any]) -> str:
    """Return an absolute path to the SQLite database, relative to repo root."""
    db_path = config["data"]["db_path"]
    if not os.path.isabs(db_path):
        db_path = os.path.join(project_root(), db_path)
    return db_path
