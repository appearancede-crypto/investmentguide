"""Coin scout tests — the whole pipeline, offline (no network)."""
import json

from crypto_tool.analysis import scout
from crypto_tool.data import database, synthetic
from crypto_tool.web import build


def _pairs(n):
    # BTCUSDT is in the TRADING universe (no ticker row) so BTCUP is
    # recognisable as a leveraged token of a listed root.
    return {f"C{i}USDT" for i in range(n)} | {"USDCUSDT", "BTCUPUSDT", "BTCUSDT", "THINUSDT"}


def _tickers(n):
    rows = [{"symbol": f"C{i}USDT", "quoteVolume": str(1e7 * (i + 1)),
             "priceChangePercent": "2.5"} for i in range(n)]
    rows += [
        {"symbol": "USDCUSDT", "quoteVolume": "9e9", "priceChangePercent": "0.01"},
        {"symbol": "BTCUPUSDT", "quoteVolume": "5e8", "priceChangePercent": "9.0"},
        {"symbol": "THINUSDT", "quoteVolume": "100", "priceChangePercent": "1.0"},
        {"symbol": "NOTAPAIRUSDT", "quoteVolume": "8e8", "priceChangePercent": "1.0"},
    ]
    return rows


def _fetch(cfg, bars=450):
    def fetch(sym):
        return synthetic.generate_ohlcv(sym, interval=cfg["data"]["interval"],
                                        n=bars, seed=abs(hash(sym)) % 10_000)
    return fetch


def test_stablecoins_are_filtered_by_name_and_behaviour(cfg):
    pairs = {"AAAUSDT", "RLUSDUSDT", "PEGUSDT"}
    tickers = [
        {"symbol": "AAAUSDT", "quoteVolume": "5e7", "priceChangePercent": "3.1",
         "lastPrice": "1.01"},                       # ~$1 but MOVING -> a real coin
        {"symbol": "RLUSDUSDT", "quoteVolume": "9e8", "priceChangePercent": "0.4",
         "lastPrice": "1.0"},                        # …USD base -> peg by design
        {"symbol": "PEGUSDT", "quoteVolume": "9e8", "priceChangePercent": "0.01",
         "lastPrice": "0.9998"},                     # glued to $1, dead flat -> peg
    ]
    rows = scout.eligible_pairs(pairs, tickers, min_quote_volume=1e6, max_symbols=0)
    assert [r["symbol"] for r in rows] == ["AAAUSDT"]


def test_eligible_pairs_filters_and_ranks(cfg):
    rows = scout.eligible_pairs(_pairs(6), _tickers(6),
                                min_quote_volume=1_000_000, max_symbols=4)
    syms = [r["symbol"] for r in rows]
    # exact membership AND order: volume-desc, top-4 cut
    assert syms == ["C5USDT", "C4USDT", "C3USDT", "C2USDT"]
    assert "USDCUSDT" not in syms          # stablecoin pair
    assert "BTCUPUSDT" not in syms         # leveraged token
    assert "THINUSDT" not in syms          # below the volume floor
    assert "NOTAPAIRUSDT" not in syms      # not a TRADING pair


def test_leveraged_filter_spares_real_coins_ending_in_up(cfg):
    """JUP (Jupiter) and SYRUP are real coins whose names merely END in a
    leveraged-token suffix — they must survive; BTCUP must not."""
    pairs = {"JUPUSDT", "SYRUPUSDT", "BTCUPUSDT", "BTCUSDT"}
    tickers = [{"symbol": s, "quoteVolume": "5e7", "priceChangePercent": "2.0",
                "lastPrice": "0.5"} for s in pairs]
    rows = scout.eligible_pairs(pairs, tickers, min_quote_volume=1e6, max_symbols=0)
    syms = {r["symbol"] for r in rows}
    assert "JUPUSDT" in syms and "SYRUPUSDT" in syms
    assert "BTCUPUSDT" not in syms         # BTC trades, so BTCUP is leveraged
    assert "BTCUSDT" in syms


def test_run_scout_offline_snapshot(cfg):
    snap = scout.run_scout(cfg, pairs=_pairs(6), tickers=_tickers(6), markets=[],
                           fetch_klines=_fetch(cfg))
    assert snap["scored"] == 6 and snap["errors"] == 0
    assert snap["eligible"] == 6
    rows = snap["rows"]
    assert 0 < len(rows) <= cfg["scout"]["top"]
    comps = [r["composite"] for r in rows]
    assert comps == sorted(comps, reverse=True)
    for r in rows:
        for k in ["symbol", "name", "price", "composite", "confidence", "flag",
                  "risk", "riskFlags", "record", "outlook", "quoteVolume", "tracked"]:
            assert k in r, k
        assert r["risk"] in ("MEDIUM", "HIGH", "EXTREME")
    json.dumps(snap, allow_nan=False)      # nothing NaN may reach the page


def test_run_scout_survives_bad_pairs(cfg):
    def fetch(sym):
        if sym == "C1USDT":
            raise RuntimeError("exchange hiccup")
        return _fetch(cfg)(sym)
    snap = scout.run_scout(cfg, pairs=_pairs(4), tickers=_tickers(4), markets=[],
                           fetch_klines=fetch)
    assert snap["errors"] == 1
    assert snap["scored"] == 3


def test_outage_never_clobbers_last_good_snapshot(tmp_path, cfg):
    import pytest
    conn = database.connect(str(tmp_path / "s.db"))
    try:
        good = scout.run_scout(cfg, pairs=_pairs(4), tickers=_tickers(4), markets=[],
                               fetch_klines=_fetch(cfg))
        database.save_scout(conn, good)
        # empty universe -> refuse to sweep at all
        with pytest.raises(RuntimeError):
            scout.run_scout(cfg, pairs=set(), tickers=[], markets=[],
                            fetch_klines=_fetch(cfg))
        assert database.load_scout(conn) == good
        # every kline fetch failing -> a 0-scored sweep must not overwrite
        def bad_fetch(sym):
            raise RuntimeError("down")
        with pytest.raises(RuntimeError):
            scout.run_and_save(conn, cfg, pairs=_pairs(4), tickers=_tickers(4),
                               markets=[], fetch_klines=bad_fetch)
        assert database.load_scout(conn) == good
    finally:
        conn.close()


def test_scout_snapshot_roundtrip_and_payload(tmp_path, cfg):
    conn = database.connect(str(tmp_path / "s.db"))
    try:
        for sym in cfg["data"]["symbols"]:
            database.upsert_ohlcv(conn, synthetic.generate_ohlcv(
                sym, interval=cfg["data"]["interval"], n=300))
        snap = scout.run_scout(cfg, pairs=_pairs(4), tickers=_tickers(4), markets=[],
                               fetch_klines=_fetch(cfg))
        database.save_scout(conn, snap)
        assert database.load_scout(conn) == snap
        payload = build.build_payload(conn, cfg, history=200)
        assert payload["scout"] == snap
        json.dumps(payload, allow_nan=False)
        # saving again replaces, not duplicates
        database.save_scout(conn, {**snap, "scored": 1})
        assert database.load_scout(conn)["scored"] == 1
    finally:
        conn.close()
