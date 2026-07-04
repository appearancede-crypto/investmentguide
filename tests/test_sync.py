"""Watchlist synchronization tests (offline)."""
from crypto_tool.data import database, ingest, synthetic
from crypto_tool.web import build


def _pairs():
    return {f"T{i}USDT" for i in range(10)} | {"BTCUSDT"}


def _tickers():
    return [{"symbol": f"T{i}USDT", "quoteVolume": str(1e7 * (i + 1)),
             "priceChangePercent": "2.0", "lastPrice": "3.0"} for i in range(10)]


def test_resolve_symbols_tops_up_and_keeps_explicit(cfg):
    cfg = {**cfg, "data": {**cfg["data"], "symbols": ["BTCUSDT"], "symbols_auto": 5}}
    syms = ingest.resolve_symbols(cfg, pairs=_pairs(), tickers=_tickers())
    assert syms[0] == "BTCUSDT"            # explicit picks always first, never dropped
    assert len(syms) == 5
    assert syms[1:] == ["T9USDT", "T8USDT", "T7USDT", "T6USDT"]  # most-traded fill


def test_resolve_symbols_offline_falls_back_to_explicit(cfg):
    cfg = {**cfg, "data": {**cfg["data"], "symbols": ["BTCUSDT", "ETHUSDT"],
                           "symbols_auto": 50}}
    assert ingest.resolve_symbols(cfg, pairs=set(), tickers=[]) == ["BTCUSDT", "ETHUSDT"]


def test_resolve_symbols_disabled(cfg):
    cfg = {**cfg, "data": {**cfg["data"], "symbols_auto": 0}}
    assert ingest.resolve_symbols(cfg) == [s.upper() for s in cfg["data"]["symbols"]]


def test_build_payload_prefers_tracked_list(tmp_path, cfg):
    conn = database.connect(str(tmp_path / "t.db"))
    try:
        for sym in ["AAAUSDT", "BBBUSDT", "CCCUSDT"]:
            database.upsert_ohlcv(conn, synthetic.generate_ohlcv(
                sym, interval=cfg["data"]["interval"], n=300))
        database.save_kv(conn, "tracked_symbols", ["AAAUSDT", "CCCUSDT"])
        payload = build.build_payload(conn, cfg, history=200)
        assert payload["names"] == ["AAAUSDT", "CCCUSDT"]   # BBB in DB but not tracked
        # no tracked list recorded -> everything in the DB (old behaviour)
        database.save_kv(conn, "tracked_symbols", None)
        payload = build.build_payload(conn, cfg, history=200)
        assert payload["names"] == ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
    finally:
        conn.close()


def test_kv_roundtrip(tmp_path):
    conn = database.connect(str(tmp_path / "kv.db"))
    try:
        assert database.load_kv(conn, "nope") is None
        database.save_kv(conn, "x", {"a": [1, 2]})
        assert database.load_kv(conn, "x") == {"a": [1, 2]}
        database.save_kv(conn, "x", [3])
        assert database.load_kv(conn, "x") == [3]
    finally:
        conn.close()
