"""Data-layer regression tests for the review fixes."""
import numpy as np
import pytest

from crypto_tool.data import database, synthetic


def test_seed_demo_deterministic_default_seed():
    """generate_ohlcv with no explicit seed must be reproducible across calls
    (zlib.crc32 seed, not the PYTHONHASHSEED-salted builtin hash)."""
    a = synthetic.generate_ohlcv("BTCUSDT", interval="1h", n=200)
    b = synthetic.generate_ohlcv("BTCUSDT", interval="1h", n=200)
    assert np.array_equal(a["close"].to_numpy(), b["close"].to_numpy())
    assert np.array_equal(a["volume"].to_numpy(), b["volume"].to_numpy())


def test_interval_to_ms_valid():
    assert synthetic.interval_to_ms("1h") == 3_600_000
    assert synthetic.interval_to_ms("15m") == 900_000
    assert synthetic.interval_to_ms("1d") == 86_400_000


@pytest.mark.parametrize("bad", ["h", "d", "xh", "1x", ""])
def test_interval_to_ms_rejects_malformed(bad):
    with pytest.raises(ValueError):
        synthetic.interval_to_ms(bad)


def test_load_ohlcv_limit_semantics(tmp_path):
    db = tmp_path / "t.db"
    conn = database.connect(str(db))
    try:
        df = synthetic.generate_ohlcv("ETHUSDT", interval="1h", n=50, seed=1)
        database.upsert_ohlcv(conn, df)

        full = database.load_ohlcv(conn, "ETHUSDT", "1h")
        assert len(full) == 50
        # limit=0 must mean "zero rows", not "no limit" (falsy-0 trap).
        assert len(database.load_ohlcv(conn, "ETHUSDT", "1h", limit=0)) == 0
        assert len(database.load_ohlcv(conn, "ETHUSDT", "1h", limit=10)) == 10
        with pytest.raises(ValueError):
            database.load_ohlcv(conn, "ETHUSDT", "1h", limit=-5)
    finally:
        conn.close()


def test_upsert_is_idempotent(tmp_path):
    db = tmp_path / "t.db"
    conn = database.connect(str(db))
    try:
        df = synthetic.generate_ohlcv("SOLUSDT", interval="1h", n=30, seed=2)
        database.upsert_ohlcv(conn, df)
        database.upsert_ohlcv(conn, df)  # same rows again
        assert len(database.load_ohlcv(conn, "SOLUSDT", "1h")) == 30
    finally:
        conn.close()
