"""Paper-trading portfolio + learner tests."""
import numpy as np
import pytest

from crypto_tool.data import database, synthetic
from crypto_tool.paper import portfolio


@pytest.fixture()
def paper_conn(tmp_path, cfg):
    conn = database.connect(str(tmp_path / "p.db"))
    for sym in cfg["data"]["symbols"]:
        df = synthetic.generate_ohlcv(sym, interval=cfg["data"]["interval"], n=400)
        database.upsert_ohlcv(conn, df)
    yield conn
    conn.close()


def test_run_paper_basic(paper_conn, cfg):
    res = portfolio.run_paper(paper_conn, cfg, starting_cash=10_000, learn=False)
    assert res is not None
    m = res["metrics"]
    for key in ["total_return_pct", "benchmark_return_pct", "sharpe",
                "max_drawdown_pct", "num_trades", "turnover_x", "avg_exposure_pct"]:
        assert key in m
    assert len(res["equity_rows"]) > 0
    assert res["metrics"]["final_equity"] > 0


def test_equity_stays_positive(paper_conn, cfg):
    res = portfolio.run_paper(paper_conn, cfg, starting_cash=10_000, learn=True)
    eq = np.array([r[1] for r in res["equity_rows"]])
    assert (eq > 0).all()


def test_paper_deterministic(paper_conn, cfg):
    a = portfolio.run_paper(paper_conn, cfg, starting_cash=10_000, learn=True)
    b = portfolio.run_paper(paper_conn, cfg, starting_cash=10_000, learn=True)
    assert a["metrics"] == b["metrics"]
    assert a["weights"] == b["weights"]


def test_learned_weights_nonneg_and_sum_preserved(paper_conn, cfg):
    res = portfolio.run_paper(paper_conn, cfg, starting_cash=10_000, learn=True)
    w = res["weights"]
    base_sum = sum(cfg["signals"]["weights"].values())
    assert all(v >= -1e-9 for v in w.values())          # never bets on anti-rules
    # total weight (scale) preserved up to 4-dp rounding of the stored weights
    assert abs(sum(w.values()) - base_sum) < 1e-3


def test_learning_changes_weights(paper_conn, cfg):
    """The adaptive book should actually move weights away from baseline."""
    res = portfolio.run_paper(paper_conn, cfg, starting_cash=10_000, learn=True)
    base = cfg["signals"]["weights"]
    moved = any(abs(res["weights"][k] - base[k]) > 1e-3 for k in base)
    assert moved


def test_run_and_save_roundtrip(paper_conn, cfg):
    database.create_paper_account(paper_conn, "t", 10_000, True,
                                  cfg["data"]["interval"], cfg["paper"])
    res = portfolio.run_and_save(paper_conn, cfg, "t")
    assert res is not None
    eq = database.load_paper_equity(paper_conn, "t")
    trades = database.load_paper_trades(paper_conn, "t")
    acc = database.get_paper_account(paper_conn, "t")
    assert len(eq) == len(res["equity_rows"])
    assert len(trades) == res["metrics"]["num_trades"]
    assert acc["metrics"]["final_equity"] == res["metrics"]["final_equity"]
