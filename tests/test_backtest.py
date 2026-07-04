"""Backtester: no look-ahead, determinism, and metric integrity."""
import numpy as np

from crypto_tool.analysis import signals
from crypto_tool.backtest import engine, metrics
from crypto_tool.data import synthetic


def test_backtest_runs_and_metrics_present(cfg):
    df = synthetic.generate_ohlcv("BTCUSDT", interval="1h", n=600, seed=5)
    bt = engine.run_backtest(df, cfg)
    m = metrics.compute_metrics(bt)
    for key in ["total_return_pct", "buy_hold_return_pct", "sharpe",
                "max_drawdown_pct", "num_trades", "win_rate_pct", "exposure_pct"]:
        assert key in m
    # Equity must stay positive (long/flat, costs bounded).
    assert (bt["result"]["equity"] > 0).all()
    # Drawdown is non-positive by definition.
    assert m["max_drawdown_pct"] <= 0


def test_backtest_no_lookahead(cfg):
    """Each entry must fill at the open *after* a qualifying signal bar."""
    df = synthetic.generate_ohlcv("ETHUSDT", interval="1h", n=600, seed=9)
    params = {"entry_score": 40, "exit_score": 0, "stop_loss_pct": 8.0}
    bt = engine.run_backtest(df, cfg, params)
    e = signals.enrich(df, cfg).reset_index(drop=True)
    fee_slip = (cfg["backtest"]["fee_pct"] + cfg["backtest"]["slippage_pct"]) / 100.0
    otime = e["open_time"].to_numpy()

    trades = bt["trades"]
    assert not trades.empty, "expected at least one trade on this seed"
    for _, tr in trades.iterrows():
        idx = int(np.where(otime == tr["entry_time"])[0][0])
        assert idx >= 1, "cannot enter on the first bar — no prior signal"
        # The decision bar is the prior bar; it must have qualified.
        assert e["composite"].iloc[idx - 1] >= params["entry_score"] - 1e-9
        # Entry fills at this bar's open plus costs — not at the signal bar's close.
        expected = e["open"].iloc[idx] * (1.0 + fee_slip)
        assert abs(tr["entry_price"] - expected) < 1e-6


def test_backtest_deterministic(cfg):
    df = synthetic.generate_ohlcv("SOLUSDT", interval="1h", n=500, seed=2)
    m1 = metrics.compute_metrics(engine.run_backtest(df, cfg))
    m2 = metrics.compute_metrics(engine.run_backtest(df, cfg))
    assert m1 == m2


def test_accuracy_gates_filter_entries(cfg):
    """Stricter gates must never produce MORE trades — accuracy over activity."""
    df = synthetic.generate_ohlcv("BTCUSDT", interval="1h", n=600, seed=5)
    loose = engine.run_backtest(df, cfg, params={"conf_gate": 0.0, "confirm_bars": 1})
    default = engine.run_backtest(df, cfg)  # ships with gate 0.6 + confirm 2
    strict = engine.run_backtest(df, cfg, params={"conf_gate": 0.95, "confirm_bars": 4})
    assert len(default["trades"]) <= len(loose["trades"])
    assert len(strict["trades"]) <= len(default["trades"])
