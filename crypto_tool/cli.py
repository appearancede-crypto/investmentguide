"""Command-line interface.

Examples
--------
    python -m crypto_tool.cli seed-demo          # offline synthetic data
    python -m crypto_tool.cli ingest             # pull live Binance candles
    python -m crypto_tool.cli scan               # ranked signal table
    python -m crypto_tool.cli backtest BTCUSDT   # backtest one symbol
    python -m crypto_tool.cli coverage           # what's in the database
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser
from typing import Any, Dict

from .analysis import discovery, exits, signals
from .backtest import engine, metrics
from .config import load_config, resolve_db_path
from .data import binance_client, coingecko, database, ingest
from .paper import portfolio

DISCLAIMER = (
    "Educational analysis only - NOT financial advice. "
    "No tool can predict markets; past performance does not guarantee future results."
)


def _conn(cfg: Dict[str, Any]):
    return database.connect(resolve_db_path(cfg))


def cmd_ingest(cfg, args, conn) -> int:
    print(f"Ingesting {len(cfg['data']['symbols'])} symbols "
          f"[{cfg['data']['interval']}] from Binance ...")
    results = ingest.ingest_all(conn, cfg)
    ok = [r for r in results if r["ok"]]
    for r in results:
        status = f"OK   {r['rows']:>5} rows" if r["ok"] else f"FAIL {r['error']}"
        print(f"  {r['symbol']:<10} {status}")
    print(f"\n{len(ok)}/{len(results)} symbols ingested.")
    if not ok:
        print("\nLive fetch failed for everything (network/region?). "
              "Run `python -m crypto_tool.cli seed-demo` to use offline demo data.")
        return 1
    return 0


def cmd_seed_demo(cfg, args, conn) -> int:
    n = args.bars or cfg["data"]["history_limit"]
    print(f"Seeding {len(cfg['data']['symbols'])} symbols with {n} synthetic candles ...")
    results = ingest.seed_demo(conn, cfg, n=n)
    for r in results:
        print(f"  {r['symbol']:<10} OK {r['rows']} rows")
    print("\nDone. This is synthetic data for demos/tests - not real prices.")
    return 0


def cmd_scan(cfg, args, conn) -> int:
    table = signals.scan(conn, cfg)
    if table.empty:
        print("No data yet. Run `ingest` or `seed-demo` first.")
        return 1
    if args.json:
        print(table.to_json(orient="records"))
        return 0
    print(f"\nSignal scan  [{cfg['data']['interval']}]   (composite -100..+100, conf %)\n")
    header = f"{'#':>2}  {'SYMBOL':<10}{'PRICE':>12}  {'SCORE':>6}  {'CONF':>5}  {'FLAG':<11} DRIVER"
    print(header)
    print("-" * max(len(header), 88))
    for i, row in table.iterrows():
        print(f"{i+1:>2}  {row['symbol']:<10}{row['price']:>12.4g}  "
              f"{row['composite']:>6.1f}  {row['confidence']:>5.1f}  "
              f"{row['flag']:<11} {row['top_driver'][:46]}")
    print(f"\n{DISCLAIMER}")
    return 0


def cmd_backtest(cfg, args, conn) -> int:
    symbol = args.symbol.upper()
    df = database.load_ohlcv(conn, symbol, cfg["data"]["interval"])
    if df.empty:
        print(f"No data for {symbol}. Run `ingest` or `seed-demo` first.")
        return 1
    bt = engine.run_backtest(df, cfg)
    m = metrics.compute_metrics(bt)
    database.save_backtest(conn, symbol, cfg["data"]["interval"], bt["params"], m)
    if args.json:
        print(json.dumps(m, indent=2))
        return 0
    print(f"\nBacktest {symbol} [{cfg['data']['interval']}] over {m['bars']} bars "
          f"(~{m['span_years']} yr)\n")
    rows = [
        ("Strategy return", f"{m['total_return_pct']:>10.2f}%"),
        ("Buy & hold return", f"{m['buy_hold_return_pct']:>10.2f}%"),
        ("Excess vs hold", f"{m['excess_vs_hold_pct']:>10.2f}%"),
        ("Strategy CAGR", f"{m['cagr_pct']:>10.2f}%"),
        ("Sharpe (strat/hold)", f"{m['sharpe']:>6} / {m['buy_hold_sharpe']}"),
        ("Max drawdown", f"{m['max_drawdown_pct']:>10.2f}%"),
        ("Hold max drawdown", f"{m['buy_hold_max_drawdown_pct']:>10.2f}%"),
        ("Exposure (time in mkt)", f"{m['exposure_pct']:>10.1f}%"),
        ("Trades", f"{m['num_trades']:>10}"),
        ("Win rate", f"{m['win_rate_pct']:>10.1f}%"),
        ("Profit factor", "n/a (no losing trades)" if m['profit_factor'] is None else f"{m['profit_factor']}"),
        ("Avg / best / worst", f"{m['avg_trade_pct']}% / {m['best_trade_pct']}% / {m['worst_trade_pct']}%"),
    ]
    for label, val in rows:
        print(f"  {label:<24}{val}")
    print(f"\n{DISCLAIMER}")
    return 0


def cmd_exits(cfg, args, conn) -> int:
    if args.symbol:
        symbol = args.symbol.upper()
        df = database.load_ohlcv(conn, symbol, cfg["data"]["interval"])
        if df.empty:
            print(f"No data for {symbol}. Run `ingest` or `seed-demo` first.")
            return 1
        summ = exits.exit_summary(signals.enrich(df, cfg), cfg)
        print(f"\nExit guidance — {symbol} @ {summ['price']:.4g}\n")
        print(f"  Recommendation : {summ['recommendation']}")
        print(f"  Trailing stop  : {summ['trailing_stop']:.4g}  (risk {summ['risk_pct']}%)")
        print(f"  Take-profit    : {summ['take_profit']:.4g}  (reward {summ['reward_pct']}%)")
        print(f"  Swing low / resistance : {summ['swing_low']:.4g} / {summ['resistance']:.4g}")
        print(f"  Reward : risk  : {summ['risk_reward']}")
        print("  Why:")
        for r in summ["reasons"]:
            print(f"    - {r}")
        print(f"\n{DISCLAIMER}")
        return 0

    table = exits.exit_scan(conn, cfg)
    if table.empty:
        print("No data yet. Run `ingest` or `seed-demo` first.")
        return 1
    print(f"\nExit guidance  [{cfg['data']['interval']}]  (act-now first)\n")
    header = (f"{'SYMBOL':<10}{'PRICE':>11}  {'ACTION':<20}  {'TRAIL STOP':>12}  "
              f"{'TAKE PROFIT':>12}  {'R:R':>6}  WHY")
    print(header)
    print("-" * max(len(header), 100))
    for _, r in table.iterrows():
        rr = "-" if r["rr"] is None else f"{r['rr']:.1f}"
        print(f"{r['symbol']:<10}{r['price']:>11.4g}  {r['action']:<20}  "
              f"{r['trailing_stop']:>12.4g}  {r['take_profit']:>12.4g}  {rr:>6}  {r['why'][:32]}")
    print(f"\n{DISCLAIMER}")
    return 0


def cmd_discover(cfg, args, conn) -> int:
    d, disc = cfg["data"], cfg["discovery"]
    print("Fetching the broad market from CoinGecko ...")
    try:
        markets = coingecko.fetch_markets(pages=args.pages or disc["pages"],
                                          timeout=d["request_timeout"])
    except coingecko.CoinGeckoError as exc:
        print(f"CoinGecko fetch failed: {exc}")
        return 1
    listed = binance_client.fetch_usdt_symbols(d["base_urls"], d["request_timeout"])
    tracked = set(database.list_symbols(conn, d["interval"]))
    rows = discovery.screen(markets, listed, tracked, top=args.top or disc["top"],
                            sort=args.sort or disc["sort"], min_volume=disc["min_volume_usd"])
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    print("\n" + "!" * 92)
    print("  SPECULATION SCREENER — these are the market's MOVERS, which means the RISKIEST coins.")
    print("  Most low-cap movers are pump-and-dumps or trend to zero. Risk flags are shown per row.")
    print("  Not a buy list. Not financial advice. Never risk more than you can lose.")
    print("!" * 92)
    print(f"\nTop {len(rows)} by {args.sort or disc['sort']}   "
          f"([*]=full signals available · [B]=on Binance)\n")
    header = (f"{'#':>2}  {'COIN':<8}{'PRICE':>12}  {'1H':>7}{'24H':>8}{'7D':>8}  "
              f"{'MCAP':>8}  {'RISK':<8} FLAGS")
    print(header)
    print("-" * max(len(header), 104))
    for i, r in enumerate(rows):
        mark = "*" if r["deepDive"] else ("B" if r["onBinance"] else " ")
        mcap = _human(r["mcap"])
        flags = "; ".join(r["flags"][:3])
        print(f"{i+1:>2}{mark} {r['symbol']:<8}{_g(r['price']):>12}  "
              f"{_pct(r['p1h']):>7}{_pct(r['p24h']):>8}{_pct(r['p7d']):>8}  "
              f"{mcap:>8}  {r['risk']:<8} {flags[:46]}")
    print(f"\n{DISCLAIMER}")
    return 0


def _human(v):
    if not v:
        return "—"
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if v >= div:
            return f"${v/div:.1f}{unit}"
    return f"${v:.0f}"


def _pct(v):
    return "—" if v is None else f"{v:+.1f}%"


def _g(v):
    if v is None:
        return "—"
    if v >= 1000:
        return f"{v:,.0f}"
    if v >= 1:
        return f"{v:.3f}"
    return f"{v:.6g}"


def cmd_web(cfg, args, conn) -> int:
    from .web import build as webbuild
    if args.serve:
        from .web import server as webserver
        webserver.serve(cfg, config_path=args.config, port=args.port,
                        open_browser=not args.no_open, ensure_data=args.ensure_data,
                        refresh_min=args.refresh_min, ingest_limit=args.ingest_limit)
        return 0
    html = webbuild.build_page(conn, cfg)
    out = os.path.join(os.path.dirname(resolve_db_path(cfg)), "signal_engine.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"Wrote self-contained page: {out}")
    print("Open it in a browser, or run `python -m crypto_tool.cli web --serve` "
          "for a live URL that refreshes from the database.")
    if not args.no_open:
        try:
            webbrowser.open("file:///" + out.replace("\\", "/"))
        except Exception:
            pass
    print(f"\n{DISCLAIMER}")
    return 0


def cmd_coverage(cfg, args, conn) -> int:
    cov = database.ohlcv_coverage(conn)
    if cov.empty:
        print("Database is empty. Run `ingest` or `seed-demo`.")
        return 1
    print(cov.to_string(index=False))
    return 0


# --------------------------------------------------------------------------- #
# Paper trading
# --------------------------------------------------------------------------- #
def cmd_paper_open(cfg, args, conn) -> int:
    cash = args.cash if args.cash is not None else cfg["paper"]["starting_cash"]
    database.create_paper_account(
        conn, args.name, cash, args.learn, cfg["data"]["interval"], cfg["paper"]
    )
    mode = "ADAPTIVE (learns from realised P&L)" if args.learn else "static weights"
    print(f"Opened paper account '{args.name}' with ${cash:,.0f} — {mode}.")
    print("Run `python -m crypto_tool.cli paper-run` to simulate it over your data.")
    return 0


def cmd_paper_run(cfg, args, conn) -> int:
    acc = database.get_paper_account(conn, args.name)
    if acc is None:
        print(f"No paper account '{args.name}'. Create it with `paper-open`.")
        return 1
    res = portfolio.run_and_save(conn, cfg, args.name)
    if res is None:
        print("Not enough data. Run `ingest` or `seed-demo` first.")
        return 1
    print(f"Ran '{args.name}' over {res['metrics']['bars']} bars "
          f"across {len(res['symbols'])} coins.\n")
    return _print_paper_status(cfg, conn, args.name)


def cmd_paper_status(cfg, args, conn) -> int:
    return _print_paper_status(cfg, conn, args.name)


def _print_paper_status(cfg, conn, name) -> int:
    acc = database.get_paper_account(conn, name)
    if acc is None:
        print(f"No paper account '{name}'.")
        return 1
    if not acc["metrics"]:
        print(f"Account '{name}' has no results yet. Run `paper-run`.")
        return 1
    m = acc["metrics"]
    mode = "ADAPTIVE" if acc["learn"] else "static"
    print(f"Paper account '{name}'  [{mode} weights]   start ${acc['starting_cash']:,.0f}\n")
    rows = [
        ("Final equity", f"${m['final_equity']:,.2f}"),
        ("Total return", f"{m['total_return_pct']:>8.2f}%"),
        ("Benchmark (equal-weight hold)", f"{m['benchmark_return_pct']:>8.2f}%"),
        ("Excess vs benchmark", f"{m['excess_vs_benchmark_pct']:>8.2f}%"),
        ("Sharpe (port / bench)", f"{m['sharpe']} / {m['benchmark_sharpe']}"),
        ("Max drawdown", f"{m['max_drawdown_pct']:>8.2f}%"),
        ("Avg exposure", f"{m['avg_exposure_pct']:>8.1f}%"),
        ("Trades / turnover", f"{m['num_trades']} / {m['turnover_x']}x"),
    ]
    for label, val in rows:
        print(f"  {label:<32}{val}")

    if acc["positions"]:
        print("\n  Current holdings:")
        interval = acc["interval"]
        equity = m["final_equity"]
        held = []
        for sym, u in acc["positions"].items():
            df = database.load_ohlcv(conn, sym, interval, limit=1)
            price = float(df["close"].iloc[-1]) if not df.empty else 0.0
            val = u * price
            held.append((sym, val, 100 * val / equity if equity else 0))
        for sym, val, pct in sorted(held, key=lambda x: -x[1]):
            print(f"    {sym:<10} ${val:>12,.2f}  ({pct:4.1f}%)")

    if acc["learn"] and acc["weights"]:
        print("\n  Learned rule weights (vs baseline):")
        base = cfg["signals"]["weights"]
        for name_, val in sorted(acc["weights"].items(), key=lambda x: -x[1]):
            arrow = "↑" if val > base[name_] + 1e-6 else "↓" if val < base[name_] - 1e-6 else "="
            print(f"    {name_:<11} {val:>5.2f}  {arrow}  (baseline {base[name_]:.2f})")

    print(f"\n{DISCLAIMER}")
    return 0


def cmd_paper_compare(cfg, args, conn) -> int:
    cash = args.cash if args.cash is not None else cfg["paper"]["starting_cash"]
    static = portfolio.run_paper(conn, cfg, starting_cash=cash, learn=False)
    adaptive = portfolio.run_paper(conn, cfg, starting_cash=cash, learn=True)
    if static is None or adaptive is None:
        print("Not enough data. Run `ingest` or `seed-demo` first.")
        return 1
    s, a = static["metrics"], adaptive["metrics"]
    bench = s["benchmark_return_pct"]
    print(f"Learned vs static — ${cash:,.0f} across {len(static['symbols'])} coins, "
          f"{s['bars']} bars\n")
    print(f"  {'':<26}{'STATIC':>12}{'ADAPTIVE':>12}")
    line = lambda lbl, sv, av: print(f"  {lbl:<26}{sv:>12}{av:>12}")
    line("Total return %", f"{s['total_return_pct']:.2f}", f"{a['total_return_pct']:.2f}")
    line("Excess vs bench %", f"{s['excess_vs_benchmark_pct']:.2f}", f"{a['excess_vs_benchmark_pct']:.2f}")
    line("Sharpe", f"{s['sharpe']}", f"{a['sharpe']}")
    line("Max drawdown %", f"{s['max_drawdown_pct']:.2f}", f"{a['max_drawdown_pct']:.2f}")
    line("Trades", f"{s['num_trades']}", f"{a['num_trades']}")
    print(f"\n  Benchmark (equal-weight buy & hold): {bench:.2f}%")
    diff = a["total_return_pct"] - s["total_return_pct"]
    verdict = ("Learning HELPED here" if diff > 0.05 else
               "Learning HURT here" if diff < -0.05 else "Learning was ~neutral here")
    print(f"  {verdict}: adaptive − static = {diff:+.2f}% (one sample — not proof either way).")
    print(f"\n{DISCLAIMER}")
    return 0


def cmd_paper_reset(cfg, args, conn) -> int:
    database.delete_paper_account(conn, args.name)
    print(f"Deleted paper account '{args.name}'.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="crypto_tool", description="Crypto signal analysis tool")
    p.add_argument("--config", default=None, help="Path to config.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="Fetch live candles from Binance into the DB")
    sd = sub.add_parser("seed-demo", help="Populate DB with synthetic offline data")
    sd.add_argument("--bars", type=int, default=None, help="Candles per symbol")
    sc = sub.add_parser("scan", help="Print the ranked latest-signal table")
    sc.add_argument("--json", action="store_true")
    bt = sub.add_parser("backtest", help="Backtest a single symbol")
    bt.add_argument("symbol")
    bt.add_argument("--json", action="store_true")
    sub.add_parser("coverage", help="Show what data is stored")
    wb = sub.add_parser("web", help="Build/serve the Signal Engine web UI")
    wb.add_argument("--serve", action="store_true", help="Serve at a local URL (refreshes on load)")
    wb.add_argument("--port", type=int, default=None, help="Port (defaults to $PORT, then config)")
    wb.add_argument("--no-open", action="store_true", help="Don't auto-open a browser")
    wb.add_argument("--ensure-data", action="store_true",
                    help="Ingest candles on startup if the DB is empty (for hosting)")
    wb.add_argument("--refresh-min", type=int, default=0,
                    help="Re-ingest candles every N minutes in the background (0 = off)")
    wb.add_argument("--ingest-limit", type=int, default=None,
                    help="Candles per coin for boot/background ingest (smaller = faster cold start)")
    ex = sub.add_parser("exits", help="Exit guidance (trailing stop, take-profit, action)")
    ex.add_argument("symbol", nargs="?", default=None, help="One symbol, or omit for all")
    dc = sub.add_parser("discover", help="Risk-flagged market movers from CoinGecko")
    dc.add_argument("--top", type=int, default=None)
    dc.add_argument("--pages", type=int, default=None, help="CoinGecko pages of 250 coins")
    dc.add_argument("--sort", choices=["momentum", "gainers", "volume"], default=None)
    dc.add_argument("--json", action="store_true")

    po = sub.add_parser("paper-open", help="Create/reset a fictional-money account")
    po.add_argument("--name", default="default")
    po.add_argument("--cash", type=float, default=None)
    po.add_argument("--learn", action="store_true", help="Adapt weights from realised P&L")
    pr = sub.add_parser("paper-run", help="Simulate the account over stored data")
    pr.add_argument("--name", default="default")
    ps = sub.add_parser("paper-status", help="Show an account's equity / P&L / holdings")
    ps.add_argument("--name", default="default")
    pc = sub.add_parser("paper-compare", help="Learned vs static, side by side")
    pc.add_argument("--cash", type=float, default=None)
    px = sub.add_parser("paper-reset", help="Delete an account")
    px.add_argument("--name", default="default")
    return p


def main(argv=None) -> int:
    # Windows consoles default to cp1252; force UTF-8 so rationale text (em-dashes,
    # arrows) renders correctly instead of as mojibake.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    dispatch = {
        "ingest": cmd_ingest,
        "seed-demo": cmd_seed_demo,
        "scan": cmd_scan,
        "backtest": cmd_backtest,
        "coverage": cmd_coverage,
        "web": cmd_web,
        "exits": cmd_exits,
        "discover": cmd_discover,
        "paper-open": cmd_paper_open,
        "paper-run": cmd_paper_run,
        "paper-status": cmd_paper_status,
        "paper-compare": cmd_paper_compare,
        "paper-reset": cmd_paper_reset,
    }
    conn = _conn(cfg)
    try:
        return dispatch[args.command](cfg, args, conn)
    finally:
        # Checkpoint the WAL and release sidecar files instead of relying on
        # interpreter exit to flush.
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            conn.close()


if __name__ == "__main__":
    sys.exit(main())
