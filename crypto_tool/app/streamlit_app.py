"""Crypto Investment Analysis — Streamlit dashboard.

Run it with the launcher in the repo root:

    streamlit run crypto_tool/app/streamlit_app.py

Views:
  * Signal Scanner — every coin ranked by composite score (the "where might the
    opportunity be" screen).
  * Coin Detail    — price + indicators + curvature + exit guidance, with the
    signal broken down rule by rule so you can see *why*.
  * Backtest       — how the strategy would have performed historically vs just
    holding the coin, with realistic fees.
  * Paper trading  — fictional money on the signals, with an optional learner.

All charts are interactive (scroll/drag zoom, range buttons). Analysis only. It
never places orders. Not financial advice.
"""
from __future__ import annotations

import os
import sys

# Make `crypto_tool` importable when Streamlit runs this file by path.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from crypto_tool.analysis import exits, signals  # noqa: E402
from crypto_tool.backtest import engine, metrics  # noqa: E402
from crypto_tool.config import load_config, resolve_db_path  # noqa: E402
from crypto_tool.data import database, ingest  # noqa: E402
from crypto_tool.paper import portfolio  # noqa: E402

st.set_page_config(page_title="Crypto Signal Analysis", page_icon="📈", layout="wide")

FLAG_COLORS = {
    "STRONG BUY": "#0b8a3e", "BUY": "#36b37e", "NEUTRAL": "#8b95a5",
    "SELL": "#ff7452", "STRONG SELL": "#bf2600",
}
EXIT_COLORS = {"EXIT": "#bf2600", "TRIM / TIGHTEN STOP": "#ff8b00", "HOLD — trail the stop": "#36b37e"}

# Interactive chart config: scroll-wheel zoom, clean modebar, hi-res export.
PLOT_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {"scale": 2},
}
_st_plotly = st.plotly_chart


def plot(fig, **kwargs):
    """Render a Plotly figure with zoom/pan enabled and full-width by default."""
    kwargs.setdefault("use_container_width", True)
    kwargs.setdefault("config", PLOT_CONFIG)
    _st_plotly(fig, **kwargs)


def add_time_controls(fig, rangeslider: bool = True):
    """Add quick range buttons (1w/1m/3m/all) + optional rangeslider for easy
    navigation along the time axis."""
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(step="all", label="All"),
            ],
            bgcolor="rgba(140,149,165,0.15)", x=0, y=1.06,
        ),
        rangeslider=dict(visible=rangeslider, thickness=0.06),
    )
    fig.update_layout(hovermode="x unified")


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
@st.cache_resource
def get_config():
    return load_config()


def db_path(cfg):
    return resolve_db_path(cfg)


def db_fingerprint(_db_path: str, symbol: str, interval: str) -> tuple:
    """Cheap (count, max-open-time) signature so the cache invalidates whenever
    the underlying data changes — including writes from a separate CLI process."""
    conn = database.connect(_db_path)
    try:
        row = conn.execute(
            "SELECT COUNT(*), COALESCE(MAX(open_time), 0) FROM ohlcv "
            "WHERE symbol=? AND interval=?",
            (symbol.upper(), interval),
        ).fetchone()
        return (int(row[0]), int(row[1]))
    finally:
        conn.close()


@st.cache_data(show_spinner=False)
def load_df(_db_path: str, symbol: str, interval: str, fingerprint: tuple) -> pd.DataFrame:
    """Cached OHLCV load, keyed by a real data fingerprint (not a session counter)."""
    conn = database.connect(_db_path)
    try:
        return database.load_ohlcv(conn, symbol, interval)
    finally:
        conn.close()


def get_df(cfg, symbol: str, interval: str) -> pd.DataFrame:
    """Load candles for a symbol through the fingerprinted cache."""
    dbp = db_path(cfg)
    return load_df(dbp, symbol, interval, db_fingerprint(dbp, symbol, interval))


def all_fingerprint(_db_path: str, interval: str) -> tuple:
    """Whole-universe data signature, for caching the paper simulation."""
    conn = database.connect(_db_path)
    try:
        row = conn.execute(
            "SELECT COUNT(*), COALESCE(MAX(open_time), 0) FROM ohlcv WHERE interval=?",
            (interval,),
        ).fetchone()
        return (int(row[0]), int(row[1]))
    finally:
        conn.close()


@st.cache_data(show_spinner=False)
def run_paper_both(_db_path: str, interval: str, cash: float, fingerprint: tuple):
    """Run the paper portfolio in both static and adaptive modes (cached)."""
    conn = database.connect(_db_path)
    try:
        cfg = load_config()
        static = portfolio.run_paper(conn, cfg, starting_cash=cash, learn=False)
        adaptive = portfolio.run_paper(conn, cfg, starting_cash=cash, learn=True)
        return static, adaptive
    finally:
        conn.close()


def to_dt(open_time_ms: pd.Series) -> pd.Series:
    return pd.to_datetime(open_time_ms, unit="ms", utc=True)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def sidebar(cfg):
    st.sidebar.header("⚙️ Data & Market")
    interval = cfg["data"]["interval"]
    st.sidebar.caption(f"Interval: **{interval}**  ·  DB: `{os.path.basename(db_path(cfg))}`")

    flash = st.session_state.pop("flash", None)
    if flash:
        st.sidebar.success(flash)

    conn = database.connect(db_path(cfg))
    try:
        cov = database.ohlcv_coverage(conn)
        symbols = database.list_symbols(conn, interval)
    finally:
        conn.close()

    c1, c2 = st.sidebar.columns(2)
    if c1.button("⬇️ Fetch live", use_container_width=True, help="Pull candles from Binance"):
        with st.spinner("Fetching from Binance…"):
            conn = database.connect(db_path(cfg))
            try:
                results = ingest.ingest_all(conn, cfg)
            finally:
                conn.close()
        ok = sum(r["ok"] for r in results)
        if ok:
            load_df.clear()
            st.session_state["flash"] = f"Ingested {ok}/{len(results)} symbols"
            st.rerun()  # re-read so the now-populated DB shows up this interaction
        else:
            st.sidebar.error("Live fetch failed (network/region). Try 'Seed demo'.")

    if c2.button("🧪 Seed demo", use_container_width=True, help="Synthetic offline data"):
        with st.spinner("Generating synthetic data…"):
            conn = database.connect(db_path(cfg))
            try:
                ingest.seed_demo(conn, cfg)
            finally:
                conn.close()
        load_df.clear()
        st.session_state["flash"] = "Seeded synthetic demo data"
        st.rerun()

    if not cov.empty:
        st.sidebar.markdown("**Stored data**")
        show = cov.copy()
        show["last"] = to_dt(show["last_ms"]).dt.strftime("%Y-%m-%d %H:%M")
        st.sidebar.dataframe(
            show[["symbol", "bars", "last"]], hide_index=True, use_container_width=True
        )
    else:
        st.sidebar.info("No data yet — click **Fetch live** or **Seed demo**.")

    return interval, symbols


# --------------------------------------------------------------------------- #
# View: Scanner
# --------------------------------------------------------------------------- #
def view_scanner(cfg, interval, symbols):
    st.subheader("🔎 Signal Scanner")
    st.caption("Every coin scored from −100 (bearish) to +100 (bullish). "
               "Confidence reflects rule agreement, signal strength, volume and data depth.")
    if not symbols:
        st.info("No data loaded. Use the sidebar to fetch live or seed demo data.")
        return

    conn = database.connect(db_path(cfg))
    try:
        table = signals.scan(conn, cfg)
    finally:
        conn.close()
    if table.empty:
        st.warning("Not enough history yet to compute signals.")
        return

    disp = table.copy()
    disp.index = disp.index + 1
    styled = disp[["symbol", "price", "composite", "confidence", "flag", "top_driver"]].rename(
        columns={"composite": "score", "confidence": "conf %", "top_driver": "top driver"}
    )

    def color_flag(val):
        return f"color: white; background-color: {FLAG_COLORS.get(val, '#8b95a5')}"

    st.dataframe(
        styled.style
        .map(color_flag, subset=["flag"])
        .background_gradient(cmap="RdYlGn", subset=["score"], vmin=-100, vmax=100)
        .format({"price": "{:.4g}", "score": "{:.1f}", "conf %": "{:.0f}"}),
        use_container_width=True, height=min(560, 60 + 38 * len(styled)),
    )

    fig = go.Figure()
    fig.add_bar(
        x=table["symbol"], y=table["composite"],
        marker_color=[FLAG_COLORS.get(f, "#8b95a5") for f in table["flag"]],
        text=[f"{c:+.0f}" for c in table["composite"]], textposition="outside",
    )
    fig.update_layout(
        title="Composite score by coin", yaxis_title="score (−100…+100)",
        yaxis_range=[-105, 105], height=380, margin=dict(t=50, b=10),
    )
    fig.add_hline(y=cfg["signals"]["buy_threshold"], line_dash="dot", line_color="#36b37e")
    fig.add_hline(y=cfg["signals"]["sell_threshold"], line_dash="dot", line_color="#ff7452")
    plot(fig)


# --------------------------------------------------------------------------- #
# View: Coin detail
# --------------------------------------------------------------------------- #
def view_detail(cfg, interval, symbols):
    st.subheader("📊 Coin Detail")
    if not symbols:
        st.info("No data loaded. Use the sidebar first.")
        return
    symbol = st.selectbox("Symbol", symbols, key="detail_symbol")
    df = get_df(cfg, symbol, interval)
    min_bars = max(cfg["indicators"]["ema_slow"], cfg["indicators"]["bb_period"]) + 5
    if len(df) < min_bars:
        st.warning(f"Need ≥{min_bars} candles; have {len(df)}.")
        return

    sig = signals.latest_signal(df, cfg)
    e = exits.add_exit_levels(sig["enriched"], cfg)
    e = e.assign(dt=to_dt(e["open_time"]))
    view = e.tail(min(400, len(e)))
    ex = exits.exit_summary(sig["enriched"], cfg)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Last price", f"{sig['close']:.4g}")
    flag = sig["flag"]
    m2.markdown(
        f"<div style='font-size:0.8rem;color:#8b95a5'>Entry signal</div>"
        f"<div style='font-size:1.6rem;font-weight:700;color:{FLAG_COLORS.get(flag)}'>"
        f"{flag}</div>", unsafe_allow_html=True,
    )
    m3.metric("Composite score", f"{sig['composite']:+.1f}", help="−100 bearish … +100 bullish")
    m4.metric("Confidence", f"{sig['confidence']*100:.0f}%")

    # --- Exit guidance (if you hold) -------------------------------------- #
    st.markdown("##### 🚪 Exit guidance — if you're holding this")
    x1, x2, x3, x4 = st.columns(4)
    x1.markdown(
        f"<div style='font-size:0.8rem;color:#8b95a5'>Action</div>"
        f"<div style='font-size:1.25rem;font-weight:700;color:{EXIT_COLORS.get(ex['recommendation'], '#8b95a5')}'>"
        f"{ex['recommendation']}</div>", unsafe_allow_html=True,
    )
    x2.metric("Trailing stop", f"{ex['trailing_stop']:.4g}", f"{ex['risk_pct']:+.1f}% from price",
              delta_color="off")
    x3.metric("Take-profit", f"{ex['take_profit']:.4g}", f"{ex['reward_pct']:+.1f}% from price",
              delta_color="off")
    rr = "—" if ex["risk_reward"] is None else f"{ex['risk_reward']:.1f}×"
    x4.metric("Reward : risk", rr, help="Target distance ÷ stop distance")
    st.caption("Why: " + " · ".join(ex["reasons"]))

    _price_panel(view, symbol, cfg)

    st.markdown("#### Why this signal — rule breakdown")
    rat = pd.DataFrame(sig["rationale"])
    rat = rat[["component", "vote", "weight", "contribution", "detail"]]
    st.dataframe(
        rat.style.background_gradient(cmap="RdYlGn", subset=["contribution"], vmin=-1.2, vmax=1.2)
        .format({"vote": "{:+.2f}", "weight": "{:.1f}", "contribution": "{:+.2f}"}),
        use_container_width=True, hide_index=True,
    )
    st.caption("Composite = weighted average of the votes, rescaled to ±100. "
               "Mean-reversion rules (RSI, Bollinger) buy weakness; trend/curvature rules buy strength.")


def _price_panel(view, symbol, cfg):
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.46, 0.18, 0.18, 0.18],
        subplot_titles=("Price · EMA · Bollinger", "Composite score", "RSI", "Curvature (z-scores)"),
    )
    # Price
    fig.add_trace(go.Candlestick(
        x=view["dt"], open=view["open"], high=view["high"], low=view["low"], close=view["close"],
        name="price", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=view["dt"], y=view["ema_fast"], name="EMA fast",
                             line=dict(width=1, color="#2684ff")), row=1, col=1)
    fig.add_trace(go.Scatter(x=view["dt"], y=view["ema_slow"], name="EMA slow",
                             line=dict(width=1, color="#ff8b00")), row=1, col=1)
    fig.add_trace(go.Scatter(x=view["dt"], y=view["bb_upper"], name="BB upper",
                             line=dict(width=0.7, color="rgba(140,149,165,0.5)")), row=1, col=1)
    fig.add_trace(go.Scatter(x=view["dt"], y=view["bb_lower"], name="BB lower", fill="tonexty",
                             fillcolor="rgba(140,149,165,0.08)",
                             line=dict(width=0.7, color="rgba(140,149,165,0.5)")), row=1, col=1)
    # Buy/Sell markers from flags
    buys = view[view["flag"].isin(["BUY", "STRONG BUY"])]
    sells = view[view["flag"].isin(["SELL", "STRONG SELL"])]
    fig.add_trace(go.Scatter(x=buys["dt"], y=buys["low"] * 0.995, mode="markers", name="buy flag",
                             marker=dict(symbol="triangle-up", size=9, color="#0b8a3e")), row=1, col=1)
    fig.add_trace(go.Scatter(x=sells["dt"], y=sells["high"] * 1.005, mode="markers", name="sell flag",
                             marker=dict(symbol="triangle-down", size=9, color="#bf2600")), row=1, col=1)
    # Exit levels: trailing stop (chandelier) and take-profit target
    fig.add_trace(go.Scatter(x=view["dt"], y=view["trailing_stop"], name="trailing stop",
                             line=dict(width=1, color="#bf2600", dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=view["dt"], y=view["take_profit"], name="take-profit",
                             line=dict(width=1, color="#0b8a3e", dash="dot")), row=1, col=1)
    # Composite
    fig.add_trace(go.Scatter(x=view["dt"], y=view["composite"], name="composite",
                             line=dict(color="#6554c0")), row=2, col=1)
    fig.add_hline(y=cfg["signals"]["buy_threshold"], line_dash="dot", line_color="#36b37e", row=2, col=1)
    fig.add_hline(y=cfg["signals"]["sell_threshold"], line_dash="dot", line_color="#ff7452", row=2, col=1)
    # RSI
    fig.add_trace(go.Scatter(x=view["dt"], y=view["rsi"], name="RSI",
                             line=dict(color="#00a3bf")), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#ff7452", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#36b37e", row=3, col=1)
    # Curvature
    fig.add_trace(go.Scatter(x=view["dt"], y=view["vel_z"], name="velocity z",
                             line=dict(color="#2684ff")), row=4, col=1)
    fig.add_trace(go.Scatter(x=view["dt"], y=view["acc_z"], name="curvature z",
                             line=dict(color="#ff5630")), row=4, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#8b95a5", row=4, col=1)

    fig.update_layout(height=820, margin=dict(t=40, b=10, l=10, r=10), hovermode="x unified",
                      xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.04))
    fig.update_xaxes(rangeslider_visible=False)
    plot(fig, config={**PLOT_CONFIG, "scrollZoom": True})


# --------------------------------------------------------------------------- #
# View: Backtest
# --------------------------------------------------------------------------- #
def view_backtest(cfg, interval, symbols):
    st.subheader("🧮 Backtest")
    st.caption("Replays the same signal engine over history with realistic fees and "
               "**next-bar** fills (no look-ahead). Benchmarked against buy-&-hold.")
    if not symbols:
        st.info("No data loaded. Use the sidebar first.")
        return

    cset = st.columns(5)
    symbol = cset[0].selectbox("Symbol", symbols, key="bt_symbol")
    bt_cfg = cfg["backtest"]
    entry = cset[1].slider("Entry score ≥", 0, 100, int(bt_cfg["entry_score"]), 5)
    exit_ = cset[2].slider("Exit score ≤", -100, 50, int(bt_cfg["exit_score"]), 5)
    stop = cset[3].slider("Stop loss %", 0.0, 25.0, float(bt_cfg["stop_loss_pct"]), 0.5)
    fee = cset[4].slider("Fee %/side", 0.0, 0.5, float(bt_cfg["fee_pct"]), 0.01)

    df = get_df(cfg, symbol, interval)
    if len(df) < 60:
        st.warning("Need more history to backtest.")
        return

    params = {"entry_score": entry, "exit_score": exit_, "stop_loss_pct": stop, "fee_pct": fee}
    bt = engine.run_backtest(df, cfg, params)
    m = metrics.compute_metrics(bt)
    res = bt["result"].assign(dt=to_dt(bt["result"]["open_time"]))

    beat = m["excess_vs_hold_pct"]
    a, b, c, d = st.columns(4)
    a.metric("Strategy return", f"{m['total_return_pct']:.1f}%", f"{beat:+.1f}% vs hold")
    b.metric("Buy & hold", f"{m['buy_hold_return_pct']:.1f}%")
    c.metric("Max drawdown", f"{m['max_drawdown_pct']:.1f}%",
             f"hold {m['buy_hold_max_drawdown_pct']:.1f}%", delta_color="off")
    d.metric("Sharpe", f"{m['sharpe']}", f"hold {m['buy_hold_sharpe']}", delta_color="off")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res["dt"], y=res["equity"], name="Strategy",
                             line=dict(color="#6554c0", width=2)))
    fig.add_trace(go.Scatter(x=res["dt"], y=res["buy_hold"], name="Buy & hold",
                             line=dict(color="#8b95a5", width=1.5, dash="dash")))
    fig.update_layout(title="Equity curve (normalised to 1.0)", height=420,
                      margin=dict(t=50, b=10), legend=dict(orientation="h", y=1.1))
    add_time_controls(fig)
    plot(fig)

    e1, e2 = st.columns([1, 1])
    with e1:
        st.markdown("**Performance**")
        st.dataframe(pd.DataFrame({
            "metric": ["CAGR", "Sharpe", "Sortino", "Exposure", "Win rate",
                       "Profit factor", "Trades", "Avg trade"],
            # Keep this column uniformly string-typed so Arrow serialisation
            # (used by st.dataframe) does not choke on mixed float/str values.
            "value": [f"{m['cagr_pct']}%", f"{m['sharpe']}", f"{m['sortino']}",
                      f"{m['exposure_pct']}%", f"{m['win_rate_pct']}%",
                      ("n/a" if m['profit_factor'] is None else f"{m['profit_factor']}"),
                      f"{m['num_trades']}", f"{m['avg_trade_pct']}%"],
        }), hide_index=True, use_container_width=True)
    with e2:
        st.markdown("**Trades**")
        trades = bt["trades"]
        if trades.empty:
            st.info("No trades triggered with these settings.")
        else:
            t = trades.copy()
            t["entry"] = to_dt(t["entry_time"]).dt.strftime("%Y-%m-%d %H:%M")
            t["exit"] = to_dt(t["exit_time"]).dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(
                t[["entry", "exit", "return_pct", "reason"]].rename(columns={"return_pct": "return %"})
                .style.background_gradient(cmap="RdYlGn", subset=["return %"], vmin=-15, vmax=15)
                .format({"return %": "{:+.2f}"}),
                hide_index=True, use_container_width=True, height=320,
            )

    st.warning("⚠️ Backtests flatter strategies: they ignore liquidity limits, assume the rules "
               "were fixed in advance, and are easy to overfit. A good past curve is **not** a "
               "promise of future returns. Treat this as a sanity check, not a guarantee.")


# --------------------------------------------------------------------------- #
# View: Paper trading
# --------------------------------------------------------------------------- #
def _equity_df(res):
    rows = res["equity_rows"]
    df = pd.DataFrame(rows, columns=["ts", "equity", "cash", "invested", "benchmark"])
    df["dt"] = to_dt(df["ts"])
    return df


def view_paper(cfg, interval, symbols):
    st.subheader("💵 Paper trading — fictional money on the signals")
    st.caption("Allocates virtual capital across the coins by signal strength (long-only, "
               "next-bar fills, real fees), rebalanced each bar. The **adaptive** book learns "
               "from realised P&L; the **static** book uses fixed weights. Both are shown so you "
               "can judge whether the learning actually helped.")
    if not symbols:
        st.info("No data loaded. Use the sidebar first.")
        return

    c1, c2 = st.columns([1, 3])
    cash = c1.number_input("Starting fictional capital ($)", min_value=100.0,
                           max_value=10_000_000.0, value=float(cfg["paper"]["starting_cash"]),
                           step=1000.0)
    dbp = db_path(cfg)
    with st.spinner("Simulating both books over your data…"):
        static, adaptive = run_paper_both(dbp, interval, float(cash),
                                          all_fingerprint(dbp, interval))
    if static is None or adaptive is None:
        st.warning("Not enough history yet to paper-trade. Fetch or seed more data.")
        return

    sm, am = static["metrics"], adaptive["metrics"]
    bench = sm["benchmark_return_pct"]
    diff = am["total_return_pct"] - sm["total_return_pct"]

    a, b, c, d = st.columns(4)
    a.metric("Static return", f"{sm['total_return_pct']:.1f}%", f"{sm['excess_vs_benchmark_pct']:+.1f}% vs hold")
    b.metric("Adaptive (learning) return", f"{am['total_return_pct']:.1f}%",
             f"{am['excess_vs_benchmark_pct']:+.1f}% vs hold")
    c.metric("Equal-weight hold", f"{bench:.1f}%")
    d.metric("Did learning help?", f"{diff:+.2f}%",
             "adaptive − static", delta_color="normal" if diff >= 0 else "inverse")

    # Equity curves
    sdf, adf = _equity_df(static), _equity_df(adaptive)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sdf["dt"], y=sdf["equity"], name="Static",
                             line=dict(color="#2684ff", width=2)))
    fig.add_trace(go.Scatter(x=adf["dt"], y=adf["equity"], name="Adaptive (learning)",
                             line=dict(color="#6554c0", width=2)))
    fig.add_trace(go.Scatter(x=sdf["dt"], y=sdf["benchmark"], name="Equal-weight hold",
                             line=dict(color="#8b95a5", width=1.5, dash="dash")))
    fig.update_layout(title=f"Fictional equity from ${cash:,.0f}", height=420,
                      margin=dict(t=50, b=10), legend=dict(orientation="h", y=1.1))
    add_time_controls(fig)
    plot(fig)

    left, right = st.columns(2)
    with left:
        st.markdown("**Adaptive book — current allocation**")
        alloc = dict(adaptive["final_alloc"])
        alloc_rows = [{"holding": k, "value": v} for k, v in alloc.items()]
        alloc_rows.append({"holding": "CASH", "value": adaptive["final_cash"]})
        adf2 = pd.DataFrame(alloc_rows)
        adf2 = adf2[adf2["value"] > 0.01]
        if len(adf2) > 1:
            pie = go.Figure(go.Pie(labels=adf2["holding"], values=adf2["value"], hole=0.45))
            pie.update_layout(height=300, margin=dict(t=10, b=10))
            plot(pie)
        else:
            st.info("Currently all in cash (no coin met the allocation threshold).")
    with right:
        st.markdown("**What the system learned to trust**")
        wrows = adaptive["weight_rows"]
        wdf = pd.DataFrame([{"dt": to_dt(pd.Series([ts]))[0], **w} for ts, w in wrows])
        wfig = go.Figure()
        for comp in signals.COMPONENT_NAMES:
            wfig.add_trace(go.Scatter(x=wdf["dt"], y=wdf[comp], name=comp, mode="lines"))
        wfig.update_layout(height=300, margin=dict(t=10, b=10), hovermode="x unified",
                           yaxis_title="weight", legend=dict(orientation="h", y=-0.2))
        plot(wfig)
        st.caption("Each line is a rule's weight over time. The learner raises rules whose recent "
                   "votes predicted returns and lowers the ones that lost money.")

    # Side-by-side metrics
    st.markdown("**Static vs adaptive**")
    def _col(m):
        # Pre-format as strings (keeps Trades an int, avoids the float "5.0" upcast).
        return [f"{m['total_return_pct']:.2f}", f"{m['excess_vs_benchmark_pct']:.2f}",
                f"{m['sharpe']:.2f}", f"{m['max_drawdown_pct']:.2f}",
                f"{m['avg_exposure_pct']:.1f}", f"{m['num_trades']:d}", f"{m['turnover_x']:.2f}"]

    comp_df = pd.DataFrame({
        "metric": ["Total return %", "Excess vs hold %", "Sharpe", "Max drawdown %",
                   "Avg exposure %", "Trades", "Turnover x"],
        "static": _col(sm),
        "adaptive": _col(am),
    })
    st.dataframe(comp_df, hide_index=True, use_container_width=True)

    verdict = ("✅ Here, learning **beat** the static book by "
               f"{diff:+.2f}%." if diff > 0.05 else
               "❌ Here, learning **underperformed** the static book by "
               f"{diff:.2f}%." if diff < -0.05 else
               "➖ Here, learning was **about even** with the static book.")
    st.warning(f"{verdict}  This is **one historical sample**, not proof. Adapting to what "
               "recently made money is a feedback loop that can just as easily chase noise — which "
               "is exactly why both books are shown. Past results never guarantee future returns, "
               "and no real money is involved.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    cfg = get_config()
    st.title("📈 Crypto Investment Analysis")
    st.markdown(
        "<div style='background:#3b2c00;border:1px solid #8a6d00;border-radius:8px;"
        "padding:8px 14px;color:#ffd97a;font-size:0.9rem'>"
        "<b>Not financial advice.</b> This is a systematic, transparent analysis tool. It is "
        "<i>unemotional and consistent</i> — not clairvoyant. No model can reliably predict crypto "
        "prices. It never trades on your behalf. Do your own research and never risk more than you "
        "can afford to lose.</div>", unsafe_allow_html=True,
    )
    with st.expander("ℹ️  How to use this — start here", expanded=False):
        st.markdown(
            "**1. Load data** — in the sidebar click **Fetch live** (real Binance candles) "
            "or **Seed demo** (offline sample). Do this once.\n\n"
            "**2. 🔎 Scanner** — see every coin ranked from −100 (bearish) to +100 (bullish). "
            "This is your *where's the opportunity* screen.\n\n"
            "**3. 📊 Coin detail** — pick a coin for charts, the rule-by-rule *why*, and "
            "**🚪 Exit guidance** (trailing stop, take-profit, and whether to hold/trim/exit).\n\n"
            "**4. 🧮 Backtest** — see how the strategy would have done historically vs buy-&-hold.\n\n"
            "**5. 💵 Paper trading** — drop fictional money on the signals and watch it learn "
            "from the P&L (shown against a static baseline so you can judge if learning helped).\n\n"
            "*Every chart is interactive:* drag to box-zoom, scroll to zoom, use the **1w/1m/3m/All** "
            "buttons or the slider to move through time, and double-click to reset."
        )
    st.write("")

    interval, symbols = sidebar(cfg)
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔎 Scanner", "📊 Coin detail + exits", "🧮 Backtest", "💵 Paper trading"])
    with tab1:
        view_scanner(cfg, interval, symbols)
    with tab2:
        view_detail(cfg, interval, symbols)
    with tab3:
        view_backtest(cfg, interval, symbols)
    with tab4:
        view_paper(cfg, interval, symbols)


# Streamlit executes this file top-to-bottom on every interaction.
main()
