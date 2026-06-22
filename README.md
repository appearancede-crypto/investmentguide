# 📈 Crypto Investment Analysis Tool

A transparent, rule-based engine that analyses crypto markets from the **shape of
price history** — its *curvature* (velocity + acceleration) and changes — combined
with classic technical indicators, and flags potential opportunities across every
coin you track. It is linked to a live SQLite database and refreshes from the
Binance public API.

> ### ⚠️ Read this first — honest expectations
> **This is not financial advice, and it is not a money printer.** No tool — this
> one included — can reliably predict crypto prices. Strategies fit to past
> "curvatures" overfit easily and can fail on new data.
>
> What it *can* do better than a human is be **systematic, unemotional, tireless
> and complete**: it applies the exact same rules to hundreds of coins, 24/7, with
> no FOMO or panic, and it **shows its reasoning** for every signal. That is the
> only sense in which it is "wiser than humans" — consistency, not clairvoyance.
>
> It performs **analysis only**. It never places an order, moves funds, or connects
> to your exchange account. Never risk more than you can afford to lose.

---

## What it does

| Capability | How |
|---|---|
| **Live data → active DB** | Pulls OHLCV candles from Binance's public API (no key, no trading access) into a local SQLite database, idempotently. |
| **Curvature analysis** | Smooths price, then reads its 1st derivative (velocity / *change*) and 2nd derivative (*curvature* / acceleration) to spot bends like decelerating sell-offs (bottoming) before they show in price. |
| **Technical indicators** | RSI, MACD, Bollinger Bands, SMA/EMA trend, ATR, OBV, momentum — all causal (no look-ahead). |
| **Signal engine** | Blends the rules into one explainable composite score (−100…+100) plus a separate **confidence** based on rule agreement, signal strength, volume and data depth. Every signal is broken down rule-by-rule. |
| **Opportunity scanner** | Ranks all tracked coins so the strongest setups float to the top. |
| **Honest backtester** | Replays the engine over history with realistic fees and **next-bar fills (no look-ahead)**, benchmarked against buy-&-hold. |
| **Exit points** | For any coin you hold: a trailing **chandelier stop**, an ATR **take-profit** target, nearest support/resistance, reward:risk, and a clear **EXIT / TRIM / HOLD** call with reasons. |
| **Paper trading + learning** | A fictional-money portfolio that allocates virtual capital across the coins by signal, tracks P&L, and *learns from the money made or lost* by tilting rule weights toward what's been profitable — shown side-by-side with a static baseline so you can see if it actually helped. |
| **Wide history** | Pages backward through Binance (1000/request) to assemble thousands of candles per coin — more regimes for the engine, backtester and learner to work from. |
| **Dashboard** | A Streamlit web UI: scanner, per-coin charts + exits, interactive backtests, and the paper-trading desk. Every chart zooms (scroll/drag), pans, and has 1w/1m/3m/All range buttons. |

## Quick start

```powershell
# 1. install dependencies
pip install -r requirements.txt

# 2. get data — EITHER live from Binance ...
python -m crypto_tool.cli ingest
#    ... OR, if Binance is blocked in your region / you're offline:
python -m crypto_tool.cli seed-demo

# 3. see the ranked signals in the terminal
python -m crypto_tool.cli scan

# 4. launch the dashboard
./run_dashboard.ps1
#    (or:  streamlit run crypto_tool/app/streamlit_app.py )
```

### CLI reference

```
python -m crypto_tool.cli ingest            # fetch live Binance candles
python -m crypto_tool.cli seed-demo [--bars N]   # offline synthetic data
python -m crypto_tool.cli scan [--json]     # ranked latest-signal table
python -m crypto_tool.cli backtest BTCUSDT [--json]
python -m crypto_tool.cli exits [SYMBOL]    # exit guidance (all coins, or one)
python -m crypto_tool.cli coverage          # what's stored in the DB

# Paper trading with fictional money
python -m crypto_tool.cli paper-open --cash 10000 --learn   # create an adaptive account
python -m crypto_tool.cli paper-run                         # simulate it over your data
python -m crypto_tool.cli paper-status                      # equity, P&L, holdings, learned weights
python -m crypto_tool.cli paper-compare                     # learned vs static, side by side
```

## Finding the right exit points

Entries are only half the job. For any coin you hold, `exits` answers *when to get out*:

```powershell
python -m crypto_tool.cli exits          # all coins, "act now" first
python -m crypto_tool.cli exits BTCUSDT  # one coin, full detail
```

It computes a **trailing chandelier stop** (recent high − 3×ATR, which ratchets up
as price rises), an **ATR take-profit** target, the nearest swing low/resistance, the
implied **reward:risk**, and a clear call — **EXIT**, **TRIM / TIGHTEN STOP**, or
**HOLD** — with reasons (momentum rolled over, overbought, trailing stop breached,
rally fading, bearish MACD…). In the dashboard's **Coin detail** tab the stop and
target are drawn right on the price chart.

## Widest possible data range

`data.history_limit` in `config.yaml` is the number of candles per coin. Binance
serves 1000 per request, so the client **pages backward** to assemble far more —
the default 5000 (~7 months of hourly bars) gives the engine, backtester and learner
several market regimes to work from. Raise it for an even wider window (the fetch
just takes a little longer).

## Paper trading & "learning from the money made or lost"

You can drop fictional money on the tool's predictions and watch it work:

```powershell
python -m crypto_tool.cli paper-open --cash 10000 --learn
python -m crypto_tool.cli paper-run
```

Each bar, capital is allocated across the coins in proportion to how bullish (and
confident) the signals are — long-only, capped per coin, rest in cash — with real
fees and **next-bar fills (no look-ahead)**. P&L is tracked against an equal-weight
buy-&-hold benchmark.

**How it "learns":** every `retrain_every` bars the system scores each rule by its
recent realised profitability (the average of *its vote × the next bar's return*),
then tilts the weights toward the rules that have been *making money* and away from
those *losing money* — walk-forward, using only past data.

**The honest part:** this is a feedback loop, **not foresight**. Chasing what
recently worked can just as easily chase noise. So the tool always runs the
**adaptive** book against a **static** baseline and tells you, in dollars, whether
learning actually helped:

```
python -m crypto_tool.cli paper-compare
#   ...
#   Learning HURT here: adaptive − static = -2.24% (one sample — not proof either way).
```

That "learning hurt" message is a feature, not a bug — the tool measures the value
of adaptation instead of assuming it. No real money is ever involved.

## Configuration

Everything is driven by [`config.yaml`](config.yaml) — coins, timeframe, indicator
periods, **signal weights**, and backtest rules (fees, entry/exit scores, stops).
No code changes needed to retune the strategy. Defaults live in
`crypto_tool/config.py` and are used if the YAML is missing.

```yaml
signals:
  weights:
    curvature: 1.2      # bump this to lean harder on the velocity/acceleration read
    trend: 1.0
    macd: 1.0
    ...
  buy_threshold: 35
```

## How the signal is built

Each rule casts a vote in **[−1, +1]** (bearish…bullish):

- **Trend** (follow): EMA-fast vs EMA-slow, price vs EMA, 50/200 cross.
- **MACD** (follow): histogram magnitude + signal-line crossover.
- **RSI** (revert): oversold = bullish, overbought = bearish.
- **Bollinger** (revert): below the lower band = bullish.
- **Curvature** (the headline): z-scored acceleration + velocity of smoothed price.
- **Momentum** (follow): normalised rate of change.
- **Volume** (confirm): OBV slope; relative volume also scales confidence.

`composite = weighted average of votes × 100`. `confidence` is separate: it is high
only when the rules *agree*, the signal is strong, volume confirms, and there's
enough history. A loud-but-flimsy signal is shown as low-confidence on purpose.

## Architecture

```
crypto_tool/
  config.py            # YAML + defaults
  data/                # binance_client, database (SQLite), synthetic, ingest
  analysis/            # indicators, curvature, signals (the engine)
  backtest/            # engine (event-driven), metrics
  app/streamlit_app.py # dashboard
  cli.py               # ingest / seed-demo / scan / backtest / coverage
tests/                 # causality (no look-ahead), correctness, backtest integrity
```

## Tests

```powershell
pytest -q
```

The suite specifically verifies the property that makes the analysis trustworthy:
**no look-ahead** — appending future candles never changes an earlier indicator,
curvature feature, or signal, and the backtester always fills *after* the bar that
generated the signal.

## Going live with real data

`ingest` uses Binance's public endpoints, which need no API key and grant no
trading ability. If `api.binance.com` is blocked in your region, the client
automatically falls back to `data-api.binance.vision`. To track different coins or
timeframes, edit `data.symbols` and `data.interval` in `config.yaml`.

**Live vs. demo data:** `ingest` upserts incrementally (safe to run on a schedule).
`seed-demo` is a clean-slate reset — it clears the demo symbols and writes fresh
synthetic candles, so live and synthetic data never interleave in one database.
Use one source per database.

## Deploy to a public website

The **Signal Engine** web UI (`python -m crypto_tool.cli web --serve`) ships ready
to host. There's a small [`Dockerfile`](Dockerfile) (web deps only — no Streamlit),
a [`render.yaml`](render.yaml) one-click Render Blueprint, and a full guide in
[`DEPLOY.md`](DEPLOY.md) covering Render, Railway and Fly.io.

```bash
docker build -t signal-engine . && docker run --rm -p 8787:8787 signal-engine
# then open http://localhost:8787
```

The hosted server reads `$PORT`, ingests candles on boot, refreshes in the
background, and caches the page so public traffic stays light on the data APIs.
It's analysis-only with **no login** — anyone with the URL can view it.

---

*Built for systematic, transparent market analysis. Not investment advice.*
