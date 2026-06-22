# Deploying the Signal Engine to a public website

The Signal Engine ships as a small Docker image that serves the live page +
deep-dive API. It reads `$PORT`, binds `0.0.0.0`, ingests candles on first boot,
and refreshes them in the background — so any container host runs it as-is.

> **Before you publish:** this site has **no login** — anyone with the URL can
> view it. It's analysis-only (it never trades, holds no keys, takes no input),
> so that's fine, but keep the "not financial advice" framing front and centre.

## What's in the box
- [`Dockerfile`](Dockerfile) — builds the image (only `requirements-web.txt`, no Streamlit).
- [`render.yaml`](render.yaml) — one-click Render Blueprint.
- Server: caches the page (~45 s) and each deep-dive (~120 s) so public traffic
  doesn't re-run a 24-coin analysis on every request, and stays light on the
  Binance/CoinGecko APIs.

## Option A — Render (easiest, free tier)
1. Push this repo to GitHub.
2. On [render.com](https://render.com): **New + → Blueprint**, pick the repo. Render
   reads `render.yaml`, builds the Dockerfile, and gives you a `…onrender.com` URL.
   (Or **New + → Web Service → Docker** and skip the blueprint.)
3. Health check is `/health`; first boot ingests data (~20–40 s), then it's live.

## Option B — Railway
1. Push to GitHub. On [railway.app](https://railway.app): **New Project → Deploy from
   GitHub repo**. Railway detects the Dockerfile and injects `$PORT`. Deploy.
   (CLI alternative: `npm i -g @railway/cli && railway up`.)

## Option C — Fly.io
```bash
fly launch --no-deploy        # detects the Dockerfile; accept defaults
fly deploy
```
Fly sets `PORT=8080`; the server reads it automatically.

## Test the image locally first
```bash
docker build -t signal-engine .
docker run --rm -p 8787:8787 -e PORT=8787 signal-engine
# open http://localhost:8787
```

## Hosting notes & caveats
- **Binance geo-blocking.** Some hosts/regions block `api.binance.com`. The client
  automatically falls back to `data-api.binance.vision`. If both are blocked,
  boot ingest seeds synthetic demo data so the site still renders (clearly fake
  prices) — pick a host region where Binance market data is reachable.
- **Free-tier cold starts.** Free instances sleep when idle and have an ephemeral
  disk, so the candle DB is rebuilt on each cold start (that's what `--ensure-data`
  is for). First request after a sleep takes ~20–40 s. For an always-warm site
  with a persistent DB, use a paid instance + a mounted disk and drop `--ensure-data`.
- **Tuning.** `PAGE_CACHE_TTL` (env, default 45 s) trades freshness vs. load.
  The container ingests 1500 candles/coin for fast cold starts and refreshes every
  20 min — change the `CMD` in the `Dockerfile` (`--ingest-limit`, `--refresh-min`)
  to taste.
- **Watchlist.** Edit `data.symbols` in `config.yaml` before deploying to change
  which coins appear.
