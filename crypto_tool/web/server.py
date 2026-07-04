"""HTTP server for the Signal Engine page (stdlib only — no extra dependencies).

Local use and small public deployments share this server. For public hosting it
adds the things traffic needs but a localhost run doesn't:

  * a short-lived **page cache** so concurrent visitors don't each trigger a full
    24-coin re-analysis (builds are serialized; the rest are served from cache);
  * a per-symbol **deep-dive cache**;
  * reads ``$PORT`` (set by Render/Railway/Fly) and binds ``0.0.0.0``;
  * optional **boot-time ingest** (so a fresh container has candles) and a
    **background refresh** thread to keep the data current.

Read-only and analysis-only. It never trades. Not financial advice.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

from ..config import load_config, resolve_db_path
from ..data import database, ingest
from . import build

_ROUTES = {"/", "/index.html", "/signal-engine"}
_SYMBOL_RE = re.compile(r"^[A-Z0-9]{2,20}$")

# Caches (the server is multi-threaded; guard shared state).
_PAGE_TTL = float(os.environ.get("PAGE_CACHE_TTL", "45"))
_COIN_TTL = float(os.environ.get("COIN_CACHE_TTL", "120"))
_COIN_MAX = int(os.environ.get("COIN_CACHE_MAX", "48"))   # bound deep-dive cache RAM
_page = {"ts": 0.0, "body": None, "gz": None, "building": False}   # type: Dict[str, Any]
_SCOUT_MIN = {"v": 0}                  # active background-sweep cadence (0 = off)
_BOOT = {"done": False}                # boot ingest finished? gates request-side rebuilds

# Shown while the first data sync / first page build is still running. The
# port binds immediately at boot (so the host's health check and proxy are
# happy) and this page refreshes itself until the real one is ready.
_WARMING_HTML = ("""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Signal Engine — warming up</title></head>
<body style="margin:0;min-height:100vh;display:grid;place-items:center;background:#0b0b0c;color:#f2f0ea;font-family:'Helvetica Neue',Arial,sans-serif">
<div style="text-align:center;max-width:440px;padding:24px">
<div style="width:34px;height:34px;border:2px solid #f2f0ea;margin:0 auto 22px;display:grid;place-items:center">
<div style="width:13px;height:13px;background:#ff453a;box-shadow:0 0 14px #ff453a"></div></div>
<div style="font-size:15px;font-weight:600;letter-spacing:.06em">SIGNAL ENGINE IS WARMING UP</div>
<div style="font-size:13px;color:#86857f;margin-top:14px;line-height:1.6">Pulling fresh candles from Binance and crunching all tracked markets.
This page refreshes itself — the full app appears in a minute or two.
(Free hosting sleeps when idle; the first visitor wakes it.)</div>
</div></body></html>""").encode("utf-8")
_coins: Dict[str, tuple] = {}
_page_lock = threading.Lock()
_build_lock = threading.Lock()
_coin_lock = threading.Lock()


def _build_page_now(config_path) -> None:
    """Build and cache the page. Heavy (tens of seconds on small hosts) —
    call ONLY from background threads, never from a request handler."""
    with _build_lock:                       # serialize concurrent rebuilds
        cfg = load_config(config_path)
        cfg.setdefault("scout", {})["active_min"] = _SCOUT_MIN["v"]
        conn = database.connect(resolve_db_path(cfg))
        try:
            body = build.build_page(conn, cfg).encode("utf-8")
        finally:
            conn.close()
        gz = gzip.compress(body, 6)          # compress once; every visitor benefits
        with _page_lock:
            _page["ts"] = time.time()
            _page["body"] = body
            _page["gz"] = gz


def _kick_rebuild(config_path) -> None:
    """Start a background page rebuild unless one is already running."""
    with _page_lock:
        if _page["building"]:
            return
        _page["building"] = True

    def run():
        try:
            _build_page_now(config_path)
        except Exception as exc:  # noqa: BLE001 — keep serving the stale page
            print(f"Page rebuild failed (still serving previous page): {exc}")
        finally:
            with _page_lock:
                _page["building"] = False

    threading.Thread(target=run, daemon=True).start()


def _has_data(config_path) -> bool:
    try:
        cfg = load_config(config_path)
        conn = database.connect(resolve_db_path(cfg))
        try:
            return bool(database.list_symbols(conn, cfg["data"]["interval"]))
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        return False


def _get_coin(config_path, symbol: str) -> bytes:
    now = time.time()
    with _coin_lock:
        hit = _coins.get(symbol)
        if hit and now - hit[0] < _COIN_TTL:
            return hit[1]
    cfg = load_config(config_path)
    conn = database.connect(resolve_db_path(cfg))
    try:
        coin = build.build_coin_payload(conn, cfg, symbol)
    finally:
        conn.close()
    body = json.dumps({"ok": True, "symbol": symbol, "coin": coin}, allow_nan=False).encode("utf-8")
    with _coin_lock:
        _coins[symbol] = (time.time(), body)
        if len(_coins) > _COIN_MAX:                  # evict oldest to bound RAM
            for k, _ in sorted(_coins.items(), key=lambda kv: kv[1][0])[:len(_coins) - _COIN_MAX]:
                _coins.pop(k, None)
    return body


def _make_handler(config_path: Optional[str]):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):   # silence default stderr logging
            pass

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/health", "/_stcore/health"):
                self._send(200, b"ok", "text/plain")
                return
            if path == "/api/coin":
                self._serve_coin(config_path)
                return
            if path == "/api/scout":
                self._serve_scout(config_path)
                return
            if path not in _ROUTES:
                self._send(404, b"Not found", "text/plain")
                return
            try:
                # Stale-while-revalidate: ALWAYS answer instantly from cache.
                # A stale (or missing) page only ever triggers a BACKGROUND
                # rebuild — a request must never wait minutes on a small host.
                with _page_lock:
                    body, gz, ts = _page["body"], _page["gz"], _page["ts"]
                if body is None:
                    # Only kick a build once boot ingest is done — building
                    # mid-ingest would publish a page with half the coins.
                    if _BOOT["done"] and _has_data(config_path):
                        _kick_rebuild(config_path)
                    self._send(200, _WARMING_HTML, "text/html; charset=utf-8")
                    return
                if time.time() - ts > _PAGE_TTL:
                    _kick_rebuild(config_path)
                use_gz = gz is not None and self._gzip_ok()
                self._send(200, gz if use_gz else body, "text/html; charset=utf-8", gz=use_gz)
            except Exception as exc:  # noqa: BLE001 — surface errors in the browser
                self._send(500, f"Error building page: {exc}".encode("utf-8"), "text/plain")

        def _serve_scout(self, config_path):
            try:
                cfg = load_config(config_path)
                conn = database.connect(resolve_db_path(cfg))
                try:
                    snap = database.load_scout(conn)
                finally:
                    conn.close()
                self._send_json(json.dumps({"ok": True, "scout": snap},
                                           allow_nan=False).encode("utf-8"))
            except Exception as exc:  # noqa: BLE001
                self._send(200, json.dumps({"ok": False, "error": str(exc)}).encode("utf-8"),
                           "application/json")

        def _serve_coin(self, config_path):
            symbol = (parse_qs(urlparse(self.path).query).get("symbol") or [""])[0].upper()
            if not _SYMBOL_RE.match(symbol):
                self._send(200, b'{"ok":false,"error":"invalid symbol"}', "application/json")
                return
            try:
                self._send_json(_get_coin(config_path, symbol))
            except Exception as exc:  # noqa: BLE001
                self._send(200, json.dumps({"ok": False, "error": str(exc)}).encode("utf-8"),
                           "application/json")

        def _gzip_ok(self):
            return "gzip" in (self.headers.get("Accept-Encoding") or "")

        def _send_json(self, body):
            if self._gzip_ok() and len(body) > 2048:
                self._send(200, gzip.compress(body, 6), "application/json", gz=True)
            else:
                self._send(200, body, "application/json")

        def _send(self, code, body, ctype, gz=False):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            if gz:
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


# --------------------------------------------------------------------------- #
# Data lifecycle (for hosted deployments)
# --------------------------------------------------------------------------- #
def _with_limit(cfg: Dict[str, Any], ingest_limit: Optional[int]) -> Dict[str, Any]:
    if ingest_limit:
        cfg = {**cfg, "data": {**cfg["data"], "history_limit": int(ingest_limit)}}
    return cfg


def _ensure_data(cfg: Dict[str, Any], ingest_limit: Optional[int]):
    """If the DB has no candles, ingest from Binance (seed synthetic if blocked)."""
    conn = database.connect(resolve_db_path(cfg))
    try:
        if database.list_symbols(conn, cfg["data"]["interval"]):
            return
        print("No candle data yet — ingesting from Binance on startup …")
        results = ingest.ingest_all(conn, _with_limit(cfg, ingest_limit))
        if not sum(r["ok"] for r in results):
            print("Live ingest unavailable (region/network) — seeding synthetic demo data.")
            ingest.seed_demo(conn, cfg)
    finally:
        conn.close()


def _scout_loop(cfg: Dict[str, Any], config_path, minutes: int):
    """Background whole-exchange sweep. Waits out the boot rush first (tiny
    hosts need their CPU for the initial ingest + page build), then sweeps
    whenever the stored snapshot is missing/stale."""
    from ..analysis import scout
    time.sleep(120)                                 # let boot ingest/build win the CPU
    while True:
        try:
            conn = database.connect(resolve_db_path(cfg))
            try:
                snap = database.load_scout(conn)
                stale = (snap is None
                         or time.time() * 1000 - snap.get("asOfMs", 0) > minutes * 60_000)
                if stale:
                    print("Scout sweep starting (whole-exchange scan) …")
                    snap = scout.run_and_save(conn, cfg)
                    print(f"Scout sweep done: {snap['scored']} pairs scored, "
                          f"kept top {len(snap['rows'])}.")
                    _kick_rebuild(config_path)      # fold it into the page off-request
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001
            print(f"Scout sweep failed (will retry): {exc}")
        time.sleep(max(5.0, minutes * 60 / 10.0))   # re-check staleness often; sweep when due


def _refresh_loop(cfg: Dict[str, Any], config_path, minutes: int, ingest_limit: Optional[int]):
    while True:
        time.sleep(minutes * 60)
        try:
            conn = database.connect(resolve_db_path(cfg))
            try:
                ingest.ingest_all(conn, _with_limit(cfg, ingest_limit))
            finally:
                conn.close()
            print("Refreshed candle data.")
            _kick_rebuild(config_path)      # rebuild off-request with the fresh candles
        except Exception as exc:  # noqa: BLE001
            print(f"Background refresh failed (will retry): {exc}")


def serve(cfg: Dict[str, Any], config_path: Optional[str] = None,
          port: Optional[int] = None, open_browser: bool = True,
          ensure_data: bool = False, refresh_min: int = 0,
          ingest_limit: Optional[int] = None, scout_min: int = 0) -> None:
    port = int(port or os.environ.get("PORT") or cfg.get("web", {}).get("port", 8787))

    # Bind the port FIRST: hosted proxies (and their "application loading"
    # splash screens) need a listening socket within seconds. All heavy work —
    # boot ingest, the first page build, refresh/scout loops — happens in
    # background threads while visitors see the self-refreshing warming page.
    if scout_min and scout_min > 0:
        _SCOUT_MIN["v"] = int(scout_min)
    httpd = ThreadingHTTPServer(("0.0.0.0", port), _make_handler(config_path))

    def _boot():
        try:
            if ensure_data:
                try:
                    _ensure_data(cfg, ingest_limit)
                except Exception as exc:  # noqa: BLE001
                    print(f"Boot ingest failed (serving whatever data exists): {exc}")
        finally:
            _BOOT["done"] = True
        if _has_data(config_path):
            print("Building the page (pre-warm) …")
            try:
                # Direct (not _kick_rebuild): waits out any in-flight build and
                # guarantees a page built from the COMPLETE boot dataset.
                _build_page_now(config_path)
            except Exception as exc:  # noqa: BLE001
                print(f"Pre-warm page build failed: {exc}")

    threading.Thread(target=_boot, daemon=True).start()
    if refresh_min and refresh_min > 0:
        threading.Thread(target=_refresh_loop,
                         args=(cfg, config_path, refresh_min, ingest_limit),
                         daemon=True).start()
        print(f"Background data refresh every {refresh_min} min.")
    if scout_min and scout_min > 0:
        threading.Thread(target=_scout_loop, args=(cfg, config_path, scout_min),
                         daemon=True).start()
        print(f"Background whole-exchange scout sweep every {scout_min} min.")

    print(f"Signal Engine serving on 0.0.0.0:{port}  ·  page cache {int(_PAGE_TTL)}s   (Ctrl-C to stop)")
    if open_browser:
        try:
            webbrowser.open(f"http://localhost:{port}")
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Signal Engine server.")
    finally:
        httpd.server_close()
