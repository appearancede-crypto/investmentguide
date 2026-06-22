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
_page = {"ts": 0.0, "body": None}      # type: Dict[str, Any]
_coins: Dict[str, tuple] = {}
_page_lock = threading.Lock()
_build_lock = threading.Lock()
_coin_lock = threading.Lock()


def _get_page(config_path) -> bytes:
    now = time.time()
    with _page_lock:
        if _page["body"] is not None and now - _page["ts"] < _PAGE_TTL:
            return _page["body"]
    # Serialize rebuilds so a burst of cache-miss requests triggers only one.
    with _build_lock:
        now = time.time()
        if _page["body"] is not None and now - _page["ts"] < _PAGE_TTL:
            return _page["body"]
        cfg = load_config(config_path)
        conn = database.connect(resolve_db_path(cfg))
        try:
            body = build.build_page(conn, cfg).encode("utf-8")
        finally:
            conn.close()
        with _page_lock:
            _page["ts"] = time.time()
            _page["body"] = body
        return body


def _bust_page_cache():
    with _page_lock:
        _page["body"] = None


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
            if path not in _ROUTES:
                self._send(404, b"Not found", "text/plain")
                return
            try:
                self._send(200, _get_page(config_path), "text/html; charset=utf-8")
            except Exception as exc:  # noqa: BLE001 — surface errors in the browser
                self._send(500, f"Error building page: {exc}".encode("utf-8"), "text/plain")

        def _serve_coin(self, config_path):
            symbol = (parse_qs(urlparse(self.path).query).get("symbol") or [""])[0].upper()
            if not _SYMBOL_RE.match(symbol):
                self._send(200, b'{"ok":false,"error":"invalid symbol"}', "application/json")
                return
            try:
                self._send(200, _get_coin(config_path, symbol), "application/json")
            except Exception as exc:  # noqa: BLE001
                self._send(200, json.dumps({"ok": False, "error": str(exc)}).encode("utf-8"),
                           "application/json")

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
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


def _refresh_loop(cfg: Dict[str, Any], minutes: int, ingest_limit: Optional[int]):
    while True:
        time.sleep(minutes * 60)
        try:
            conn = database.connect(resolve_db_path(cfg))
            try:
                ingest.ingest_all(conn, _with_limit(cfg, ingest_limit))
            finally:
                conn.close()
            _bust_page_cache()
            print("Refreshed candle data.")
        except Exception as exc:  # noqa: BLE001
            print(f"Background refresh failed (will retry): {exc}")


def serve(cfg: Dict[str, Any], config_path: Optional[str] = None,
          port: Optional[int] = None, open_browser: bool = True,
          ensure_data: bool = False, refresh_min: int = 0,
          ingest_limit: Optional[int] = None) -> None:
    port = int(port or os.environ.get("PORT") or cfg.get("web", {}).get("port", 8787))

    if ensure_data:
        _ensure_data(cfg, ingest_limit)
    if refresh_min and refresh_min > 0:
        threading.Thread(target=_refresh_loop, args=(cfg, refresh_min, ingest_limit),
                         daemon=True).start()
        print(f"Background data refresh every {refresh_min} min.")

    httpd = ThreadingHTTPServer(("0.0.0.0", port), _make_handler(config_path))
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
