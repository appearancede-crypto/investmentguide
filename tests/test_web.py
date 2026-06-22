"""Signal Engine web-page builder tests."""
import json
import re

import pandas as pd

from crypto_tool.data import database, synthetic
from crypto_tool.web import build


def test_signal_eval_scores_calls():
    n = 60
    close = [100.0] * n
    flag = ["NEUTRAL"] * n
    flag[10] = "BUY"
    for k in range(10, 40):
        close[k] = 100.0 + (k - 9)          # rising after the BUY call
    flag[45] = "SELL"
    for k in range(40, n):
        close[k] = 130.0 + (k - 39)         # still rising after the SELL call
    ev = build.signal_eval(pd.DataFrame({"close": close, "flag": flag}), horizon=10, band=1.0)
    assert ev["call"][10] == "buy" and ev["outcome"][10] == 1      # rose -> right
    assert ev["call"][45] == "sell" and ev["outcome"][45] == -1    # rose but called sell -> wrong
    # only the first bar of a run is marked
    assert ev["call"][11] is None
    s = ev["summary"]
    assert s["hits"] >= 1 and s["misses"] >= 1
    assert s["accuracy"] is not None


def _seed(conn, cfg, n=300):
    for sym in cfg["data"]["symbols"]:
        database.upsert_ohlcv(conn, synthetic.generate_ohlcv(sym, interval=cfg["data"]["interval"], n=n))


def test_build_payload_shape(tmp_path, cfg):
    conn = database.connect(str(tmp_path / "w.db"))
    try:
        _seed(conn, cfg)
        payload = build.build_payload(conn, cfg, history=200)
        assert payload["markets"] == len(cfg["data"]["symbols"])
        assert payload["interval"] == cfg["data"]["interval"]
        assert set(payload["thresholds"]) == {"buy", "strongBuy", "sell", "strongSell"}
        sym = payload["names"][0]
        c = payload["coins"][sym]
        for k in ["o", "h", "l", "c", "emaF", "emaS", "bbUp", "bbLo", "rsi", "comp",
                  "velZ", "accZ", "flag", "t", "latest", "rationale",
                  "call", "outcome", "fwd", "eval"]:
            assert k in c, k
        assert set(["calls", "hits", "misses", "accuracy", "horizon"]).issubset(c["eval"])
        assert len(c["c"]) == 200
        assert len(c["rationale"]) == 7
        assert set(["price", "composite", "confidence", "flag", "comps"]).issubset(c["latest"])
    finally:
        conn.close()


def test_payload_is_json_safe(tmp_path, cfg):
    """No NaN/inf may reach the page (would break JSON parsing in the browser)."""
    conn = database.connect(str(tmp_path / "w.db"))
    try:
        _seed(conn, cfg)
        payload = build.build_payload(conn, cfg, history=200)
        json.dumps(payload, allow_nan=False)  # raises if any NaN/inf slipped through
    finally:
        conn.close()


def test_render_page_embeds_data(tmp_path, cfg):
    conn = database.connect(str(tmp_path / "w.db"))
    try:
        _seed(conn, cfg)
        html = build.build_page(conn, cfg, history=200)
        assert "/*__SIGNAL_DATA__*/" not in html        # marker replaced
        assert "window.__SIGNAL_DATA__" in html
        assert html.count("</script>") == 1             # JSON didn't inject a closing tag
        m = re.search(r"window\.__SIGNAL_DATA__\s*=\s*(\{.*?\});\n", html, re.S)
        assert m is not None
        data = json.loads(m.group(1))
        assert data["markets"] >= 1
    finally:
        conn.close()
