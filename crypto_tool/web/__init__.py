"""Standalone 'Signal Engine' web UI (design-handoff recreation).

A pixel-faithful HTML/CSS/JS recreation of the Signal Engine design, wired to the
**real** crypto_tool engine: the page is rendered with live signal/indicator data
computed by ``crypto_tool.analysis`` over the candles in the database. Served as a
self-contained file (and optionally over HTTP). Analysis only — not financial advice.
"""
