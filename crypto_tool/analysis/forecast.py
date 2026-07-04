"""Outlook engine: project how the trend COULD unfold, from prior observations.

HOW IT WORKS (historical analogues, honestly framed)
----------------------------------------------------
"Predicting" a chart is impossible; what *is* possible is answering a humbler,
useful question: **when this coin looked like it does right now, what happened
next?** Every bar of history is described by a small feature vector — the
composite signal score, the velocity and acceleration of smoothed price
(curvature), RSI and Bollinger %B. We find the K past moments most similar to
the present one, collect the price paths that actually followed each of them,
and report their distribution:

* a **fan of quantile paths** (10/25/50/75/90th percentile) projected forward
  from the current price — "the shaded cone" on the chart;
* **checkpoint stats** ("in 62% of similar past moments, price was higher a
  day later; typical move +0.8%");
* a **calibration check**: we replay the same method at past moments spaced so
  their outcome windows do not overlap, and count how often the real outcome
  landed inside the projected 10–90% band. If that coverage is far from 80%,
  the cone is drawn too tight or too wide — and we say so instead of hiding it.

Two honesty details worth knowing:

* **Analogues are de-clustered.** Features move slowly, so the raw nearest
  neighbours of "now" are runs of consecutive bars from the same few episodes.
  We enforce a minimum time separation between accepted analogues, so K counts
  *distinct* past moments, not the same moment sampled K times.
* **The calibration replays are non-overlapping**, so the reported sample
  count is an honest count of independent trials, not overlapping windows
  dressed up as evidence.

Strictly causal: an analogue only qualifies if its entire forward window was
observed, and every input is derived from candles up to "now". This is a tally
of the past — NOT foresight, NOT financial advice.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Feature weights for the similarity metric. Each feature is pre-squashed to a
# comparable ±1-ish scale, so these express relative importance only.
_FEATURE_WEIGHTS = np.array([
    1.5,   # composite score /100 — the headline "does the setup look the same"
    1.0,   # tanh(velocity z)    — is price rising or falling, and how hard
    1.0,   # tanh(acceleration z)— is the move speeding up or bending over
    0.8,   # (RSI-50)/50         — stretched or not
    0.8,   # Bollinger %B - 0.5  — where price sits inside its recent envelope
])

QUANTS = (10, 25, 50, 75, 90)

# Fewer distinct analogues than this and we refuse to draw a cone at all.
_MIN_EPISODES = 15


# --------------------------------------------------------------------------- #
# Feature construction
# --------------------------------------------------------------------------- #
def _features(e: pd.DataFrame) -> np.ndarray:
    """(bars, 5) feature matrix on a comparable ±1 scale. NaN where not live."""
    comp = e["composite"].to_numpy(dtype=float) / 100.0
    vel = np.tanh(e["vel_z"].to_numpy(dtype=float))
    acc = np.tanh(e["acc_z"].to_numpy(dtype=float))
    rsi = (e["rsi"].to_numpy(dtype=float) - 50.0) / 50.0
    pctb = e["bb_pctb"].to_numpy(dtype=float).clip(-0.5, 1.5) - 0.5
    return np.column_stack([comp, vel, acc, rsi, pctb])


def _min_sep(horizon: int) -> int:
    """Minimum bar separation between accepted analogues (distinct episodes)."""
    return max(2, horizon // 4)


def _knn(X: np.ndarray, target: np.ndarray, candidates: np.ndarray,
         k: int, min_sep: int) -> np.ndarray:
    """De-clustered nearest neighbours: accept candidates nearest-first, but
    skip any within ``min_sep`` bars of an already-accepted one, so each
    analogue is a distinct episode rather than the same moment repeated."""
    if len(candidates) == 0:
        return candidates
    diff = (X[candidates] - target) * _FEATURE_WEIGHTS
    dist = np.sqrt((diff * diff).sum(axis=1))
    order = np.argsort(dist, kind="stable")
    blocked = np.zeros(int(X.shape[0]), dtype=bool)
    chosen: List[int] = []
    for oi in order:
        i = int(candidates[oi])
        if blocked[i]:
            continue
        chosen.append(i)
        if len(chosen) >= k:
            break
        blocked[max(0, i - min_sep + 1): i + min_sep] = True
    return np.array(chosen, dtype=int)


def _pick_k(n_candidates: int, fcfg: Dict[str, Any]) -> int:
    k = int(round(n_candidates * float(fcfg["k_frac"])))
    k = max(int(fcfg["k_min"]), min(int(fcfg["k_max"]), k))
    return min(k, n_candidates)          # never claim more analogues than exist


def _forward_clean(ok: np.ndarray, horizon: int) -> np.ndarray:
    """fwd_clean[i] is True when every close in (i, i+horizon] is usable."""
    n = len(ok)
    csum = np.concatenate([[0], np.cumsum(~ok)])
    out = np.zeros(n, dtype=bool)
    idx = np.arange(n)
    valid = idx + horizon <= n - 1
    vi = idx[valid]
    out[vi] = (csum[vi + horizon + 1] - csum[vi + 1]) == 0
    return out


# --------------------------------------------------------------------------- #
# The projection
# --------------------------------------------------------------------------- #
def project(e: pd.DataFrame, cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build the outlook block for the LAST bar of an enriched frame.

    Returns None when there is not enough comparable history to say anything
    honest (better silence than a fabricated cone).
    """
    fcfg = {**_DEFAULTS, **(cfg.get("web", {}).get("forecast") or {})}
    horizon = int(fcfg["bars"])
    # Dedup + sort: a duplicated checkpoint in user config must not double-count
    # calibration tallies (coverage could otherwise read >100%).
    checkpoints = sorted({int(c) for c in fcfg["checkpoints"] if int(c) <= horizon})
    n = len(e)
    if n < horizon + 50:
        return None

    close = e["close"].to_numpy(dtype=float)
    X = _features(e)
    now = n - 1
    target = X[now]
    if not np.all(np.isfinite(target)):
        return None

    ok = np.isfinite(close) & (close > 0)
    feat_ok = np.all(np.isfinite(X), axis=1) & ok
    # An analogue must have its FULL forward window already observed (causal)
    # and clean (no NaN/zero close anywhere inside it).
    idx = np.arange(n)
    cand_ok = feat_ok & _forward_clean(ok, horizon)
    candidates = idx[cand_ok & (idx + horizon <= now)]
    if len(candidates) < int(fcfg["min_candidates"]):
        return None

    sep = _min_sep(horizon)
    nearest = _knn(X, target, candidates, _pick_k(len(candidates), fcfg), sep)
    k = int(len(nearest))
    if k < _MIN_EPISODES:
        return None

    # Relative forward paths of the analogues: (k, horizon).
    steps = np.arange(1, horizon + 1)
    paths = close[nearest[:, None] + steps[None, :]] / close[nearest, None]
    if not np.isfinite(paths).all():     # belt-and-braces; mask should prevent this
        return None
    price_now = float(close[now])

    bands = {
        f"p{q}": [round(float(v) * price_now, 8)
                  for v in np.percentile(paths, q, axis=0)]
        for q in QUANTS
    }

    bar_hours = _bar_hours(e)
    cps = []
    for h in checkpoints:
        r = paths[:, h - 1] - 1.0
        cps.append({
            "bars": h,
            "hours": round(h * bar_hours, 1) if bar_hours else None,
            "prob_up": round(float((r > 0).mean()) * 100.0, 1),
            "median_pct": round(float(np.median(r)) * 100.0, 2),
            "p10_pct": round(float(np.percentile(r, 10)) * 100.0, 2),
            "p90_pct": round(float(np.percentile(r, 90)) * 100.0, 2),
        })

    calib = _calibrate(close, X, cand_ok, horizon, checkpoints or [horizon], fcfg)

    quality = _quality(k, len(candidates), calib)
    summary = _summary(cps, k, sep, bar_hours, quality)

    return {
        "horizon": horizon,
        "barHours": bar_hours,
        "analogues": k,
        "candidates": int(len(candidates)),
        "minSepBars": sep,
        "quality": quality,
        "asofPrice": price_now,
        "bands": bands,
        "checkpoints": cps,
        "calibration": calib,
        "summary": summary,
    }


def _bar_hours(e: pd.DataFrame) -> Optional[float]:
    ts = e["open_time"].to_numpy(dtype="int64")
    if len(ts) < 3:
        return None
    ms = float(np.median(np.diff(ts)))
    return round(ms / 3_600_000.0, 4) if ms > 0 else None


# --------------------------------------------------------------------------- #
# Calibration — replay the method on this coin's own past and score the cone
# --------------------------------------------------------------------------- #
def _band_at(t: int, close: np.ndarray, X: np.ndarray, cand_ok: np.ndarray,
             horizon: int, h_star: int, fcfg: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """The 10–90% band for the ``h_star``-step move, built exactly as it would
    have been at bar ``t`` — candidates must have fully resolved by ``t``.
    (Kept as the single-horizon form: unit tests poison the future against it.)"""
    bands = _bands_at(t, close, X, cand_ok, horizon, [h_star], fcfg)
    return None if bands is None else bands[h_star][:2]


def _bands_at(t: int, close: np.ndarray, X: np.ndarray, cand_ok: np.ndarray,
              horizon: int, hs: List[int], fcfg: Dict[str, Any]
              ) -> Optional[Dict[int, Tuple[float, float, float]]]:
    """(lo, hi, median) of the analogue moves at each step in ``hs``, built
    exactly as ``project`` would have at bar ``t`` (one KNN, many horizons)."""
    idx = np.arange(len(close))
    cand = idx[cand_ok & (idx + horizon <= t)]
    if len(cand) < int(fcfg["min_candidates"]):
        return None
    near = _knn(X, X[t], cand, _pick_k(len(cand), fcfg), _min_sep(horizon))
    if len(near) < _MIN_EPISODES:
        return None
    out: Dict[int, Tuple[float, float, float]] = {}
    for h in hs:
        r = close[near + h] / close[near] - 1.0
        out[h] = (float(np.percentile(r, 10)), float(np.percentile(r, 90)),
                  float(np.median(r)))
    return out


def _calibrate(close: np.ndarray, X: np.ndarray, cand_ok: np.ndarray,
               horizon: int, checkpoints: List[int],
               fcfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """At up to ``calib_samples`` past bars — spaced at least ``h_star`` (the
    longest checkpoint) apart so outcome windows do NOT overlap — rebuild the
    bands exactly as ``project`` would have at that time, and measure at EVERY
    checkpoint: how often reality landed inside the 10–90% band (target 80%),
    and the typical miss — the median absolute gap between the projected
    middle path and what actually happened. The miss is the number a quick
    flipper actually needs: how wrong the middle of the cone tends to be."""
    n = len(close)
    samples = int(fcfg["calib_samples"])
    h_star = max(checkpoints) if checkpoints else horizon

    # A bar t is testable when its bands can be built and its outcome at
    # t+h_star was observed; keep only non-overlapping outcome windows.
    testable: List[int] = []
    for t in np.arange(n):
        if t + h_star >= n:
            break
        if not cand_ok[t]:                        # its own features must be live
            continue
        if testable and t - testable[-1] < h_star:
            continue
        n_cand = int(np.count_nonzero(cand_ok[: max(0, t - horizon + 1)]))
        if n_cand >= int(fcfg["min_candidates"]):
            testable.append(int(t))
    if len(testable) < 10:
        return None
    if len(testable) > samples:
        sel = np.linspace(0, len(testable) - 1, samples).round().astype(int)
        testable = [testable[i] for i in np.unique(sel)]

    scored = 0
    inside = {h: 0 for h in checkpoints}
    misses = {h: [] for h in checkpoints}         # |realized − projected median|
    for t in testable:
        bands = _bands_at(t, close, X, cand_ok, horizon, checkpoints, fcfg)
        if bands is None:
            continue
        scored += 1
        for h in checkpoints:
            lo, hi, med = bands[h]
            realized = close[t + h] / close[t] - 1.0
            if lo <= realized <= hi:
                inside[h] += 1
            misses[h].append(abs(realized - med))
    if scored < 10:
        return None
    h_primary = h_star
    return {
        "samples": scored,
        "coverage_pct": round(inside[h_primary] / scored * 100.0, 1),
        "target_pct": 80,
        "horizon_bars": h_primary,
        "horizons": [
            {"bars": h,
             "coverage_pct": round(inside[h] / scored * 100.0, 1),
             "typical_miss_pct": round(float(np.median(misses[h])) * 100.0, 2)}
            for h in checkpoints
        ],
    }


def _quality(k: int, candidates: int, calib: Optional[Dict[str, Any]]) -> str:
    """Comparison-quality tier: how deep the look-alike history is and how well
    the band fitted this coin's own past. It says NOTHING about the future —
    a HIGH tier means the comparison is well-fed, not that it will be right."""
    if candidates < 150 or k < 30:
        return "LOW"
    if calib is None:
        return "MEDIUM"                      # plenty of history, but unverified fit
    # Tolerance is sample-aware: with few (independent) trials, a coverage far
    # from 80% is still within noise, so demand ~2 binomial SDs before demoting.
    tol = max(12.0, 200.0 * float(np.sqrt(0.8 * 0.2 / calib["samples"])))
    if abs(calib["coverage_pct"] - calib["target_pct"]) > tol:
        return "MEDIUM"
    return "HIGH"


# --------------------------------------------------------------------------- #
# Plain-English summary
# --------------------------------------------------------------------------- #
def _hours_word(hours: Optional[float], bars: int) -> str:
    if hours is None:
        return f"{bars} bars"
    if hours < 48:
        v = int(round(hours))
        return f"{v} hour{'s' if v != 1 else ''}"
    v = int(round(hours / 24.0))
    return f"{v} day{'s' if v != 1 else ''}"


def _summary(cps: List[Dict[str, Any]], k: int, sep: int,
             bar_hours: Optional[float], quality: str) -> str:
    if not cps:
        return ""
    cp = next((c for c in cps if c["bars"] == 24), cps[0])
    when = _hours_word(cp["hours"], cp["bars"])
    p = cp["prob_up"]
    if p >= 60:
        lean = f"price was HIGHER {when} later in {p:.0f}% of them"
    elif p <= 40:
        lean = f"price was LOWER {when} later in {100 - p:.0f}% of them"
    else:
        lean = f"it was close to a coin flip {when} later ({p:.0f}% up)"
    med = cp["median_pct"]
    qual_note = {
        "LOW": " Comparable history is thin here, so treat this outlook as weak.",
        "MEDIUM": " The band has not been verified as a tight fit on this coin's past, so lean on it lightly.",
        "HIGH": " Even with plenty of look-alike history, this only describes the past.",
    }[quality]
    sep_word = _hours_word(round(sep * bar_hours, 1) if bar_hours else None, sep)
    return (
        f"Looking at the {k} separate past moments (each at least {sep_word} apart) "
        f"when this chart looked most like it does right now — similar score, trend, "
        f"curvature and stretch — {lean}; the typical outcome was {med:+.1f}%, and "
        f"the middle 80% of outcomes fell between {cp['p10_pct']:+.1f}% and "
        f"{cp['p90_pct']:+.1f}%.{qual_note} History rhymes, it does not repeat — "
        f"this is a tally of the past, not a promise about the future."
    )


_DEFAULTS: Dict[str, Any] = {
    "bars": 48,             # how far forward the cone is drawn
    "checkpoints": [4, 12, 24, 48],   # 4h serves the quick flippers honestly
    "min_candidates": 60,   # below this, refuse to draw a cone at all
    "k_frac": 0.08,         # analogues used = this fraction of candidates …
    "k_min": 25,            # … clamped to this range (and to the pool size)
    "k_max": 150,
    "calib_samples": 60,    # max non-overlapping replays for the coverage check
}
