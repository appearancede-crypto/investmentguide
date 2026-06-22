"""Price *curvature* features — the headline analysis the tool is built around.

The user asked for analysis based on "previous curvatures and changes". We make
that precise by treating a smoothed price series as a function of time and
reading its derivatives:

  * velocity      = 1st difference of smoothed price  -> the *change* (slope)
  * curvature     = 2nd difference of smoothed price  -> the *acceleration*
  * regime        = up / sideways / down, from the z-scored velocity
  * inflection    = a sign-flip in curvature (a bend in the trend)

Why curvature matters: a falling market that is *decelerating* (price still
dropping but curvature turning positive) is the classic shape of a bottom
forming — visible in the 2nd derivative before it shows up in price. Reading
that shape consistently, without hope or fear, is something a machine genuinely
does better than a human eye.

All features are causal (rolling / ewm only look backwards) and scale-invariant
(normalised by price), so they are comparable across coins of very different
nominal prices.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def _zscore(series: pd.Series, window: int) -> pd.Series:
    # Require a full window before emitting a z-score: a half-window std (as few
    # as ~25 points) of a noisy 2nd-difference series is too unstable and can
    # saturate the curvature vote on a thin, unrepresentative sample.
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    z = (series - mean) / std.replace(0.0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan)


def compute_curvature(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Append velocity / curvature / regime / inflection columns to ``df``."""
    ic = cfg["indicators"]
    span = ic["curvature_smooth"]
    out = df.copy().reset_index(drop=True)

    # Smoothed price (an EMA) — differentiating raw price is too noisy.
    sp = out["close"].ewm(span=span, adjust=False).mean()
    out["sp"] = sp

    # First and second differences, normalised by price for scale-invariance.
    velocity = sp.diff()
    accel = velocity.diff()
    out["velocity"] = velocity
    out["accel"] = accel
    out["rel_velocity"] = velocity / sp.replace(0.0, np.nan)
    out["rel_accel"] = accel / sp.replace(0.0, np.nan)

    # Z-score the normalised derivatives over a rolling window so they become
    # dimensionless "how unusual is this slope/bend" signals.
    window = max(span * 5, 50)
    out["vel_z"] = _zscore(out["rel_velocity"], window)
    out["acc_z"] = _zscore(out["rel_accel"], window)

    # Regime from z-scored velocity: clear up/down vs sideways chop.
    vel_z = out["vel_z"].fillna(0.0)
    regime = np.where(vel_z > 0.5, 1, np.where(vel_z < -0.5, -1, 0))
    out["regime"] = regime.astype("int8")

    # Inflection: curvature changed sign vs the previous bar (a bend).
    sign = np.sign(out["accel"].fillna(0.0))
    out["inflection"] = (sign != sign.shift(1)).fillna(False) & (sign != 0)
    return out
