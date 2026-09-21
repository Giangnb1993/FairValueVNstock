from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def _finite(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _interp(x: Optional[float], points: list[tuple[float, float]]) -> float:
    if x is None:
        return 0.0
    pts = sorted(points)
    if x <= pts[0][0]:
        return float(pts[0][1])
    if x >= pts[-1][0]:
        return float(pts[-1][1])
    for (a, sa), (b, sb) in zip(pts, pts[1:]):
        if a <= x <= b:
            return float(sa + (sb - sa) * (x - a) / (b - a))
    return 0.0


def _prepare_price_history(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None

    df = df.copy()
    # V8.6 raw uses time/open/high/low/close/volume. Keep aliases defensive.
    aliases = {
        "date": "time",
        "datetime": "time",
        "trading_date": "time",
        "vol": "volume",
        "total_volume": "volume",
    }
    lower = {str(c).strip().lower(): c for c in df.columns}
    for src, dst in aliases.items():
        if dst not in lower and src in lower:
            df = df.rename(columns={lower[src]: dst})
            lower = {str(c).strip().lower(): c for c in df.columns}

    required = ["open", "high", "low", "close", "volume"]
    if not all(c in lower for c in required):
        return None

    rename = {lower[c]: c for c in required}
    if "time" in lower:
        rename[lower["time"]] = "time"
    df = df.rename(columns=rename)

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.sort_values("time")

    df = df.dropna(subset=["high", "low", "close", "volume"])
    df = df[(df["close"] > 0) & (df["volume"] >= 0)]
    if len(df) < 25:
        return None
    return df.reset_index(drop=True)


def _load_price_history(ticker_dir: Path) -> pd.DataFrame | None:
    p = ticker_dir / "raw" / "price_history.csv"
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        raw = pd.read_csv(p)
    except (pd.errors.EmptyDataError, OSError, ValueError):
        return None
    return _prepare_price_history(raw)


def volume_accumulation_score_from_df(price_history: pd.DataFrame | None) -> Dict[str, Any]:
    """Score a recent base/accumulation -> breakout pattern from daily OHLCV.

    Locked V8.7 components (100 points):
      20  volume dries up inside the 20-session base
      20  up-day volume dominates down-day volume inside the base
      20  breakout volume expands vs prior 20-session average
      15  price breaks the prior 20-session base high (candidate within last 10 sessions)
      15  post-breakout retest holds with volume <= 0.8x prior-20 average
      10  breakout candle closes near its high

    This score is informational. It does not change V8.6 fair value, screening
    weights, ranking, or the Score>75 & Price/FV<0.85 selection rule.
    """
    df = _prepare_price_history(price_history)
    if df is None:
        return {
            "volume_score": None,
            "volume_signal": "NO_DATA",
            "volume_data_sessions": 0,
        }

    n = len(df)
    search_start = max(20, n - 10)
    breakout_idx = None
    breakout_level = None
    breakout_vol_ma20 = None
    base_range_pct = None

    # Search the most recent valid breakout first. Base = 20 sessions immediately before it.
    for i in range(n - 1, search_start - 1, -1):
        base = df.iloc[i - 20:i]
        if len(base) < 20:
            continue
        base_high = _finite(base["high"].max())
        base_low = _finite(base["low"].min())
        close_i = _finite(df.at[i, "close"])
        if not base_high or not base_low or close_i is None:
            continue
        rng = (base_high / base_low - 1.0) * 100.0
        # A base should be bounded; 25% keeps the rule usable across volatile VN equities.
        if rng > 25.0:
            continue
        latest_close = _finite(df.iloc[-1]["close"])
        if close_i >= base_high * 1.005 and latest_close is not None and latest_close >= base_high * 0.98:
            breakout_idx = i
            breakout_level = base_high
            breakout_vol_ma20 = _finite(base["volume"].mean())
            base_range_pct = rng
            break

    # If no breakout, evaluate the latest 20 sessions as a forming base.
    anchor = breakout_idx if breakout_idx is not None else n
    base = df.iloc[max(0, anchor - 20):anchor].copy()
    if len(base) < 20:
        return {
            "volume_score": None,
            "volume_signal": "INSUFFICIENT_DATA",
            "volume_data_sessions": int(n),
        }

    first10 = _finite(base.iloc[:10]["volume"].mean())
    last10 = _finite(base.iloc[10:]["volume"].mean())
    dryup_ratio = (last10 / first10) if first10 and last10 is not None else None
    dryup_score = _interp(dryup_ratio, [
        (0.50, 20), (0.65, 18), (0.75, 15), (0.90, 10), (1.00, 6), (1.15, 2), (1.30, 0),
    ])

    # Up/down volume uses close-to-close direction within the base.
    b = base.copy()
    b["prev_close"] = b["close"].shift(1)
    up_vol = _finite(b.loc[b["close"] > b["prev_close"], "volume"].sum()) or 0.0
    down_vol = _finite(b.loc[b["close"] < b["prev_close"], "volume"].sum()) or 0.0
    if down_vol <= 0:
        up_down_ratio = 3.0 if up_vol > 0 else None
    else:
        up_down_ratio = up_vol / down_vol
    up_down_score = _interp(up_down_ratio, [
        (0.60, 0), (0.80, 4), (1.00, 8), (1.20, 13), (1.50, 17), (2.00, 20),
    ])

    breakout_price_score = 15.0 if breakout_idx is not None else 0.0
    breakout_volume_ratio = None
    breakout_volume_score = 0.0
    close_near_high = None
    close_near_high_score = 0.0
    breakout_date = None
    retest_status = "NO_BREAKOUT"
    retest_volume_ratio = None
    retest_score = 0.0

    if breakout_idx is not None:
        row = df.iloc[breakout_idx]
        vol = _finite(row["volume"])
        if breakout_vol_ma20 and vol is not None:
            breakout_volume_ratio = vol / breakout_vol_ma20
            breakout_volume_score = _interp(breakout_volume_ratio, [
                (0.80, 0), (1.00, 3), (1.20, 8), (1.50, 15), (2.00, 20),
            ])

        hi, lo, cl = _finite(row["high"]), _finite(row["low"]), _finite(row["close"])
        if hi is not None and lo is not None and cl is not None:
            if hi > lo:
                close_near_high = (cl - lo) / (hi - lo)
            else:
                close_near_high = 1.0
            close_near_high_score = _interp(close_near_high, [
                (0.30, 0), (0.50, 3), (0.65, 6), (0.80, 10),
            ])

        if "time" in df.columns and pd.notna(row.get("time")):
            breakout_date = pd.Timestamp(row["time"]).strftime("%Y-%m-%d")

        after = df.iloc[breakout_idx + 1:].copy()
        if after.empty:
            retest_status = "NOT_YET"
        else:
            # A valid retest approaches the breakout level, holds it (2% tolerance),
            # and trades on <=80% of the pre-breakout 20-session average volume.
            approached = after[after["low"] <= float(breakout_level) * 1.03]
            if approached.empty:
                retest_status = "NOT_YET"
            else:
                valid = approached[
                    (approached["close"] >= float(breakout_level) * 0.98)
                    & (approached["volume"] <= float(breakout_vol_ma20 or 0) * 0.80)
                ] if breakout_vol_ma20 else approached.iloc[0:0]
                if not valid.empty:
                    retest_status = "HELD_LOW_VOLUME"
                    rv = _finite(valid.iloc[0]["volume"])
                    if breakout_vol_ma20 and rv is not None:
                        retest_volume_ratio = rv / breakout_vol_ma20
                    retest_score = 15.0
                elif (approached["close"] < float(breakout_level) * 0.98).any():
                    retest_status = "FAILED"
                else:
                    retest_status = "HELD_BUT_VOLUME_HIGH"

    total = min(100.0, max(0.0,
        dryup_score + up_down_score + breakout_volume_score
        + breakout_price_score + retest_score + close_near_high_score
    ))

    if breakout_idx is not None and retest_status == "HELD_LOW_VOLUME" and total >= 70:
        signal = "UPTREND_EARLY"
    elif breakout_idx is not None and total >= 60:
        signal = "BREAKOUT_CONFIRMED"
    elif breakout_idx is not None:
        signal = "BREAKOUT_WEAK"
    elif total >= 30:
        signal = "ACCUMULATION"
    else:
        signal = "NEUTRAL"

    return {
        "volume_score": round(total, 2),
        "volume_signal": signal,
        "volume_data_sessions": int(n),
        "vol_dryup_score": round(dryup_score, 2),
        "vol_up_down_score": round(up_down_score, 2),
        "vol_breakout_volume_score": round(breakout_volume_score, 2),
        "vol_breakout_price_score": round(breakout_price_score, 2),
        "vol_retest_score": round(retest_score, 2),
        "vol_close_near_high_score": round(close_near_high_score, 2),
        "vol_dryup_ratio": round(dryup_ratio, 4) if dryup_ratio is not None else None,
        "vol_up_down_ratio": round(up_down_ratio, 4) if up_down_ratio is not None else None,
        "breakout_volume_ratio": round(breakout_volume_ratio, 4) if breakout_volume_ratio is not None else None,
        "breakout_date": breakout_date,
        "breakout_level": round(float(breakout_level), 4) if breakout_level is not None else None,
        "base_range_pct": round(float(base_range_pct), 2) if base_range_pct is not None else None,
        "retest_status": retest_status,
        "retest_volume_ratio": round(retest_volume_ratio, 4) if retest_volume_ratio is not None else None,
        "breakout_close_location": round(close_near_high, 4) if close_near_high is not None else None,
    }


def volume_accumulation_score(ticker_dir: str | Path) -> Dict[str, Any]:
    """File-based wrapper used by the batch report layer."""
    ticker_dir = Path(ticker_dir)
    df = _load_price_history(ticker_dir)
    return volume_accumulation_score_from_df(df)
