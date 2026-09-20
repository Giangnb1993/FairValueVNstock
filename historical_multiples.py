from __future__ import annotations

import math
import re
import unicodedata
from typing import Any, Iterable

import numpy as np
import pandas as pd


def _norm(x: Any) -> str:
    if x is None:
        return ""
    s = unicodedata.normalize("NFD", str(x).strip().lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def flatten_columns(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = ["__".join(str(x) for x in tup if str(x) != "") for tup in out.columns]
    else:
        out.columns = [str(c) for c in out.columns]
    return out


def _year_from_value(x: Any) -> int | None:
    if x is None:
        return None
    s = str(x)
    m = re.search(r"(20\d{2})", s)
    if m:
        return int(m.group(1))
    try:
        ts = pd.to_datetime(x, errors="coerce")
        if pd.notna(ts):
            return int(ts.year)
    except Exception:
        pass
    return None


def timeseries_metric(
    df: pd.DataFrame | None,
    aliases: Iterable[str],
    *,
    exclude: Iterable[str] = (),
) -> pd.Series:
    """Extract year->metric from orient='time_series' output using explicit row years."""
    if df is None or len(df) == 0:
        return pd.Series(dtype='float64')
    df = flatten_columns(df)
    aliases_n = [_norm(a) for a in aliases]
    excludes_n = [_norm(a) for a in exclude]

    # Identify a year/period column. Schemas may use year, reportYear, report_year, Năm, etc.
    year_col = None
    preferred = (
        'year','fiscal_year','report_year','year_report','yearreport',
        'nam','period','report_period','fiscal_period',
        'time','date','period_end','report_date','reporting_date'
    )
    norm_cols = {_norm(c): c for c in df.columns}
    for k in preferred:
        if k in norm_cols:
            year_col = norm_cols[k]
            break
    if year_col is None:
        best = (0, None)
        for c in df.columns:
            count = 0
            for v in df[c].head(20):
                y = _year_from_value(v)
                if y is not None and 2000 <= y <= 2100:
                    count += 1
            if count > best[0]:
                best = (count, c)
        if best[0] >= max(2, min(4, len(df))):
            year_col = best[1]

    # Score metric columns: exact > token containment.
    candidates = []
    for c in df.columns:
        if c == year_col:
            continue
        nc = _norm(c)
        if any(e and e in nc for e in excludes_n):
            continue
        exact = any(nc == a for a in aliases_n)
        contains = sum(1 for a in aliases_n if a and (a in nc or nc in a))
        if exact or contains:
            numeric = pd.to_numeric(df[c], errors='coerce').notna().sum()
            score = numeric * 100 + (10000 if exact else 0) + contains
            candidates.append((score, c))
    if not candidates:
        return pd.Series(dtype='float64')
    candidates.sort(reverse=True)
    metric_col = candidates[0][1]

    out = {}
    for idx, row in df.iterrows():
        y = _year_from_value(row.get(year_col)) if year_col is not None else None
        if y is None:
            y = _year_from_value(idx)
        v = _safe_float(row.get(metric_col))
        if y is not None and v is not None:
            out[str(y)] = v
    return pd.Series(dict(sorted(out.items(), key=lambda kv:int(kv[0]), reverse=True)), dtype='float64')


def normalize_price_history(history: pd.DataFrame | None, current_price: float | None) -> tuple[pd.DataFrame | None, float, str]:
    """Normalize OHLCV close units to the current quote scale (e.g. 73.6 -> 73,600)."""
    if history is None or history.empty:
        return history, 1.0, 'no_history'
    h = flatten_columns(history)
    norms = {_norm(c): c for c in h.columns}
    time_col = next((norms[k] for k in ('time','date','trading_date') if k in norms), None)
    close_col = next((norms[k] for k in ('close','close_price') if k in norms), None)
    if time_col is None or close_col is None:
        return h, 1.0, 'missing_time_or_close'
    h = h.copy()
    h[time_col] = pd.to_datetime(h[time_col], errors='coerce')
    h[close_col] = pd.to_numeric(h[close_col], errors='coerce')
    h = h.dropna(subset=[time_col, close_col]).sort_values(time_col)
    if h.empty or not current_price:
        return h, 1.0, 'no_scale_check'
    recent = float(h[close_col].tail(min(20,len(h))).median())
    factors = (1.0,10.0,100.0,1000.0,10000.0)
    factor = min(factors, key=lambda f: abs(recent*f-current_price)/max(current_price,1.0))
    rel = abs(recent*factor-current_price)/max(current_price,1.0)
    # Recent history need not equal today's quote exactly, so allow broad 35% sanity band.
    if rel > 0.35:
        return h, 1.0, f'scale_unverified_rel_diff={rel:.3f}'
    h[close_col] = h[close_col] * factor
    h.attrs['time_col'] = time_col
    h.attrs['close_col'] = close_col
    return h, factor, f'scale_verified_rel_diff={rel:.3f}'


def year_end_prices(history: pd.DataFrame | None, current_price: float | None, max_years: int | None = None) -> tuple[pd.Series, dict]:
    h, factor, status = normalize_price_history(history, current_price)
    if h is None or h.empty:
        return pd.Series(dtype='float64'), {'factor':factor,'status':status}
    tc = h.attrs.get('time_col')
    cc = h.attrs.get('close_col')
    if not tc or not cc:
        return pd.Series(dtype='float64'), {'factor':factor,'status':status}
    h = h.copy()
    h['_year'] = h[tc].dt.year
    out = {}
    for y, g in h.groupby('_year'):
        # Median of last 20 trading sessions reduces one-day year-end noise.
        vals = g.sort_values(tc)[cc].dropna().tail(20)
        if not vals.empty:
            out[str(int(y))] = float(vals.median())
    ordered = sorted(out.items(), key=lambda kv:int(kv[0]), reverse=True)
    if max_years is not None:
        ordered = ordered[:max_years]
    out = dict(ordered)
    return pd.Series(out,dtype='float64'), {'factor':factor,'status':status,'method':'median_last_20_trading_days'}



def normalize_historical_shares(raw: pd.Series, current_shares: float | None, latest_year: str | None = None) -> tuple[pd.Series, dict]:
    s = pd.to_numeric(raw, errors='coerce').dropna().astype(float)
    if s.empty:
        return s, {'status':'NO_DATA','transform':None,'relative_diff':None}
    if not current_shares or current_shares <= 0:
        return s, {'status':'NO_CURRENT_SHARES','transform':'raw','relative_diff':None}
    if latest_year and latest_year in s.index:
        ref_year = latest_year
    else:
        try: ref_year = sorted(s.index, key=lambda x:int(str(x)[:4]), reverse=True)[0]
        except Exception: ref_year = s.index[0]
    raw_ref = _safe_float(s.get(ref_year))
    if raw_ref is None or raw_ref <= 0:
        return s, {'status':'NO_REFERENCE_VALUE','transform':None,'relative_diff':None}
    candidates=[]
    for f in (1.0,10.0,100.0,1000.0,10000.0,100000.0,1000000.0):
        for mode in ('mul','div'):
            tv = raw_ref*f if mode=='mul' else raw_ref/f
            if tv>0:
                rel=abs(tv-current_shares)/max(current_shares,1.0); candidates.append((rel,mode,f))
    candidates.sort(key=lambda x:x[0]); rel,mode,factor=candidates[0]
    out=s*factor if mode=='mul' else s/factor
    overlap=bool(latest_year and latest_year in s.index); threshold=0.25 if overlap else 0.60
    return out, {'status':'PASS' if rel<=threshold else 'FAIL','transform':f'{mode}_{factor:g}','relative_diff':rel,'reference_year':str(ref_year),'latest_year_overlap':overlap,'raw_reference':raw_ref,'normalized_reference':float(out.get(ref_year))}

def clean_multiple_series(series: pd.Series, lo: float, hi: float, max_points: int | None = None) -> tuple[pd.Series, dict]:
    s = pd.to_numeric(series, errors='coerce').dropna().astype(float)
    s = s[(s >= lo) & (s <= hi)]
    before_iqr = len(s)
    if len(s) >= 5:
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3-q1
        if iqr > 0:
            low, high = q1-1.5*iqr, q3+1.5*iqr
            s = s[(s >= low) & (s <= high)]
    # Use every valid historical period returned by the provider.
    # No fixed 4/8-year cap: account/API tier determines available history.
    try:
        ordered = sorted(s.index, key=lambda x:int(str(x)[:4]), reverse=True)
        if max_points is not None:
            ordered = ordered[:max_points]
        s = s.loc[ordered]
    except Exception:
        if max_points is not None:
            s = s.iloc[:max_points]
    return s, {'valid_before_iqr':before_iqr,'valid_after_iqr':len(s),'hard_bounds':[lo,hi]}


def historical_target(series: pd.Series, *, lo: float, hi: float, min_points: int = 4) -> dict:
    clean, meta = clean_multiple_series(series, lo, hi)
    if len(clean) < min_points:
        return {'status':'NOT_ENOUGH_DATA','target':None,'series':clean.to_dict(),'count':len(clean),'cleaning':meta}
    return {
        'status':'OK',
        'target':float(clean.median()),
        'series':clean.to_dict(),
        'count':len(clean),
        'cleaning':meta,
        'method':'median_cleaned_history'
    }


def fixed_blend(dcf: dict | None, pe: dict | None, ev_ebitda: dict | None) -> dict:
    """Exact 60/25/15 blend. No silent weight renormalization; all scenarios must be numeric."""
    required = {'dcf_fcff':dcf,'pe':pe,'ev_ebitda':ev_ebitda}
    scenarios=('bear','base','bull')
    missing=[]; invalid={}
    for k,d in required.items():
        if not isinstance(d,dict):
            missing.append(k); continue
        bad=[]
        for name in scenarios:
            x=_safe_float(d.get(name))
            if x is None or x < 0:
                bad.append(name)
        if bad:
            invalid[k]=bad
    if missing or invalid:
        return {'status':'PARTIAL','missing_methods':missing,'invalid_method_scenarios':invalid,
                'bear':None,'base':None,'bull':None}
    weights = {'dcf_fcff':0.60,'pe':0.25,'ev_ebitda':0.15}
    def case(name):
        return sum(float(required[k][name])*weights[k] for k in weights)
    return {'status':'OK_FULL','missing_methods':[],'invalid_method_scenarios':{},
            'bear':case('bear'),'base':case('base'),'bull':case('bull'),'weights':weights}
