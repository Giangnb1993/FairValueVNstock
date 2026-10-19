from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from historical_multiples import timeseries_metric, year_end_prices, historical_target, fixed_blend, normalize_historical_shares
from latest_fundamentals import (
    build_latest_snapshot,
    quarter_metric,
    latest_quarter_value,
)
from volume_score import volume_accumulation_score_from_df


try:
    import vnstock
    from vnstock import Market, Reference, Fundamental
    from vnstock.core.utils.auth import change_api_key, check_status
except ImportError as exc:
    raise SystemExit("Hay cai: pip install -U vnstock pandas numpy") from exc


# ============================================================
# Generic helpers
# ============================================================

def norm_text(x: Any) -> str:
    if x is None:
        return ""
    s = unicodedata.normalize("NFD", str(x).strip().lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def norm_id(x: Any) -> str:
    return norm_text(x).replace(" ", "_")


def safe_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        try:
            y = float(x)
            return y if math.isfinite(y) else None
        except Exception:
            return None
    s = str(x).strip().replace(",", "")
    if not s or s.lower() in {"nan", "none", "null", "-"}:
        return None
    try:
        y = float(s)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def fmt_num(x: float | None) -> str:
    return "N/A" if x is None else f"{x:,.2f}"


def fmt_pct(x: float | None) -> str:
    return "N/A" if x is None else f"{x * 100:.2f}%"


def annual_columns(df: pd.DataFrame | None) -> list[Any]:
    """
    Provider annual-statement columns, newest -> oldest.

    A current-calendar-year annual column may be a completed fiscal year for
    a non-calendar-year company. Exclude only future years.
    """
    if df is None or df.empty:
        return []
    current_year = datetime.now().year
    cols = []
    for c in df.columns:
        s = str(c).strip()
        if re.fullmatch(r"20\d{2}", s):
            y = int(s)
            if y <= current_year:
                cols.append((y, c))
    cols.sort(reverse=True)
    return [c for _, c in cols]


def strip_prefix(x: Any) -> str:
    tokens = norm_text(x).split()
    roman = {"i","ii","iii","iv","v","vi","vii","viii","ix","x"}
    letters = set("abcdefghij")
    while tokens and (tokens[0].isdigit() or tokens[0] in roman or tokens[0] in letters):
        tokens.pop(0)
    return " ".join(tokens)


def row_meta(row: pd.Series | None) -> dict | None:
    if row is None:
        return None
    out = {}
    for c in ("item", "item_en", "item_id", "unit", "levels", "level", "row_number"):
        if c in row.index and pd.notna(row.get(c)):
            v = row.get(c)
            out[c] = v.item() if hasattr(v, "item") else v
    return out


def numeric_count(df: pd.DataFrame, row: pd.Series) -> int:
    return sum(safe_float(row.get(c)) is not None for c in annual_columns(df))


def find_row(
    df: pd.DataFrame | None,
    item_ids: Iterable[str] = (),
    exact_names: Iterable[str] = (),
    include_any: Iterable[str] = (),
    exclude_any: Iterable[str] = (),
    strict: bool = True,
) -> pd.Series | None:
    """Strict row resolver. IDs/exact names dominate; fuzzy is fallback only when strict=False."""
    if df is None or df.empty:
        return None

    ids = {norm_id(x) for x in item_ids if norm_id(x)}
    names = {norm_text(x) for x in exact_names if norm_text(x)}
    stripped_names = {strip_prefix(x) for x in exact_names if strip_prefix(x)}
    includes = [norm_text(x) for x in include_any if norm_text(x)]
    excludes = [norm_text(x) for x in exclude_any if norm_text(x)]

    candidates = []
    for idx, row in df.iterrows():
        rid = norm_id(row.get("item_id", ""))
        txts = [norm_text(row.get(c, "")) for c in ("item", "item_en", "name") if c in df.columns]
        strips = [strip_prefix(row.get(c, "")) for c in ("item", "item_en", "name") if c in df.columns]
        hay = " | ".join(txts)

        if any(e in hay for e in excludes):
            continue

        id_exact = rid in ids if ids else False
        id_collision = any(re.fullmatch(re.escape(w) + r"_\d+", rid) for w in ids) if ids and rid else False
        name_exact = bool(set(txts) & names)
        name_stripped = bool(set(strips) & stripped_names)
        fuzzy = sum(1 for x in includes if x in hay)

        if not (id_exact or id_collision or name_exact or name_stripped or (not strict and fuzzy)):
            continue

        n = numeric_count(df, row)
        level = safe_float(row.get("levels"))
        if level is None:
            level = safe_float(row.get("level"))
        if level is None:
            level = 999

        score = (
            n * 1000
            + 500 * id_exact
            + 450 * id_collision
            + 400 * name_exact
            + 350 * name_stripped
            + 20 * fuzzy
            - float(level)
        )
        candidates.append((score, n, idx, row))

    if not candidates:
        return None
    candidates.sort(key=lambda z: (z[0], z[1]), reverse=True)
    return candidates[0][3]


def report_series(df: pd.DataFrame | None, row: pd.Series | None) -> pd.Series:
    if df is None or row is None:
        return pd.Series(dtype="float64")
    d = {}
    for c in annual_columns(df):
        v = safe_float(row.get(c))
        if v is not None:
            d[str(c)] = v
    return pd.Series(d, dtype="float64")


def reverse_values(series: pd.Series) -> pd.Series:
    s = series.dropna()
    if len(s) < 2:
        return series.copy()
    return pd.Series(dict(zip(list(s.index), list(s.values)[::-1])), dtype="float64")


def apply_orientation(series: pd.Series, orientation: str) -> pd.Series:
    if orientation == "reversed":
        return reverse_values(series)
    return series.copy()


def latest(s: pd.Series) -> float | None:
    s = s.dropna()
    return None if s.empty else float(s.iloc[0])


def avg_latest_two(s: pd.Series) -> float | None:
    s = s.dropna()
    if len(s) >= 2:
        return (float(s.iloc[0]) + float(s.iloc[1])) / 2
    return latest(s)


def cagr(s: pd.Series) -> float | None:
    pairs = []
    for y, v in s.dropna().items():
        if re.fullmatch(r"20\d{2}", str(y)):
            pairs.append((int(y), float(v)))
    if len(pairs) < 2:
        return None
    pairs.sort()
    y0, v0 = pairs[0]
    y1, v1 = pairs[-1]
    if y1 <= y0 or v0 <= 0 or v1 <= 0:
        return None
    return (v1 / v0) ** (1 / (y1 - y0)) - 1


def clip(x: float, lo: float, hi: float) -> float:
    return min(max(float(x), lo), hi)


# ============================================================
# API result
# ============================================================

@dataclass
class FetchResult:
    name: str
    data: pd.DataFrame | None
    error: str | None = None


# V8.2 validated family map for audited 201-symbol universe.
VALIDATED_TICKER_FAMILY_MAP = {
    "AAA": "GENERAL_NORMAL",
    "AAS": "SECURITIES",
    "ABB": "BANK",
    "ABI": "INSURANCE",
    "ACB": "BANK",
    "ACV": "TRANSPORT_LOGISTICS",
    "AGR": "SECURITIES",
    "ANV": "CONSUMER_RETAIL",
    "APG": "SECURITIES",
    "BAB": "BANK",
    "BAF": "CONSUMER_RETAIL",
    "BCG": "HOLDING_CONGLOMERATE",
    "BCM": "REAL_ESTATE",
    "BFC": "CYCLICAL_COMMODITY",
    "BIC": "INSURANCE",
    "BID": "BANK",
    "BMI": "INSURANCE",
    "BMP": "GENERAL_NORMAL",
    "BSI": "SECURITIES",
    "BSR": "CYCLICAL_COMMODITY",
    "BVH": "INSURANCE",
    "BVS": "SECURITIES",
    "BWE": "UTILITIES",
    "C4G": "INDUSTRIAL_CONSTRUCTION",
    "CAP": "GENERAL_NORMAL",
    "CC1": "INDUSTRIAL_CONSTRUCTION",
    "CEO": "REAL_ESTATE",
    "CHP": "UTILITIES",
    "CII": "INDUSTRIAL_CONSTRUCTION",
    "CLW": "UTILITIES",
    "CMG": "TECHNOLOGY_ASSET_LIGHT",
    "CSV": "CYCLICAL_COMMODITY",
    "CTD": "INDUSTRIAL_CONSTRUCTION",
    "CTG": "BANK",
    "CTI": "INDUSTRIAL_CONSTRUCTION",
    "CTS": "SECURITIES",
    "DBC": "CONSUMER_RETAIL",
    "DBD": "GENERAL_NORMAL",
    "DCM": "CYCLICAL_COMMODITY",
    "DGC": "CYCLICAL_COMMODITY",
    "DGW": "CONSUMER_RETAIL",
    "DHA": "CYCLICAL_COMMODITY",
    "DHC": "GENERAL_NORMAL",
    "DHG": "GENERAL_NORMAL",
    "DIG": "REAL_ESTATE",
    "DPM": "CYCLICAL_COMMODITY",
    "DXG": "REAL_ESTATE",
    "DXP": "TRANSPORT_LOGISTICS",
    "EIB": "BANK",
    "ELC": "TECHNOLOGY_ASSET_LIGHT",
    "EVS": "SECURITIES",
    "FCN": "INDUSTRIAL_CONSTRUCTION",
    "FIT": "HOLDING_CONGLOMERATE",
    "FMC": "CONSUMER_RETAIL",
    "FOC": "TECHNOLOGY_ASSET_LIGHT",
    "FPT": "TECHNOLOGY_ASSET_LIGHT",
    "FRT": "CONSUMER_RETAIL",
    "FTS": "SECURITIES",
    "GAS": "UTILITIES",
    "GEG": "UTILITIES",
    "GEL": "HOLDING_CONGLOMERATE",
    "GEX": "HOLDING_CONGLOMERATE",
    "GIL": "GENERAL_NORMAL",
    "GMD": "TRANSPORT_LOGISTICS",
    "GSP": "TRANSPORT_LOGISTICS",
    "HAH": "TRANSPORT_LOGISTICS",
    "HAX": "GENERAL_NORMAL",
    "HBC": "INDUSTRIAL_CONSTRUCTION",
    "HCM": "SECURITIES",
    "HDB": "BANK",
    "HDG": "HOLDING_CONGLOMERATE",
    "HHV": "INDUSTRIAL_CONSTRUCTION",
    "HMH": "TRANSPORT_LOGISTICS",
    "HND": "UTILITIES",
    "HPG": "CYCLICAL_COMMODITY",
    "HPT": "TECHNOLOGY_ASSET_LIGHT",
    "HSG": "CYCLICAL_COMMODITY",
    "HTN": "INDUSTRIAL_CONSTRUCTION",
    "HUT": "HOLDING_CONGLOMERATE",
    "HVN": "TRANSPORT_LOGISTICS",
    "IDC": "REAL_ESTATE",
    "IDJ": "HOLDING_CONGLOMERATE",
    "IDP": "CONSUMER_RETAIL",
    "IMP": "GENERAL_NORMAL",
    "IPA": "HOLDING_CONGLOMERATE",
    "ITD": "TECHNOLOGY_ASSET_LIGHT",
    "KBC": "REAL_ESTATE",
    "KDC": "CONSUMER_RETAIL",
    "KDH": "REAL_ESTATE",
    "KSB": "CYCLICAL_COMMODITY",
    "KSF": "REAL_ESTATE",
    "KSV": "CYCLICAL_COMMODITY",
    "LAS": "CYCLICAL_COMMODITY",
    "LCG": "INDUSTRIAL_CONSTRUCTION",
    "LPB": "BANK",
    "MBB": "BANK",
    "MBS": "SECURITIES",
    "MCH": "CONSUMER_RETAIL",
    "MIG": "INSURANCE",
    "MPC": "CONSUMER_RETAIL",
    "MSB": "BANK",
    "MSH": "GENERAL_NORMAL",
    "MSN": "HOLDING_CONGLOMERATE",
    "MSR": "CYCLICAL_COMMODITY",
    "MWG": "CONSUMER_RETAIL",
    "NAB": "BANK",
    "NET": "CONSUMER_RETAIL",
    "NKG": "CYCLICAL_COMMODITY",
    "NLG": "REAL_ESTATE",
    "NNC": "CYCLICAL_COMMODITY",
    "NT2": "UTILITIES",
    "NTC": "REAL_ESTATE",
    "NTP": "GENERAL_NORMAL",
    "NVL": "REAL_ESTATE",
    "OCB": "BANK",
    "ORS": "SECURITIES",
    "PAC": "GENERAL_NORMAL",
    "PAN": "CONSUMER_RETAIL",
    "PC1": "HOLDING_CONGLOMERATE",
    "PDR": "REAL_ESTATE",
    "PET": "CONSUMER_RETAIL",
    "PGI": "INSURANCE",
    "PGV": "UTILITIES",
    "PHP": "TRANSPORT_LOGISTICS",
    "PNJ": "CONSUMER_RETAIL",
    "POT": "TECHNOLOGY_ASSET_LIGHT",
    "POW": "UTILITIES",
    "PPC": "UTILITIES",
    "PRE": "INSURANCE",
    "PTB": "GENERAL_NORMAL",
    "PTI": "INSURANCE",
    "PVD": "CYCLICAL_COMMODITY",
    "PVI": "INSURANCE",
    "PVS": "CYCLICAL_COMMODITY",
    "PVT": "TRANSPORT_LOGISTICS",
    "QCG": "REAL_ESTATE",
    "QNS": "CONSUMER_RETAIL",
    "QTP": "UTILITIES",
    "RAL": "GENERAL_NORMAL",
    "REE": "UTILITIES",
    "SAB": "CONSUMER_RETAIL",
    "SAM": "TECHNOLOGY_ASSET_LIGHT",
    "SBT": "CONSUMER_RETAIL",
    "SCI": "INDUSTRIAL_CONSTRUCTION",
    "SFI": "TRANSPORT_LOGISTICS",
    "SGP": "TRANSPORT_LOGISTICS",
    "SGT": "TECHNOLOGY_ASSET_LIGHT",
    "SHB": "BANK",
    "SHP": "UTILITIES",
    "SHS": "SECURITIES",
    "SIP": "REAL_ESTATE",
    "SJD": "UTILITIES",
    "SSB": "BANK",
    "SSI": "SECURITIES",
    "STB": "BANK",
    "SZC": "REAL_ESTATE",
    "TBC": "UTILITIES",
    "TCB": "BANK",
    "TCD": "INDUSTRIAL_CONSTRUCTION",
    "TCH": "REAL_ESTATE",
    "TCL": "TRANSPORT_LOGISTICS",
    "TCM": "GENERAL_NORMAL",
    "TDM": "UTILITIES",
    "THD": "HOLDING_CONGLOMERATE",
    "TIG": "HOLDING_CONGLOMERATE",
    "TIS": "CYCLICAL_COMMODITY",
    "TMP": "UTILITIES",
    "TMS": "TRANSPORT_LOGISTICS",
    "TNG": "GENERAL_NORMAL",
    "TPB": "BANK",
    "TRA": "GENERAL_NORMAL",
    "TVS": "SECURITIES",
    "VCB": "BANK",
    "VCG": "INDUSTRIAL_CONSTRUCTION",
    "VCI": "SECURITIES",
    "VCS": "GENERAL_NORMAL",
    "VDS": "SECURITIES",
    "VEA": "HOLDING_CONGLOMERATE",
    "VFS": "SECURITIES",
    "VGC": "HOLDING_CONGLOMERATE",
    "VHC": "CONSUMER_RETAIL",
    "VHM": "REAL_ESTATE",
    "VIB": "BANK",
    "VIC": "HOLDING_CONGLOMERATE",
    "VIP": "TRANSPORT_LOGISTICS",
    "VIX": "SECURITIES",
    "VJC": "TRANSPORT_LOGISTICS",
    "VND": "SECURITIES",
    "VNE": "INDUSTRIAL_CONSTRUCTION",
    "VNM": "CONSUMER_RETAIL",
    "VNR": "INSURANCE",
    "VNZ": "TECHNOLOGY_ASSET_LIGHT",
    "VOS": "TRANSPORT_LOGISTICS",
    "VPB": "BANK",
    "VRE": "REAL_ESTATE",
    "VSC": "TRANSPORT_LOGISTICS",
    "VSH": "UTILITIES",
    "VTE": "TECHNOLOGY_ASSET_LIGHT",
    "VTO": "TRANSPORT_LOGISTICS",
    "VTP": "TRANSPORT_LOGISTICS",
    "YEG": "TECHNOLOGY_ASSET_LIGHT"
}


class StockAudit:
    """
    CANONICAL BUILD

    The crucial rule:
      Determine ONE statement-year orientation, once.
      ALL financial-statement rows use that same orientation.

    This prevents Revenue=2025 while EPS/EBIT/Debt accidentally use 2022.
    """

    def __init__(self, symbol: str, output_dir: str = "output", api_key: str | None = None):
        self.symbol = symbol.upper().strip()
        self.base_dir = Path(output_dir) / self.symbol
        self.base_dir.mkdir(parents=True, exist_ok=True)

        if api_key:
            change_api_key(api_key.strip())

        try:
            self.auth_status = check_status()
        except Exception as exc:
            self.auth_status = {"status_error": repr(exc)}

        self.market = Market()
        self.reference = Reference()
        self.fundamental = Fundamental()
        self.results: dict[str, FetchResult] = {}
        self.matched: dict[str, Any] = {}

    # --------------------------------------------------------
    # Fetch: report orientation is the only statement source.
    # Normal run = 6 API calls; OHLCV is only fallback if quote price missing.
    # --------------------------------------------------------

    def _fetch(self, name: str, fn):
        try:
            d = fn()
            if d is None:
                raise ValueError("API returned None")
            if not isinstance(d, pd.DataFrame):
                d = pd.DataFrame(d)
            self.results[name] = FetchResult(name, d)
            return d
        except Exception as exc:
            self.results[name] = FetchResult(name, None, repr(exc))
            return None

    def fetch_all(self) -> dict[str, pd.DataFrame | None]:
        eq = self.fundamental.equity(self.symbol)
        quote = self._fetch("quote", lambda: self.market.quote(self.symbol))
        info = self._fetch("company_info", lambda: self.reference.company(self.symbol).info())
        income = self._fetch("income_year", lambda: eq.income_statement(period="year", orient="report"))
        balance = self._fetch("balance_year", lambda: eq.balance_sheet(period="year", orient="report"))
        cashflow = self._fetch("cashflow_year", lambda: eq.cash_flow(period="year", orient="report"))
        ratio = self._fetch("ratio_year", lambda: eq.ratio(period="year", orient="report"))

        # Annual history: use every period returned by the provider. No fixed 4/8-year assumption.
        income_ts = self._fetch("income_year_ts", lambda: eq.income_statement(period="year", orient="time_series"))
        balance_ts = self._fetch("balance_year_ts", lambda: eq.balance_sheet(period="year", orient="time_series"))
        cashflow_ts = self._fetch("cashflow_year_ts", lambda: eq.cash_flow(period="year", orient="time_series"))
        ratio_ts = self._fetch("ratio_year_ts", lambda: eq.ratio(period="year", orient="time_series"))

        # Latest-data layer: quarterly statements + quarterly ratios.
        # These calls make the engine roll forward automatically as new quarters are published.
        income_q_ts = self._fetch("income_quarter_ts", lambda: eq.income_statement(period="quarter", orient="time_series"))
        balance_q_ts = self._fetch("balance_quarter_ts", lambda: eq.balance_sheet(period="quarter", orient="time_series"))
        cashflow_q_ts = self._fetch("cashflow_quarter_ts", lambda: eq.cash_flow(period="quarter", orient="time_series"))
        ratio_q_ts = self._fetch("ratio_quarter_ts", lambda: eq.ratio(period="quarter", orient="time_series"))

        # 8-year daily history for point-in-time historical multiples.
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = f"{datetime.now().year - 9}-01-01"
        price_history = self._fetch(
            "price_history_8y",
            lambda: self.market.equity(self.symbol).ohlcv(start=start_date, end=end_date, interval="1D"),
        )

        data = {
            "quote": quote,
            "company_info": info,
            "income": income,
            "balance": balance,
            "cashflow": cashflow,
            "ratio": ratio,
            "income_ts": income_ts,
            "balance_ts": balance_ts,
            "cashflow_ts": cashflow_ts,
            "ratio_ts": ratio_ts,
            "income_q_ts": income_q_ts,
            "balance_q_ts": balance_q_ts,
            "cashflow_q_ts": cashflow_q_ts,
            "ratio_q_ts": ratio_q_ts,
            "price_history": price_history,
        }
        return data

    def save_raw(self, data):
        raw = self.base_dir / "raw"
        raw.mkdir(exist_ok=True)
        for k, df in data.items():
            if df is not None:
                df.to_csv(raw / f"{k}.csv", index=False, encoding="utf-8-sig")
        errs = {k: v.error for k, v in self.results.items() if v.error}
        (raw / "fetch_errors.json").write_text(json.dumps(errs, ensure_ascii=False, indent=2), encoding="utf-8")

    # --------------------------------------------------------
    # Market / ratio
    # --------------------------------------------------------

    def market_price(self, quote, history):
        if quote is not None and not quote.empty:
            cols = {norm_id(c): c for c in quote.columns}
            for k in ("close_price", "match_price", "last_price"):
                if k in cols:
                    v = safe_float(quote.iloc[0][cols[k]])
                    if v and v > 0:
                        return v, f"quote.{cols[k]}"
            ref=cols.get("reference_price")
            if ref is not None:
                v=safe_float(quote.iloc[0][ref])
                if v and v>0:
                    return v,f"quote.{ref}"
        if history is not None and not history.empty:
            cols = {norm_id(c): c for c in history.columns}
            c = cols.get("close") or cols.get("close_price")
            if c is not None:
                vals = pd.to_numeric(history[c], errors="coerce").dropna()
                if not vals.empty:
                    v=float(vals.iloc[-1])
                    if 0 < v < 1000:
                        return v*1000.0, f"price_history.{c}*1000"
                    if v>0:
                        return v, f"price_history.{c}"
        return None, "not_found"

    def ratio_value(self, ratio, ids=(), names=()):
        row = find_row(ratio, item_ids=ids, exact_names=names, strict=True)
        s = report_series(ratio, row)
        return latest(s), s, row_meta(row)

    # --------------------------------------------------------
    # Canonical orientation
    # --------------------------------------------------------

    def choose_orientation(self, income, balance, ratio, ratio_ts=None) -> tuple[str, dict]:
        """
        Determine normal vs reversed ONCE. Prefer an independent annual
        revenue-growth reconciliation when available; ROE/ROA is fallback.
        """
        ni_row = find_row(
            income,
            item_ids=["net_profit", "net_income", "profit_after_tax"],
            exact_names=["Lợi nhuận sau thuế", "Net profit", "Net income", "Profit after tax"],
            include_any=["loi nhuan sau thue"],
            exclude_any=["co dong", "minority", "non controlling"],
            strict=False,
        )
        eq_row = find_row(
            balance,
            item_ids=["owner_equity", "owners_equity", "shareholders_equity", "total_equity", "capital_and_reserves"],
            exact_names=["Vốn chủ sở hữu", "Tổng vốn chủ sở hữu", "Vốn và các quỹ", "Total equity", "Shareholders' equity"],
            exclude_any=["von gop", "minority", "loi ich", "retained"],
            strict=True,
        )
        asset_row = find_row(
            balance,
            item_ids=["total_assets"],
            exact_names=["Tổng cộng tài sản", "Tổng tài sản", "Total assets"],
            strict=True,
        )

        ni_raw = report_series(income, ni_row)
        eq_raw = report_series(balance, eq_row)
        asset_raw = report_series(balance, asset_row)

        roe_ref, _, roe_meta = self.ratio_value(
            ratio, ["roe", "return_on_equity"], ["ROE", "Return on equity"]
        )
        roa_ref, _, roa_meta = self.ratio_value(
            ratio, ["roa", "return_on_assets"], ["ROA", "Return on assets"]
        )
        if roe_ref is not None:
            roe_ref /= 100.0
        if roa_ref is not None:
            roa_ref /= 100.0

        def score(orientation):
            ni = apply_orientation(ni_raw, orientation)
            eq = apply_orientation(eq_raw, orientation)
            assets = apply_orientation(asset_raw, orientation)
            ni0 = latest(ni)
            avgeq = avg_latest_two(eq)
            avgassets = avg_latest_two(assets)
            roe = ni0 / avgeq if ni0 is not None and avgeq not in (None, 0) else None
            roa = ni0 / avgassets if ni0 is not None and avgassets not in (None, 0) else None

            parts = []
            if roe_ref is not None and roe is not None:
                parts.append(abs(roe - roe_ref))
            if roa_ref is not None and roa is not None:
                parts.append(abs(roa - roa_ref))

            penalty = 0
            if roe is None or not (-1 < roe < 1.5):
                penalty += 5
            if roa is None or not (-0.5 < roa < 0.8):
                penalty += 5
            if latest(eq) and latest(assets) and latest(eq) > latest(assets):
                penalty += 5
            return (sum(parts) if parts else 100) + penalty, {
                "roe_calc": roe,
                "roa_calc": roa,
                "roe_ref": roe_ref,
                "roa_ref": roa_ref,
            }

        sn, mn = score("normal")
        sr, mr = score("reversed")

        rev_meta={"status":"UNAVAILABLE"}; rev_choice=None
        try:
            rev_row=find_row(income,item_ids=["revenue","net_revenue","sales_revenue"],
                exact_names=["Doanh thu thuần","Net revenue","Revenue"],
                include_any=["doanh thu thuan"],strict=False)
            rev_raw=report_series(income,rev_row)
            if ratio_ts is not None and not getattr(ratio_ts,"empty",True) and len(rev_raw)>=3:
                cmap={norm_id(c):c for c in ratio_ts.columns}; pc=cmap.get("period"); rc=cmap.get("net_revenue")
                refs={}
                if pc is not None and rc is not None:
                    for _,rr in ratio_ts.iterrows():
                        y=str(rr.get(pc,"")).strip(); v=safe_float(rr.get(rc))
                        if re.fullmatch(r"20\d{2}",y) and v is not None: refs[y]=v
                def growth_score(ori):
                    ss=apply_orientation(rev_raw,ori); mp={str(k):float(v) for k,v in ss.dropna().items()}; errs=[]
                    for y,ref in refs.items():
                        py=str(int(y)-1)
                        if y in mp and py in mp and mp[py]!=0:
                            g=(mp[y]/mp[py]-1.0)*100.0
                            if np.isfinite(g): errs.append(abs(g-ref))
                    return (float(np.median(errs)) if len(errs)>=2 else None),len(errs)
                gn,nn=growth_score("normal"); gr,nr=growth_score("reversed")
                if gn is not None and gr is not None:
                    best="normal" if gn<gr else "reversed"; be=min(gn,gr); oe=max(gn,gr)
                    if be<=0.05 and (oe-be)>=0.50: rev_choice=best
                    rev_meta={"status":"VERIFIED" if rev_choice else "AMBIGUOUS",
                              "normal_growth_error_pp":gn,"reversed_growth_error_pp":gr,
                              "comparisons":min(nn,nr),"chosen":rev_choice}
        except Exception as exc:
            rev_meta={"status":"ERROR","error":str(exc)}

        if rev_choice in {"normal","reversed"}:
            orientation=rev_choice; chosen=mr if orientation=="reversed" else mn
            safe_verified=True; verification_status="VERIFIED_REVENUE_GROWTH"
        else:
            orientation="reversed" if sr<sn else "normal"; chosen=mr if orientation=="reversed" else mn
            safe_verified=min(sn,sr)<1; verification_status="VERIFIED_ROE_ROA" if safe_verified else "LOW_CONFIDENCE_UNVERIFIED"
        return orientation, {
            "normal_score": sn,"reversed_score": sr,"chosen": orientation,
            "safe_verified": safe_verified,"verification_status": verification_status,
            "revenue_growth_orientation":rev_meta,**chosen,"roe_row": roe_meta,"roa_row": roa_meta,
        }

    def statement_series(
        self,
        df,
        orientation,
        *,
        key,
        item_ids=(),
        exact_names=(),
        include_any=(),
        exclude_any=(),
        strict=True,
    ):
        row = find_row(
            df,
            item_ids=item_ids,
            exact_names=exact_names,
            include_any=include_any,
            exclude_any=exclude_any,
            strict=strict,
        )
        raw = report_series(df, row)
        s = apply_orientation(raw, orientation)
        self.matched[key] = row_meta(row)
        return s

    # --------------------------------------------------------
    # Company info
    # --------------------------------------------------------

    def shares_outstanding(self, info, net_income_parent=None, eps=None):
        if info is not None and not info.empty:
            for i, c in enumerate(info.columns):
                n = norm_id(c)
                if any(k in n for k in ("outstanding_shares", "shares_outstanding", "outstanding_share")):
                    vals = pd.to_numeric(info.iloc[:, i], errors="coerce").dropna()
                    if not vals.empty and float(vals.iloc[0]) > 1_000_000:
                        return float(vals.iloc[0]), f"company_info.{c}"
        if net_income_parent and eps and eps > 0:
            x = net_income_parent / eps
            if 1e7 < x < 1e11:
                return x, "derived:parent_profit/EPS"
        return None, "not_found"

    def sector(self, info):
        """Valuation-family classifier with audited ticker-map priority."""
        t=self.symbol.upper()
        fam=VALIDATED_TICKER_FAMILY_MAP.get(t)
        if fam:
            return fam,f"validated_ticker_map:{t}"
        txt=""
        if info is not None and not getattr(info,"empty",True):
            fields={
                "industry","industry_name","sector","sector_name",
                "icb_industry","icb_sector","nganh","nganh_nghe",
                "business_model","businessmodel"
            }
            vals=[]
            for i,c in enumerate(info.columns):
                if norm_id(c) in fields:
                    vals.extend(str(x) for x in info.iloc[:,i].dropna().tolist())
            txt=norm_text(" ".join(vals))

        def has(words):
            return any(norm_text(x) in txt for x in words)

        rules=[
            ("SECURITIES",("securities","chung khoan","brokerage","moi gioi chung khoan")),
            ("BANK",("commercial bank","banking","ngan hang")),
            ("INSURANCE",("insurance","bao hiem")),
            ("REAL_ESTATE",("real estate","bat dong san","property development","phat trien bat dong san")),
            ("UTILITIES",("electricity","power","dien luc","water utility","cap nuoc","utilities","utility")),
            ("TECHNOLOGY_ASSET_LIGHT",("technology","information technology","software","cong nghe","phan mem","it services")),
            ("CONSUMER_RETAIL",("retail","ban le","consumer","tieu dung","food","beverage","dairy","jewelry","pharmaceutical")),
            ("INDUSTRIAL_CONSTRUCTION",("construction","xay dung","engineering","infrastructure","ha tang","industrial","cong nghiep")),
            ("TRANSPORT_LOGISTICS",("transport","logistics","port","shipping","airline","aviation","cang","van tai")),
            ("HOLDING_CONGLOMERATE",("holding","conglomerate","tap doan da nganh","multi industry","diversified")),
            ("CYCLICAL_COMMODITY",("steel","thep","fertilizer","phan bon","chemical","hoa chat","oil","gas","petroleum","coal","mining","commodity","cement","xi mang")),
        ]
        for fam,words in rules:
            if has(words):
                return fam,txt

        maps={
            "SECURITIES":{"SSI","VND","VCI","HCM","MBS","SHS","FTS","VIX","CTS","BSI","VDS","AGR","ORS","TVS","APG","DSC"},
            "BANK":{"VCB","BID","CTG","TCB","MBB","ACB","VPB","HDB","STB","SHB","VIB","LPB","TPB","MSB","OCB","EIB","NAB","BAB","SGB","KLB"},
            "INSURANCE":{"BVH","PVI","BMI","BIC","MIG","ABI","PTI"},
            "REAL_ESTATE":{"VHM","NVL","PDR","DXG","KDH","NLG","DIG","CEO","KBC","IDC","SZC","NTL","TCH"},
            "UTILITIES":{"POW","NT2","PPC","VSH","REE","BWE","TDM","GEG","PC1","SJD"},
            "TECHNOLOGY_ASSET_LIGHT":{"FPT","CMG","ELC","FOX"},
            "CONSUMER_RETAIL":{"MWG","PNJ","VNM","MSN","SAB","MCH","FRT","DGW","PET","DHG","IMP","TRA"},
            "INDUSTRIAL_CONSTRUCTION":{"CTD","VCG","HHV","C4G","LCG","FCN","DPG","HBC","GEX"},
            "TRANSPORT_LOGISTICS":{"GMD","HAH","VSC","VOS","PVT","VJC","HVN","SCS","PHP","VTP"},
            "HOLDING_CONGLOMERATE":{"VIC"},
            "CYCLICAL_COMMODITY":{"HPG","HSG","NKG","DGC","DCM","DPM","GAS","PVD","PVS","BSR","PLX"},
        }
        for fam,tickers in maps.items():
            if t in tickers:
                return fam,f"ticker:{t}"

        return "GENERAL_NORMAL",txt or "unknown"

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    def calculate(self, data):
        income, balance, cf, ratio = data["income"], data["balance"], data["cashflow"], data["ratio"]
        price, price_source = self.market_price(data["quote"], data["price_history"])

        orientation, orientation_meta = self.choose_orientation(income, balance, ratio, data.get("ratio_ts"))

        S = lambda df, **kw: self.statement_series(df, orientation, **kw)

        revenue = S(
            income, key="revenue",
            item_ids=["revenue","net_revenue","sales_revenue"],
            exact_names=["Doanh thu thuần", "Doanh thu", "Net revenue", "Revenue"],
            include_any=["doanh thu thuan"], strict=False
        )
        net_income = S(
            income, key="net_income",
            item_ids=["net_profit","net_income","profit_after_tax"],
            exact_names=["Lợi nhuận sau thuế", "Net profit", "Profit after tax"],
            include_any=["loi nhuan sau thue"],
            exclude_any=["co dong","minority","non controlling"], strict=False
        )
        parent_profit = S(
            income, key="parent_profit",
            item_ids=["net_profit_parent","profit_attributable_to_parent","profit_after_tax_parent","profit_after_tax_for_shareholders_of_the_parent_company","profit_after_tax_for_shareholders_of_parent_company"],
            exact_names=[
                "Lợi nhuận sau thuế của cổ đông công ty mẹ",
                "Lợi nhuận sau thuế thuộc về cổ đông công ty mẹ",
                "Profit attributable to owners of parent",
                "Net profit attributable to parent shareholders",
            ],
            include_any=["co dong cong ty me", "owners of parent"],
            exclude_any=["minority","khong kiem soat"], strict=False
        )
        pbt = S(
            income, key="profit_before_tax",
            item_ids=["profit_before_tax","pre_tax_profit","pretax_profit"],
            exact_names=["Tổng lợi nhuận kế toán trước thuế", "Lợi nhuận trước thuế", "Profit before tax"],
            include_any=["loi nhuan", "truoc thue"], strict=False
        )
        interest_income = S(
            income, key="interest_expense_income",
            item_ids=["interest_expense","finance_interest_expense","borrowing_costs"],
            exact_names=["Chi phí lãi vay", "Chi phí lãi vay và thuê tài chính", "Interest expense", "Borrowing costs"],
            include_any=["chi phi lai vay", "interest expense", "borrowing costs"], strict=False
        )
        interest_cf = S(
            cf, key="interest_expense_cashflow",
            item_ids=["interest_expense","finance_interest_expense","borrowing_costs"],
            exact_names=["Chi phí lãi vay", "Interest expense", "Borrowing costs"],
            include_any=["chi phi lai vay", "interest expense", "borrowing costs"], strict=False
        )
        interest = interest_income if not interest_income.empty else interest_cf
        self.matched["interest_expense"] = (
            self.matched.get("interest_expense_income")
            if not interest_income.empty
            else self.matched.get("interest_expense_cashflow")
        )

        operating_profit = S(
            income, key="operating_profit",
            item_ids=[
                "operating_profit","profit_from_business_activities",
                "profit_from_operating_activities","net_operating_profit",
                "net_profit_from_operating_activities"
            ],
            exact_names=[
                "Lợi nhuận từ hoạt động kinh doanh",
                "Lợi nhuận thuần từ hoạt động kinh doanh",
                "Lợi nhuận thuần từ hoạt động kinh doanh trước chi phí lãi vay",
                "Operating profit",
                "Profit from business operations",
                "Profit from operating activities"
            ],
            include_any=["loi nhuan", "hoat dong kinh doanh"],
            exclude_any=["sau thue","co dong"], strict=False
        )
        tax_exp = S(
            income, key="tax_expense",
            item_ids=["income_tax_expense","corporate_income_tax","tax_expense"],
            exact_names=["Chi phí thuế TNDN", "Chi phí thuế thu nhập doanh nghiệp", "Income tax expense"],
            include_any=["chi phi thue"], strict=False
        )
        eps = S(
            income, key="basic_eps",
            item_ids=["basic_eps","earnings_per_share","eps"],
            exact_names=["Lãi cơ bản trên cổ phiếu", "Basic earnings per share", "Earnings per share"],
            include_any=["lai co ban tren co phieu", "basic earnings per share"],
            exclude_any=["suy giam","diluted"], strict=False
        )

        equity = S(
            balance, key="equity",
            item_ids=["owner_equity","owners_equity","shareholders_equity","total_equity","capital_and_reserves"],
            exact_names=["Vốn chủ sở hữu", "Tổng vốn chủ sở hữu", "Vốn và các quỹ", "Total equity", "Shareholders' equity"],
            exclude_any=["von gop","minority","loi ich","retained"], strict=True
        )
        assets = S(
            balance, key="total_assets",
            item_ids=["total_assets"],
            exact_names=["Tổng cộng tài sản", "Tổng tài sản", "Total assets"], strict=True
        )
        cash = S(
            balance, key="cash",
            item_ids=["cash_and_cash_equivalents"],
            exact_names=["Tiền và các khoản tương đương tiền", "Cash and cash equivalents"],
            include_any=["tien va cac khoan tuong duong tien"], strict=False
        )
        current_assets = S(
            balance, key="current_assets",
            item_ids=["current_assets","total_current_assets"],
            exact_names=["Tài sản ngắn hạn", "Current assets"],
            include_any=["tai san ngan han"], strict=False
        )
        current_liab = S(
            balance, key="current_liabilities",
            item_ids=["current_liabilities","total_current_liabilities"],
            exact_names=["Nợ ngắn hạn", "Current liabilities"],
            include_any=["no ngan han"], strict=False
        )
        short_debt = S(
            balance, key="short_debt",
            item_ids=["short_term_borrowings","short_term_debt","short_term_loans",
                      "short_term_borrowings_and_financial_leases"],
            exact_names=["Vay và nợ thuê tài chính ngắn hạn", "Vay ngắn hạn", "Short-term borrowings"],
            include_any=(), exclude_any=["tai san","asset"], strict=True
        )
        long_debt = S(
            balance, key="long_debt",
            item_ids=["long_term_borrowings","long_term_debt","long_term_loans",
                      "long_term_borrowings_and_financial_leases"],
            exact_names=["Vay và nợ thuê tài chính dài hạn", "Vay dài hạn", "Long-term borrowings"],
            include_any=(), exclude_any=["tai san","asset"], strict=True
        )

        cfo = S(
            cf, key="operating_cash_flow",
            item_ids=["operating_cash_flow","cash_from_operations","net_cash_from_operating_activities"],
            exact_names=["Lưu chuyển tiền thuần từ hoạt động kinh doanh", "Net cash flow from operating activities"],
            include_any=["luu chuyen tien thuan tu hoat dong kinh doanh"], strict=False
        )
        capex = S(
            cf, key="capex",
            item_ids=["payment_for_fixed_assets_constructions_and_other_long_term_assets","purchase_of_fixed_assets","capital_expenditure","capex"],
            exact_names=[
                "Tiền chi để mua sắm, xây dựng TSCĐ và các tài sản dài hạn khác",
                "Purchase of fixed assets", "Capital expenditure"
            ],
            include_any=["mua sam", "tscd"], exclude_any=["thanh ly","proceeds"], strict=False
        )
        da = S(
            cf, key="depreciation",
            item_ids=["depreciation","depreciation_and_amortization","depreciation_amortization"],
            exact_names=["Khấu hao TSCĐ", "Khấu hao tài sản cố định", "Depreciation and amortization"],
            include_any=["khau hao"], strict=False
        )

        # Derived series, always after canonical orientation.
        debt_years = sorted(set(short_debt.index) | set(long_debt.index), reverse=True)
        total_debt = pd.Series({
            y: float(short_debt.get(y, 0) or 0) + float(long_debt.get(y, 0) or 0)
            for y in debt_years
        }, dtype=float)

        # EBIT hierarchy:
        # 1) explicit Operating Profit / Profit from business operations;
        # 2) PBT + explicit Interest Expense.
        if not operating_profit.empty:
            ebit = operating_profit.copy()
            ebit_source = "operating_profit_statement"
        else:
            ebit_years = sorted(set(pbt.index) & set(interest.index), reverse=True)
            ebit = pd.Series({
                y: float(pbt[y]) + abs(float(interest[y]))
                for y in ebit_years
            }, dtype=float)
            ebit_source = "pbt_plus_interest" if not ebit.empty else "not_found"

        ebitda_years = sorted(set(ebit.index) & set(da.index), reverse=True)
        ebitda = pd.Series({
            y: float(ebit[y]) + abs(float(da[y]))
            for y in ebitda_years
        }, dtype=float)

        # Operating NWC approximation:
        # (Current Assets - Cash) - (Current Liabilities - Short-term interest-bearing debt).
        # This avoids treating cash and financing debt as operating working capital.
        nwc_years = sorted(
            set(current_assets.index) & set(current_liab.index)
            & set(cash.index) & set(short_debt.index),
            reverse=True
        )
        nwc = pd.Series({
            y: (
                float(current_assets[y]) - float(cash[y])
                - (float(current_liab[y]) - float(short_debt[y]))
            )
            for y in nwc_years
        }, dtype=float)
        pairs = sorted((int(y), float(v)) for y, v in nwc.items())
        delta_nwc_d = {}
        for (y0,v0),(y1,v1) in zip(pairs,pairs[1:]):
            delta_nwc_d[str(y1)] = v1-v0
        delta_nwc = pd.Series(dict(sorted(delta_nwc_d.items(), key=lambda kv:int(kv[0]), reverse=True)), dtype=float)

        tax_rate = pd.Series(dtype=float)
        tx = {}
        for y in set(pbt.index) & set(tax_exp.index):
            if float(pbt[y]) > 0:
                tx[y] = clip(abs(float(tax_exp[y]))/float(pbt[y]), 0, 0.35)
        if tx:
            tax_rate = pd.Series(dict(sorted(tx.items(), reverse=True)), dtype=float)

        # ----------------------------------------------------
        # FCFF
        # Primary method:
        #   FCFF = CFO + InterestExpense*(1-T) - CAPEX
        # CFO already includes the real working-capital movement reported by the company.
        # This is therefore more robust than rebuilding Delta NWC from broad balance-sheet
        # subtotals such as Current Assets / Current Liabilities.
        #
        # Cross-check only:
        #   EBIT*(1-T) + D&A - CAPEX - Delta Operating NWC
        # This is retained for audit, not as the production DCF input.
        # ----------------------------------------------------
        fcff_primary_d = {}
        fcff_crosscheck_d = {}

        primary_years = set(cfo.index) & set(capex.index)
        for y in primary_years:
            t = safe_float(tax_rate.get(y))
            if t is None:
                t = 0.20
            after_tax_interest = 0.0
            if y in interest.index and safe_float(interest.get(y)) is not None:
                after_tax_interest = abs(float(interest[y])) * (1 - t)
            fcff_primary_d[y] = (
                float(cfo[y])
                + after_tax_interest
                - abs(float(capex[y]))
            )

        cross_years = set(ebit.index) & set(da.index) & set(capex.index) & set(delta_nwc.index)
        for y in cross_years:
            t = safe_float(tax_rate.get(y))
            if t is None:
                t = 0.20
            fcff_crosscheck_d[y] = (
                float(ebit[y])*(1-t)
                + abs(float(da[y]))
                - abs(float(capex[y]))
                - float(delta_nwc[y])
            )

        fcff = pd.Series(dict(sorted(fcff_primary_d.items(), reverse=True)), dtype=float)
        fcff_crosscheck = pd.Series(dict(sorted(fcff_crosscheck_d.items(), reverse=True)), dtype=float)

        fcf_d = {}
        for y in set(cfo.index) & set(capex.index):
            fcf_d[y] = float(cfo[y]) - abs(float(capex[y]))
        fcf = pd.Series(dict(sorted(fcf_d.items(), reverse=True)), dtype=float)

        pe, pe_s, _ = self.ratio_value(ratio, ["pe","price_to_earning","price_to_earnings"], ["P/E","Price to earnings"])
        pb, pb_s, _ = self.ratio_value(ratio, ["pb","price_to_book"], ["P/B","Price to book"])
        roe_ref, roe_s, _ = self.ratio_value(ratio, ["roe","return_on_equity"], ["ROE","Return on equity"])
        roa_ref, roa_s, _ = self.ratio_value(ratio, ["roa","return_on_assets"], ["ROA","Return on assets"])
        beta, beta_s, _ = self.ratio_value(ratio, ["beta"], ["Beta"])
        ev_ebitda_ratio, ev_ebitda_s, _ = self.ratio_value(
            ratio, ["ev_ebitda","enterprise_value_to_ebitda"], ["EV/EBITDA"]
        )
        if roe_ref is not None: roe_ref/=100.0
        if roa_ref is not None: roa_ref/=100.0

        latest_year = max(int(y) for y in revenue.index) if not revenue.empty else None
        eps_raw = safe_float(eps.get(str(latest_year))) if latest_year else None
        parent_latest = safe_float(parent_profit.get(str(latest_year))) if latest_year else None

        # Normalize common upstream EPS unit error (e.g. 5,216,000 returned for 5,216 VND).
        # Divide by 1,000 only while raw EPS is implausibly larger than the stock price.
        eps_latest = eps_raw
        eps_scale_factor = 1.0
        if eps_latest is not None and price is not None and price > 0:
            while eps_latest > price and eps_scale_factor < 1_000_000:
                eps_latest /= 1000.0
                eps_scale_factor *= 1000.0

        shares, shares_source = self.shares_outstanding(data["company_info"], parent_latest, eps_latest)

        # EPS validation:
        # - price/EPS must be in a broad plausible range;
        # - if parent profit is available, reconcile to parent profit / shares.
        eps_reconcile = None
        eps_validated = eps_latest
        eps_validation_status = "not_tested"
        implied_pe = (price / eps_latest) if (price and eps_latest and eps_latest > 0) else None

        if eps_latest is not None and implied_pe is not None and not (1.0 <= implied_pe <= 100.0):
            eps_validated = None
            eps_validation_status = "reject_implausible_price_to_eps"
        elif eps_latest is not None and parent_latest is not None and shares not in (None, 0):
            eps_implied = parent_latest / shares
            eps_reconcile = abs(eps_latest - eps_implied) / max(abs(eps_implied), 1.0)
            if eps_reconcile <= 0.15:
                eps_validation_status = "pass_parent_profit_per_share"
            else:
                eps_validation_status = "reject_parent_profit_per_share_mismatch"
                eps_validated = None
        elif eps_latest is not None:
            eps_validation_status = "pass_price_eps_sanity"

        net_income_latest = latest(net_income)
        roe_calc = net_income_latest / avg_latest_two(equity) if net_income_latest and avg_latest_two(equity) else None
        roa_calc = net_income_latest / avg_latest_two(assets) if net_income_latest and avg_latest_two(assets) else None

        # Transparent current multiples if provider ratio rows are unavailable.
        pe_calculated = (price / eps_validated) if (price and eps_validated and eps_validated > 0) else None
        bvps = (latest(equity) / shares) if (latest(equity) is not None and shares) else None
        pb_calculated = (price / bvps) if (price and bvps and bvps > 0) else None

        metrics = {
            "symbol": self.symbol,
            "vnstock_version": getattr(vnstock, "__version__", "unknown"),
            "auth_status": self.auth_status,
            "market_price": price,
            "price_source": price_source,
            "orientation": orientation,
            "orientation_meta": orientation_meta,
            "annual_periods": list(revenue.index),
            "latest_year": latest_year,
            "annual_current_calendar_year_accepted": bool(
                latest_year is not None and latest_year == datetime.now().year
            ),
            "latest_revenue": latest(revenue),
            "latest_net_income": latest(net_income),
            "latest_parent_profit": latest(parent_profit),
            "latest_pbt": latest(pbt),
            "latest_interest_expense": latest(interest),
            "interest_expense_source": (
                "income_statement" if not interest_income.empty
                else ("cash_flow" if not interest_cf.empty else "not_found")
            ),
            "latest_operating_profit": latest(operating_profit),
            "latest_ebit": latest(ebit),
            "ebit_source": ebit_source,
            "latest_da": latest(da),
            "latest_ebitda": latest(ebitda),
            "latest_equity": latest(equity),
            "avg_equity": avg_latest_two(equity),
            "latest_assets": latest(assets),
            "latest_cash": latest(cash),
            "latest_current_assets": latest(current_assets),
            "latest_current_liabilities": latest(current_liab),
            "latest_nwc": latest(nwc),
            "latest_delta_nwc": latest(delta_nwc),
            "latest_short_debt": latest(short_debt),
            "latest_long_debt": latest(long_debt),
            "latest_total_debt": latest(total_debt),
            "latest_cfo": latest(cfo),
            "latest_capex": abs(latest(capex)) if latest(capex) is not None else None,
            "latest_fcf": latest(fcf),
            "latest_tax_rate": latest(tax_rate),
            "latest_fcff": latest(fcff),
            "latest_fcff_crosscheck": latest(fcff_crosscheck),
            "fcff_method": "CFO + after-tax interest - CAPEX",
            "fcff_crosscheck_method": "EBIT*(1-T)+D&A-CAPEX-DeltaOperatingNWC",
            "revenue_cagr": cagr(revenue),
            "net_income_cagr": cagr(net_income),
            "fcff_cagr": cagr(fcff),
            "net_margin": latest(net_income)/latest(revenue) if latest(net_income) and latest(revenue) else None,
            "roe_calc": roe_calc,
            "roe_ref": roe_ref,
            "roa_calc": roa_calc,
            "roa_ref": roa_ref,
            "pe": pe if pe is not None else pe_calculated,
            "pe_provider": pe,
            "pe_calculated": pe_calculated,
            "pb": pb if pb is not None else pb_calculated,
            "pb_provider": pb,
            "pb_calculated": pb_calculated,
            "beta": beta,
            "ev_ebitda_ratio": ev_ebitda_ratio,
            "eps": eps_validated,
            "eps_raw_statement": eps_raw,
            "eps_normalized_statement": eps_latest,
            "eps_scale_factor": eps_scale_factor,
            "eps_implied_pe": implied_pe,
            "eps_source": "income_statement:basic_eps",
            "eps_validation_status": eps_validation_status,
            "eps_reconcile_relative_diff": eps_reconcile,
            "shares": shares,
            "shares_source": shares_source,
            "matched_items": self.matched,
            "series": {
                "revenue": revenue.to_dict(),
                "net_income": net_income.to_dict(),
                "parent_profit": parent_profit.to_dict(),
                "pbt": pbt.to_dict(),
                "interest_expense": interest.to_dict(),
                "operating_profit": operating_profit.to_dict(),
                "ebit": ebit.to_dict(),
                "depreciation": da.to_dict(),
                "ebitda": ebitda.to_dict(),
                "equity": equity.to_dict(),
                "assets": assets.to_dict(),
                "cash": cash.to_dict(),
                "current_assets": current_assets.to_dict(),
                "current_liabilities": current_liab.to_dict(),
                "nwc": nwc.to_dict(),
                "delta_nwc": delta_nwc.to_dict(),
                "short_debt": short_debt.to_dict(),
                "long_debt": long_debt.to_dict(),
                "total_debt": total_debt.to_dict(),
                "cfo": cfo.to_dict(),
                "capex": capex.to_dict(),
                "fcf": fcf.to_dict(),
                "tax_rate": tax_rate.to_dict(),
                "fcff": fcff.to_dict(),
                "fcff_crosscheck": fcff_crosscheck.to_dict(),
                "pe": pe_s.to_dict(),
                "pb": pb_s.to_dict(),
                "roe": roe_s.to_dict(),
                "roa": roa_s.to_dict(),
                "beta": beta_s.to_dict(),
                "ev_ebitda": ev_ebitda_s.to_dict(),
                "eps": ({str(k):(safe_float(v)/eps_scale_factor if safe_float(v) is not None else None)
                         for k,v in eps.to_dict().items()} if eps_scale_factor else eps.to_dict()),
            }
        }
        # Historical multiples remain annual anchors and use every valid period returned.
        metrics["historical_multiples"] = self.historical_multiples(data, metrics)

        # Dynamic latest fundamentals. This layer never assumes a calendar year.
        # Flow metrics use validated TTM when four consecutive quarters are available;
        # balance-sheet metrics use the latest reported quarter. If quarterly data fail
        # reconciliation, annual values remain in place and the reason is recorded.
        metrics["eps_annual_audit"] = metrics.get("eps")
        latest_snapshot = build_latest_snapshot(data, metrics)
        metrics["latest_snapshot"] = latest_snapshot
        metrics["latest_data_period"] = latest_snapshot.get("latest_period")
        metrics["current_basis"] = latest_snapshot.get("basis")

        if latest_snapshot.get("status") in {"OK","PARTIAL"}:
            metrics["reported_shares_estimate"]=latest_snapshot.get("reported_shares_estimate")
            metrics["share_change_ratio"]=latest_snapshot.get("share_change_ratio")
            metrics["post_report_share_change_detected"]=bool(
                metrics.get("share_change_ratio") is not None and abs(metrics.get("share_change_ratio")-1.0)>0.01
            )
            ov = latest_snapshot.get("overrides", {})

            fiscal_year_flow_fallback={}
            same_completed_fiscal_year = bool(
                metrics.get("annual_current_calendar_year_accepted")
                and metrics.get("latest_year") is not None
                and latest_snapshot.get("basis")=="TTM_PLUS_LATEST_QUARTER"
                and latest_snapshot.get("latest_period")==f"{metrics.get('latest_year')}-Q4"
            )
            if same_completed_fiscal_year:
                for key in [
                    "latest_interest_expense","latest_da","latest_cfo",
                    "latest_capex","latest_fcff_crosscheck"
                ]:
                    if key not in ov and metrics.get(key) is not None:
                        ov[key]=metrics.get(key)
                        fiscal_year_flow_fallback[key]="completed_annual_same_fiscal_year"
                if fiscal_year_flow_fallback:
                    latest_snapshot["fiscal_year_flow_fallback"]=fiscal_year_flow_fallback
                    latest_snapshot["overrides"]=ov
                    fm=latest_snapshot.get("flow_mode") or {}
                    key_to_mode={
                        "latest_interest_expense":"interest","latest_da":"da",
                        "latest_cfo":"cfo","latest_capex":"capex",
                        "latest_fcff_crosscheck":"fcff_crosscheck"
                    }
                    for key in fiscal_year_flow_fallback:
                        if key in key_to_mode:
                            fm[key_to_mode[key]]="completed_annual_same_fiscal_year"
                    latest_snapshot["flow_mode"]=fm

            for k, v in ov.items():
                if v is not None:
                    metrics[k] = v

            # Recompute all dependent current-basis metrics AFTER applying overrides.
            # Never mix annual and TTM/latest-quarter values under one current label.
            ov_keys = set(ov.keys())

            if metrics.get("current_basis") == "TTM_PLUS_LATEST_QUARTER":
                # Flow metrics: if TTM did not supply the metric, current value is N/A.
                for key in [
                    "latest_revenue","latest_net_income","latest_parent_profit","latest_pbt",
                    "latest_operating_profit","latest_interest_expense","latest_da",
                    "latest_cfo","latest_capex","latest_fcff_crosscheck"
                ]:
                    if key not in ov_keys:
                        metrics[key]=None

                # Balance metrics: if latest quarter did not supply the component, do not
                # carry an annual value forward.
                freshness=latest_snapshot.get("balance_freshness",{})
                for src,key in [
                    ("equity","latest_equity"),("assets","latest_assets"),("cash","latest_cash"),
                    ("short_debt","latest_short_debt"),("long_debt","latest_long_debt")
                ]:
                    if freshness.get(src)!="latest_quarter":
                        metrics[key]=None

            # EBITDA: if TTM EBIT and TTM D&A are both available, recompute exactly.
            # If only one side is current, do not silently retain stale annual EBITDA.
            if metrics.get("current_basis") == "TTM_PLUS_LATEST_QUARTER":
                if "latest_ebit" in ov_keys and "latest_da" in ov_keys:
                    metrics["latest_ebitda"] = metrics["latest_ebit"] + abs(metrics["latest_da"])
                else:
                    metrics["latest_ebitda"] = None

            # Debt: use a latest-quarter total only when both latest components exist.
            # If only one component is fresh, total debt becomes N/A rather than mixing periods.
            if "latest_short_debt" in ov_keys or "latest_long_debt" in ov_keys:
                if "latest_short_debt" in ov_keys and "latest_long_debt" in ov_keys:
                    metrics["latest_total_debt"] = metrics["latest_short_debt"] + metrics["latest_long_debt"]
                else:
                    metrics["latest_total_debt"] = None

            # Current FCFF must be recomputed only from current-basis CFO/CAPEX.
            if metrics.get("current_basis") == "TTM_PLUS_LATEST_QUARTER":
                if "latest_cfo" in ov_keys and "latest_capex" in ov_keys:
                    tr = metrics.get("latest_tax_rate")
                    if tr is None:
                        tr = 0.20
                    ati = abs(metrics.get("latest_interest_expense") or 0) * (1-tr)
                    metrics["latest_fcff"] = metrics["latest_cfo"] + ati - metrics["latest_capex"]
                else:
                    metrics["latest_fcff"] = None

            # Recompute per-share metrics only on a consistent basis.
            # Annual fallback profit / current shares is NOT TTM EPS.
            sh = metrics.get("shares")
            if metrics.get("current_basis") == "TTM_PLUS_LATEST_QUARTER" and sh and sh > 0:
                if metrics.get("latest_parent_profit") is not None:
                    metrics["eps_ttm_calculated"] = metrics["latest_parent_profit"] / sh
                elif metrics.get("latest_net_income") is not None:
                    metrics["eps_ttm_calculated"] = metrics["latest_net_income"] / sh

            # Latest ROE/ROA use TTM profit over average balance sheet when possible.
            if metrics.get("latest_net_income") is not None:
                ae = latest_snapshot.get("average_equity")
                aa = latest_snapshot.get("average_assets")
                if ae and ae > 0:
                    metrics["roe_calc"] = metrics["latest_net_income"] / ae
                if aa and aa > 0:
                    metrics["roa_calc"] = metrics["latest_net_income"] / aa

            # If flow data fell back to annual while balance-sheet data are latest-quarter,
            # a self-calculated ROE/ROA would mix periods. Preserve annual values only as audit
            # metadata and do not expose them as current calculated ratios.
            if metrics.get("current_basis") != "TTM_PLUS_LATEST_QUARTER":
                metrics["roe_calc_annual_audit"] = metrics.get("roe_calc")
                metrics["roa_calc_annual_audit"] = metrics.get("roa_calc")
                metrics["roe_calc"] = None
                metrics["roa_calc"] = None

            # Quarterly ratio provider may expose trailing ROE/EPS directly; use only as references.
            qref = latest_snapshot.get("ratio_references", {})
            if qref.get("roe_trailing") is not None:
                metrics["roe_ref"] = qref["roe_trailing"]
            if qref.get("roa_trailing") is not None:
                metrics["roa_ref"] = qref["roa_trailing"]
            if qref.get("trailing_eps") is not None:
                metrics["eps_provider_trailing"] = qref["trailing_eps"]
                metrics["eps_current"] = qref["trailing_eps"]
                metrics["eps"] = qref["trailing_eps"]
                metrics["eps_current_source"] = "provider_latest_quarter.trailing_eps"
                if metrics.get("market_price") and qref["trailing_eps"] > 0:
                    metrics["pe_calculated"] = metrics["market_price"] / qref["trailing_eps"]
                    metrics["pe"] = metrics["pe_calculated"]
            elif metrics.get("current_basis") == "TTM_PLUS_LATEST_QUARTER" and metrics.get("eps_ttm_calculated") is not None:
                metrics["eps_current"] = metrics["eps_ttm_calculated"]
                metrics["eps_current_source"] = "verified_ttm_profit/current_shares"
                metrics["eps"] = metrics["eps_ttm_calculated"]
                if metrics.get("market_price") and metrics["eps_ttm_calculated"] > 0:
                    metrics["pe_calculated"] = metrics["market_price"] / metrics["eps_ttm_calculated"]
                    metrics["pe"] = metrics["pe_calculated"]
            else:
                metrics["eps_current"] = metrics.get("eps")
                metrics["eps_current_source"] = "latest_annual_statement"
            if qref.get("bvps") is not None and qref["bvps"] > 0:
                metrics["bvps_provider_current"] = qref["bvps"]
                metrics["bvps_current"] = qref["bvps"]
                metrics["bvps_current_source"] = "provider_latest_quarter.book_value_per_share_bvps"
            elif sh and sh > 0 and metrics.get("latest_equity") is not None:
                metrics["bvps_current"] = metrics["latest_equity"] / sh
                metrics["bvps_current_source"] = "latest_equity/current_shares_fallback"
            if metrics.get("bvps_current") and metrics["bvps_current"] > 0 and metrics.get("market_price"):
                metrics["pb_calculated"] = metrics["market_price"] / metrics["bvps_current"]
                metrics["pb"] = metrics["pb_calculated"]
            if qref.get("beta") is not None:
                metrics["beta"] = qref["beta"]

        return metrics


    # --------------------------------------------------------
    # Historical Multiples (isolated from canonical current-year engine)
    # --------------------------------------------------------

    def historical_multiples(self, data, m):
        price_y, price_meta = year_end_prices(data.get("price_history"), m.get("market_price"), max_years=None)

        # Prefer explicit time-series fields. We validate overlap with canonical current-year values.
        eps_ts = timeseries_metric(
            data.get("income_ts"),
            ["earnings_per_share_vnd","basic_eps","earnings_per_share","lai_co_ban_tren_co_phieu"],
            exclude=["diluted","suy_giam"]
        )
        op_ts = timeseries_metric(
            data.get("income_ts"),
            ["operating_profit","profit_from_business_activities","profit_from_operating_activities","loi_nhuan_thuan_tu_hoat_dong_kinh_doanh"]
        )
        da_ts = timeseries_metric(
            data.get("cashflow_ts"),
            ["depreciation_of_fixed_assets_and_investment_properties","depreciation_and_amortization","depreciation","khau_hao"]
        )
        cash_ts = timeseries_metric(
            data.get("balance_ts"),
            ["cash_and_cash_equivalents","tien_va_cac_khoan_tuong_duong_tien"]
        )
        short_debt_ts = timeseries_metric(
            data.get("balance_ts"),
            ["short_term_borrowings_and_financial_leases","short_term_borrowings","short_term_debt"]
        )
        long_debt_ts = timeseries_metric(
            data.get("balance_ts"),
            ["long_term_borrowings_and_financial_leases","long_term_borrowings","long_term_debt"]
        )
        share_capital_ts = timeseries_metric(
            data.get("balance_ts"),
            ["owners_capital","owners_contributed_capital","owners_contributed_capital_2","contributed_capital","share_capital","paid_in_capital","charter_capital","von_gop_cua_chu_so_huu","von_co_phan","von_dieu_le"]
        )
        shares_ratio_ts = timeseries_metric(
            data.get("ratio_ts"),
            ["outstanding_shares","shares_outstanding","listed_shares","number_of_outstanding_shares","number_of_shares_outstanding","outstanding_share","shares"]
        )
        market_cap_ts = timeseries_metric(
            data.get("ratio_ts"),
            ["market_cap","market_capitalization","market_capitalisation","market_value","marketcap"]
        )
        eps_provider_ratio_ts = timeseries_metric(
            data.get("ratio_ts"),
            ["trailing_eps","eps_ttm","earnings_per_share"]
        )
        bvps_provider_ratio_ts = timeseries_metric(
            data.get("ratio_ts"),
            ["book_value_per_share_bvps","bvps","book_value_per_share"]
        )
        pe_provider_ts = timeseries_metric(
            data.get("ratio_ts"),
            ["pe_ratio","priceToEarning","price_to_earning","price_to_earnings","pe"]
        )
        pb_provider_ts = timeseries_metric(
            data.get("ratio_ts"),
            ["pb_ratio","priceToBook","price_to_book","pb"]
        )
        ev_provider_ts = timeseries_metric(
            data.get("ratio_ts"),
            ["ev_ebitda","ev_to_ebitda","enterprise_value_to_ebitda","enterprise_value_ebitda"]
        )
        ev_ebit_provider_ts = timeseries_metric(
            data.get("ratio_ts"),
            ["ev_ebit","ev_to_ebit","enterprise_value_to_ebit","enterprise_value_ebit"]
        )
        ebit_margin_provider_ts = timeseries_metric(
            data.get("ratio_ts"), ["ebit_margin"]
        )
        ebitda_margin_provider_ts = timeseries_metric(
            data.get("ratio_ts"), ["ebitda_net_revenue","ebitda_margin"]
        )

        # EPS normalization uses the already-validated current-year scale factor.
        eps_scale = float(m.get("eps_scale_factor") or 1.0)
        eps_ts = eps_ts / eps_scale if not eps_ts.empty else eps_ts

        # Validate time-series accounting fields against the canonical latest-year values.
        latest_year = str(m.get("latest_year"))
        validation = {}
        def overlap_check(name, s, canonical, tolerance=0.08):
            if canonical is None or s is None or s.empty or latest_year not in s.index:
                validation[name] = {"status":"NO_OVERLAP"}
                return False
            x = safe_float(s.get(latest_year))
            if x is None:
                validation[name] = {"status":"NO_OVERLAP"}
                return False
            rel = abs(x-canonical)/max(abs(canonical),1.0)
            ok = rel <= tolerance
            validation[name] = {"status":"PASS" if ok else "FAIL", "relative_diff":rel, "ts_value":x, "canonical":canonical}
            return ok

        eps_ts_ok = overlap_check("eps_ts", eps_ts, m.get("eps"), 0.08)
        op_ts_ok = overlap_check("operating_profit_ts", op_ts, m.get("latest_operating_profit"), 0.08)
        da_ts_ok = overlap_check("da_ts", da_ts, m.get("latest_da"), 0.10)
        cash_ts_ok = overlap_check("cash_ts", cash_ts, m.get("latest_cash"), 0.08)
        short_ts_ok = overlap_check("short_debt_ts", short_debt_ts, m.get("latest_short_debt"), 0.08)
        long_ts_ok = overlap_check("long_debt_ts", long_debt_ts, m.get("latest_long_debt"), 0.08)

        # Build BOTH provider and computed P/E histories, choose the valid one with more periods.
        provider_pe_target = historical_target(pe_provider_ts, lo=3.0, hi=80.0, min_points=4)
        pe_vals = {}
        if eps_ts_ok:
            for y in set(price_y.index) & set(eps_ts.index):
                e=safe_float(eps_ts.get(y)); px=safe_float(price_y.get(y))
                if e is not None and e>0 and px is not None and px>0: pe_vals[y]=px/e
        computed_pe_hist=pd.Series(dict(sorted(pe_vals.items(),reverse=True)),dtype=float)
        computed_pe_target=historical_target(computed_pe_hist,lo=3.0,hi=80.0,min_points=4)
        candidates=[]
        if provider_pe_target["status"]=="OK": candidates.append((provider_pe_target["count"],"provider_ratio_time_series",pe_provider_ts,provider_pe_target))
        if computed_pe_target["status"]=="OK": candidates.append((computed_pe_target["count"],"computed_year_end_price_over_eps",computed_pe_hist,computed_pe_target))
        if candidates:
            candidates.sort(key=lambda x:(x[0],x[1]=="computed_year_end_price_over_eps"),reverse=True); _,pe_source,pe_hist,pe_target=candidates[0]
        elif computed_pe_target.get("count",0)>=provider_pe_target.get("count",0):
            pe_source,pe_hist,pe_target="computed_year_end_price_over_eps",computed_pe_hist,computed_pe_target
        else:
            pe_source,pe_hist,pe_target="provider_ratio_time_series",pe_provider_ts,provider_pe_target

        # Historical shares: adaptive unit normalization; no hard-coded par-value division.
        shares_source="ratio_time_series"
        shares_hist,shares_norm=normalize_historical_shares(shares_ratio_ts,m.get("shares"),latest_year)
        if shares_hist.empty or shares_norm.get("status")!="PASS":
            capital_hist,capital_norm=normalize_historical_shares(share_capital_ts,m.get("shares"),latest_year)
            if not capital_hist.empty and capital_norm.get("status")=="PASS":
                shares_hist,shares_norm,shares_source=capital_hist,capital_norm,"share_capital_adaptive_scale"
            elif shares_hist.empty:
                shares_hist,shares_norm,shares_source=capital_hist,capital_norm,"share_capital_adaptive_scale"
        shares_ok=shares_norm.get("status")=="PASS"; shares_diff=shares_norm.get("relative_diff")

        # Build internally-consistent EBITDA history only if its components reconcile.
        ebitda_ts = pd.Series(dtype=float)
        if op_ts_ok and da_ts_ok:
            vals = {}
            for y in set(op_ts.index) & set(da_ts.index):
                vals[y] = float(op_ts[y]) + abs(float(da_ts[y]))
            ebitda_ts = pd.Series(dict(sorted(vals.items(), reverse=True)), dtype=float)

        # Historical EV/EBITDA target:
        # The real VNStock ratio_ts schema exposes an explicit `ev_ebitda` column.
        # Use that provider history directly, exactly as we already do for P/E.
        # This avoids inventing historical shares/market-cap from unavailable fields.
        ev_provider_target = historical_target(
            ev_provider_ts, lo=2.0, hi=35.0, min_points=4
        )

        # Optional computed EV/EBITDA remains audit-only when all inputs happen to exist.
        ev_computed_source = "computed_year_end_ev_over_canonical_ebitda_audit"
        ev_vals = {}
        direct_mc = market_cap_ts.copy()
        direct_mc_ok = False

        if latest_year in direct_mc.index and m.get("market_price") and m.get("shares"):
            expected_mc = m["market_price"] * m["shares"]
            mc = safe_float(direct_mc.get(latest_year))
            if mc and mc > 0:
                factors = (1.0,10.0,100.0,1000.0,10000.0,1000000.0)
                f = min(
                    factors,
                    key=lambda x: abs(mc*x-expected_mc)/max(expected_mc,1.0)
                )
                direct_mc = direct_mc * f
                direct_mc_ok = (
                    abs(float(direct_mc[latest_year])-expected_mc)
                    / max(expected_mc,1.0)
                    <= 0.25
                )

        common = (
            set(cash_ts.index)
            & set(short_debt_ts.index)
            & set(long_debt_ts.index)
            & set(ebitda_ts.index)
        )
        if cash_ts_ok and short_ts_ok and long_ts_ok and not ebitda_ts.empty:
            for y in common:
                ca = safe_float(cash_ts.get(y))
                sd = safe_float(short_debt_ts.get(y))
                ld = safe_float(long_debt_ts.get(y))
                eb = safe_float(ebitda_ts.get(y))
                mc = None

                if direct_mc_ok and y in direct_mc.index:
                    mc = safe_float(direct_mc.get(y))
                elif shares_ok and y in price_y.index and y in shares_hist.index:
                    px = safe_float(price_y.get(y))
                    sh = safe_float(shares_hist.get(y))
                    if px is not None and sh is not None and px > 0 and sh > 0:
                        mc = px * sh

                if None not in (mc,ca,sd,ld,eb) and mc > 0 and eb > 0:
                    ev = mc + sd + ld - ca
                    if ev > 0:
                        ev_vals[y] = ev / eb

        ev_computed_hist = pd.Series(
            dict(sorted(ev_vals.items(), reverse=True)), dtype=float
        )
        ev_computed_target = historical_target(
            ev_computed_hist, lo=2.0, hi=35.0, min_points=4
        )

        # Conservative provider-ratio bridge: when direct EV/EBITDA is zero/missing,
        # use the algebraic identity EV/EBITDA = (EV/EBIT) * (EBIT margin / EBITDA margin).
        # All three inputs come from the same annual ratio_ts row, avoiding mixed-period
        # statement reconciliation. Require 3+ cleaned annual observations.
        ev_bridge_vals={}
        common_bridge=set(ev_ebit_provider_ts.index) & set(ebit_margin_provider_ts.index) & set(ebitda_margin_provider_ts.index)
        for y in common_bridge:
            eve=safe_float(ev_ebit_provider_ts.get(y)); em=safe_float(ebit_margin_provider_ts.get(y)); ebm=safe_float(ebitda_margin_provider_ts.get(y))
            if eve is not None and eve>0 and em is not None and em>0 and ebm is not None and ebm>0:
                x=eve*(em/ebm)
                if np.isfinite(x) and x>0: ev_bridge_vals[y]=x
        ev_bridge_hist=pd.Series(dict(sorted(ev_bridge_vals.items(),reverse=True)),dtype=float)
        ev_bridge_target=historical_target(ev_bridge_hist,lo=2.0,hi=35.0,min_points=3)
        # Preserve every valid direct provider observation and fill only missing/zero years
        # from the algebraic bridge. This avoids replacing good provider history.
        ev_filled_vals={}
        years=set(ev_provider_ts.index) | set(ev_bridge_hist.index)
        for y in years:
            direct=safe_float(ev_provider_ts.get(y))
            bridge=safe_float(ev_bridge_hist.get(y))
            if direct is not None and direct>0:
                ev_filled_vals[y]=direct
            elif bridge is not None and bridge>0:
                ev_filled_vals[y]=bridge
        ev_filled_hist=pd.Series(dict(sorted(ev_filled_vals.items(),reverse=True)),dtype=float)
        ev_filled_target=historical_target(ev_filled_hist,lo=2.0,hi=35.0,min_points=3)
        if ev_provider_target.get("status")=="OK":
            ev_target=ev_provider_target; ev_source="provider_ratio_time_series"; ev_target_confidence="STANDARD_4_PLUS_PERIODS"; ev_recovery_status="NOT_NEEDED"
        elif ev_provider_target.get("count",0)>=2:
            # Preserve the already-approved direct-provider 2/3-period recovery path.
            # The bridge is for genuinely missing direct history, not for replacing good observations.
            ev_target=ev_provider_target; ev_source="provider_ratio_time_series"; ev_target_confidence=None; ev_recovery_status="NOT_NEEDED_DIRECT_2_PLUS"
        elif ev_filled_target.get("status")=="OK" and ev_filled_target.get("count",0)>=3:
            ev_target=ev_filled_target; ev_source="provider_ratio_with_ev_ebit_bridge_fill"; ev_target_confidence="RECOVERY_3_PLUS_EV_EBIT_BRIDGE"; ev_recovery_status="USED"
        else:
            ev_target=ev_provider_target; ev_source="provider_ratio_time_series"; ev_target_confidence=None; ev_recovery_status="UNAVAILABLE"

        pb_provider_target = historical_target(
            pb_provider_ts, lo=0.2, hi=20.0, min_points=4
        )

        return {
            "price_history": price_y.to_dict(),
            "price_history_meta": price_meta,
            "validation": validation,
            "pe": {**pe_target, "source":pe_source, "raw_series":pe_hist.to_dict(), "provider_series":pe_provider_ts.to_dict()},
            "ev_ebitda": {
                **ev_target,"source":ev_source,"target_confidence":ev_target_confidence,
                "recovery_status":ev_recovery_status,
                "raw_series":(ev_filled_hist.to_dict() if ev_recovery_status=="USED" else ev_provider_ts.to_dict()),
                "provider_series":ev_provider_ts.to_dict(),"provider_ev_ebit_series":ev_ebit_provider_ts.to_dict(),
                "recovered_series":ev_bridge_hist.to_dict(),"recovered_target":ev_bridge_target,
                "filled_series":ev_filled_hist.to_dict(),"filled_target":ev_filled_target,
                "recovery_formula":"EV_EBITDA = EV_EBIT * EBIT_MARGIN / EBITDA_MARGIN",
                "provider_ebit_margin_series":ebit_margin_provider_ts.to_dict(),
                "provider_ebitda_margin_series":ebitda_margin_provider_ts.to_dict(),
                "computed_audit":ev_computed_target,"computed_series":ev_computed_hist.to_dict(),
            },
            "pb_audit": {
                **pb_provider_target,
                "source":"provider_ratio_time_series",
                "raw_series":pb_provider_ts.to_dict(),
            },
            "eps_provider": {"status":"OK" if len(eps_provider_ratio_ts.dropna()) >= 1 else "NO_DATA",
                "series":eps_provider_ratio_ts.to_dict(),"count":len(eps_provider_ratio_ts.dropna()),
                "source":"provider_ratio_time_series.trailing_eps"},
            "bvps_provider": {"status":"OK" if len(bvps_provider_ratio_ts.dropna()) >= 1 else "NO_DATA",
                "series":bvps_provider_ratio_ts.to_dict(),"count":len(bvps_provider_ratio_ts.dropna()),
                "source":"provider_ratio_time_series.book_value_per_share_bvps"},
            "shares_history": shares_hist.to_dict(),
            "shares_history_source": shares_source,
            "shares_history_validation": {"status":"PASS" if shares_ok else "FAIL", "relative_diff":shares_diff, "normalization":shares_norm},
            "market_cap_history": direct_mc.to_dict(),
            "market_cap_history_validated": direct_mc_ok,
            "ebitda_history": ebitda_ts.to_dict(),
            "period_diagnostics": {
                "income_ts_rows":0 if data.get("income_ts") is None else len(data.get("income_ts")),
                "balance_ts_rows":0 if data.get("balance_ts") is None else len(data.get("balance_ts")),
                "cashflow_ts_rows":0 if data.get("cashflow_ts") is None else len(data.get("cashflow_ts")),
                "ratio_ts_rows":0 if data.get("ratio_ts") is None else len(data.get("ratio_ts")),
                "eps_periods":len(eps_ts),
                "provider_pe_periods":len(pe_provider_ts),
                "provider_eps_periods":len(eps_provider_ratio_ts),
                "provider_bvps_periods":len(bvps_provider_ratio_ts),
                "provider_pb_periods":len(pb_provider_ts),
                "provider_ev_ebitda_periods":len(ev_provider_ts),
                "provider_ev_ebit_periods":len(ev_ebit_provider_ts),
                "provider_ebit_margin_periods":len(ebit_margin_provider_ts),
                "provider_ebitda_margin_periods":len(ebitda_margin_provider_ts),
                "recovered_ev_ebitda_periods":len(ev_bridge_hist),
                "computed_pe_periods":len(computed_pe_hist),
                "share_capital_periods":len(share_capital_ts),
                "shares_ratio_periods":len(shares_ratio_ts),
                "ebitda_periods":len(ebitda_ts),
                "price_years":len(price_y),
            },
        }

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    def _validate_normal(self, m):
        checks = []
        def add(name, ok, detail, critical=False):
            checks.append({"check":name,"passed":bool(ok),"critical":critical,"detail":detail})

        add("Gia thi truong", m["market_price"] is not None, m["price_source"], True)
        add("Canonical orientation", m["orientation"] in {"normal","reversed"} and m.get("orientation_meta",{}).get("safe_verified",True) is not False, str(m["orientation_meta"]), True)
        add("Revenue", m["latest_revenue"] is not None, str(self.matched.get("revenue")), True)
        add("Net income", m["latest_net_income"] is not None, str(self.matched.get("net_income")), True)
        add("Equity", m["latest_equity"] is not None, str(self.matched.get("equity")), True)
        add("Assets", m["latest_assets"] is not None, str(self.matched.get("total_assets")), True)
        add("PBT", m["latest_pbt"] is not None, str(self.matched.get("profit_before_tax")), True)
        add(
            "Interest expense",
            m["latest_interest_expense"] is not None,
            f"source={m.get('interest_expense_source')}; row={self.matched.get('interest_expense')}",
            False,
        )
        add("D&A", m["latest_da"] is not None, str(self.matched.get("depreciation")), True)
        add(
            "EBIT resolved",
            m["latest_ebit"] is not None,
            f"source={m.get('ebit_source')}; EBIT={fmt_num(m['latest_ebit'])}; operating_profit={fmt_num(m.get('latest_operating_profit'))}",
            True,
        )
        add("EBITDA derived", m["latest_ebitda"] is not None, f"EBIT+D&A={fmt_num(m['latest_ebitda'])}", True)
        add("CFO", m["latest_cfo"] is not None, str(self.matched.get("operating_cash_flow")), True)
        add("CAPEX", m["latest_capex"] is not None, str(self.matched.get("capex")), True)
        add(
            "Delta Operating NWC cross-check",
            m["latest_delta_nwc"] is not None,
            f"{fmt_num(m['latest_delta_nwc'])}; audit-only",
            False
        )
        add(
            "FCFF primary",
            m["latest_fcff"] is not None,
            f"{fmt_num(m['latest_fcff'])}; method={m.get('fcff_method')}",
            True
        )
        add(
            "FCFF cross-check",
            m.get("latest_fcff_crosscheck") is not None,
            f"{fmt_num(m.get('latest_fcff_crosscheck'))}; method={m.get('fcff_crosscheck_method')}",
            False
        )
        add("Debt components", m["latest_total_debt"] is not None, f"short={fmt_num(m['latest_short_debt'])}, long={fmt_num(m['latest_long_debt'])}", True)
        add("Shares outstanding", m["shares"] is not None, m["shares_source"], True)

        # EPS is optional for the total valuation; if not trustworthy P/E is skipped.
        eps_ok = m["eps"] is not None and m["eps"] > 0
        add(
            "Basic EPS reconciled",
            eps_ok,
            f"used={fmt_num(m['eps'])}; raw={fmt_num(m.get('eps_raw_statement'))}; normalized={fmt_num(m.get('eps_normalized_statement'))}; scale={m.get('eps_scale_factor')}x; implied_PE={fmt_num(m.get('eps_implied_pe'))}; status={m.get('eps_validation_status')}; diff={fmt_pct(m.get('eps_reconcile_relative_diff'))}; row={self.matched.get('basic_eps')}",
            False,
        )

        if m["roe_ref"] is not None and m["roe_calc"] is not None:
            diff = abs(m["roe_ref"]-m["roe_calc"])
            add("ROE cross-check", diff <= 0.08, f"calc={fmt_pct(m['roe_calc'])}, ref={fmt_pct(m['roe_ref'])}, diff={diff*100:.2f}pp", True)
        if m["roa_ref"] is not None and m["roa_calc"] is not None:
            diff = abs(m["roa_ref"]-m["roa_calc"])
            add("ROA cross-check", diff <= 0.05, f"calc={fmt_pct(m['roa_calc'])}, ref={fmt_pct(m['roa_ref'])}, diff={diff*100:.2f}pp", False)

        add("Equity <= Assets", (m["latest_equity"] or 0) <= (m["latest_assets"] or -1),
            f"equity={fmt_num(m['latest_equity'])}, assets={fmt_num(m['latest_assets'])}", True)

        critical = [x for x in checks if x["critical"]]
        status = "PASS" if all(x["passed"] for x in critical) else "FAIL_CRITICAL"
        score = round(sum(x["passed"] for x in checks)/len(checks)*100,1)
        return {"status":status,"score":score,"checks":checks}

    def validate(self,m,sector="NORMAL"):
        if sector in {"GENERAL_NORMAL","TECHNOLOGY_ASSET_LIGHT","CONSUMER_RETAIL","UTILITIES","INDUSTRIAL_CONSTRUCTION","TRANSPORT_LOGISTICS","HOLDING_CONGLOMERATE","CYCLICAL_COMMODITY","REAL_ESTATE"}:
            if sector=="GENERAL_NORMAL":
                return self._validate_normal(m)

        checks=[]
        def add(name,ok,detail,critical=False):
            checks.append({"check":name,"passed":bool(ok),"critical":critical,"detail":detail})

        add("Gia thi truong",m.get("market_price") is not None,m.get("price_source"),True)
        add("Canonical orientation",m.get("orientation") in {"normal","reversed"} and m.get("orientation_meta",{}).get("safe_verified",True) is not False,str(m.get("orientation_meta")),True)
        add("Equity",bool(m.get("latest_equity") and m.get("latest_equity")>0),str(self.matched.get("equity")),True)
        add("Assets",bool(m.get("latest_assets") and m.get("latest_assets")>0),str(self.matched.get("total_assets")),True)
        if sector in {"SECURITIES","BANK","INSURANCE"}:
            add("Current BVPS",bool(m.get("bvps_current") and m.get("bvps_current")>0),m.get("bvps_current_source"),True)
            add("Current trailing EPS",bool(m.get("eps_provider_trailing") and m.get("eps_provider_trailing")>0),m.get("eps_current_source"),False)
        else:
            add("Shares outstanding",bool(m.get("shares") and m.get("shares")>0),m.get("shares_source"),True)
        roe=m.get("roe_ref") if m.get("roe_ref") is not None else m.get("roe_calc")
        add("ROE available",roe is not None,f"calc={fmt_pct(m.get('roe_calc'))}, ref={fmt_pct(m.get('roe_ref'))}",True)

        if sector in {"REAL_ESTATE","CYCLICAL_COMMODITY","INDUSTRIAL_CONSTRUCTION","TRANSPORT_LOGISTICS","CONSUMER_RETAIL","TECHNOLOGY_ASSET_LIGHT","UTILITIES","HOLDING_CONGLOMERATE"}:
            add("Revenue",m.get("latest_revenue") is not None,str(self.matched.get("revenue")),True)
            add("Net income",m.get("latest_net_income") is not None,str(self.matched.get("net_income")),True)
        else:
            add("Net income",m.get("latest_net_income") is not None,str(self.matched.get("net_income")),False)
            add("EPS",bool(m.get("eps") and m.get("eps")>0),str(self.matched.get("basic_eps")),False)

        # CFO/FCFF availability controls whether the DCF method can be built.
        # It is not a canonical-data hard failure for multi-method sector models:
        # the blend layer must return PARTIAL when DCF is unavailable.
        add("CFO",m.get("latest_cfo") is not None,str(self.matched.get("operating_cash_flow")),False)
        add("FCFF",m.get("latest_fcff") is not None,fmt_num(m.get("latest_fcff")),False)
        add("EBITDA",m.get("latest_ebitda") is not None,fmt_num(m.get("latest_ebitda")),False)
        add("Debt mapping",m.get("latest_total_debt") is not None,fmt_num(m.get("latest_total_debt")),False)

        h=m.get("historical_multiples") or {}
        pe=h.get("pe") or {}; pb=h.get("pb_audit") or {}; ev=h.get("ev_ebitda") or {}
        add("Historical P/E",pe.get("status")=="OK",str(pe.get("series")),False)
        add("Historical P/B",pb.get("status")=="OK",str(pb.get("series")),sector in {"SECURITIES","BANK","INSURANCE","REAL_ESTATE"})
        # Historical EV/EBITDA availability belongs to valuation completeness, not canonical data quality.
        add("Historical EV/EBITDA",ev.get("status")=="OK",str(ev.get("series")),False)

        crit=[x for x in checks if x["critical"]]
        status="PASS" if all(x["passed"] for x in crit) else "FAIL_CRITICAL"
        score=round(sum(x["passed"] for x in checks)/len(checks)*100,1)
        return {"status":status,"score":score,"checks":checks,"sector":sector}

    # --------------------------------------------------------
    # Valuation
    # --------------------------------------------------------

    def _valuation_normal(self, m, q, info):
        out = {
            "status":"NOT_RUN","reason":None,"method_values":{},
            "fair_value":None,"bear":None,"base":None,"bull":None,
            "market_price":m["market_price"],"upside":None,"margin_of_safety":None
        }
        if q["status"] != "PASS":
            out["reason"] = "DATA_QUALITY_NOT_PASS"
            return out

        sector, industry = self.sector(info)
        out["sector_type"] = sector
        out["industry"] = industry

        if not m.get("shares") or m.get("shares")<=0:
            out["reason"] = "MISSING_POSITIVE_SHARES"
            return out

        dcf_input_ok = m.get("latest_fcff") is not None and m.get("latest_fcff")>0

        # Growth: cash flow dominates. Never discard a negative FCFF CAGR.
        parts=[]
        if m["fcff_cagr"] is not None: parts.append((clip(m["fcff_cagr"],-0.40,0.40),0.50))
        if m["revenue_cagr"] is not None: parts.append((clip(m["revenue_cagr"],-0.20,0.35),0.25))
        if m["net_income_cagr"] is not None: parts.append((clip(m["net_income_cagr"],-0.25,0.40),0.25))
        raw_growth = sum(v*w for v,w in parts)/sum(w for _,w in parts) if parts else 0.08
        base_growth = clip(raw_growth,0.02,0.15)

        rf=0.045
        erp=0.065
        beta=clip(m["beta"] if m["beta"] is not None else 1.0,0.65,1.60)
        ke=clip(rf+beta*erp,0.085,0.16)
        debt=max(m["latest_total_debt"] or 0,0)
        cash=max(m["latest_cash"] or 0,0)
        market_cap=m["market_price"]*m["shares"]

        # Validate EBITDA against the market EV/EBITDA ratio when available.
        # This does not define accounting EBITDA; it is a defensive cross-check to
        # prevent an obviously mis-mapped D&A/EBIT row from entering relative valuation.
        ev_ebitda_validation = "not_tested"
        ev_ebitda_rel_diff = None
        allow_ev_ebitda = True
        if m.get("ev_ebitda_ratio") and m.get("latest_ebitda") and m.get("market_price"):
            implied_ev = market_cap + max(m.get("latest_total_debt") or 0, 0) - max(m.get("latest_cash") or 0, 0)
            if implied_ev > 0 and m["ev_ebitda_ratio"] > 0:
                implied_ebitda = implied_ev / m["ev_ebitda_ratio"]
                ev_ebitda_rel_diff = abs(m["latest_ebitda"] - implied_ebitda) / max(abs(implied_ebitda), 1.0)
                if ev_ebitda_rel_diff <= 0.25:
                    ev_ebitda_validation = "pass_market_multiple_reconciliation"
                else:
                    ev_ebitda_validation = "reject_market_multiple_mismatch"
                    allow_ev_ebitda = False

        # Keep rejection evidence even when missing methods lead to fallback/PARTIAL.
        out["input_validation"] = {"ev_ebitda": {
            "status": ev_ebitda_validation,
            "relative_diff": ev_ebitda_rel_diff,
            "allowed": allow_ev_ebitda,
        }}

        debt_s=pd.Series(m["series"]["total_debt"],dtype=float)
        avg_debt=avg_latest_two(debt_s)
        kd=clip(abs(m["latest_interest_expense"])/avg_debt,0.04,0.14) if avg_debt and m["latest_interest_expense"] is not None else 0.08
        kd_source="interest_expense/avg_debt" if avg_debt and m["latest_interest_expense"] is not None else "fallback_8pct"
        tax=clip(m["latest_tax_rate"] if m["latest_tax_rate"] is not None else 0.20,0,0.35)
        total_cap=market_cap+debt
        we=market_cap/total_cap if total_cap else 1
        wd=debt/total_cap if total_cap else 0
        wacc=clip(we*ke+wd*kd*(1-tax),0.075,0.18)
        tg=0.03

        def dcf_ps(g,w,tg_):
            if w<=tg_: return None
            f=m["latest_fcff"]
            pv=0
            path=np.linspace(g,max(tg_+0.02,0.05),5)
            for year,gg in enumerate(path,1):
                f*=1+float(gg)
                pv+=f/(1+w)**year
            tv=f*(1+tg_)/(w-tg_)
            ev=pv+tv/(1+w)**5
            eqv=ev+cash-debt
            return eqv/m["shares"] if eqv>0 else None

        if dcf_input_ok:
            dcf_bear=dcf_ps(max(0.02,base_growth-0.04),min(0.18,wacc+0.015),max(0.02,tg-0.005))
            dcf_base=dcf_ps(base_growth,wacc,tg)
            dcf_bull=dcf_ps(min(0.20,base_growth+0.04),max(0.07,wacc-0.01),min(0.04,tg+0.005))
            if dcf_base:
                out["method_values"]["dcf_fcff"]={"bear":dcf_bear,"base":dcf_base,"bull":dcf_bull,"weight":0.60,
                                                    "input_confidence":"PRIMARY_POSITIVE_FCFF"}

        # ----------------------------------------------------
        # Historical relative valuation. Target multiples must come from cleaned
        # historical data, NEVER from the current market multiple.
        # ----------------------------------------------------
        hist = m.get("historical_multiples") or {}
        pe_hist = hist.get("pe") or {}
        ev_hist = hist.get("ev_ebitda") or {}

        pe_target,pe_target_confidence = self._resolve_historical_target(pe_hist,"pe")
        # Prefer the resolved current EPS; a non-positive current EPS must not
        # silently fall back to an older profitable year.
        pe_eps = m.get("eps_current") if m.get("eps_current") is not None else m.get("eps")
        if pe_eps is not None and pe_eps>0 and pe_target:
            out["method_values"]["pe"]={
                "bear":pe_eps*pe_target*0.85,
                "base":pe_eps*pe_target,
                "bull":pe_eps*pe_target*1.15,
                "weight":0.25,
                "eps_used":pe_eps,
                "eps_source":m.get("eps_current_source") or ("eps_current" if m.get("eps_current") is not None else "latest_annual_statement"),
                "target_multiple":pe_target,
                "history_count":pe_hist.get("count"),
                "history_source":pe_hist.get("source"),
                "target_confidence":pe_target_confidence,
            }

        ev_target,ev_target_confidence = self._resolve_historical_target(ev_hist,"ev_ebitda")
        if allow_ev_ebitda and m.get("latest_ebitda") and ev_target and m.get("shares"):
            def evps_target(x):
                eqv=m["latest_ebitda"]*x+cash-debt
                return eqv/m["shares"] if eqv>0 else None
            out["method_values"]["ev_ebitda"]={
                "bear":evps_target(ev_target*0.85),
                "base":evps_target(ev_target),
                "bull":evps_target(ev_target*1.15),
                "weight":0.15,
                "target_multiple":ev_target,
                "history_count":ev_hist.get("count"),
                "history_source":ev_hist.get("source"),
                "target_confidence":ev_target_confidence,
            }

        # Exact requested formula. No silent re-weighting if one method is missing.
        blend = fixed_blend(
            out["method_values"].get("dcf_fcff"),
            out["method_values"].get("pe"),
            out["method_values"].get("ev_ebitda"),
        )
        out["blend"] = blend
        if blend["status"] != "OK_FULL":
            out["missing_methods"] = blend.get("missing_methods")
            fb=self._try_fallback("GENERAL_NORMAL",out["method_values"],"MISSING_FULL_METHODS",out["missing_methods"])
            out["fallback"] = fb
            if fb.get("status")=="OK_FALLBACK":
                for k in ("status","fair_value_type","confidence","fallback_methodology","fallback_weights",
                          "fallback_methods","fallback_trigger","method_agreement_ratio","bear","base","bull","fair_value"):
                    out[k]=fb.get(k)
                out["reason"]="APPROVED_SECTOR_FALLBACK"
                if out["fair_value"] and m.get("market_price"):
                    out["upside"]=out["fair_value"]/m["market_price"]-1
                    out["margin_of_safety"]=1-m["market_price"]/out["fair_value"]
                return out
            out["status"] = "PARTIAL"
            out["reason"] = "NO_APPROVED_FALLBACK_FOR_MISSING_FULL_METHODS"
            out["bear"] = out["base"] = out["bull"] = out["fair_value"] = None
            return out

        sanity=self._cross_method_sanity(out["method_values"])
        out["cross_method_sanity"]=sanity
        if sanity.get("status")=="FAIL_DCF_EXTREME":
            fb=self._try_fallback("GENERAL_NORMAL",out["method_values"],"CROSS_METHOD_DIVERGENCE",exclude_methods={"dcf_fcff"})
            out["fallback"]=fb
            if fb.get("status")=="OK_FALLBACK":
                for k in ("status","fair_value_type","confidence","fallback_methodology","fallback_weights",
                          "fallback_methods","fallback_trigger","method_agreement_ratio","bear","base","bull","fair_value"):
                    out[k]=fb.get(k)
                out["reason"]="APPROVED_FALLBACK_AFTER_DCF_DIVERGENCE"
                if out["fair_value"] and m.get("market_price"):
                    out["upside"]=out["fair_value"]/m["market_price"]-1
                    out["margin_of_safety"]=1-m["market_price"]/out["fair_value"]
                return out
            out["status"]="PARTIAL"; out["reason"]="CROSS_METHOD_DIVERGENCE"
            out["bear"]=out["base"]=out["bull"]=out["fair_value"]=None
            return out
        out["bear"] = blend["bear"]
        out["base"] = blend["base"]
        out["bull"] = blend["bull"]
        out["fair_value"] = out["base"]
        if out["fair_value"] and m["market_price"]:
            out["upside"] = out["fair_value"]/m["market_price"]-1
            out["margin_of_safety"] = 1-m["market_price"]/out["fair_value"]
        out["assumptions"]={
            "base_growth":base_growth,"raw_growth":raw_growth,
            "fcff_cagr":m["fcff_cagr"],"revenue_cagr":m["revenue_cagr"],"net_income_cagr":m["net_income_cagr"],
            "risk_free":rf,"erp":erp,"beta":beta,"cost_of_equity":ke,
            "cost_of_debt":kd,"cost_of_debt_source":kd_source,
            "tax_rate":tax,"wacc":wacc,"terminal_growth":tg,
            "debt":debt,"cash":cash,"shares":m["shares"],
            "year_orientation":m["orientation"],
            "fcff_method":m.get("fcff_method"),
            "fcff_crosscheck":m.get("latest_fcff_crosscheck"),
            "eps_validation_status":m.get("eps_validation_status"),
            "eps_reconcile_relative_diff":m.get("eps_reconcile_relative_diff"),
            "ev_ebitda_validation":ev_ebitda_validation,
            "ev_ebitda_reconcile_relative_diff":ev_ebitda_rel_diff,
            "historical_pe_target":(m.get("historical_multiples") or {}).get("pe",{}).get("target"),
            "historical_ev_ebitda_target":(m.get("historical_multiples") or {}).get("ev_ebitda",{}).get("target"),
            "blend_weights":{"dcf_fcff":0.60,"pe":0.25,"ev_ebitda":0.15},
        }
        out["status"]="OK_FULL"
        out["fair_value_type"]="FULL_FAIR_VALUE"
        out["confidence"]=self._full_confidence(out["method_values"])
        self._apply_dispersion_confidence(out)
        return out

    def _resolve_historical_target(self, hist, kind):
        """Conservative historical-multiple recovery; never uses current multiple as fair target."""
        hist=hist or {}
        target=safe_float(hist.get("target")) if hist.get("status")=="OK" else None
        if target is not None and target>0:
            return float(target),str(hist.get("target_confidence") or "STANDARD_4_PLUS_PERIODS")

        vals=[safe_float(x) for x in (hist.get("series") or {}).values()]
        vals=[float(x) for x in vals if x is not None and x>0]
        raw=hist.get("provider_series") or hist.get("raw_series") or {}
        raw_n=len(raw)
        if kind=="pe" and len(vals)>=3:
            return float(np.median(vals)),"FALLBACK_3_VALID_PERIODS"
        if len(vals)>=2 and raw_n>=4:
            return float(np.median(vals)),"LOW_CONFIDENCE_RECOVERY_2_VALID_PERIODS"
        return None,None

    def _method_evidence_confidence(self, method):
        if not isinstance(method,dict):
            return "LOW"
        tags=[]
        for k in ("target_confidence","recovery_status"):
            if method.get(k):
                tags.append(str(method.get(k)))
        tm=method.get("target_multiple_method")
        if isinstance(tm,dict) and tm.get("confidence"):
            tags.append(str(tm.get("confidence")))
        text=" ".join(tags).upper()
        if "LOW_CONFIDENCE" in text or "2_VALID_PERIODS" in text:
            return "LOW"
        if "FALLBACK_3" in text or "RECOVER" in text:
            return "MEDIUM"
        return "HIGH"

    def _full_confidence(self, methods, proxy=False):
        levels=[self._method_evidence_confidence(x) for x in (methods or {}).values() if isinstance(x,dict)]
        if "LOW" in levels:
            return "LOW" if proxy else "MEDIUM"
        if "MEDIUM" in levels or proxy:
            return "MEDIUM"
        return "HIGH"

    def _fallback_specs(self, sector):
        """Fixed sector fallbacks. They are not generic re-weighting of missing full-model weights."""
        specs={
            "BANK":[
                ("BANK_FALLBACK_PB_RI",{"pb":0.60,"residual_income":0.40}),
                ("BANK_FALLBACK_PB_PE",{"pb":0.65,"pe":0.35}),
            ],
            "SECURITIES":[("SECURITIES_FALLBACK_PB_PE",{"pb":0.60,"pe":0.40})],
            "INSURANCE":[("INSURANCE_FALLBACK_PB_PE",{"pb":0.60,"pe":0.40})],
            "REAL_ESTATE":[("REAL_ESTATE_FALLBACK_PB_PE",{"pb":0.65,"pe":0.35})],
            "CYCLICAL_COMMODITY":[
                ("CYCLICAL_FALLBACK_EV_PE",{"ev_ebitda":0.65,"pe":0.35}),
                ("CYCLICAL_FALLBACK_PE_PB",{"pe":0.60,"pb":0.40}),
                ("CYCLICAL_FALLBACK_EV_PB",{"ev_ebitda":0.70,"pb":0.30}),
            ],
            "UTILITIES":[
                ("UTILITIES_FALLBACK_DCF_EV",{"dcf_fcff":0.60,"ev_ebitda":0.40}),
                ("UTILITIES_FALLBACK_DCF_PB",{"dcf_fcff":0.65,"pb":0.35}),
                ("UTILITIES_FALLBACK_EV_PB",{"ev_ebitda":0.65,"pb":0.35}),
            ],
            "TECHNOLOGY_ASSET_LIGHT":[
                ("TECH_FALLBACK_DCF_PE",{"dcf_fcff":0.60,"pe":0.40}),
                ("TECH_FALLBACK_PE_EV",{"pe":0.70,"ev_ebitda":0.30}),
                ("TECH_FALLBACK_DCF_EV",{"dcf_fcff":0.70,"ev_ebitda":0.30}),
                ("TECH_FALLBACK_PE_PB_LOW",{"pe":0.75,"pb":0.25}),
            ],
            "CONSUMER_RETAIL":[
                ("CONSUMER_FALLBACK_DCF_PE",{"dcf_fcff":0.55,"pe":0.45}),
                ("CONSUMER_FALLBACK_PE_EV",{"pe":0.70,"ev_ebitda":0.30}),
                ("CONSUMER_FALLBACK_DCF_EV",{"dcf_fcff":0.70,"ev_ebitda":0.30}),
                ("CONSUMER_FALLBACK_PE_PB_LOW",{"pe":0.70,"pb":0.30}),
            ],
            "INDUSTRIAL_CONSTRUCTION":[
                ("INDUSTRIAL_FALLBACK_EV_PE",{"ev_ebitda":0.60,"pe":0.40}),
                ("INDUSTRIAL_FALLBACK_EV_DCF",{"ev_ebitda":0.65,"dcf_fcff":0.35}),
                ("INDUSTRIAL_FALLBACK_PE_DCF",{"pe":0.60,"dcf_fcff":0.40}),
                ("INDUSTRIAL_FALLBACK_PE_PB_LOW",{"pe":0.65,"pb":0.35}),
            ],
            "TRANSPORT_LOGISTICS":[
                ("TRANSPORT_FALLBACK_EV_DCF",{"ev_ebitda":0.60,"dcf_fcff":0.40}),
                ("TRANSPORT_FALLBACK_EV_PE",{"ev_ebitda":0.70,"pe":0.30}),
                ("TRANSPORT_FALLBACK_DCF_PE",{"dcf_fcff":0.65,"pe":0.35}),
                ("TRANSPORT_FALLBACK_PE_PB_LOW",{"pe":0.70,"pb":0.30}),
            ],
            "HOLDING_CONGLOMERATE":[
                ("HOLDING_FALLBACK_PB_PE",{"pb":0.65,"pe":0.35}),
                ("HOLDING_FALLBACK_PB_DCF",{"pb":0.70,"dcf_fcff":0.30}),
                ("HOLDING_FALLBACK_PE_DCF",{"pe":0.60,"dcf_fcff":0.40}),
            ],
            "GENERAL_NORMAL":[
                ("GENERAL_FALLBACK_DCF_PE",{"dcf_fcff":0.65,"pe":0.35}),
                ("GENERAL_FALLBACK_PE_EV",{"pe":0.70,"ev_ebitda":0.30}),
                ("GENERAL_FALLBACK_DCF_EV",{"dcf_fcff":0.75,"ev_ebitda":0.25}),
            ],
        }
        return specs.get(sector,[])

    def _try_fallback(self, sector, methods, trigger, full_missing=None, exclude_methods=None):
        exclude_methods=set(exclude_methods or [])
        attempts=[]
        for name,weights in self._fallback_specs(sector):
            if any(k in exclude_methods for k in weights):
                attempts.append({"methodology":name,"status":"EXCLUDED_TRIGGER_METHOD","weights":weights})
                continue
            blend=self._blend_sector(methods,weights)
            if blend.get("status")!="OK_FULL":
                attempts.append({"methodology":name,"status":"MISSING_METHODS","weights":weights,
                                 "missing_methods":blend.get("missing_methods"),
                                 "invalid_method_scenarios":blend.get("invalid_method_scenarios")})
                continue
            bases=[float(methods[k]["base"]) for k in weights]
            lo=min(bases); hi=max(bases)
            ratio=hi/lo if lo>0 else None
            if ratio is None or ratio>2.25:
                attempts.append({"methodology":name,"status":"METHOD_DISAGREEMENT","weights":weights,
                                 "method_agreement_ratio":ratio,"max_allowed_ratio":2.25})
                continue
            confidence="MEDIUM" if ratio<=1.50 else "LOW"
            if any(self._method_evidence_confidence(methods[k])=="LOW" for k in weights):
                confidence="LOW"
            return {
                "status":"OK_FALLBACK","fair_value_type":"FALLBACK_FAIR_VALUE",
                "confidence":confidence,"fallback_methodology":name,
                "fallback_weights":weights,"fallback_methods":list(weights),
                "fallback_trigger":trigger,"full_missing_methods":list(full_missing or []),
                "method_agreement_ratio":ratio,"blend":blend,
                "bear":blend["bear"],"base":blend["base"],"bull":blend["bull"],
                "fair_value":blend["base"],"fallback_attempts":attempts,
            }
        return {"status":"REJECTED","fallback_trigger":trigger,
                "full_missing_methods":list(full_missing or []),"fallback_attempts":attempts}

    def _method_dispersion_audit(self, method_values, blend):
        weights=(blend or {}).get("weights") if isinstance(blend,dict) else None
        if not isinstance(weights,dict) or len(weights)<2: return {"status":"NOT_APPLICABLE","ratio":None,"bases":{}}
        bases={}
        for k in weights:
            d=(method_values or {}).get(k); v=safe_float(d.get("base")) if isinstance(d,dict) else None
            if v is not None and np.isfinite(v) and v>0: bases[k]=float(v)
        if len(bases)<2: return {"status":"NOT_ENOUGH_METHODS","ratio":None,"bases":bases}
        lo=min(bases.values()); hi=max(bases.values()); ratio=hi/lo if lo>0 else None
        return {"status":"HIGH_DISPERSION" if ratio is not None and ratio>2.25 else "PASS","ratio":ratio,"threshold":2.25,
                "bases":bases,"min_method":min(bases,key=bases.get),"max_method":max(bases,key=bases.get)}

    def _apply_dispersion_confidence(self,out):
        if not isinstance(out,dict) or out.get("status") not in {"OK_FULL","OK_PROXY_NO_RNAV","OK_PROXY_NO_SOTP"}: return out
        audit=self._method_dispersion_audit(out.get("method_values"),out.get("blend")); out["method_dispersion"]=audit
        if audit.get("status")=="HIGH_DISPERSION":
            ratio=audit.get("ratio") or 0; old=str(out.get("confidence") or "HIGH"); cap="LOW" if ratio>4.0 else "MEDIUM"; rank={"HIGH":3,"MEDIUM":2,"LOW":1}
            out["confidence_before_dispersion"]=old; out["confidence"]=old if rank.get(old,1)<=rank[cap] else cap
        return out

    def _cross_method_sanity(self, method_values, threshold=8.0):
        bases={}
        for k,d in (method_values or {}).items():
            if isinstance(d,dict):
                v=safe_float(d.get("base"))
                if v is not None and np.isfinite(v) and v>0:
                    bases[k]=float(v)
        if len(bases)<3 or "dcf_fcff" not in bases:
            return {"status":"PASS_OR_NOT_APPLICABLE","bases":bases}
        others=[v for k,v in bases.items() if k!="dcf_fcff"]
        if len(others)<2:
            return {"status":"PASS_OR_NOT_APPLICABLE","bases":bases}
        anchor=float(np.median(others)); ratio=bases["dcf_fcff"]/anchor if anchor>0 else None
        if ratio is not None and ratio>threshold:
            return {"status":"FAIL_DCF_EXTREME","bases":bases,"relative_anchor_median":anchor,"dcf_to_relative_ratio":ratio,"threshold":threshold}
        return {"status":"PASS","bases":bases,"relative_anchor_median":anchor,"dcf_to_relative_ratio":ratio,"threshold":threshold}

    def _median_positive(self,m,key):
        xs=[]
        for v in (m.get("series",{}).get(key,{}) or {}).values():
            x=safe_float(v)
            if x is not None and x>0: xs.append(float(x))
        return float(np.median(xs)) if xs else None

    def _sustainable_roe_financial(self,m):
        vals=[]; points=[]
        raw=(m.get("series",{}).get("roe",{}) or {})
        def ykey(k):
            try: return int(str(k)[:4])
            except Exception: return -1
        for k in sorted(raw.keys(),key=ykey,reverse=True):
            x=safe_float(raw.get(k))
            if x is None: continue
            if abs(x)>2: x/=100.0
            if -0.50 <= x <= 1.00:
                vals.append(float(x)); points.append({"period":str(k),"value":float(x),"source":"annual_provider_roe"})
        cur=safe_float(m.get("roe_ref"))
        if cur is not None:
            if abs(cur)>2: cur/=100.0
            if -0.50 <= cur <= 1.00:
                vals.append(float(cur)); points.append({"period":m.get("latest_data_period"),"value":float(cur),"source":"latest_provider_trailing_roe"})
        if not vals: return None,{"method":"median_annual_plus_latest_trailing","status":"NO_DATA","points":[]}
        value=float(np.median(vals))
        return value,{"method":"median_annual_plus_latest_trailing","status":"OK","points":points,"count":len(vals),"median":value}


    def _normalized_eps_financial(self,m):
        h=(m.get("historical_multiples") or {}).get("eps_provider") or {}
        raw=h.get("series") or {}; vals=[]; points=[]
        def ykey(k):
            try: return int(str(k)[:4])
            except Exception: return -1
        for k in sorted(raw.keys(),key=ykey,reverse=True):
            x=safe_float(raw.get(k))
            if x is not None and x>0:
                vals.append(float(x)); points.append({"period":str(k),"value":float(x),"source":"annual_provider_trailing_eps"})
        cur=safe_float(m.get("eps_provider_trailing"))
        if cur is not None and cur>0:
            vals.append(float(cur)); points.append({"period":m.get("latest_data_period"),"value":float(cur),"source":"latest_provider_trailing_eps"})
        if not vals:
            fallback=safe_float(m.get("eps")); return fallback,{"method":"provider_trailing_eps_median","status":"FALLBACK_ANNUAL_STATEMENT" if fallback else "NO_DATA","points":[],"median":fallback}
        value=float(np.median(vals)); return value,{"method":"provider_trailing_eps_median","status":"OK","points":points,"count":len(vals),"median":value}



    def _scenario(self,base):
        return None if base is None or base<=0 else {"bear":base*0.85,"base":base,"bull":base*1.15}

    def _blend_sector(self,methods,weights):
        scenarios=("bear","base","bull")
        missing=[]; invalid={}
        for k in weights:
            d=methods.get(k)
            if not isinstance(d,dict):
                missing.append(k); continue
            bad=[]
            for c in scenarios:
                v=safe_float(d.get(c))
                if v is None or not np.isfinite(v) or v < 0:
                    bad.append(c)
            if bad: invalid[k]=bad
        if missing or invalid:
            return {"status":"PARTIAL","missing_methods":missing,
                    "invalid_method_scenarios":invalid,"weights":weights,
                    "bear":None,"base":None,"bull":None}
        z={"status":"OK_FULL","missing_methods":[],"invalid_method_scenarios":{},"weights":weights}
        for c in scenarios:
            z[c]=sum(float(methods[k][c])*weights[k] for k in weights)
        return z

    def _normal_dcf_scenario_only(self,m,info):
        """
        Reuse the unchanged normal-company valuation engine only to obtain
        its DCF scenario. Relative-value components are ignored here.
        """
        q={"status":"PASS"}
        v=self._valuation_normal(m,q,info)
        d=(v.get("method_values") or {}).get("dcf_fcff")
        return d if isinstance(d,dict) else None

    def _valuation_sector(self,m,q,sector,industry):
        out={"status":"NOT_RUN","reason":None,"method_values":{},"fair_value":None,
             "bear":None,"base":None,"bull":None,"market_price":m.get("market_price"),
             "upside":None,"margin_of_safety":None,"sector_type":sector,"industry":industry}
        if q.get("status")!="PASS":
            out["reason"]="DATA_QUALITY_NOT_PASS_FOR_SECTOR_MODEL"; return out

        sh=m.get("shares"); eq=m.get("latest_equity")
        financial_sector=sector in {"SECURITIES","BANK","INSURANCE"}
        if financial_sector:
            if not m.get("bvps_current") or m.get("bvps_current")<=0:
                out["reason"]="MISSING_POSITIVE_CURRENT_BVPS"; return out
        elif not sh or not eq or sh<=0 or eq<=0:
            out["reason"]="MISSING_POSITIVE_EQUITY_OR_SHARES"; return out

        h=m.get("historical_multiples") or {}
        peh=h.get("pe") or {}; pbh=h.get("pb_audit") or {}; evh=h.get("ev_ebitda") or {}

        pe_t,pe_target_confidence=self._resolve_historical_target(peh,"pe")

        pb_hist=pbh.get("target") if pbh.get("status")=="OK" else None

        ev_t,ev_target_confidence=self._resolve_historical_target(evh,"ev_ebitda")

        # Reuse existing DCF engine for families where DCF is conceptually appropriate.
        dcf_s=None
        if sector in {"TECHNOLOGY_ASSET_LIGHT","CONSUMER_RETAIL","UTILITIES","TRANSPORT_LOGISTICS","INDUSTRIAL_CONSTRUCTION","HOLDING_CONGLOMERATE"}:
            dcf_s=self._normal_dcf_scenario_only(m, None)
            if dcf_s:
                out["method_values"]["dcf_fcff"]=dcf_s

        bvps=m.get("bvps_current") if financial_sector else (eq/sh)
        if financial_sector:
            roe,roe_norm_meta=self._sustainable_roe_financial(m)
        else:
            roe=m.get("roe_ref") if m.get("roe_ref") is not None else m.get("roe_calc")
            if roe is not None and abs(roe)>2: roe/=100.0
            roe_norm_meta={"method":"existing_nonfinancial_reference","value":roe}
        rf,erp,tg=0.045,0.065,0.03
        beta=clip(m.get("beta") if m.get("beta") is not None else 1.0,0.65,1.60)
        ke=clip(rf+beta*erp,0.085,0.16)

        pb_just=None
        if roe is not None and ke>tg:
            x=(roe-tg)/(ke-tg)
            if x>0: pb_just=x
        pb_t=None
        if pb_hist:
            adj=clip(pb_just,pb_hist*0.75,pb_hist*1.50) if pb_just else pb_hist
            pb_t=0.70*pb_hist+0.30*adj
        elif pb_just:
            pb_t=pb_just

        pb_base=bvps*pb_t if pb_t else None
        pb=self._scenario(pb_base)
        if pb:
            pb.update({"target_multiple":pb_t,"historical_target":pb_hist,"justified_pb":pb_just,"bvps":bvps})
            out["method_values"]["pb"]=pb

        eps_norm_meta=None
        if financial_sector:
            norm_eps,eps_norm_meta=self._normalized_eps_financial(m)
        else:
            if sector=="REAL_ESTATE":
                nni=self._median_positive(m,"parent_profit")
                earnings_basis="parent_profit"
                if nni is None:
                    nni=self._median_positive(m,"net_income")
                    earnings_basis="net_income_fallback"
                norm_eps=nni/sh if nni else None
                eps_norm_meta={"method":"median_annual_parent_profit_on_current_share_basis",
                               "earnings_basis":earnings_basis,
                               "current_shares":sh,
                               "normalized_eps":norm_eps}
            else:
                nni=self._median_positive(m,"net_income")
                norm_eps=nni/sh if nni else None
        pe_target_used=pe_t
        pe_target_meta=({"method":"cleaned_historical_median_after_earnings_normalization",
                              "historical_median":pe_t,
                              "confidence":pe_target_confidence} if pe_t else None)
        pe_raw=norm_eps*pe_target_used if norm_eps and pe_target_used else None
        pe_base=pe_raw
        if pe_base and pb_base and sector in {"SECURITIES","BANK","INSURANCE","REAL_ESTATE"}:
            pe_base=clip(pe_base,pb_base*0.60,pb_base*1.80)
        pe=self._scenario(pe_base)
        if pe:
            pe.update({"target_multiple":pe_target_used,"historical_median_target":pe_t,
                       "target_multiple_method":pe_target_meta,"target_confidence":pe_target_confidence,
                       "normalized_eps":norm_eps,"raw_base":pe_raw,
                       "eps_normalization":eps_norm_meta})
            out["method_values"]["pe"]=pe

        if sector in {"SECURITIES","BANK","INSURANCE","REAL_ESTATE"} and roe is not None and ke>tg:
            ri_raw=bvps+bvps*(roe-ke)/(ke-tg)
            ri_base=ri_raw if ri_raw>0 else None
            if ri_base and pb_base: ri_base=clip(ri_base,pb_base*0.60,pb_base*1.60)
            ri=self._scenario(ri_base)
            if ri:
                ri.update({"normalized_roe":roe,"cost_of_equity":ke,"terminal_growth":tg,"raw_base":ri_raw})
                out["method_values"]["residual_income"]=ri

        if sector in {"CYCLICAL_COMMODITY","UTILITIES","INDUSTRIAL_CONSTRUCTION","TRANSPORT_LOGISTICS",
                           "TECHNOLOGY_ASSET_LIGHT","CONSUMER_RETAIL"}:
            neb=self._median_positive(m,"ebitda")
            debt=max(m.get("latest_total_debt") or 0,0); cash=max(m.get("latest_cash") or 0,0)
            ev_base=None
            if neb and ev_t:
                eqv=neb*ev_t+cash-debt
                ev_base=eqv/sh if eqv>0 else None
            ev=self._scenario(ev_base)
            if ev:
                ev.update({"target_multiple":ev_t,"normalized_ebitda":neb,
                           "target_confidence":ev_target_confidence})
                out["method_values"]["ev_ebitda"]=ev

        if sector=="SECURITIES":
            w={"pb":0.50,"pe":0.30,"residual_income":0.20}
            method="SECURITIES_PB_ROE_PE_RESIDUAL_INCOME"

        elif sector=="BANK":
            w={"pb":0.50,"residual_income":0.30,"pe":0.20}
            method="BANK_PB_ROE_RESIDUAL_INCOME_PE"

        elif sector=="INSURANCE":
            w={"pb":0.45,"pe":0.30,"residual_income":0.25}
            method="INSURANCE_PB_PE_RESIDUAL_INCOME"

        elif sector=="REAL_ESTATE":
            w={"pb":0.50,"pe":0.30,"residual_income":0.20}
            method="REAL_ESTATE_PROXY_PB_PE_RESIDUAL_INCOME"
            out["rnav_status"]="RNAV_NOT_AVAILABLE_FROM_GENERIC_VNSTOCK"

        elif sector=="CYCLICAL_COMMODITY":
            w={"ev_ebitda":0.50,"pe":0.30,"pb":0.20}
            method="CYCLICAL_COMMODITY_NORMALIZED_EV_EBITDA_PE_PB"

        elif sector=="UTILITIES":
            w={"dcf_fcff":0.50,"ev_ebitda":0.30,"pb":0.20}
            method="UTILITIES_DCF_EV_EBITDA_PB"

        elif sector=="TECHNOLOGY_ASSET_LIGHT":
            w={"dcf_fcff":0.50,"pe":0.35,"ev_ebitda":0.15}
            method="TECH_ASSET_LIGHT_DCF_PE_EV_EBITDA"

        elif sector=="CONSUMER_RETAIL":
            w={"dcf_fcff":0.45,"pe":0.40,"ev_ebitda":0.15}
            method="CONSUMER_RETAIL_DCF_PE_EV_EBITDA"

        elif sector=="INDUSTRIAL_CONSTRUCTION":
            w={"ev_ebitda":0.40,"pe":0.35,"dcf_fcff":0.25}
            method="INDUSTRIAL_CONSTRUCTION_EV_EBITDA_PE_DCF"

        elif sector=="TRANSPORT_LOGISTICS":
            w={"ev_ebitda":0.45,"dcf_fcff":0.35,"pe":0.20}
            method="TRANSPORT_LOGISTICS_EV_EBITDA_DCF_PE"

        elif sector=="HOLDING_CONGLOMERATE":
            # SOTP is preferred, but generic VNStock data is not segment/project level.
            # Use a transparent proxy rather than pretending a SOTP was calculated.
            w={"pb":0.50,"pe":0.30,"dcf_fcff":0.20}
            method="HOLDING_PROXY_PB_PE_DCF"
            out["sotp_status"]="SOTP_NOT_AVAILABLE_FROM_GENERIC_VNSTOCK"

        else:
            out["reason"]="UNSUPPORTED_SECTOR"
            return out

        blend=self._blend_sector(out["method_values"],w)
        out["blend"]=blend; out["methodology"]=method
        out["assumptions"]={"normalized_roe":roe,"bvps":bvps,"risk_free":rf,"erp":erp,"beta":beta,
                            "cost_of_equity":ke,"terminal_growth":tg,"historical_pe_target":pe_t,
                            "pe_target_used":pe_target_used,"pe_target_method":pe_target_meta,
                            "historical_pb_target":pb_hist,"historical_ev_ebitda_target":ev_t,
                            "historical_pe_target_confidence":pe_target_confidence,
                            "historical_ev_ebitda_target_confidence":ev_target_confidence,
                            "sector_weights":w,"roe_normalization":roe_norm_meta,
                            "eps_normalization":eps_norm_meta,
                            "share_basis":{"current_shares":m.get("shares"),
                                           "reported_shares_estimate":m.get("reported_shares_estimate"),
                                           "share_change_ratio":m.get("share_change_ratio"),
                                           "post_report_share_change_detected":m.get("post_report_share_change_detected"),
                                           "valuation_bvps_source":"latest_equity/current_shares" if not financial_sector else m.get("bvps_current_source")},
                            "note":"Financial-sector P/E uses independent normalized EPS; REAL_ESTATE is a proxy without RNAV; normal-company FCFF is not used unless sector=GENERAL_NORMAL."}
        if blend["status"]!="OK_FULL":
            out["missing_methods"]=blend.get("missing_methods")
            out["invalid_method_scenarios"]=blend.get("invalid_method_scenarios")
            fb=self._try_fallback(sector,out["method_values"],"MISSING_FULL_METHODS",out["missing_methods"])
            out["fallback"]=fb
            if fb.get("status")=="OK_FALLBACK":
                for k in ("status","fair_value_type","confidence","fallback_methodology","fallback_weights",
                          "fallback_methods","fallback_trigger","method_agreement_ratio","bear","base","bull","fair_value"):
                    out[k]=fb.get(k)
                out["reason"]="APPROVED_SECTOR_FALLBACK"
                if out.get("market_price") and out.get("fair_value"):
                    out["upside"]=out["fair_value"]/out["market_price"]-1
                    out["margin_of_safety"]=1-out["market_price"]/out["fair_value"]
                return out
            out["status"]="PARTIAL"; out["reason"]="NO_APPROVED_FALLBACK_FOR_MISSING_FULL_METHODS"
            return out

        if sector=="UTILITIES":
            # Apply the existing 2.25 agreement limit to the methods actually
            # used in the utility blend; the audit-only P/E is not an anchor.
            active_methods={k:out["method_values"][k] for k in w}
            sanity=self._cross_method_sanity(active_methods,threshold=2.25)
        else:
            sanity=self._cross_method_sanity(out["method_values"])
        out["cross_method_sanity"]=sanity
        if sanity.get("status")=="FAIL_DCF_EXTREME":
            if sector=="UTILITIES":
                out["full_blend_audit"]=out["blend"]
            fb=self._try_fallback(sector,out["method_values"],"CROSS_METHOD_DIVERGENCE",exclude_methods={"dcf_fcff"})
            out["fallback"]=fb
            if fb.get("status")=="OK_FALLBACK":
                for k in ("status","fair_value_type","confidence","fallback_methodology","fallback_weights",
                          "fallback_methods","fallback_trigger","method_agreement_ratio","bear","base","bull","fair_value"):
                    out[k]=fb.get(k)
                out["reason"]="APPROVED_FALLBACK_AFTER_DCF_DIVERGENCE"
                if sector=="UTILITIES":
                    out["blend"]=fb["blend"]
                if out.get("market_price") and out.get("fair_value"):
                    out["upside"]=out["fair_value"]/out["market_price"]-1
                    out["margin_of_safety"]=1-out["market_price"]/out["fair_value"]
                return out
            out["status"]="PARTIAL"; out["reason"]="CROSS_METHOD_DIVERGENCE"
            out["bear"]=out["base"]=out["bull"]=out["fair_value"]=None
            if sector=="UTILITIES":
                out["blend"]={**blend,"status":"PARTIAL","bear":None,"base":None,"bull":None,
                              "reason":"CROSS_METHOD_DIVERGENCE"}
            return out
        out["bear"],out["base"],out["bull"]=blend["bear"],blend["base"],blend["bull"]
        out["fair_value"]=out["base"]
        if sector=="REAL_ESTATE":
            out["status"]="OK_PROXY_NO_RNAV"
        elif sector=="HOLDING_CONGLOMERATE":
            out["status"]="OK_PROXY_NO_SOTP"
        else:
            out["status"]="OK_FULL"
        out["fair_value_type"]="FULL_FAIR_VALUE"
        out["confidence"]=self._full_confidence(out["method_values"],proxy=sector in {"REAL_ESTATE","HOLDING_CONGLOMERATE"})
        self._apply_dispersion_confidence(out)
        if out["market_price"] and out["fair_value"]:
            out["upside"]=out["fair_value"]/out["market_price"]-1
            out["margin_of_safety"]=1-out["market_price"]/out["fair_value"]
        return out

    def valuation(self,m,q,info):
        sector,industry=self.sector(info)
        if sector=="GENERAL_NORMAL":
            return self._valuation_normal(m,q,info)
        return self._valuation_sector(m,q,sector,industry)

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    def save(self,m,q,v):
        (self.base_dir/"metrics.json").write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding="utf-8")
        (self.base_dir/"data_quality.json").write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding="utf-8")
        (self.base_dir/"valuation.json").write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding="utf-8")

    def print_report(self,m,q,v):
        print("\n"+"="*78)
        print(f"VN STOCK FAIR VALUE V8.4 FALLBACK-CONFIDENCE - {self.symbol}")
        print("="*78)
        print(f"Auth                     : {m['auth_status']}")
        print(f"Market price             : {fmt_num(m['market_price'])} ({m['price_source']})")
        print(f"Year orientation         : {m['orientation']}  {m['orientation_meta']}")
        print(f"Annual periods available : {m['annual_periods']}")
        print(f"Current-year annual used : {m.get('annual_current_calendar_year_accepted')}")
        print(f"Latest data period       : {m.get('latest_data_period') or m.get('latest_year')}")
        print(f"Current fundamentals     : {m.get('current_basis') or 'LATEST_ANNUAL'}")
        print(f"Sector model             : {m.get('sector_type')} ({m.get('industry')})")
        print("-"*78)
        print(f"Revenue                  : {fmt_num(m['latest_revenue'])}")
        print(f"Net income               : {fmt_num(m['latest_net_income'])}")
        print(f"Parent profit            : {fmt_num(m['latest_parent_profit'])}")
        print(f"PBT                      : {fmt_num(m['latest_pbt'])}")
        print(f"Operating profit         : {fmt_num(m.get('latest_operating_profit'))}")
        print(f"Interest expense         : {fmt_num(m['latest_interest_expense'])} ({m.get('interest_expense_source')})")
        print(f"EBIT                     : {fmt_num(m['latest_ebit'])} ({m.get('ebit_source')})")
        print(f"D&A                      : {fmt_num(m['latest_da'])}")
        print(f"EBITDA = EBIT + D&A      : {fmt_num(m['latest_ebitda'])}")
        print(f"EPS statement raw/norm   : {fmt_num(m.get('eps_raw_statement'))} / {fmt_num(m.get('eps_normalized_statement'))}")
        print(f"EPS current / source     : {fmt_num(m.get('eps_current'))} / {m.get('eps_current_source')}")
        print(f"BVPS current / source    : {fmt_num(m.get('bvps_current'))} / {m.get('bvps_current_source')}")
        print(f"EPS scale / implied P-E  : {m.get('eps_scale_factor')}x / {fmt_num(m.get('eps_implied_pe'))}")
        print(f"EPS validation           : {m.get('eps_validation_status')} diff={fmt_pct(m.get('eps_reconcile_relative_diff'))}")
        print("-"*78)
        print(f"Equity                   : {fmt_num(m['latest_equity'])}")
        print(f"Assets                   : {fmt_num(m['latest_assets'])}")
        print(f"Cash                     : {fmt_num(m['latest_cash'])}")
        print(f"Short debt               : {fmt_num(m['latest_short_debt'])}")
        print(f"Long debt                : {fmt_num(m['latest_long_debt'])}")
        print(f"Total debt               : {fmt_num(m['latest_total_debt'])}")
        print(f"CFO                      : {fmt_num(m['latest_cfo'])}")
        print(f"CAPEX                    : {fmt_num(m['latest_capex'])}")
        print(f"Delta Operating NWC      : {fmt_num(m['latest_delta_nwc'])}  [audit only]")
        print(f"FCFF primary             : {fmt_num(m['latest_fcff'])}")
        print(f"FCFF method              : {m.get('fcff_method')}")
        print(f"FCFF cross-check         : {fmt_num(m.get('latest_fcff_crosscheck'))}")
        print("-"*78)
        print(f"Revenue CAGR             : {fmt_pct(m['revenue_cagr'])}")
        print(f"Net income CAGR          : {fmt_pct(m['net_income_cagr'])}")
        print(f"FCFF CAGR                : {fmt_pct(m['fcff_cagr'])}")
        print(f"ROE calc / ref           : {fmt_pct(m['roe_calc'])} / {fmt_pct(m['roe_ref'])}")
        print(f"ROA calc / ref           : {fmt_pct(m['roa_calc'])} / {fmt_pct(m['roa_ref'])}")
        print(f"P/E provider / calculated: {fmt_num(m.get('pe_provider'))} / {fmt_num(m.get('pe_calculated'))}")
        print(f"P/B provider / calculated: {fmt_num(m.get('pb_provider'))} / {fmt_num(m.get('pb_calculated'))}")
        print(f"Shares current           : {fmt_num(m['shares'])} ({m['shares_source']})")
        print(f"Shares report estimate   : {fmt_num(m.get('reported_shares_estimate'))}")
        print(f"Post-report share ratio  : {fmt_num(m.get('share_change_ratio'))}x | detected={m.get('post_report_share_change_detected')}")
        print("-"*78)
        print("MATCHED ITEMS")
        for k,val in m["matched_items"].items():
            print(f"  {k:24}: {val}")
        print("-"*78)
        snap=m.get("latest_snapshot") or {}
        if snap:
            print("-"*78)
            print("LATEST / TTM DATA")
            print(f"Status                   : {snap.get('status')}")
            print(f"Latest quarter           : {snap.get('latest_period')}")
            print(f"Balance stock period     : {snap.get('balance_stock_period')}  {snap.get('balance_period_resolution')}")
            print(f"Debt derivation          : {snap.get('debt_derivation')}")
            print(f"Quarter orientation      : {snap.get('quarter_statement_orientation')}  {snap.get('quarter_orientation_meta')}")
            print(f"TTM quarters             : {snap.get('ttm_periods')}")
            print(f"Flow mode                : {snap.get('flow_mode')}")
            print(f"Reconciliation           : {snap.get('reconciliation')}")
            print(f"Fallback reason          : {snap.get('fallback_reason')}")
            print(f"Fiscal-year flow fallback: {snap.get('fiscal_year_flow_fallback')}")
        print("-"*78)
        print("VOLUME ACCUMULATION (V8.7)")
        print(f"Volume score             : {fmt_num(m.get('volume_score'))}/100")
        print(f"Volume signal            : {m.get('volume_signal')}")
        print(f"Dry-up / Up-Down         : {fmt_num(m.get('vol_dryup_score'))}/20 / {fmt_num(m.get('vol_up_down_score'))}/20")
        print(f"Breakout Vol / Price     : {fmt_num(m.get('vol_breakout_volume_score'))}/20 / {fmt_num(m.get('vol_breakout_price_score'))}/15")
        print(f"Retest / Close near high : {fmt_num(m.get('vol_retest_score'))}/15 / {fmt_num(m.get('vol_close_near_high_score'))}/10")
        print(f"Breakout date / retest   : {m.get('breakout_date')} / {m.get('retest_status')}")
        print("-"*78)
        print(f"DATA QUALITY             : {q['status']}  score={q['score']}/100")
        for c in q["checks"]:
            print(f"[{'OK' if c['passed'] else 'FAIL'}]{' [CRITICAL]' if c['critical'] else ''} {c['check']}: {c['detail']}")
        print("-"*78)
        print("HISTORICAL MULTIPLES")
        hm=m.get("historical_multiples") or {}
        hp=hm.get("pe") or {}; he=hm.get("ev_ebitda") or {}
        print(f"Historical P/E target    : {fmt_num(hp.get('target'))} | periods={hp.get('count')} | source={hp.get('source')} | status={hp.get('status')}")
        print(f"P/E history              : {hp.get('series')}")
        print(f"Historical EV/EBITDA     : {fmt_num(he.get('target'))} | periods={he.get('count')} | source={he.get('source')} | status={he.get('status')}")
        print(f"EV/EBITDA history        : {he.get('series')}")
        pb_audit = hm.get("pb_audit") or {}
        print(f"Historical P/B audit     : {fmt_num(pb_audit.get('target'))} | periods={pb_audit.get('count')} | status={pb_audit.get('status')}")
        print(f"Historical shares source : {hm.get('shares_history_source')} | validation={hm.get('shares_history_validation')}")
        print(f"Period diagnostics       : {hm.get('period_diagnostics')}")
        print("="*78)
        print("AUTO FAIR VALUE")
        print("-"*78)
        print(f"Status                   : {v['status']}")
        if v.get("methodology"): print(f"Methodology              : {v.get('methodology')}")
        if v.get("rnav_status"): print(f"RNAV                     : {v.get('rnav_status')}")
        if v.get("sotp_status"): print(f"SOTP                     : {v.get('sotp_status')}")
        if v.get("reason"): print(f"Reason                   : {v['reason']}")
        if v.get("missing_methods"): print(f"Missing methods          : {v['missing_methods']}")
        for k,x in v.get("method_values",{}).items():
            print(f"{k:24}: Bear {fmt_num(x.get('bear'))} | Base {fmt_num(x.get('base'))} | Bull {fmt_num(x.get('bull'))}")
        if v["status"] in {"OK_FULL","OK_PROXY_NO_RNAV","OK_PROXY_NO_SOTP"}:
            print("-"*78)
            print(f"BEAR                     : {fmt_num(v['bear'])}")
            print(f"BASE / FAIR VALUE        : {fmt_num(v['fair_value'])}")
            print(f"BULL                     : {fmt_num(v['bull'])}")
            print(f"Market                   : {fmt_num(v['market_price'])}")
            print(f"Upside/(Downside)        : {fmt_pct(v['upside'])}")
            print(f"Margin of Safety         : {fmt_pct(v['margin_of_safety'])}")
            print(f"Assumptions              : {v['assumptions']}")
        print("="*78)
        print(f"Saved to: {self.base_dir.resolve()}")

    def run(self):
        print(f"Dang lay du lieu {self.symbol}...")
        data=self.fetch_all()
        self.save_raw(data)
        m=self.calculate(data)
        # V8.7 report-only signal: derive from the already-fetched daily OHLCV.
        # This does not make another VNStock API request and does not alter fair value.
        m.update(volume_accumulation_score_from_df(data.get("price_history")))
        sector,industry=self.sector(data["company_info"])
        m["sector_type"]=sector; m["industry"]=industry
        q=self.validate(m,sector)
        v=self.valuation(m,q,data["company_info"])
        self.save(m,q,v)
        self.print_report(m,q,v)
        return m,q,v


