from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd


# Screening score is a screening aid, not a buy/sell recommendation.
# Locked top-level weights agreed for the report layer.
SCREENING_WEIGHTS = {
    "valuation": 0.35,
    "quality": 0.30,
    "risk": 0.20,
    "confidence": 0.15,
}

DEFAULT_MIN_SCORE = 75.0
DEFAULT_MAX_PRICE_TO_FV = 0.85

FINANCIAL_SECTORS = {"BANK", "SECURITIES", "INSURANCE"}


def _finite(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _interp(x: Optional[float], points: Iterable[tuple[float, float]]) -> Optional[float]:
    if x is None:
        return None
    pts = sorted(points)
    if x <= pts[0][0]:
        return float(pts[0][1])
    if x >= pts[-1][0]:
        return float(pts[-1][1])
    for (a, sa), (b, sb) in zip(pts, pts[1:]):
        if a <= x <= b:
            return float(sa + (sb - sa) * (x - a) / (b - a))
    return None


def _weighted_available(items: Iterable[tuple[Optional[float], float]], default: float = 50.0) -> float:
    valid = [(v, w) for v, w in items if v is not None]
    if not valid:
        return default
    sw = sum(w for _, w in valid)
    return sum(v * w for v, w in valid) / sw


def _latest_ratio_row(ticker_dir: Path) -> Dict[str, Any]:
    p = ticker_dir / "raw" / "ratio_ts.csv"
    if not p.exists() or p.stat().st_size == 0:
        return {}
    try:
        df = pd.read_csv(p)
    except (pd.errors.EmptyDataError, OSError, ValueError):
        return {}
    if df.empty:
        return {}
    if "period" in df.columns:
        years = pd.to_numeric(df["period"], errors="coerce")
        if years.notna().any():
            return df.loc[years.idxmax()].to_dict()
    return df.iloc[-1].to_dict()


def _load_metrics(ticker_dir: Path) -> Dict[str, Any]:
    p = ticker_dir / "metrics.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def valuation_score(row: pd.Series) -> tuple[Optional[float], Optional[float]]:
    price = _finite(row.get("market_price"))
    fv = _finite(row.get("fair_value"))
    if price is None or fv is None or price <= 0 or fv <= 0:
        return None, None
    price_to_fv = price / fv

    # Price/FV is the dominant valuation input. 0.85 is intentionally not
    # an automatic "good" verdict; it only corresponds to ~76/100 here.
    pf_score = _interp(price_to_fv, [
        (0.50, 100), (0.60, 95), (0.70, 90), (0.80, 82),
        (0.85, 76), (0.90, 68), (1.00, 50), (1.15, 20), (1.30, 0),
    ])

    dispersion = _finite(row.get("method_dispersion_ratio"))
    if dispersion is None:
        dispersion = _finite(row.get("method_agreement_ratio"))
    agreement_score = 100.0 if dispersion is None else _interp(dispersion, [
        (1.00, 100), (1.25, 95), (1.50, 85), (2.00, 65),
        (2.25, 50), (3.00, 25), (5.00, 0),
    ])
    return 0.80 * pf_score + 0.20 * agreement_score, price_to_fv


def quality_score(sector: str, ratio: Dict[str, Any]) -> float:
    roe = _finite(ratio.get("roe"))
    roa = _finite(ratio.get("roa"))
    revenue_growth = _finite(ratio.get("net_revenue"))
    profit_growth = _finite(ratio.get("profit_after_tax_for_shareholders_of_the_parent_company"))
    net_margin = _finite(ratio.get("net_margin"))

    if sector == "BANK":
        return _weighted_available([
            (_interp(roe, [(0, 0), (8, 35), (12, 55), (16, 75), (20, 90), (25, 100)]), 0.40),
            (_interp(roa, [(0, 0), (0.5, 30), (1.0, 60), (1.5, 80), (2.0, 95), (2.5, 100)]), 0.30),
            (_interp(profit_growth, [(-20, 0), (0, 40), (5, 55), (10, 70), (20, 90), (30, 100)]), 0.30),
        ])

    if sector in {"SECURITIES", "INSURANCE"}:
        return _weighted_available([
            (_interp(roe, [(0, 0), (5, 30), (10, 55), (15, 75), (20, 90), (25, 100)]), 0.40),
            (_interp(roa, [(0, 0), (1, 30), (3, 55), (5, 75), (8, 90), (12, 100)]), 0.25),
            (_interp(profit_growth, [(-20, 0), (0, 40), (10, 60), (20, 80), (30, 100)]), 0.35),
        ])

    return _weighted_available([
        (_interp(roe, [(0, 0), (5, 25), (10, 50), (15, 70), (20, 85), (25, 100)]), 0.25),
        (_interp(roa, [(0, 0), (2, 30), (5, 55), (8, 75), (12, 90), (15, 100)]), 0.15),
        (_interp(revenue_growth, [(-15, 0), (0, 35), (5, 50), (10, 65), (15, 80), (25, 100)]), 0.20),
        (_interp(profit_growth, [(-30, 0), (0, 35), (10, 55), (20, 75), (30, 90), (50, 100)]), 0.25),
        (_interp(net_margin, [(0, 20), (3, 40), (5, 55), (10, 75), (15, 90), (25, 100)]), 0.15),
    ])


def risk_score(sector: str, ratio: Dict[str, Any], metrics: Dict[str, Any]) -> float:
    beta = _finite(ratio.get("beta"))
    debt_to_assets = _finite(ratio.get("debt_to_assets"))
    share_change_ratio = _finite(metrics.get("share_change_ratio"))

    # Share-basis change is treated as dilution/capital-change risk only in the
    # report score. It does not change fair value or valuation methodology.
    dilution_pct = abs(share_change_ratio - 1.0) * 100.0 if share_change_ratio and share_change_ratio > 0 else 0.0

    beta_score = _interp(beta, [
        (0.50, 100), (0.80, 90), (1.00, 80), (1.20, 65),
        (1.50, 45), (2.00, 20), (3.00, 0),
    ]) if beta is not None else 70.0
    dilution_score = _interp(dilution_pct, [
        (0, 100), (2, 95), (5, 85), (10, 70),
        (20, 45), (40, 10), (60, 0),
    ])

    # Debt/asset is not comparable for banks/brokers/insurers, so it is excluded
    # rather than penalizing financial companies for their business model.
    if sector in FINANCIAL_SECTORS or debt_to_assets is None:
        return 0.55 * dilution_score + 0.45 * beta_score

    leverage_score = _interp(debt_to_assets, [
        (0, 100), (10, 95), (20, 88), (30, 75),
        (40, 60), (50, 40), (60, 20), (75, 0),
    ])
    return 0.50 * leverage_score + 0.30 * dilution_score + 0.20 * beta_score


def confidence_score(row: pd.Series) -> float:
    confidence = str(row.get("confidence") or "").upper()
    base = {"HIGH": 100.0, "MEDIUM": 75.0, "LOW": 50.0}.get(confidence, 40.0)
    dq = _finite(row.get("data_quality_score"))
    if dq is None:
        dq = 50.0
    return 0.75 * base + 0.25 * dq


def screening_score(row: pd.Series, ticker_dir: Path) -> Dict[str, Any]:
    vscore, price_to_fv = valuation_score(row)
    if vscore is None or price_to_fv is None:
        return {"screening_score": None, "price_to_fv": None}

    ratio = _latest_ratio_row(ticker_dir)
    metrics = _load_metrics(ticker_dir)
    qscore = quality_score(str(row.get("sector") or ""), ratio)
    rscore = risk_score(str(row.get("sector") or ""), ratio, metrics)
    cscore = confidence_score(row)

    total = (
        SCREENING_WEIGHTS["valuation"] * vscore
        + SCREENING_WEIGHTS["quality"] * qscore
        + SCREENING_WEIGHTS["risk"] * rscore
        + SCREENING_WEIGHTS["confidence"] * cscore
    )
    return {
        "screening_score": round(total, 2),
        "valuation_score": round(vscore, 2),
        "quality_score": round(qscore, 2),
        "risk_score": round(rscore, 2),
        "confidence_score": round(cscore, 2),
        "price_to_fv": price_to_fv,
        "upside_pct": (1.0 / price_to_fv - 1.0) * 100.0,
    }


def _fmt_k(v: float) -> str:
    k = float(v) / 1000.0
    if abs(k - round(k)) < 0.05:
        return f"{int(round(k))}k"
    return f"{k:.1f}k"


def generate_reports(
    output_root: str | Path,
    *,
    min_score: float = DEFAULT_MIN_SCORE,
    max_price_to_fv: float = DEFAULT_MAX_PRICE_TO_FV,
) -> Dict[str, Any]:
    root = Path(output_root)
    summary_csv = root / "batch_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing batch summary: {summary_csv}")

    df = pd.read_csv(summary_csv)
    rows = []
    for _, row in df.iterrows():
        ticker = str(row.get("ticker") or "").upper()
        detail = screening_score(row, root / ticker)
        price = _finite(row.get("market_price"))
        fv = _finite(row.get("fair_value"))
        rec = row.to_dict()
        rec.update(detail)
        rec["ticker"] = ticker
        rec["market_price"] = price
        rec["fair_value"] = fv
        rows.append(rec)

    report = pd.DataFrame(rows)
    if "screening_score" in report.columns:
        report = report.sort_values(["screening_score", "ticker"], ascending=[False, True], na_position="last")

    market_csv = root / "market_report.csv"
    report.to_csv(market_csv, index=False, encoding="utf-8-sig")

    usable = report[
        report["screening_score"].notna()
        & (report["screening_score"] > float(min_score))
        & report["price_to_fv"].notna()
        & (report["price_to_fv"] < float(max_price_to_fv))
    ].copy()

    # Telegram payload intentionally stays compact. Component scores remain in
    # market_report.csv for drill-down but are not shown in the list message.
    telegram_rows = []
    lines = []
    for rank, (_, r) in enumerate(usable.iterrows(), 1):
        item = {
            "rank": rank,
            "ticker": r["ticker"],
            "score": round(float(r["screening_score"]), 1),
            "market_price": float(r["market_price"]),
            "fair_value": float(r["fair_value"]),
            "price_to_fv": round(float(r["price_to_fv"]), 4),
            "upside_pct": round(float(r["upside_pct"]), 1),
            "confidence": str(r.get("confidence") or ""),
            "fair_value_type": str(r.get("fair_value_type") or ""),
        }
        telegram_rows.append(item)
        lines.append(
            f"{rank}. {item['ticker']} | {item['score']:.0f}đ | "
            f"Giá {_fmt_k(item['market_price'])} | FV {_fmt_k(item['fair_value'])} | "
            f"+{item['upside_pct']:.1f}% | {item['confidence']}"
        )

    payload = {
        "filter": {
            "screening_score_gt": float(min_score),
            "price_to_fv_lt": float(max_price_to_fv),
        },
        "screening_weights": SCREENING_WEIGHTS,
        "total_universe": int(len(report)),
        "selected_count": int(len(usable)),
        "stocks": telegram_rows,
    }
    (root / "telegram_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (root / "telegram_report.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    return payload
