from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import math
import json
import pandas as pd


@dataclass
class Issue:
    code: str
    severity: str
    stage: str
    message: str
    auto_fixed: bool = False
    detail: Any = None

    def to_dict(self):
        return asdict(self)


class ErrorCodes:
    FETCH_FAILURE = "E001_FETCH_FAILURE"
    FETCH_PARTIAL = "E002_FETCH_PARTIAL"

    ANNUAL_ORIENTATION_AMBIGUOUS = "E101_ANNUAL_ORIENTATION_AMBIGUOUS"
    QUARTER_ORIENTATION_AMBIGUOUS = "E102_QUARTER_ORIENTATION_AMBIGUOUS"
    ANNUAL_ORIENTATION_UNVERIFIED = "E103_ANNUAL_ORIENTATION_UNVERIFIED"

    ACCOUNTING_IDENTITY = "E201_ACCOUNTING_IDENTITY_MISMATCH"
    DEBT_INCONSISTENT = "E202_DEBT_INCONSISTENT"
    SHARE_BASIS_CHANGE = "E203_SHARE_BASIS_CHANGE"
    SHARE_HISTORY_MISMATCH = "E204_SHARE_HISTORY_MISMATCH"
    DEBT_ZERO_EVIDENCE = "E205_DEBT_ZERO_EVIDENCE"

    PERIOD_REMAP = "E301_PERIOD_REMAP"
    PERIOD_REMAP_WEAK_ANCHOR = "E305_PERIOD_REMAP_WEAK_ANCHOR"
    CURRENT_FISCAL_YEAR = "E302_CURRENT_FISCAL_YEAR"
    QUARTER_FLOW_UNVERIFIED = "E303_QUARTER_FLOW_UNVERIFIED"
    FISCAL_FLOW_FALLBACK = "E304_FISCAL_FLOW_FALLBACK"

    HIST_PE_INSUFFICIENT = "E401_HIST_PE_INSUFFICIENT"
    HIST_EV_INSUFFICIENT = "E402_HIST_EV_EBITDA_INSUFFICIENT"
    VALUATION_METHOD_MISSING = "E403_VALUATION_METHOD_MISSING"
    VALUATION_PARTIAL = "E404_VALUATION_PARTIAL"
    VALUATION_INVALID = "E405_VALUATION_INVALID"
    VALUATION_NOT_RUN = "E406_VALUATION_NOT_RUN"
    CROSS_METHOD_DIVERGENCE = "E407_CROSS_METHOD_DIVERGENCE"
    FALLBACK_FAIR_VALUE = "E408_FALLBACK_FAIR_VALUE"
    FALLBACK_REJECTED = "E409_FALLBACK_REJECTED"
    LOW_CONFIDENCE_RECOVERY = "E410_LOW_CONFIDENCE_METHOD_RECOVERY"
    METHOD_DISPERSION_HIGH = "E411_METHOD_DISPERSION_HIGH"
    HIST_EV_BRIDGE_RECOVERY = "E412_HIST_EV_EBITDA_BRIDGE_RECOVERY"

    DATA_QUALITY_FAIL = "E501_DATA_QUALITY_FAIL"
    DATA_QUALITY_PARTIAL_MISMATCH = "E502_DATA_QUALITY_PARTIAL_MISMATCH"

    STAGE_EXCEPTION = "E901_STAGE_EXCEPTION"


def _finite(x):
    try:
        return x is not None and math.isfinite(float(x))
    except Exception:
        return False


def _ratio_gap(meta: dict | None):
    if not isinstance(meta, dict):
        return None
    n = meta.get("normal_score")
    r = meta.get("reversed_score")
    if not (_finite(n) and _finite(r)):
        return None
    return abs(float(n)-float(r))


def _find_col(df, candidates):
    if df is None or getattr(df, "empty", True):
        return None
    norm = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c in norm:
            return norm[c]
    return None


class ValidationEngine:
    """
    Validation-first layer.

    This layer NEVER changes valuation formulas. It classifies anomalies and
    auto-fixes already performed by the canonical engine, so batch runs can be
    triaged by ERROR CLASS instead of ticker-by-ticker.
    """

    def __init__(self, symbol: str):
        self.symbol = symbol.upper().strip()

    def evaluate(self, data: dict, metrics: dict | None, quality: dict | None,
                 valuation: dict | None, stage_exception: dict | None = None):
        issues: list[Issue] = []

        if stage_exception:
            issues.append(Issue(
                ErrorCodes.STAGE_EXCEPTION, "ERROR", stage_exception.get("stage","unknown"),
                stage_exception.get("message","Unhandled stage exception"),
                detail=stage_exception,
            ))
            return self._result(issues, metrics, quality, valuation)

        # ---------------- Fetch layer ----------------
        fetch_errors = {}
        for k, v in (data or {}).items():
            if v is None:
                fetch_errors[k] = "missing"
        if fetch_errors:
            critical = [k for k in ("quote","company_info","income","balance","ratio") if k in fetch_errors]
            issues.append(Issue(
                ErrorCodes.FETCH_FAILURE if critical else ErrorCodes.FETCH_PARTIAL,
                "ERROR" if critical else "WARN",
                "fetch",
                f"Missing fetched datasets: {sorted(fetch_errors)}",
                detail={"missing":sorted(fetch_errors),"critical":critical},
            ))

        if not isinstance(metrics, dict):
            issues.append(Issue(ErrorCodes.STAGE_EXCEPTION,"ERROR","calculate","No metrics produced"))
            return self._result(issues, metrics, quality, valuation)

        # ---------------- Orientation ----------------
        am = metrics.get("orientation_meta") or {}
        gap = _ratio_gap(am)
        if gap is not None and gap < 0.003:
            issues.append(Issue(
                ErrorCodes.ANNUAL_ORIENTATION_AMBIGUOUS, "WARN", "canonicalize",
                "Annual orientation evidence is close; review if other contracts fail.",
                detail={"gap":gap,"meta":am},
            ))
        if am.get("safe_verified") is False:
            issues.append(Issue(
                ErrorCodes.ANNUAL_ORIENTATION_UNVERIFIED, "ERROR", "canonicalize",
                "Annual orientation could not be independently verified; valuation is blocked.",
                detail=am,
            ))

        snap = metrics.get("latest_snapshot") or {}
        qm = snap.get("quarter_orientation_meta") or {}
        reason = str(qm.get("reason") or "")
        if "fallback_ambiguous" in reason or "no_strong" in reason:
            issues.append(Issue(
                ErrorCodes.QUARTER_ORIENTATION_AMBIGUOUS, "WARN", "canonicalize",
                "Quarter orientation relied on fallback evidence.",
                detail=qm,
            ))

        # ---------------- Accounting contracts ----------------
        sd = metrics.get("latest_short_debt")
        ld = metrics.get("latest_long_debt")
        td = metrics.get("latest_total_debt")
        if _finite(sd) and _finite(ld):
            expected = float(sd)+float(ld)
            if not _finite(td):
                issues.append(Issue(
                    ErrorCodes.DEBT_INCONSISTENT, "ERROR", "contract",
                    "Short debt and long debt exist but total debt is missing.",
                    detail={"short":sd,"long":ld,"total":td},
                ))
            else:
                den=max(abs(expected),1.0)
                err=abs(float(td)-expected)/den
                if err > 0.005:
                    issues.append(Issue(
                        ErrorCodes.DEBT_INCONSISTENT, "ERROR", "contract",
                        "Total debt does not reconcile with short + long debt.",
                        detail={"short":sd,"long":ld,"total":td,"relative_error":err},
                    ))

        # Balance identity if provider ratio exposes liabilities on latest quarter.
        rq = (data or {}).get("ratio_q_ts")
        assets = metrics.get("latest_assets")
        equity = metrics.get("latest_equity")
        if rq is not None and not getattr(rq,"empty",True) and _finite(assets) and _finite(equity):
            lc = _find_col(rq, ["liabilities"])
            pc = _find_col(rq, ["period"])
            if lc is not None:
                rr=rq.copy()
                if pc is not None:
                    rr=rr.sort_values(pc)
                try:
                    liab=float(pd.to_numeric(rr[lc],errors="coerce").dropna().iloc[-1])
                    # Ratio endpoint often scales balance values; only check when magnitude is comparable.
                    if abs(liab) > 1e9:
                        err=abs(float(assets)-(float(equity)+liab))/max(abs(float(assets)),1.0)
                        if err>0.02:
                            issues.append(Issue(
                                ErrorCodes.ACCOUNTING_IDENTITY,"WARN","contract",
                                "Assets do not reconcile to equity + liabilities within 2%.",
                                detail={"assets":assets,"equity":equity,"liabilities":liab,"relative_error":err},
                            ))
                except Exception:
                    pass

        # ---------------- Share basis ----------------
        scr=metrics.get("share_change_ratio")
        if _finite(scr) and abs(float(scr)-1.0)>0.01:
            issues.append(Issue(
                ErrorCodes.SHARE_BASIS_CHANGE, "INFO", "canonicalize",
                "Report-period and current share counts differ; canonical engine detected a share-basis change.",
                auto_fixed=True,
                detail={"share_change_ratio":scr,
                        "report_shares":metrics.get("reported_shares_estimate"),
                        "current_shares":metrics.get("shares")},
            ))

        hsv=((metrics.get("historical_multiples") or {}).get("shares_history_validation") or {})
        if hsv.get("status")=="FAIL":
            issues.append(Issue(
                ErrorCodes.SHARE_HISTORY_MISMATCH, "WARN", "historical",
                "Historical share-capital normalization does not reconcile to current shares.",
                detail=hsv,
            ))

        dd=snap.get("debt_derivation") or {}
        if "derived_zero_from_latest_ratio_debt_to_assets_zero" in str(dd):
            issues.append(Issue(
                ErrorCodes.DEBT_ZERO_EVIDENCE,"INFO","canonicalize",
                "Latest-quarter borrowing fields were null and provider debt_to_assets was explicitly zero; interest-bearing debt was set to zero.",
                auto_fixed=True,detail=dd,
            ))

        # ---------------- Period/fiscal auto-fixes ----------------
        br=snap.get("balance_period_resolution") or {}
        if br.get("status")=="REMAPPED":
            issues.append(Issue(
                ErrorCodes.PERIOD_REMAP, "INFO", "canonicalize",
                "Quarterly balance stock period was automatically remapped using capital-history evidence.",
                auto_fixed=True, detail=br,
            ))
            bvps=metrics.get("bvps_current"); rshares=metrics.get("reported_shares_estimate"); eq=metrics.get("latest_equity")
            if _finite(bvps) and _finite(rshares) and _finite(eq) and float(eq)>0:
                err=abs(float(bvps)*float(rshares)-float(eq))/abs(float(eq))
                # Current BVPS can belong to a different provider period than the remapped balance row.
                # Keep this as a diagnostic warning only; it is not a period-matched hard contract.
                sev="WARN" if err>0.10 else None
                if sev:
                    issues.append(Issue(ErrorCodes.PERIOD_REMAP_WEAK_ANCHOR,sev,"canonicalize",
                        "Remapped balance period has a weak non-period-matched BVPS×shares≈equity diagnostic anchor.",
                        detail={"anchor_relative_error":err,
                                "anchor_scope":"diagnostic_only_latest_ratio_vs_remapped_balance",
                                "balance_period_resolution":br}))

        if metrics.get("annual_current_calendar_year_accepted"):
            issues.append(Issue(
                ErrorCodes.CURRENT_FISCAL_YEAR, "INFO", "canonicalize",
                "Current-calendar-year annual statement accepted as completed fiscal year.",
                auto_fixed=True,
                detail={"latest_year":metrics.get("latest_year")},
            ))

        ff=snap.get("fiscal_year_flow_fallback")
        if ff:
            issues.append(Issue(
                ErrorCodes.FISCAL_FLOW_FALLBACK, "INFO", "canonicalize",
                "Incomplete quarterly cash-flow was filled only from the same completed annual fiscal year.",
                auto_fixed=True, detail=ff,
            ))

        fallback=str(snap.get("fallback_reason") or "")
        if fallback.startswith("QUARTER_FLOW_NOT_VERIFIED"):
            issues.append(Issue(
                ErrorCodes.QUARTER_FLOW_UNVERIFIED, "INFO", "latest_data",
                "Quarterly flow data could not be verified as TTM; annual-flow fallback remains explicit.",
                detail={"fallback_reason":fallback,"flow_mode":snap.get("flow_mode")},
            ))

        # ---------------- Historical multiples ----------------
        hm=metrics.get("historical_multiples") or {}
        hp=hm.get("pe") or {}
        he=hm.get("ev_ebitda") or {}
        if hp.get("status")=="NOT_ENOUGH_DATA":
            issues.append(Issue(
                ErrorCodes.HIST_PE_INSUFFICIENT, "INFO", "historical",
                f"Historical P/E has only {hp.get('count')} cleaned periods.",
                detail=hp,
            ))
        if he.get("status")=="NOT_ENOUGH_DATA":
            issues.append(Issue(ErrorCodes.HIST_EV_INSUFFICIENT,"INFO","historical",
                f"Historical EV/EBITDA has only {he.get('count')} cleaned periods.",detail=he))
        if he.get("recovery_status")=="USED":
            issues.append(Issue(ErrorCodes.HIST_EV_BRIDGE_RECOVERY,"INFO","historical",
                "Historical EV/EBITDA recovered from EV/EBIT and statement EBIT/EBITDA for 3+ years.",
                detail={"source":he.get("source"),"target":he.get("target"),"count":he.get("count"),
                        "formula":he.get("recovery_formula"),"recovered_series":he.get("recovered_series")}))

        # ---------------- Existing quality layer ----------------
        if isinstance(quality,dict):
            if quality.get("status")!="PASS":
                issues.append(Issue(
                    ErrorCodes.DATA_QUALITY_FAIL, "ERROR", "quality",
                    f"Existing data-quality layer returned {quality.get('status')}.",
                    detail={"score":quality.get("score"),"checks":quality.get("checks")},
                ))

        # ---------------- Valuation ----------------
        if isinstance(valuation,dict):
            vs=valuation.get("status")
            missing=valuation.get("missing_methods") or []
            invalid=valuation.get("invalid_method_scenarios") or {}
            if missing or invalid:
                issues.append(Issue(
                    ErrorCodes.VALUATION_METHOD_MISSING, "WARN", "valuation",
                    "One or more required sector valuation methods are missing/invalid.",
                    detail={"missing_methods":missing,"invalid_method_scenarios":invalid},
                ))
            if vs=="OK_FALLBACK":
                issues.append(Issue(
                    ErrorCodes.FALLBACK_FAIR_VALUE,"INFO","valuation",
                    "Approved sector-specific fallback fair value used; full-model weights were not renormalized.",
                    detail={"methodology":valuation.get("fallback_methodology"),
                            "methods":valuation.get("fallback_methods"),
                            "weights":valuation.get("fallback_weights"),
                            "confidence":valuation.get("confidence"),
                            "method_agreement_ratio":valuation.get("method_agreement_ratio"),
                            "trigger":valuation.get("fallback_trigger")},
                ))
            fb=valuation.get("fallback") or {}
            if vs=="PARTIAL" and fb.get("status")=="REJECTED":
                issues.append(Issue(
                    ErrorCodes.FALLBACK_REJECTED,"WARN","valuation",
                    "No approved fallback passed minimum-method and method-agreement gates.",
                    detail=fb,
                ))
            recoveries=[]
            for mk,md in (valuation.get("method_values") or {}).items():
                if isinstance(md,dict):
                    tm=md.get("target_multiple_method") if isinstance(md.get("target_multiple_method"),dict) else {}
                    tag=str(md.get("target_confidence") or tm.get("confidence") or "")
                    if "FALLBACK_3" in tag or "2_VALID_PERIODS" in tag or "RECOVERY" in tag:
                        recoveries.append({"method":mk,"confidence":tag})
            if recoveries:
                issues.append(Issue(
                    ErrorCodes.LOW_CONFIDENCE_RECOVERY,"INFO","valuation",
                    "One or more methods use an explicit historical-data recovery rule.",
                    detail={"recoveries":recoveries},
                ))

            md=valuation.get("method_dispersion") or {}
            if md.get("status")=="HIGH_DISPERSION":
                issues.append(Issue(ErrorCodes.METHOD_DISPERSION_HIGH,"WARN","valuation",
                    "Full/proxy valuation methods have high base-value dispersion; fair value is retained but confidence is capped.",detail=md))

            if vs=="NOT_RUN":
                issues.append(Issue(ErrorCodes.VALUATION_NOT_RUN,"WARN","valuation",
                    "Valuation did not run to completion; ticker cannot be reported as PASS.",
                    detail={"reason":valuation.get("reason")}))
            elif vs=="PARTIAL":
                issues.append(Issue(
                    ErrorCodes.VALUATION_PARTIAL, "WARN", "valuation",
                    "Valuation is PARTIAL; no full sector blend should be treated as final fair value.",
                    detail={"reason":valuation.get("reason")},
                ))
                if valuation.get("reason")=="CROSS_METHOD_DIVERGENCE":
                    issues.append(Issue(ErrorCodes.CROSS_METHOD_DIVERGENCE,"WARN","valuation",
                        "DCF materially diverges from relative methods; fair value withheld.",
                        detail=valuation.get("cross_method_sanity")))
                if isinstance(quality,dict) and quality.get("status")=="PASS":
                    issues.append(Issue(
                        ErrorCodes.DATA_QUALITY_PARTIAL_MISMATCH, "INFO", "quality",
                        "Data quality passed but valuation is partial. These are separate statuses.",
                    ))
            elif vs in {"OK_FULL","OK_PROXY_NO_RNAV","OK_PROXY_NO_SOTP","OK_FALLBACK"}:
                fv=valuation.get("fair_value")
                if not _finite(fv) or float(fv)<=0:
                    issues.append(Issue(
                        ErrorCodes.VALUATION_INVALID, "ERROR", "valuation",
                        "Valuation status is OK but fair value is non-finite/non-positive.",
                        detail={"fair_value":fv,"status":vs},
                    ))
        else:
            issues.append(Issue(ErrorCodes.VALUATION_INVALID,"ERROR","valuation","No valuation object produced"))

        return self._result(issues, metrics, quality, valuation)

    def _result(self, issues, metrics, quality, valuation):
        severities=[i.severity for i in issues]
        auto_fixed=sum(1 for i in issues if i.auto_fixed)
        if "ERROR" in severities:
            status="NEEDS_REVIEW"
        elif isinstance(valuation,dict) and valuation.get("status") in {"PARTIAL","NOT_RUN"}:
            status="PARTIAL"
        elif any(i.auto_fixed for i in issues):
            status="AUTO_FIXED"
        else:
            status="PASS"

        return {
            "symbol": self.symbol,
            "validation_status": status,
            "auto_fixed_count": auto_fixed,
            "review_required": status=="NEEDS_REVIEW",
            "issue_codes":[i.code for i in issues],
            "issues":[i.to_dict() for i in issues],
            "data_quality_status": quality.get("status") if isinstance(quality,dict) else None,
            "data_quality_score": quality.get("score") if isinstance(quality,dict) else None,
            "valuation_status": valuation.get("status") if isinstance(valuation,dict) else None,
            "fair_value": valuation.get("fair_value") if isinstance(valuation,dict) else None,
            "fair_value_type": valuation.get("fair_value_type") if isinstance(valuation,dict) else None,
            "confidence": valuation.get("confidence") if isinstance(valuation,dict) else None,
            "fallback_methodology": valuation.get("fallback_methodology") if isinstance(valuation,dict) else None,
            "method_agreement_ratio": valuation.get("method_agreement_ratio") if isinstance(valuation,dict) else None,
            "method_dispersion_ratio": ((valuation.get("method_dispersion") or {}).get("ratio") if isinstance(valuation,dict) else None),
            "market_price": metrics.get("market_price") if isinstance(metrics,dict) else None,
            "sector": metrics.get("sector_type") if isinstance(metrics,dict) else None,
            "latest_period": metrics.get("latest_data_period") if isinstance(metrics,dict) else None,
        }
