from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import time
import traceback
from collections import Counter
import pandas as pd

from stock_audit import StockAudit
from validation_engine import ValidationEngine


RAW_FILES={
    "quote":"quote.csv",
    "company_info":"company_info.csv",
    "income":"income.csv",
    "balance":"balance.csv",
    "cashflow":"cashflow.csv",
    "ratio":"ratio.csv",
    "income_ts":"income_ts.csv",
    "balance_ts":"balance_ts.csv",
    "cashflow_ts":"cashflow_ts.csv",
    "ratio_ts":"ratio_ts.csv",
    "income_q_ts":"income_q_ts.csv",
    "balance_q_ts":"balance_q_ts.csv",
    "cashflow_q_ts":"cashflow_q_ts.csv",
    "ratio_q_ts":"ratio_q_ts.csv",
    "price_history":"price_history.csv",
}


def load_raw_dir(raw_dir: str | Path):
    raw_dir=Path(raw_dir)
    data={}
    for key,fn in RAW_FILES.items():
        p=raw_dir/fn
        if not p.exists():
            data[key]=None
            continue
        try:
            data[key]=pd.read_csv(p)
        except pd.errors.EmptyDataError:
            data[key]=None
        except Exception:
            data[key]=None
    return data


def evaluate_data(symbol: str, data: dict, output_root: str | Path="batch_output",
                  auth_status=None, save_artifacts=True):
    symbol=symbol.upper().strip()
    outdir=Path(output_root)/symbol
    outdir.mkdir(parents=True,exist_ok=True)

    obj=StockAudit.__new__(StockAudit)
    obj.symbol=symbol
    obj.base_dir=outdir
    obj.auth_status=auth_status or {"offline_raw":True}
    obj.matched={}
    obj.results={}

    stage="calculate"
    try:
        m=obj.calculate(data)
        stage="sector"
        sector,industry=obj.sector(data.get("company_info"))
        m["sector_type"]=sector
        m["industry"]=industry

        stage="quality"
        q=obj.validate(m,sector)

        stage="valuation"
        v=obj.valuation(m,q,data.get("company_info"))

        stage="validation_first"
        vr=ValidationEngine(symbol).evaluate(data,m,q,v)

        if save_artifacts:
            obj.save(m,q,v)
            (outdir/"validation_report.json").write_text(
                json.dumps(vr,ensure_ascii=False,indent=2),encoding="utf-8"
            )
        return {"ok":True,"metrics":m,"quality":q,"valuation":v,"validation":vr}

    except Exception as exc:
        detail={
            "stage":stage,
            "message":repr(exc),
            "traceback":traceback.format_exc(limit=12),
        }
        vr=ValidationEngine(symbol).evaluate(
            data, None, None, None, stage_exception=detail
        )
        if save_artifacts:
            (outdir/"validation_report.json").write_text(
                json.dumps(vr,ensure_ascii=False,indent=2),encoding="utf-8"
            )
            (outdir/"stage_exception.json").write_text(
                json.dumps(detail,ensure_ascii=False,indent=2),encoding="utf-8"
            )
        return {"ok":False,"metrics":None,"quality":None,"valuation":None,"validation":vr}


def _raw_cache_dir(output_root, symbol):
    return Path(output_root)/symbol/"raw"


def evaluate_live(symbol: str, output_root="batch_output", api_key=None, reuse_raw=True):
    symbol=symbol.upper().strip()
    cache=_raw_cache_dir(output_root,symbol)

    if reuse_raw and cache.exists() and any(cache.glob("*.csv")):
        data=load_raw_dir(cache)
        r=evaluate_data(symbol,data,output_root,auth_status={"cache":"reused"},save_artifacts=True)
        r["source"]="RAW_CACHE"
        return r

    obj=StockAudit(symbol,output_root,api_key)
    data=obj.fetch_all()
    obj.save_raw(data)
    r=evaluate_data(symbol,data,output_root,auth_status=obj.auth_status,save_artifacts=True)
    r["source"]="LIVE_API"
    return r


def _row(result):
    vr=result["validation"]; v=result.get("valuation") or {}; issues=vr.get("issues") or []
    by_sev={"ERROR":[],"WARN":[],"INFO":[]}; auto=[]
    for i in issues:
        sev=i.get("severity"); code=i.get("code")
        if sev in by_sev and code: by_sev[sev].append(code)
        if i.get("auto_fixed") and code: auto.append(code)
    return {
        "ticker":vr.get("symbol"),"sector":vr.get("sector"),"validation_status":vr.get("validation_status"),
        "data_quality_status":vr.get("data_quality_status"),"data_quality_score":vr.get("data_quality_score"),
        "valuation_status":vr.get("valuation_status"),"fair_value_type":vr.get("fair_value_type"),
        "confidence":vr.get("confidence"),"fair_value":vr.get("fair_value"),"market_price":vr.get("market_price"),
        "fallback_methodology":vr.get("fallback_methodology"),"method_agreement_ratio":vr.get("method_agreement_ratio"),
        "method_dispersion_ratio":vr.get("method_dispersion_ratio"),
        "latest_period":vr.get("latest_period"),"auto_fixed_count":vr.get("auto_fixed_count"),"review_required":vr.get("review_required"),
        "hard_error_class":";".join(by_sev["ERROR"]),"warning_class":";".join(by_sev["WARN"]),
        "info_class":";".join(by_sev["INFO"]),"auto_fix_class":";".join(auto),
        "issue_class":";".join(vr.get("issue_codes") or []),"missing_methods":";".join(v.get("missing_methods") or []),
        "source":result.get("source"),
    }


def write_batch_reports(results: list[dict], output_root: str | Path):
    output_root=Path(output_root)
    output_root.mkdir(parents=True,exist_ok=True)
    rows=[_row(r) for r in results]

    csv_path=output_root/"batch_summary.csv"
    if rows:
        with csv_path.open("w",newline="",encoding="utf-8-sig") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    codes=Counter()
    statuses=Counter()
    valuation_statuses=Counter()
    fair_value_types=Counter()
    confidences=Counter()
    sectors=Counter()
    for r in results:
        vr=r["validation"]
        statuses[vr.get("validation_status")]+=1
        valuation_statuses[vr.get("valuation_status")]+=1
        if vr.get("fair_value_type"): fair_value_types[vr.get("fair_value_type")]+=1
        if vr.get("confidence"): confidences[vr.get("confidence")]+=1
        if vr.get("sector"): sectors[vr.get("sector")]+=1
        for code in vr.get("issue_codes") or []: codes[code]+=1

    summary={
        "total":len(results),
        "status_counts":dict(statuses),
        "valuation_status_counts":dict(valuation_statuses),
        "fair_value_type_counts":dict(fair_value_types),
        "confidence_counts":dict(confidences),
        "usable_fair_value_count":sum(1 for r in results if (r.get("validation") or {}).get("fair_value") not in (None,0)),
        "issue_code_counts":dict(codes),
        "sector_counts":dict(sectors),
        "crash_count":sum(1 for r in results if not r.get("ok")),
        "rows":rows,
    }
    (output_root/"batch_summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8"
    )

    issue_groups={}; error_groups={}; warning_groups={}; info_groups={}; auto_fix_groups={}
    for r in results:
        sym=r["validation"]["symbol"]
        for i in (r["validation"].get("issues") or []):
            code=i.get("code")
            if not code: continue
            issue_groups.setdefault(code,[]).append(sym)
            sev=i.get("severity")
            if sev=="ERROR": error_groups.setdefault(code,[]).append(sym)
            elif sev=="WARN": warning_groups.setdefault(code,[]).append(sym)
            elif sev=="INFO": info_groups.setdefault(code,[]).append(sym)
            if i.get("auto_fixed"): auto_fix_groups.setdefault(code,[]).append(sym)
    for name,obj in [("issue_groups.json",issue_groups),("error_groups.json",error_groups),("warning_groups.json",warning_groups),("info_groups.json",info_groups),("auto_fix_groups.json",auto_fix_groups)]:
        (output_root/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")
    return summary


def run_batch(symbols, output_root="batch_output", api_key=None,
              raw_root=None, reuse_raw=True, delay_seconds=16.0):
    results=[]
    symbols=[s.upper().strip() for s in symbols if s and s.strip()]
    for i,symbol in enumerate(symbols,1):
        print(f"[{i}/{len(symbols)}] {symbol}")
        if raw_root:
            raw_dir=Path(raw_root)/symbol/"raw"
            if not raw_dir.exists():
                # Also accept raw_root/SYMBOL directly.
                raw_dir=Path(raw_root)/symbol
            if not raw_dir.exists():
                vr=ValidationEngine(symbol).evaluate(
                    {},None,None,None,
                    stage_exception={"stage":"raw_loader","message":f"Raw directory not found: {raw_dir}"}
                )
                r={"ok":False,"metrics":None,"quality":None,"valuation":None,
                   "validation":vr,"source":"RAW_ROOT_MISSING"}
            else:
                r=evaluate_data(symbol,load_raw_dir(raw_dir),output_root)
                r["source"]="RAW_ROOT"
        else:
            r=evaluate_live(symbol,output_root,api_key,reuse_raw=reuse_raw)

        results.append(r)
        vr=r["validation"]
        print("   ->",vr["validation_status"],
              "| valuation=",vr.get("valuation_status"),
              "| issues=",",".join(vr.get("issue_codes") or []) or "NONE")

        # Free/community VNStock currently has a tight call budget and each
        # ticker uses many endpoints. Delay only after live API calls.
        if not raw_root and r.get("source")=="LIVE_API" and i<len(symbols) and delay_seconds>0:
            time.sleep(delay_seconds)

    return write_batch_reports(results,output_root)


def parse_symbols(text):
    text=text.replace("\n",",").replace(";",",")
    return [x.strip().upper() for x in text.split(",") if x.strip()]


def main():
    p=argparse.ArgumentParser(description="V8.4 Fallback + Confidence batch audit")
    p.add_argument("--symbols",help="Comma-separated tickers")
    p.add_argument("--file",help="Text file: one ticker per line or comma-separated")
    p.add_argument("--output",default="batch_output")
    p.add_argument("--api-key",default=None)
    p.add_argument("--raw-root",default=None,
                   help="Offline raw root containing SYMBOL/raw/*.csv")
    p.add_argument("--no-reuse-raw",action="store_true")
    p.add_argument("--delay",type=float,default=16.0,
                   help="Seconds between live tickers; ignored for --raw-root")
    args=p.parse_args()

    symbols=[]
    if args.symbols:
        symbols.extend(parse_symbols(args.symbols))
    if args.file:
        symbols.extend(parse_symbols(Path(args.file).read_text(encoding="utf-8")))
    # preserve order, remove duplicates
    symbols=list(dict.fromkeys(symbols))
    if not symbols:
        raise SystemExit("No tickers supplied. Use --symbols or --file.")

    summary=run_batch(
        symbols,output_root=args.output,api_key=args.api_key,
        raw_root=args.raw_root,reuse_raw=not args.no_reuse_raw,
        delay_seconds=args.delay,
    )
    print("\nBATCH DONE")
    print(json.dumps({
        "total":summary["total"],
        "status_counts":summary["status_counts"],
        "crash_count":summary["crash_count"],
        "top_issue_codes":dict(Counter(summary["issue_code_counts"]).most_common(12)),
    },ensure_ascii=False,indent=2))
    from report_engine import generate_reports
    report=generate_reports(args.output,min_score=75.0,max_price_to_fv=0.85)
    print("SCREENING REPORT:",report["selected_count"],"/",report["total_universe"])
    print("Saved:",Path(args.output).resolve())


if __name__=="__main__":
    main()
