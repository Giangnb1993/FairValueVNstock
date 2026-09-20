from __future__ import annotations

import math
import re
import unicodedata
from typing import Any, Iterable

import pandas as pd


def _norm(x: Any) -> str:
    if x is None:
        return ""
    s=unicodedata.normalize("NFD",str(x).strip().lower())
    s="".join(c for c in s if unicodedata.category(c)!="Mn")
    s=re.sub(r"[^a-z0-9]+","_",s)
    return re.sub(r"_+","_",s).strip("_")


def _f(x):
    try:
        v=float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def parse_quarter(x: Any):
    """Return (year, quarter, canonical_label) for common provider formats."""
    if x is None: return None
    s=str(x).strip().upper()
    patterns=[
        r"(20\d{2})\s*[-_/ ]?Q([1-4])",
        r"Q([1-4])\s*[-_/ ]?(20\d{2})",
        r"(20\d{2})\s*[-_/ ]?([1-4])$",
        r"(20\d{2}).*QUARTER\s*([1-4])",
        r"QUARTER\s*([1-4]).*(20\d{2})",
    ]
    for i,p in enumerate(patterns):
        m=re.search(p,s)
        if not m: continue
        if i in (1,4): q=int(m.group(1)); y=int(m.group(2))
        else: y=int(m.group(1)); q=int(m.group(2))
        return y,q,f"{y}-Q{q}"
    return None


def _period_column_pos(df: pd.DataFrame | None):
    """Return the physical column position containing quarter labels.

    Position-based selection is required because provider DataFrames can contain
    duplicate column names; df['name'] would then return a 2-D DataFrame.
    """
    if df is None or df.empty:
        return None

    preferred=("period","quarter","report_period","fiscal_period","time","date","report_date")
    normalized=[_norm(c) for c in df.columns]
    for k in preferred:
        for i,nc in enumerate(normalized):
            if nc==k:
                return i

    best=(0,None)
    for i in range(df.shape[1]):
        s=df.iloc[:,i]
        n=sum(parse_quarter(v) is not None for v in s.head(32).tolist())
        if n>best[0]:
            best=(n,i)
    return best[1] if best[0]>=1 else None


def _period_column(df: pd.DataFrame | None):
    """Backward-compatible label helper."""
    pos=_period_column_pos(df)
    return None if pos is None else df.columns[pos]


def quarter_metric(df: pd.DataFrame | None, aliases: Iterable[str], exclude: Iterable[str]=()):
    """Extract one quarterly metric safely, including duplicate column names.

    Candidates are evaluated by physical column position. This avoids pandas
    returning a DataFrame when multiple columns have the same label.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)

    ppos=_period_column_pos(df)
    if ppos is None:
        return pd.Series(dtype=float)

    an=[_norm(a) for a in aliases]
    ex=[_norm(e) for e in exclude]
    cand=[]

    for i,c in enumerate(df.columns):
        if i==ppos:
            continue
        nc=_norm(c)
        if any(e and e in nc for e in ex):
            continue
        exact=any(nc==a for a in an)
        contains=sum(1 for a in an if a and (a in nc or nc in a))
        if not (exact or contains):
            continue

        col=df.iloc[:,i]
        numeric_count=int(pd.to_numeric(col,errors='coerce').notna().sum())
        if numeric_count <= 0:
            continue

        exact_ranks=[j for j,a in enumerate(an) if nc==a]
        contains_ranks=[j for j,a in enumerate(an) if a and (a in nc or nc in a)]
        rank=min(exact_ranks or contains_ranks)
        # Numeric availability is mandatory; alias order then decides semantic preference.
        alias_priority=(len(an)-rank)*1_000_000
        score=(10_000_000 if exact else 0) + alias_priority + numeric_count*100 + contains
        cand.append((score,numeric_count,-i,i,str(c)))

    if not cand:
        return pd.Series(dtype=float)

    cand.sort(reverse=True)
    mpos=cand[0][3]
    period_col=df.iloc[:,ppos]
    metric_col=df.iloc[:,mpos]

    out={}
    for pv,mv in zip(period_col.tolist(),metric_col.tolist()):
        p=parse_quarter(pv)
        v=_f(mv)
        if p and v is not None:
            out[p[2]]=v

    return pd.Series(
        dict(sorted(out.items(),key=lambda kv:(int(kv[0][:4]),int(kv[0][-1])),reverse=True)),
        dtype=float,
    )



def canonicalize_quarter_statement(df: pd.DataFrame | None, orientation: str):
    """Reuse validated annual statement orientation for quarterly statements."""
    if df is None or df.empty or orientation != "reversed":
        return df
    ppos=_period_column_pos(df)
    if ppos is None:
        return df
    out=df.copy()
    valid=[]
    for r in range(len(out)):
        p=parse_quarter(out.iloc[r,ppos])
        if p:
            valid.append((r,p[0]*4+p[1]))
    if len(valid)<=1:
        return out
    rows=[r for r,_ in sorted(valid,key=lambda x:x[1])]
    for c in range(out.shape[1]):
        if c==ppos:
            continue
        vals=[out.iloc[r,c] for r in rows][::-1]
        for r,v in zip(rows,vals):
            out.iat[r,c]=v
    return out

def _q_ord(label):
    p=parse_quarter(label)
    return None if not p else p[0]*4+p[1]


def consecutive_latest(series: pd.Series, n=4):
    if series is None or series.empty: return []
    labs=sorted(series.index,key=lambda x:_q_ord(x) or -1,reverse=True)
    if len(labs)<n: return []
    chosen=labs[:n]
    orders=[_q_ord(x) for x in chosen]
    if any(x is None for x in orders): return []
    return chosen if all(orders[i]-orders[i+1]==1 for i in range(n-1)) else []


def latest_quarter_value(series: pd.Series):
    if series is None or series.empty: return None
    labs=sorted(series.index,key=lambda x:_q_ord(x) or -1,reverse=True)
    return _f(series.get(labs[0])) if labs else None


def _annual_value(annual_dict, year):
    if not annual_dict: return None
    return _f(annual_dict.get(str(year)))


def _flow_mode(series: pd.Series, annual_dict: dict):
    """Detect standalone-quarter vs cumulative-YTD, including partial completed-year evidence."""
    if series is None or series.empty:
        return 'unknown',None
    by_year={}
    for lab,v in series.items():
        p=parse_quarter(lab)
        if p:
            by_year.setdefault(p[0],{})[p[1]]=_f(v)

    full_standalone=[]; q4_ytd=[]; partial_standalone=[]
    for y,qm in by_year.items():
        av=_annual_value(annual_dict,y)
        if av is None:
            continue
        denom=max(abs(av),1.0)
        if 4 in qm and qm[4] is not None:
            yerr=abs(qm[4]-av)/denom
            q4_ytd.append(yerr)
            if yerr>0.35:
                partial_standalone.append(yerr)
        if all(q in qm and qm[q] is not None for q in (1,2,3,4)):
            serr=abs(sum(qm[q] for q in (1,2,3,4))-av)/denom
            full_standalone.append(serr)

    if full_standalone:
        se=sum(full_standalone)/len(full_standalone)
        ye=sum(q4_ytd)/len(q4_ytd) if q4_ytd else 999
        if min(se,ye)>0.35:
            return 'unreconciled',min(se,ye)
        return ('standalone',se) if se<=ye else ('ytd_cumulative',ye)

    if q4_ytd:
        ye=min(q4_ytd)
        if ye<=0.20:
            # Q4 approximately equals FY -> evidence for cumulative/YTD.
            return 'ytd_cumulative_partial_evidence',ye
        # IMPORTANT: Q4 being far from FY only proves "not YTD".
        # It does NOT prove that provider rows are standalone quarters.
        return 'not_ytd_unverified',ye
    return 'unknown',None

def _standalone_series(raw: pd.Series, mode: str, annual_dict: dict):
    if raw is None or raw.empty: return pd.Series(dtype=float)
    if mode == 'standalone':
        return raw.copy()
    if mode not in {'ytd_cumulative','ytd_cumulative_partial_evidence'}:
        return pd.Series(dtype=float)
    by_year={}
    for lab,v in raw.items():
        p=parse_quarter(lab)
        if p: by_year.setdefault(p[0],{})[p[1]]=_f(v)
    out={}
    for y,qm in by_year.items():
        if 1 in qm and qm[1] is not None: out[f'{y}-Q1']=qm[1]
        if 2 in qm and qm[2] is not None and 1 in qm and qm[1] is not None: out[f'{y}-Q2']=qm[2]-qm[1]
        if 3 in qm and qm[3] is not None and 2 in qm and qm[2] is not None: out[f'{y}-Q3']=qm[3]-qm[2]
        if 4 in qm and qm[4] is not None:
            if 3 in qm and qm[3] is not None: out[f'{y}-Q4']=qm[4]-qm[3]
            else:
                av=_annual_value(annual_dict,y)
                if av is not None and all(q in out for q in (f'{y}-Q1',f'{y}-Q2',f'{y}-Q3')):
                    out[f'{y}-Q4']=av-out[f'{y}-Q1']-out[f'{y}-Q2']-out[f'{y}-Q3']
    return pd.Series(dict(sorted(out.items(),key=lambda kv:_q_ord(kv[0]),reverse=True)),dtype=float)


def _ttm(raw: pd.Series, annual_dict: dict):
    mode,err=_flow_mode(raw,annual_dict)
    if mode in {'unknown','not_ytd_unverified','unreconciled'}:
        clean=pd.Series(dtype=float)
        mode_used='unknown_unverified' if mode=='unknown' else mode
    else:
        clean=_standalone_series(raw,mode,annual_dict); mode_used=mode
    periods=consecutive_latest(clean,4)
    if len(periods)!=4: return None,periods,mode_used,err
    vals=[_f(clean.get(p)) for p in periods]
    if any(v is None for v in vals): return None,periods,mode_used,err
    return float(sum(vals)),periods,mode_used,err


def _ratio_latest(df, aliases):
    s=quarter_metric(df,aliases)
    v=latest_quarter_value(s)
    if v is not None and any('roe' in _norm(a) or 'roa' in _norm(a) for a in aliases):
        v/=100.0
    return v



def _avg_latest_two_quarters(s: pd.Series):
    if s is None or s.empty: return None
    labs=sorted(s.index,key=lambda x:_q_ord(x) or -1,reverse=True)
    vals=[_f(s.get(l)) for l in labs[:2]]
    vals=[v for v in vals if v is not None]
    return (sum(vals)/len(vals)) if vals else None

def choose_quarter_statement_orientation(iq,bq,rq,annual_orientation='normal',shares=None):
    roe_ref=_ratio_latest(rq,['roe'])
    roa_ref=_ratio_latest(rq,['roa'])
    bvps_ref=_ratio_latest(rq,['book_value_per_share_bvps','bvps','book_value_per_share'])

    def inspect(orientation):
        ci=canonicalize_quarter_statement(iq,orientation)
        cb=canonicalize_quarter_statement(bq,orientation)
        ni=quarter_metric(ci,['net_profit','net_income','profit_after_tax'])
        eq=quarter_metric(cb,['equity','owners_equity_2','owners_equity_3','owners_equity',
                              'shareholders_equity','total_equity','capital_and_reserves'])
        assets=quarter_metric(cb,['total_assets'])
        paid=quarter_metric(cb,['capital_from_owners','owners_capital','contributed_capital','charter_capital'])
        ni0=latest_quarter_value(ni); eq0=latest_quarter_value(eq); paid0=latest_quarter_value(paid)
        report_shares=(paid0/10000.0) if paid0 is not None and paid0>0 else None
        anchor_shares=report_shares if report_shares else shares
        implied_equity=(bvps_ref*anchor_shares) if (
            bvps_ref is not None and anchor_shares is not None and anchor_shares>0 and bvps_ref>0
        ) else None
        ae=_avg_latest_two_quarters(eq); aa=_avg_latest_two_quarters(assets)
        roe=ni0/ae if ni0 is not None and ae not in (None,0) else None
        roa=ni0/aa if ni0 is not None and aa not in (None,0) else None
        eq_err=abs(eq0-implied_equity)/abs(implied_equity) if (
            implied_equity not in (None,0) and eq0 is not None
        ) else None
        parts=[]
        if roe_ref is not None and roe is not None: parts.append(abs(roe-roe_ref))
        if roa_ref is not None and roa is not None: parts.append(abs(roa-roa_ref))
        return {'roe_calc':roe,'roa_calc':roa,'roe_ref':roe_ref,'roa_ref':roa_ref,
                'latest_equity':eq0,'bvps_ref':bvps_ref,
                'report_shares_estimate':report_shares,
                'anchor_shares':anchor_shares,
                'share_anchor_source':'latest_paid_in_capital_div_10000' if report_shares else 'company_info_current_shares_fallback',
                'current_shares':shares,'latest_paid_in_capital':paid0,
                'implied_equity_from_bvps':implied_equity,
                'equity_anchor_error':eq_err,
                'ratio_error':sum(parts) if parts else None}
    mn=inspect('normal'); mr=inspect('reversed')
    en=mn.get('equity_anchor_error'); er=mr.get('equity_anchor_error')
    if en is not None and er is not None:
        best=min(en,er); diff=abs(en-er)
        if best<=0.15 and diff>=0.02:
            chosen='normal' if en<er else 'reversed'; reason='bvps_times_report_shares_equity_anchor'
        elif best<=0.05 and diff>=0.005:
            chosen='normal' if en<er else 'reversed'; reason='bvps_times_report_shares_equity_anchor_close'
        else:
            rn=mn.get('ratio_error'); rr=mr.get('ratio_error')
            if rn is not None and rr is not None and min(rn,rr)<=0.02 and abs(rn-rr)>=0.01:
                chosen='normal' if rn<rr else 'reversed'; reason='strong_quarter_roe_roa_secondary_evidence'
            else:
                chosen=annual_orientation; reason='annual_orientation_fallback_ambiguous_quarter_evidence'
    else:
        rn=mn.get('ratio_error'); rr=mr.get('ratio_error')
        if rn is not None and rr is not None and min(rn,rr)<=0.02 and abs(rn-rr)>=0.01:
            chosen='normal' if rn<rr else 'reversed'; reason='strong_quarter_roe_roa_secondary_evidence'
        else:
            chosen=annual_orientation; reason='annual_orientation_fallback_no_strong_quarter_anchor'
    meta=mn if chosen=='normal' else mr
    return chosen,{'chosen':chosen,'reason':reason,'normal':mn,'reversed':mr,**meta}



def _quarter_end_month(label):
    p=parse_quarter(label)
    if not p: return None
    y,q,_=p
    return y,q*3

def _capital_change_months(history):
    out=[]
    if not isinstance(history,str): return out
    for mm,yy,verb in re.findall(
        r"Tháng\s+(\d{1,2})/(\d{4})\s*:\s*(Tăng|Giảm)\s+vốn\s+điều\s+lệ",
        history, flags=re.IGNORECASE):
        try: out.append((int(yy),int(mm),verb.lower()))
        except Exception: pass
    return out

def _resolve_balance_stock_period(bq,latest_period,current_shares,company_info):
    r={"status":"NO_REMAP","requested_period":latest_period,"resolved_period":latest_period,"reason":"no_evidence"}
    if bq is None or getattr(bq,"empty",True) or not latest_period or not current_shares: return r
    qe=_quarter_end_month(latest_period)
    if not qe: return r
    history=None
    if company_info is not None and not getattr(company_info,"empty",True):
        for c in company_info.columns:
            if _norm(c)=="history":
                try: history=company_info.iloc[0][c]
                except Exception: history=None
                break
    changes=_capital_change_months(history)
    if not changes:
        r["reason"]="no_dated_capital_change_in_history"; return r
    yend,mend=qe
    if not any(y<yend or (y==yend and m<=mend) for y,m,_ in changes):
        r["reason"]="capital_change_after_report_period"; return r
    paid=quarter_metric(bq,['capital_from_owners','owners_capital','contributed_capital','charter_capital'])
    if paid.empty:
        r["reason"]="no_quarter_paid_in_capital"; return r
    target=float(current_shares)*10000.0
    cand=[]
    for lab,v in paid.items():
        vv=_f(v)
        if vv is not None and target>0: cand.append((abs(vv-target)/target,lab,vv))
    if not cand:
        r["reason"]="no_capital_candidates"; return r
    cand.sort(key=lambda x:x[0])
    best_err,best_lab,best_val=cand[0]
    nominal=_f(paid.get(latest_period))
    nominal_err=abs(nominal-target)/target if nominal is not None else None
    r.update({"target_current_paid_in_capital":target,"nominal_latest_paid_in_capital":nominal,
              "nominal_latest_error":nominal_err,"best_matching_period":best_lab,
              "best_matching_paid_in_capital":best_val,"best_matching_error":best_err,
              "capital_changes":changes})
    if best_err<=0.005 and best_lab!=latest_period and (nominal_err is None or nominal_err>=0.02):
        r.update({"status":"REMAPPED","resolved_period":best_lab,
                  "reason":"dated_pre_report_capital_change_and_current_capital_match"})
    else:
        r["reason"]="nominal_period_not_disproved"
    return r

def _value_at_period(series,period):
    if series is None or series.empty or not period: return None
    return _f(series.get(period))

def _explicit_field_exists(df, aliases):
    if df is None or getattr(df,"empty",True): return False
    amap={_norm(c):c for c in df.columns}
    return any(_norm(a) in amap for a in aliases)

def _ratio_value_for_period(df, period, aliases):
    if df is None or getattr(df,"empty",True) or not period: return None
    cmap={_norm(c):c for c in df.columns}; pc=cmap.get("period")
    vc=next((cmap.get(_norm(a)) for a in aliases if cmap.get(_norm(a)) is not None),None)
    if pc is None or vc is None: return None
    rows=df[df[pc].astype(str).str.strip()==str(period)]
    return None if rows.empty else _f(rows.iloc[-1][vc])

def _explicit_all_null_field(df, aliases):
    """True only when an explicit provider debt field exists and all observations are null."""
    if df is None or getattr(df, "empty", True):
        return False
    amap={_norm(c):c for c in df.columns}
    for alias in aliases:
        c=amap.get(_norm(alias))
        if c is not None:
            vals=pd.to_numeric(df[c],errors="coerce")
            if vals.notna().sum()==0:
                return True
    return False


def build_latest_snapshot(data: dict, annual_metrics: dict):
    annual_orientation=annual_metrics.get('orientation','normal')
    rq=data.get('ratio_q_ts')
    quarter_orientation,quarter_orientation_meta=choose_quarter_statement_orientation(
        data.get('income_q_ts'),data.get('balance_q_ts'),rq,annual_orientation,
        annual_metrics.get('shares'))
    iq=canonicalize_quarter_statement(data.get('income_q_ts'),quarter_orientation)
    bq=canonicalize_quarter_statement(data.get('balance_q_ts'),quarter_orientation)
    cq=canonicalize_quarter_statement(data.get('cashflow_q_ts'),quarter_orientation)
    if all(x is None or getattr(x,'empty',True) for x in (iq,bq,cq,rq)):
        return {'status':'FALLBACK_ANNUAL','basis':'LATEST_ANNUAL','fallback_reason':'NO_QUARTER_DATA','overrides':{}}

    aliases={
      'revenue':['revenue','net_revenue','net_sales','sales_revenue'],
      'net_income':['net_profit','net_income','profit_after_tax'],
      'parent_profit':['profit_after_tax_for_shareholders_of_parent_company','profit_after_tax_for_shareholders_of_the_parent_company','net_profit_parent','profit_attributable_to_parent','profit_after_tax_parent'],
      'pbt':['profit_before_tax','pre_tax_profit','pretax_profit'],
      'operating_profit':['operating_profit','gross_profit','profit_from_business_activities','profit_from_operating_activities'],
      'interest':['interest_expense','borrowing_costs','of_which_interest_expense'],
      'tax':['income_tax_expense','corporate_income_tax','current_corporate_income_tax_expenses','tax_expense'],
      'da':['depreciation','depreciation_and_amortization','depreciation_of_fixed_assets_and_investment_properties'],
      'cfo':['operating_cash_flow','cash_from_operations','net_cash_from_operating_activities'],
      'capex':['payment_for_fixed_assets_constructions_and_other_long_term_assets','purchase_of_fixed_assets','capital_expenditure','capex'],
      'equity':['equity','owners_equity_2','owners_equity','shareholders_equity','total_equity','capital_and_reserves'],
      'assets':['total_assets'],
      'cash':['cash_and_cash_equivalents','cash'],
      'short_debt':['short_term_borrowings_and_financial_leases','short_term_borrowings','short_term_debt','short_term_loans'],
      'long_debt':['long_term_borrowings_and_financial_leases','long_term_borrowings','long_term_debt','long_term_loans'],
      'total_liabilities':['total_liabilities','liabilities'],
      'current_liabilities':['current_liabilities','short_term_liabilities'],
      'paid_in_capital':['capital_from_owners','owners_capital','contributed_capital','charter_capital'],
    }
    q={}
    for k in ('revenue','net_income','parent_profit','pbt','operating_profit','interest','tax'):
        q[k]=quarter_metric(iq,aliases[k])
    # D&A/interest can be in CF; fill missing from cash flow
    q['da']=quarter_metric(cq,aliases['da'])
    if q['interest'].empty: q['interest']=quarter_metric(cq,aliases['interest'])
    q['cfo']=quarter_metric(cq,aliases['cfo'])
    q['capex']=quarter_metric(cq,aliases['capex'])
    for k in ('equity','assets','cash','short_debt','long_debt','total_liabilities','current_liabilities','paid_in_capital'):
        q[k]=quarter_metric(bq,aliases[k])

    annual=annual_metrics.get('series',{})
    flow_map={
      'revenue':'revenue','net_income':'net_income','parent_profit':'parent_profit','pbt':'pbt',
      'operating_profit':'operating_profit','interest':'interest_expense','da':'depreciation',
      'cfo':'cfo','capex':'capex'
    }
    ttm={}; modes={}; rec={}; ttm_periods=[]
    for k,ak in flow_map.items():
        val,periods,mode,err=_ttm(q[k],annual.get(ak,{}) or {})
        ttm[k]=val; modes[k]=mode; rec[k]=err
        if not ttm_periods and periods: ttm_periods=periods

    # Require core TTM income metrics to use a TTM basis. CFO/FCFF may legitimately be absent for financials.
    core_ok = ttm.get('revenue') is not None and ttm.get('net_income') is not None
    latest_period=None
    for s in list(q.values()):
        if s is not None and not s.empty:
            lab=sorted(s.index,key=lambda x:_q_ord(x) or -1,reverse=True)[0]
            if latest_period is None or (_q_ord(lab) or -1)>(_q_ord(latest_period) or -1): latest_period=lab
    balance_period_resolution=_resolve_balance_stock_period(
        bq,latest_period,annual_metrics.get('shares'),data.get('company_info'))
    stock_period=balance_period_resolution.get('resolved_period') or latest_period
    overrides={}
    for src,dst in [('equity','latest_equity'),('assets','latest_assets'),('cash','latest_cash'),('short_debt','latest_short_debt'),('long_debt','latest_long_debt'),('paid_in_capital','latest_paid_in_capital')]:
        v=_value_at_period(q[src],stock_period)
        if v is not None: overrides[dst]=v

    debt_derivation=None
    short_at_stock=_value_at_period(q['short_debt'],stock_period); long_at_stock=_value_at_period(q['long_debt'],stock_period)
    dta=_ratio_value_for_period(rq,latest_period,['debt_to_assets'])
    exact_period=(stock_period==latest_period and balance_period_resolution.get('status')!='REMAPPED')
    if (overrides.get('latest_short_debt') is None and overrides.get('latest_long_debt') is None
        and short_at_stock is None and long_at_stock is None
        and _explicit_field_exists(bq,aliases['short_debt']) and _explicit_field_exists(bq,aliases['long_debt'])
        and exact_period and dta is not None and abs(float(dta))<=0.005):
        overrides['latest_short_debt']=0.0; overrides['latest_long_debt']=0.0
        debt_derivation={'short_debt':'derived_zero_from_latest_ratio_debt_to_assets_zero',
            'long_debt':'derived_zero_from_latest_ratio_debt_to_assets_zero','ratio_period':latest_period,
            'debt_to_assets':dta,'evidence':'both_explicit_borrowing_fields_null_at_exact_latest_balance_period'}
    if overrides.get('latest_long_debt') is None:
        tl=_value_at_period(q['total_liabilities'],stock_period)
        cl=_value_at_period(q['current_liabilities'],stock_period)
        if tl is not None and cl is not None and tl>=0 and cl>=0:
            gap=tl-cl
            tol=max(1_000_000.0,0.005*max(abs(tl),1.0))
            if abs(gap)<=tol:
                overrides['latest_long_debt']=0.0
                debt_derivation={
                    'long_debt':'derived_zero_from_total_liabilities_equals_current_liabilities',
                    'total_liabilities':tl,'current_liabilities':cl,'gap':gap,'tolerance':tol
                }

    # VNStock can encode an explicit zero-debt line as an all-null column.
    # Infer zero only if that exact debt field exists in the schema; never when absent.
    if overrides.get('latest_long_debt') is None and _explicit_all_null_field(bq, aliases['long_debt']):
        overrides['latest_long_debt']=0.0
        debt_derivation=(debt_derivation or {}) | {
            'long_debt':'derived_zero_from_explicit_all_null_long_debt_field'
        }
    if overrides.get('latest_short_debt') is None and _explicit_all_null_field(bq, aliases['short_debt']):
        overrides['latest_short_debt']=0.0
        debt_derivation=(debt_derivation or {}) | {
            'short_debt':'derived_zero_from_explicit_all_null_short_debt_field'
        }

    if overrides.get('latest_short_debt') is not None and overrides.get('latest_long_debt') is not None:
        overrides['latest_total_debt']=overrides['latest_short_debt']+overrides['latest_long_debt']

    if core_ok:
        mapping={'revenue':'latest_revenue','net_income':'latest_net_income','parent_profit':'latest_parent_profit','pbt':'latest_pbt','operating_profit':'latest_operating_profit','interest':'latest_interest_expense','da':'latest_da','cfo':'latest_cfo'}
        for src,dst in mapping.items():
            if ttm.get(src) is not None: overrides[dst]=ttm[src]
        if ttm.get('capex') is not None: overrides['latest_capex']=abs(ttm['capex'])
        if overrides.get('latest_operating_profit') is not None:
            overrides['latest_ebit']=overrides['latest_operating_profit']
        elif overrides.get('latest_pbt') is not None and overrides.get('latest_interest_expense') is not None:
            overrides['latest_ebit']=overrides['latest_pbt']+abs(overrides['latest_interest_expense'])
        if overrides.get('latest_ebit') is not None and overrides.get('latest_da') is not None:
            overrides['latest_ebitda']=overrides['latest_ebit']+abs(overrides['latest_da'])
        # TTM FCFF for normal-company DCF only when CFO and CAPEX exist.
        if overrides.get('latest_cfo') is not None and overrides.get('latest_capex') is not None:
            pbt=overrides.get('latest_pbt'); tax_ttm=ttm.get('tax')
            tax_rate=0.20
            if pbt and pbt>0 and tax_ttm is not None: tax_rate=max(0,min(0.35,abs(tax_ttm)/pbt))
            ati=abs(overrides.get('latest_interest_expense') or 0)*(1-tax_rate)
            overrides['latest_tax_rate']=tax_rate
            overrides['latest_fcff']=overrides['latest_cfo']+ati-overrides['latest_capex']

    # Average equity/assets: latest quarter vs same quarter one year earlier if available; else latest two quarters.
    def avg_balance(s):
        if s is None or s.empty: return None
        labs=sorted(s.index,key=lambda x:_q_ord(x) or -1,reverse=True)
        if not labs: return None
        latest=latest_quarter_value(s); lo=_q_ord(labs[0])
        prior=next((_f(s.get(x)) for x in labs[1:] if _q_ord(x)==lo-4),None)
        if prior is None and len(labs)>1: prior=_f(s.get(labs[1]))
        return (latest+prior)/2 if latest is not None and prior is not None else latest

    ratio_refs={
      'roe_trailing':_ratio_latest(rq,['roe_trailling','roe_trailing','roe']),
      'roa_trailing':_ratio_latest(rq,['roa_trailling','roa_trailing','roa']),
      'trailing_eps':_ratio_latest(rq,['trailing_eps','eps_ttm','earnings_per_share']),
      'bvps':_ratio_latest(rq,['book_value_per_share_bvps','bvps','book_value_per_share']),
      'beta':_ratio_latest(rq,['beta']),
      'pe':_ratio_latest(rq,['pe_ratio','pe']),
      'pb':_ratio_latest(rq,['pb_ratio','pb']),
    }

    # Use TTM only when core flow metrics exist. Otherwise balance sheet may still be fresher,
    # but we do not label the whole valuation TTM.
    status='OK' if core_ok else 'PARTIAL'
    basis='TTM_PLUS_LATEST_QUARTER' if core_ok else 'LATEST_QUARTER_BALANCE_WITH_ANNUAL_FLOWS'
    if core_ok:
        reason=None
    else:
        unverified=[k for k,v in modes.items() if v in {'not_ytd_unverified','unknown_unverified','unreconciled'}]
        reason=('QUARTER_FLOW_NOT_VERIFIED:' + ','.join(unverified)) if unverified else 'INSUFFICIENT_4_CONSECUTIVE_QUARTERS_FOR_CORE_TTM'
    return {
      'status':status,'basis':basis,'latest_period':latest_period,'ttm_periods':ttm_periods,
      'flow_mode':modes,'reconciliation':rec,'fallback_reason':reason,'overrides':overrides,
      'average_equity':(_value_at_period(q['equity'],stock_period) if balance_period_resolution.get('status')=='REMAPPED' else avg_balance(q['equity'])),
      'average_assets':(_value_at_period(q['assets'],stock_period) if balance_period_resolution.get('status')=='REMAPPED' else avg_balance(q['assets'])),
      'ratio_references':ratio_refs,
      'quarter_statement_orientation':quarter_orientation,
      'quarter_orientation_meta':quarter_orientation_meta,
      'annual_statement_orientation':annual_orientation,
      'balance_period_resolution':balance_period_resolution,
      'balance_stock_period':stock_period,
      'debt_derivation':debt_derivation,
      'reported_shares_estimate':(overrides.get('latest_paid_in_capital')/10000.0 if overrides.get('latest_paid_in_capital') else quarter_orientation_meta.get('report_shares_estimate')),
      'current_shares':annual_metrics.get('shares'),
      'share_change_ratio':(
          annual_metrics.get('shares')/(overrides.get('latest_paid_in_capital')/10000.0)
          if annual_metrics.get('shares') and overrides.get('latest_paid_in_capital')
          else (annual_metrics.get('shares')/quarter_orientation_meta.get('report_shares_estimate')
                if annual_metrics.get('shares') and quarter_orientation_meta.get('report_shares_estimate') else None)
      ),
      'quarter_period_counts':{k:len(v) for k,v in q.items()},
      'balance_freshness':{
        'equity':'latest_quarter' if 'latest_equity' in overrides else 'annual_fallback',
        'assets':'latest_quarter' if 'latest_assets' in overrides else 'annual_fallback',
        'cash':'latest_quarter' if 'latest_cash' in overrides else 'annual_fallback',
        'short_debt':'latest_quarter' if 'latest_short_debt' in overrides else 'annual_fallback',
        'long_debt':'latest_quarter' if 'latest_long_debt' in overrides else 'annual_fallback',
        'total_debt':'latest_quarter_complete' if 'latest_total_debt' in overrides else 'not_recomputed_incomplete_components',
      },
    }
