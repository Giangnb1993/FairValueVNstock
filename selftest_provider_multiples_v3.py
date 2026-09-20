import pandas as pd
from historical_multiples import timeseries_metric, historical_target, fixed_blend

def main():
    ratio = pd.DataFrame({
        "period":["2025","2024","2023","2022"],
        "ticker":["FPT"]*4,
        "pe_ratio":[15.91,26.74,17.56,14.64],
        "pb_ratio":[3.10,4.20,3.60,3.00],
        "ev_ebitda":[8.7,12.1,10.0,8.3],
    })

    pe = timeseries_metric(ratio, ["pe_ratio"])
    pb = timeseries_metric(ratio, ["pb_ratio"])
    ev = timeseries_metric(ratio, ["ev_ebitda"])

    assert list(pe.index) == ["2025","2024","2023","2022"]
    assert len(pe) == 4 and len(pb) == 4 and len(ev) == 4

    pe_t = historical_target(pe, lo=3, hi=80, min_points=4)
    ev_t = historical_target(ev, lo=2, hi=35, min_points=4)
    assert pe_t["status"] == "OK"
    assert ev_t["status"] == "OK"
    assert abs(pe_t["target"] - ((15.91+17.56)/2)) < 1e-9
    assert abs(ev_t["target"] - ((8.7+10.0)/2)) < 1e-9

    dcf={"bear":55000,"base":87000,"bull":138000}
    pev={"bear":74000,"base":87300,"bull":100000}
    evv={"bear":60000,"base":75000,"bull":90000}
    blend=fixed_blend(dcf,pev,evv)
    assert blend["status"] == "OK_FULL"
    assert abs(blend["base"]-(87000*.60+87300*.25+75000*.15)) < 1e-9

    print("PROVIDER MULTIPLES V3 SELFTEST PASS")

if __name__ == "__main__":
    main()
