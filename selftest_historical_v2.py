import pandas as pd
from historical_multiples import timeseries_metric, normalize_historical_shares, historical_target, fixed_blend
df=pd.DataFrame({"reportYear":[str(y) for y in range(2025,2017,-1)],"owners_contributed_capital_2":[17143264220000,14604721310000,12709709690000,11045153210000,9075513490000,7892684570000,6789693180000,5312170920000]})
cap=timeseries_metric(df,["owners_contributed_capital","owners_contributed_capital_2"]); assert len(cap)==8,cap
sh,meta=normalize_historical_shares(cap,1714326422,"2025"); assert meta["status"]=="PASS",meta; assert len(sh)==8
pe=pd.Series({str(y):v for y,v in zip(range(2025,2017,-1),[15.9,26.7,17.6,14.6,18,16,15,14])}); t=historical_target(pe,lo=3,hi=80,min_points=4); assert t["status"]=="OK" and t["count"]>=7
b=fixed_blend({"bear":60,"base":80,"bull":100},{"bear":70,"base":90,"bull":110},{"bear":50,"base":70,"bull":90}); assert abs(b["base"]-81)<1e-9
print("HISTORICAL V2 SELFTEST PASS",meta,t["count"])
