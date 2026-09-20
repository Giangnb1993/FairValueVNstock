import pandas as pd
from historical_multiples import historical_target, fixed_blend, year_end_prices

# 1) Historical target median and outlier cleaning.
s = pd.Series({'2025':14,'2024':16,'2023':15,'2022':17,'2021':1000},dtype=float)
r = historical_target(s,lo=3,hi=80,min_points=4)
assert r['status']=='OK'
assert abs(r['target']-15.5)<1e-9

# 2) Fixed blend must be exactly 60/25/15.
d={'bear':80,'base':100,'bull':120}
p={'bear':64,'base':80,'bull':96}
e={'bear':72,'base':90,'bull':108}
b=fixed_blend(d,p,e)
assert b['status']=='OK_FULL'
assert abs(b['base']-(100*.60+80*.25+90*.15))<1e-9
assert fixed_blend(d,None,e)['status']=='PARTIAL'

# 3) Price unit normalization and 20-session year-end median.
dates=pd.date_range('2024-12-01',periods=20,freq='B')
h=pd.DataFrame({'time':dates,'close':[70.0+i*.1 for i in range(20)]})
y,meta=year_end_prices(h,72000,max_years=8)
assert meta['factor']==1000.0
assert '2024' in y.index
print('HISTORICAL SELFTEST PASS')
