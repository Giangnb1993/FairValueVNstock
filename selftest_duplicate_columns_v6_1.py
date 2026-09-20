import pandas as pd
from latest_fundamentals import quarter_metric, parse_quarter, consecutive_latest

# Duplicate metric labels: first revenue is sparse/non-numeric, second is valid.
df=pd.DataFrame([
    ['2026-Q2','bad',200.0,20.0],
    ['2026-Q1',None,180.0,18.0],
    ['2025-Q4','x',170.0,17.0],
    ['2025-Q3','y',160.0,16.0],
],columns=['period','revenue','revenue','net_profit'])

rev=quarter_metric(df,['revenue'])
assert list(rev.index)==['2026-Q2','2026-Q1','2025-Q4','2025-Q3'], rev
assert rev['2026-Q2']==200.0 and rev['2025-Q3']==160.0, rev
assert consecutive_latest(rev,4)==['2026-Q2','2026-Q1','2025-Q4','2025-Q3']

# Duplicate period label too: physical position selection must remain 1-D.
df2=pd.DataFrame([
    ['2026-Q2','ignore',10.0],
    ['2026-Q1','ignore',9.0],
    ['2025-Q4','ignore',8.0],
    ['2025-Q3','ignore',7.0],
],columns=['period','period','metric'])
s=quarter_metric(df2,['metric'])
assert len(s)==4 and s['2026-Q2']==10.0, s

print('V6.1 DUPLICATE COLUMN SELFTEST PASS')
