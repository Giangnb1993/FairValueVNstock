from pathlib import Path
import json
import tempfile

import pandas as pd

from volume_score import volume_accumulation_score, volume_accumulation_score_from_df


def _sample():
    dates = pd.bdate_range('2026-07-01', periods=40)
    rows = []
    for i, dt in enumerate(dates):
        if i < 30:
            # Tight base with drying volume.
            close = 100.0 + ((i % 4) - 1.5) * 0.25
            vol = 1_200_000 - i * 15_000
            high, low = close + 0.45, close - 0.45
        elif i == 30:
            # Break above the prior 20-session high with >2x volume.
            close, high, low, vol = 102.8, 103.0, 100.8, 2_000_000
        elif i == 31:
            # Low-volume retest that holds the breakout level.
            close, high, low, vol = 101.0, 101.6, 100.35, 500_000
        else:
            close = 102.0 + (i - 32) * 0.35
            high, low, vol = close + 0.45, close - 0.45, 1_050_000
        rows.append({
            'time': dt.strftime('%Y-%m-%d'),
            'open': close - 0.25,
            'high': high,
            'low': low,
            'close': close,
            'volume': vol,
        })
    return pd.DataFrame(rows)


df = _sample()
a = volume_accumulation_score_from_df(df)
assert a['volume_score'] is not None
assert 0 <= a['volume_score'] <= 100
assert a['vol_breakout_price_score'] == 15.0
assert a['retest_status'] == 'HELD_LOW_VOLUME'
assert a['volume_signal'] in {'BREAKOUT_CONFIRMED', 'UPTREND_EARLY'}

with tempfile.TemporaryDirectory() as td:
    raw = Path(td) / 'raw'
    raw.mkdir(parents=True)
    df.to_csv(raw / 'price_history.csv', index=False)
    b = volume_accumulation_score(td)
    assert a['volume_score'] == b['volume_score']
    assert a['volume_signal'] == b['volume_signal']

print(json.dumps(a, ensure_ascii=False, indent=2))
print('V8.7 volume selftest: PASS')
