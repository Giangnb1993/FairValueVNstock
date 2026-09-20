"""
Usage:
  python selftest_v8_batch.py /path/to/raw_fixture_root

Expected structure:
  raw_fixture_root/VCB/raw/*.csv
  raw_fixture_root/BVH/raw/*.csv
  ...
"""
from pathlib import Path
import json
import sys

from batch_audit import run_batch

symbols=["VCB","BVH","VHM","HPG","MWG","CTD","POW"]
if len(sys.argv)<2:
    raise SystemExit("Pass raw fixture root as argv[1].")
root=Path(sys.argv[1])
summary=run_batch(symbols,output_root="selftest_v8_output",raw_root=root,delay_seconds=0)
assert summary["crash_count"]==0, summary
rows={r["ticker"]:r for r in summary["rows"]}
for s in ["VCB","BVH","VHM","HPG","MWG","CTD"]:
    assert rows[s]["valuation_status"] in {"OK_FULL","OK_PROXY_NO_RNAV","OK_PROXY_NO_SOTP"}, rows[s]
assert rows["POW"]["valuation_status"]=="PARTIAL", rows["POW"]
print("V8 batch selftest PASS")
