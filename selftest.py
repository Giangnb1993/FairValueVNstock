"""Offline self-test for the canonical year mapper. Does not call VNStock."""
import pandas as pd
from stock_audit import report_series, apply_orientation, cagr

def main():
    # Mimic upstream payload with year labels newest->oldest but values oldest->newest.
    df = pd.DataFrame([{
        "item":"Revenue","item_id":"revenue",
        "2025":44.0,"2024":52.0,"2023":62.0,"2022":70.0
    }])
    raw = report_series(df, df.iloc[0])
    fixed = apply_orientation(raw, "reversed")
    assert fixed["2025"] == 70.0
    assert fixed["2022"] == 44.0
    assert cagr(fixed) > 0
    print("SELFTEST PASS")
    print(fixed.to_dict())

if __name__ == "__main__":
    main()
