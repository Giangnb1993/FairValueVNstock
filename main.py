import argparse
import os
from pathlib import Path

from stock_audit import StockAudit
from batch_audit import run_batch, parse_symbols
from report_engine import generate_reports


def main():
    parser = argparse.ArgumentParser(
        description="VN Stock Fair Value V8.7 - Volume Accumulation Score"
    )
    parser.add_argument("symbol", nargs="?", help="Ma co phieu, vi du FPT")
    parser.add_argument("--output", default="output", help="Thu muc luu ket qua")
    parser.add_argument("--api-key", default=None, help="VNStock API key")

    parser.add_argument("--batch", default=None,
                        help="Nhieu ma, cach nhau bang dau phay")
    parser.add_argument("--batch-file", default=None,
                        help="File txt chua danh sach ma")
    parser.add_argument("--raw-root", default=None,
                        help="Chay batch offline tu ROOT/SYMBOL/raw/*.csv")
    parser.add_argument("--delay", type=float, default=16.0,
                        help="Delay giua cac ma khi fetch live")
    parser.add_argument("--no-reuse-raw", action="store_true",
                        help="Khong dung raw cache da luu")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("VNSTOCK_API_KEY")

    symbols=[]
    if args.batch:
        symbols.extend(parse_symbols(args.batch))
    if args.batch_file:
        batch_path=Path(args.batch_file)
        if not batch_path.exists() or not batch_path.is_file():
            raise SystemExit(
                f"Khong tim thay batch file: {batch_path}. Thu muc hien tai: {Path.cwd()}"
            )
        symbols.extend(parse_symbols(batch_path.read_text(encoding="utf-8")))
    symbols=list(dict.fromkeys(symbols))

    if symbols:
        run_batch(
            symbols,
            output_root=args.output,
            api_key=api_key,
            raw_root=args.raw_root,
            reuse_raw=not args.no_reuse_raw,
            delay_seconds=args.delay,
        )
        report = generate_reports(args.output, min_score=75.0, max_price_to_fv=0.85)
        print(f"SCREENING REPORT: {report['selected_count']}/{report['total_universe']} ma dat Score > 75 va Price/FV < 0.85")
        print(f"Saved: {Path(args.output) / 'telegram_report.txt'}")
        return

    symbol = (args.symbol or input("Nhap ma co phieu (VD: FPT): ")).strip().upper()
    if not symbol:
        raise SystemExit("Ma co phieu khong duoc de trong.")

    StockAudit(symbol, args.output, api_key).run()


if __name__ == "__main__":
    main()
