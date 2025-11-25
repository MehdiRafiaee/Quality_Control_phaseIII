#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--lambda_value", type=str, required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    pattern = f"*results_lambda_{args.lambda_value}_shard_*.csv"
    files = sorted(input_dir.rglob(pattern))

    if not files:
        print(f"هیچ فایلی با الگوی {pattern} پیدا نشد!")
        print("فایل‌های موجود:", list(input_dir.glob("*.csv")))
        sys.exit(1)

    print(f"{len(files)} شارد پیدا شد. در حال ادغام...")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
            print(f"بارگذاری شد: {f.name} → {len(df)} ردیف")
        except Exception as e:
            print(f"خطا در {f}: {e}")

    merged = pd.concat(dfs, ignore_index=True)
    
    # میانگین‌گیری نهایی
    summary = merged.groupby("Scenario").agg({
        "ARL": ["mean", "std"],
        "h": "first"
    }).round(2)
    summary.columns = ["ARL", "SD", "h"]
    summary["ARL±SD"] = summary["ARL"].astype(str) + "±" + summary["SD"].astype(str)
    summary = summary[["h", "ARL±SD"]].reset_index()

    # ذخیره
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    
    print("\n" + "="*50)
    print("ادغام با موفقیت انجام شد!")
    print(summary.to_string(index=False))
    print(f"فایل نهایی: {args.output}")
    print("="*50)

if __name__ == "__main__":
    main()
