#!/usr/bin/env python3
# merge_results.py - سازگار با improved_simulation_engine_fixed.py
import argparse
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    # فایل‌های خروجی جدید به این شکل هستند:
    pattern = "*_MEWMA_SUMMARY_*.csv"
    files = list(input_dir.glob(pattern)) + list(input_dir.rglob(pattern))

    if not files:
        print("خطا: هیچ فایلی پیدا نشد! فایل‌ها باید شامل MEWMA_SUMMARY باشند.")
        print("فایل‌های موجود:", list(input_dir.glob("*")))
        raise FileNotFoundError("No summary files found")

    print(f"تعداد فایل‌های پیدا شده: {len(files)}")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            print(f"بارگذاری شد: {f.name} → {len(df)} ردیف")
            dfs.append(df)
        except Exception as e:
            print(f"خطا در خواندن {f}: {e}")

    combined = pd.concat(dfs, ignore_index=True)

    # استخراج lambda و گروه‌بندی
    combined["Lambda"] = combined["Lambda"].round(2)
    summary = combined.groupby(["Dataset", "Lambda", "h"]).agg({
        col: lambda x: " ± ".join(x.str.split(" ± ").map(lambda y: y[0]).astype(float).round(2).astype(str)) if "±" in " ".join(x) else x.iloc[0]
        for col in combined.columns if col not in ["Dataset", "Lambda", "h", "Seed"]
    }).reset_index()

    # مرتب‌سازی
    summary = summary.sort_values(["Lambda", "Dataset"])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(f"\nگزارش نهایی با موفقیت ذخیره شد: {args.output}")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()