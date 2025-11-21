#!/usr/bin/env python3
# merge_results.py
# تجمیع نتایج شاردهای شبیه‌سازی (Map-Reduce - Reduce Phase)
# سازگار با خروجی‌های simulation_engine.py

import argparse
import pandas as pd
from pathlib import Path
import sys

def merge_shard_results(input_dir: str, output_path: str):
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"خطا: پوشه {input_dir} پیدا نشد!")
        sys.exit(1)

    # پیدا کردن تمام فایل‌های results_shard_*.csv
    shard_files = sorted(input_path.glob("results_shard_*.csv"))
    
    if not shard_files:
        print("هشدار: هیچ فایلی با الگوی results_shard_*.csv پیدا نشد!")
        sys.exit(1)

    print(f"{len(shard_files)} شارد پیدا شد. در حال تجمیع...")

    all_dfs = []
    for shard_file in shard_files:
        try:
            df = pd.read_csv(shard_file)
            all_dfs.append(df)
            print(f"   ✓ {shard_file.name} — {len(df)} ردیف")
        except Exception as e:
            print(f"   ✗ خطا در خواندن {shard_file.name}: {e}")

    if not all_dfs:
        print("خطا: هیچ داده‌ای برای تجمیع وجود ندارد!")
        sys.exit(1)

    # تجمیع با گروه‌بندی بر اساس کلیدهای مشترک
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # میانگین‌گیری روی سناریوها برای هر ترکیب (Dataset, Lambda)
    key_cols = ["Dataset", "Lambda", "h"]
    scenario_cols = ["IC", "small", "moderate", "large", "cond", "inlier"]
    
    aggregated = combined.groupby(key_cols)[scenario_cols].mean().round(2).reset_index()
    
    # مرتب‌سازی قشنگ
    aggregated = aggregated.sort_values(["Dataset", "Lambda"])

    # ذخیره نهایی
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(output_path, index=False)
    
    print("\nتجمیع با موفقیت انجام شد!")
    print(f"فایل نهایی ذخیره شد: {output_path}")
    print("\nگزارش نهایی:")
    print(aggregated.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Simulation Shards")
    parser.add_argument("--input_dir", type=str, default="aggregated_results",
                        help="پوشه‌ای که فایل‌های شارد در آن هستند")
    parser.add_argument("--output", type=str, default="aggregated_results/final_report.csv",
                        help="مسیر فایل خروجی نهایی")
    
    args = parser.parse_args()
    merge_shard_results(args.input_dir, args.output)
