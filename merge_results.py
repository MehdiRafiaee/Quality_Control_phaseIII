import argparse
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Merge Simulation Results")

    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing shard CSV files")
    parser.add_argument("--output", type=str, required=True, help="Path to save the final merged CSV")
    parser.add_argument("--lambda_value", type=str, required=True, help="Lambda value used in simulation")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Searching for results in: {input_dir}")
    
    # الگوی دقیق جستجو بر اساس نام‌گذاری در simulation_engine.py
    # مثال: results_lambda_0.1_shard_5.csv
    pattern = f"*results_lambda_{args.lambda_value}_shard_*.csv"
    
    # جستجوی فایل‌ها (هم در روت دایرکتوری و هم زیرپوشه‌ها برای اطمینان)
    all_files = list(input_dir.rglob(pattern))

    # اگر پیدا نشد، یک جستجوی کلی‌تر انجام می‌دهیم (Fail-safe)
    if not all_files:
        print(f"Warning: No files found for pattern '{pattern}'. Searching for ANY csv file...")
        all_files = list(input_dir.rglob("*.csv"))

    if not all_files:
        print(f"Critical Error: No CSV files found in {input_dir} to merge.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(all_files)} CSV files. Merging...")

    dfs = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            # اطمینان حاصل می‌کنیم فایل خالی نباشد
            if not df.empty:
                dfs.append(df)
            else:
                print(f"Warning: File {f} is empty. Skipping.")
        except Exception as e:
            print(f"Error reading {f}: {e}. Skipping.", file=sys.stderr)

    if not dfs:
        print("Error: No valid data extracted from files.", file=sys.stderr)
        sys.exit(1)

    # اتصال دیتافریم‌ها
    merged = pd.concat(dfs, ignore_index=True)

    # مرتب‌سازی نهایی بر اساس شماره شارد (برای نظم داده‌ها)
    if 'shard' in merged.columns:
        merged = merged.sort_values('shard')

    # ذخیره خروجی
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    merged.to_csv(output_path, index=False)
    
    print("-" * 30)
    print(f"MERGE SUCCESSFUL")
    print(f"Total Files Merged: {len(dfs)}")
    print(f"Total Rows: {len(merged)}")
    print(f"Saved to: {output_path}")
    print("-" * 30)

if __name__ == "__main__":
    main()
