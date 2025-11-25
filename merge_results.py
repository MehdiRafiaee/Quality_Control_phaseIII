import argparse
import pandas as pd
from pathlib import Path
import sys

def main():
    parser = argparse.ArgumentParser(description="Merge Simulation Results")

    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing shard CSV files")
    parser.add_argument("--output", type=str, required=True, help="Path to save the final merged CSV")
    parser.add_argument("--lambda_value", type=str, required=True, help="Lambda value used in simulation (for filtering)")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # پیدا کردن فایل‌ها به صورت بازگشتی (هم در روت و هم در ساب‌فولدرها)
    # الگوی جستجو بر اساس نام فایل‌های تولید شده در شاردها
    pattern = f"**/*results_lambda_{args.lambda_value}_shard_*.csv"
    all_files = list(input_dir.rglob(f"results_lambda_{args.lambda_value}_shard_*.csv"))

    if not all_files:
        # تلاش مجدد با الگوی ساده‌تر اگر الگوی دقیق پیدا نشد (Fail-safe)
        print(f"Warning: No files found matching strict pattern. Trying generic CSV search...")
        all_files = list(input_dir.rglob("*.csv"))

    if not all_files:
        print(f"Error: No result CSV files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(all_files)} files to merge.")

    dfs = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not read {f}. Skipping. Error: {e}")

    if not dfs:
        print("Error: No valid dataframes to concat.", file=sys.stderr)
        sys.exit(1)

    merged = pd.concat(dfs, ignore_index=True)

    # مرتب‌سازی بر اساس شارد برای نظم بیشتر
    if 'shard' in merged.columns:
        merged = merged.sort_values('shard')

    # ایجاد مسیر خروجی
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    merged.to_csv(output_path, index=False)
    print(f"Successfully merged {len(dfs)} shards into '{output_path}'.")

if __name__ == "__main__":
    main()
