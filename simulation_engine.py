import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import time
import sys

def run_bootstrap_simulation(df, n_reps, base_seed, shard_index, lambda_threshold):
    """
    اجرای شبیه‌سازی بوت‌استرپ (Bootstrap) روی داده‌های واقعی.
    """
    # 1. فیلتر کردن داده‌ها
    # فقط داده‌های موفق (Success)
    if 'status' in df.columns:
        df = df[df['status'] == 'success']
    
    # اعمال فیلتر Lambda (به عنوان حد آستانه برای inlier_ratio)
    # اگر lambda_threshold برابر 0 باشد، عملاً همه داده‌ها انتخاب می‌شوند
    filtered_data = df[df['inlier_ratio'] >= lambda_threshold]['inlier_ratio'].values
    
    n_samples = len(filtered_data)
    
    if n_samples == 0:
        raise ValueError(f"No data points found with inlier_ratio >= {lambda_threshold} and status='success'")

    print(f"Shard {shard_index}: Bootstrapping with {n_samples} samples (Threshold: {lambda_threshold})")

    # 2. تنظیم تولیدکننده اعداد تصادفی (بسیار سریعتر از روش‌های قدیمی)
    rng = np.random.default_rng(seed=base_seed)

    # 3. اجرای حلقه بوت‌استرپ با بهینه‌سازی حافظه
    # به جای ایجاد یک ماتریس عظیم (n_reps x n_samples)، کار را تکه‌تکه انجام می‌دهیم
    boot_means = []
    chunk_size = 5000  # پردازش ۵۰۰۰ تایی برای جلوگیری از پر شدن حافظه
    
    for start in range(0, n_reps, chunk_size):
        # محاسبه اندازه این تکه (ممکن است آخرین تکه کوچکتر باشد)
        current_batch_size = min(chunk_size, n_reps - start)
        
        # انتخاب ایندکس‌های تصادفی با جایگذاری (With Replacement)
        indices = rng.integers(0, n_samples, size=(current_batch_size, n_samples))
        
        # استخراج داده‌ها و میانگین‌گیری در یک مرحله برداری (Vectorized)
        batch_samples = filtered_data[indices]
        batch_means = np.mean(batch_samples, axis=1)
        
        boot_means.extend(batch_means)

    # تبدیل لیست نهایی به آرایه نامپای
    boot_means = np.array(boot_means)

    # 4. آماده‌سازی خروجی
    result_df = pd.DataFrame({
        "shard": shard_index,
        "seed": base_seed,
        "lambda_threshold": lambda_threshold,
        "bootstrap_mean": boot_means,  # هر ردیف یک نمونه‌گیری است
        "original_sample_size": n_samples
    })

    return result_df

def main():
    parser = argparse.ArgumentParser(description="Real Data Bootstrap Engine")

    parser.add_argument("--input_file", type=str, required=True, help="Path to phase2_final_features.csv")
    parser.add_argument("--lambda_value", type=float, default=0.0, help="Filter threshold for inlier_ratio")
    parser.add_argument("--n_rep_final", type=int, default=50000, help="Number of bootstrap iterations")
    parser.add_argument("--base_seed", type=int, required=True)
    parser.add_argument("--shard_index", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)
    
    # آرگومان‌های زیر برای سازگاری با نسخه‌های قبلی نگه داشته شده‌اند ولی استفاده نمی‌شوند
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument("--n_rep_cal", type=int, default=0)

    args = parser.parse_args()

    # ایجاد مسیر خروجی
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.input_file}...")
    
    try:
        start_time = time.time()
        
        # خواندن فایل CSV
        if not Path(args.input_file).exists():
            raise FileNotFoundError(f"Input file not found: {args.input_file}")
            
        df_source = pd.read_csv(args.input_file)
        
        # اجرای شبیه‌سازی
        result_df = run_bootstrap_simulation(
            df_source,
            args.n_rep_final,
            args.base_seed,
            args.shard_index,
            args.lambda_value
        )

        # نام‌گذاری فایل خروجی
        out_file = out_dir / f"results_lambda_{args.lambda_value}_shard_{args.shard_index}.csv"
        result_df.to_csv(out_file, index=False)
        
        elapsed = time.time() - start_time
        print(f"Success! Shard {args.shard_index} completed in {elapsed:.4f} seconds.")

    except Exception as e:
        print(f"CRITICAL ERROR in Shard {args.shard_index}: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
