import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import time
import sys

def run_simulation(lambda_value, n_rep_cal, n_rep_final, base_seed, shard_index):
    """
    اجرای شبیه‌سازی با استفاده از Generator جدید NumPy برای سرعت و دقت بیشتر.
    """
    # استفاده از PCG64 که سریع‌تر و مدرن‌تر از MT19937 (روش قدیمی) است
    rng = np.random.default_rng(seed=base_seed)

    # تولید داده‌ها
    # نکته: در نسخه جدید نامپای، متد exponential روی شیء rng صدا زده می‌شود
    cal_data = rng.exponential(scale=1/lambda_value, size=n_rep_cal)
    final_data = rng.exponential(scale=1/lambda_value, size=n_rep_final)

    # ساخت دیتافریم نتایج
    df = pd.DataFrame({
        "lambda": [lambda_value],
        "shard": [shard_index],
        "seed": [base_seed],
        "calibration_mean": [cal_data.mean()],
        "final_mean": [final_data.mean()],
        "final_var": [final_data.var()],
        "n_samples": [n_rep_final]
    })

    return df

def main():
    parser = argparse.ArgumentParser(description="Monte Carlo Simulation Engine")

    parser.add_argument("--lambda_value", type=float, default=0.1)
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument("--n_rep_cal", type=int, default=20000)
    parser.add_argument("--n_rep_final", type=int, default=50000)
    parser.add_argument("--base_seed", type=int, required=True)
    parser.add_argument("--shard_index", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)

    args = parser.parse_args()

    # ایجاد مسیر خروجی
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting simulation for Shard {args.shard_index} with Lambda={args.lambda_value}...")
    start_time = time.time()

    try:
        df = run_simulation(
            args.lambda_value,
            args.n_rep_cal,
            args.n_rep_final,
            args.base_seed,
            args.shard_index
        )

        # ذخیره فایل CSV
        out_file = out_dir / f"results_lambda_{args.lambda_value}_shard_{args.shard_index}.csv"
        df.to_csv(out_file, index=False)
        
        elapsed = time.time() - start_time
        print(f"Shard {args.shard_index} completed in {elapsed:.4f} seconds.")

        # ذخیره لاگ اجرا (اختیاری ولی مفید برای دیباگ)
        with open(out_dir / f"log_shard_{args.shard_index}.txt", "w") as f:
            f.write(f"Shard: {args.shard_index}\n")
            f.write(f"Status: Success\n")
            f.write(f"Duration: {elapsed:.4f}s\n")
            f.write(f"Timestamp: {time.ctime()}\n")

    except Exception as e:
        print(f"Error in shard {args.shard_index}: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
