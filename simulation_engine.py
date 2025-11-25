import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import time


def run_simulation(lambda_value, n_rep_cal, n_rep_final, base_seed, shard_index):

    np.random.seed(base_seed)

    # نمونه‌سازی ساده. شما جایگزین می‌کنید با مدل خودتان
    cal_data = np.random.exponential(1/lambda_value, size=n_rep_cal)
    final_data = np.random.exponential(1/lambda_value, size[n_rep_final])

    df = pd.DataFrame({
        "lambda": lambda_value,
        "shard": shard_index,
        "seed": base_seed,
        "calibration_value": [cal_data.mean()],
        "final_mean": [final_data.mean()],
        "final_var": [final_data.var()]
    })

    return df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--lambda_value", type=float, default=0.1)
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument("--n_rep_cal", type=int, default=20000)
    parser.add_argument("--n_rep_final", type=int, default=50000)
    parser.add_argument("--base_seed", type=int, required=True)
    parser.add_argument("--shard_index", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)

    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = run_simulation(
        args.lambda_value,
        args.n_rep_cal,
        args.n_rep_final,
        args.base_seed,
        args.shard_index
    )

    out_file = out_dir / f"results_lambda_{args.lambda_value}_shard_{args.shard_index}.csv"
    df.to_csv(out_file, index=False)

    with open(out_dir / f"execution_log_shard_{args.shard_index}.txt", "w") as f:
        f.write(f"Finished shard {args.shard_index} at {time.ctime()}\n")


if __name__ == "__main__":
    main()
