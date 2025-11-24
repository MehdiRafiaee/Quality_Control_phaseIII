import argparse
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--lambda_value", type=float, required=True)

    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    all_files = list(input_dir.rglob(f"results_lambda_{args.lambda_value}_shard_*.csv"))

    dfs = [pd.read_csv(f) for f in all_files]

    merged = pd.concat(dfs, ignore_index=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
