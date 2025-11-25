import argparse
import pandas as pd
from pathlib import Path

def merge_results(input_dir, output):
    folder = Path(input_dir)
    files = folder.glob("results_lambda_*.csv")

    dfs = []
    for f in files:
        dfs.append(pd.read_csv(f))

    out = pd.concat(dfs, ignore_index=True)
    out.to_csv(output, index=False)
    print("Merged:", output)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default="all_shards")
    p.add_argument("--output", default="final_report.csv")
    args = p.parse_args()

    merge_results(args.input_dir, args.output)
