import argparse
import numpy as np
import pandas as pd
from numba import njit, prange

# -----------------------------------------
# Efficient batch-based Gaussian sampling
# -----------------------------------------
@njit(fastmath=True, nogil=True)
def generate_normal_batch(n, p, seed):
    rng = np.random.RandomState(seed)
    return rng.normal(0.0, 1.0, (n, p))

# -----------------------------------------
# Fast MEWMA simulation kernel
# -----------------------------------------
@njit(fastmath=True, nogil=True, parallel=True)
def simulate_batch(lamb, h, L, inv_mat, shift, scale, seeds, max_steps):
    n = seeds.shape[0]
    p = shift.shape[0]
    results = np.empty(n, dtype=np.int32)

    for i in prange(n):
        rng = np.random.RandomState(seeds[i])
        Z = np.zeros(p)
        for t in range(1, max_steps + 1):
            noise = rng.normal(0.0, 1.0, p)
            X = shift + scale * L.dot(noise)
            Z = (1 - lamb) * Z + lamb * X
            stat = Z @ inv_mat @ Z
            if stat > h:
                results[i] = t
                break
        else:
            results[i] = max_steps
    return results

# -----------------------------------------
# Main
# -----------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_value", type=float, required=True)
    parser.add_argument("--n_sim", type=int, default=50000)
    parser.add_argument("--base_seed", type=int, default=1234)
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()

    lamb = args.lambda_value

    # Load Phase 2 feature matrix
    df = pd.read_csv("phase2_final_features.csv")
    L = df.values
    p = L.shape[0]

    inv_mat = np.linalg.inv(np.cov(L))  # Fast estimation

    # Example shift vectors (IC, small, moderate ...)
    scenarios = {
        "IC":      np.zeros(p),
        "small":   np.ones(p) * 0.1,
        "moderate":np.ones(p) * 0.3,
        "large":   np.ones(p) * 0.6,
    }

    # Calibrated h (placeholder: set per-lambda)
    h = 8.5

    seeds = np.random.randint(0, 2**32, size=args.n_sim)

    records = []
    for name, shift in scenarios.items():
        res = simulate_batch(
            lamb, h, L, inv_mat, shift,
            scale=1.0,
            seeds=seeds,
            max_steps=50000
        )
        arl = res.mean()
        records.append([lamb, name, arl])

    out_df = pd.DataFrame(records, columns=["Lambda", "Scenario", "ARL"])
    out_df.to_csv(f"{args.out}/results_lambda_{lamb}.csv", index=False)

if __name__ == "__main__":
    main()
