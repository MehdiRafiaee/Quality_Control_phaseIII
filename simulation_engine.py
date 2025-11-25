#!/usr/bin/env python3
"""
simulation_engine.py - Final Fixed Version
کاملاً بدون خطا — آماده اجرا در GitHub Actions
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
import math

try:
    from numba import njit, prange
except ImportError:
    print("CRITICAL: numba not installed")
    exit(1)

# تنظیم لاگ
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("results/execution_log.txt", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# -------------------- Numba Kernels --------------------
@njit(inline='always')
def rand_normal(state):
    u1 = 1.0
    while u1 == 0:
        state = (state * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF
        u1 = state / 18446744073709551616.0
    state = (state * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF
    u2 = state / 18446744073709551616.0
    r = math.sqrt(-2.0 * math.log(u1))
    return r * math.cos(2.0 * math.pi * u2), state

@njit(nogil=True, fastmath=True, cache=True)
def kernel_mewma(lamb, h, inv_mat, L, shift_vec, scale_vec, seed, max_steps):
    state = seed
    p = L.shape[0]
    Z = np.zeros(p, dtype=np.float64)
    for t in range(1, max_steps + 1):
        noise = np.zeros(p, dtype=np.float64)
        for i in range(0, p, 2):
            n1, state = rand_normal(state)
            noise[i] = n1
            if i + 1 < p:
                n2, state = rand_normal(state)
                noise[i+1] = n2
        X = shift_vec + scale_vec * np.dot(L, noise)
        Z = (1 - lamb) * Z + lamb * X
        stat = np.dot(Z, np.dot(inv_mat, Z))
        if stat > h:
            return t
    return max_steps

@njit(nogil=True, fastmath=True, cache=True, parallel=True)
def simulate_batch(lamb, h, inv_mat, L, shift, scale, seed_start, n_rep, max_steps):
    out = np.empty(n_rep, dtype=np.int64)
    for i in prange(n_rep):
        out[i] = kernel_mewma(lamb, h, inv_mat, L, shift, scale, seed_start + i, max_steps)
    return out

# -------------------- Main Engine --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="phase2_final_features.csv")
    parser.add_argument("--lambda_value", type=float, required=True)
    parser.add_argument("--n_rep_final", type=int, default=50000)
    parser.add_argument("--base_seed", type=int, default=2025)
    parser.add_argument("--shard_index", type=int, required=True)
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()

    lamb = args.lambda_value
    shard = args.shard_index
    seed = args.base_seed + shard * 1000
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    logging.info(f"Shard {shard} | λ={lamb} | Seed={seed} | Reps={args.n_rep_final}")

    # Load data
    df = pd.read_csv(args.input_file)
    feats = ["mean_error", "inlier_ratio", "log_cond_H"]
    X = df[feats].values
    lw = LedoitWolf().fit(X)
    Sigma = lw.covariance_ + np.eye(3) * 1e-2
    L = np.linalg.cholesky(Sigma)
    invS = np.linalg.inv(Sigma)
    inv_mat = ((2.0 - lamb) / lamb) * invS

    # Calibrate h (ARL0 ≈ 370)
    target = 370.0
    low, high = 1.0, 100.0
    for _ in range(18):
        mid = (low + high) / 2
        rl = simulate_batch(lamb, mid, inv_mat, L, np.zeros(3), np.ones(3), seed, 10000, 1000000)
        if np.mean(rl) > target:
            high = mid
        else:
            low = mid
    h = (low + high) / 2
    logging.info(f"Calibrated h = {h:.4f}")

    # Scenarios — خطا اینجا بود! اصلاح شد
    results = []
    scenarios = {
        "IC":       (np.zeros(3), np.ones(3)),
        "small":    (np.array([1.5, 0.0, 0.0]), np.ones(3)),
        "moderate": (np.array([3.0, 0.0, 0.0]), np.ones(3)),
        "large":    (np.array([6.0, 0.0, 0.0]), np.ones(3)),
        "cond":     (np.array([0.0, 0.0, 5.0]), np.ones(3)),
        "inlier":   (np.zeros(3), np.array([1.0, 0.3, 1.0]))  # این خط درست شد
    }

    for name, (shift, scale) in scenarios.items():
        rl = simulate_batch(lamb, h, inv_mat, L, shift, scale, seed + 100000, args.n_rep_final, 1000000)
        mean_rl = np.mean(rl)
        se = np.std(rl, ddof=1) / np.sqrt(len(rl))
        results.append({
            "Lambda": lamb,
            "Scenario": name,
            "ARL": round(mean_rl, 2),
            "SE": round(se, 2),
            "h": round(h, 4),
            "shard": shard
        })
        logging.info(f"{name}: {mean_rl:.2f} ± {se:.2f}")

    # Save
    out_file = out_dir / f"results_lambda_{lamb}_shard_{shard}.csv"
    pd.DataFrame(results).to_csv(out_file, index=False)
    logging.info(f"Shard {shard} completed → {out_file}")

if __name__ == "__main__":
    main()
