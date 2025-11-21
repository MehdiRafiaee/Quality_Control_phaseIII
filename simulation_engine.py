#!/usr/bin/env python3
# simulation_engine_ver06_FINAL_FIXED.py
# 100% WORKING ON GITHUB ACTIONS - NOV 2025
# Tested: ARL0 ≈ 370, Stable Cholesky, No Numba Errors

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import argparse
import logging
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from sklearn.covariance import LedoitWolf
from pathlib import Path
import math  # <=== این خط حیاتی بود!

try:
    from numba import njit
except ImportError:
    sys.exit("CRITICAL: Numba missing.")

# ====================== LOGGING ======================
class LoggerSetup:
    def __init__(self, output_dir):
        self.logger = logging.getLogger("SimEngine")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(ch)
        log_path = output_dir / "execution_log.txt"
        fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(fh)
    def log(self, msg):
        self.logger.info(msg)

# ====================== NUMBA KERNELS ======================
@njit(inline='always')
def xorshift64_star(state: np.uint64) -> np.uint64:
    x = state
    x ^= (x >> np.uint64(12)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x ^= (x << np.uint64(25)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x ^= (x >> np.uint64(27)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return (x * np.uint64(2685821657736338717)) & np.uint64(0xFFFFFFFFFFFFFFFF)

@njit(inline='always')
def rand_normal_pair(state: np.uint64):
    state = xorshift64_star(state)
    u1 = np.uint64(state) / 18446744073709551616.0
    if u1 <= 0.0: u1 = 1e-18
    state = xorshift64_star(state)
    u2 = np.uint64(state) / 18446744073709551616.0
    r = math.sqrt(-2.0 * math.log(u1))
    theta = 2.0 * math.pi * u2
    return r * math.cos(theta), r * math.sin(theta), state

@njit(nogil=True, fastmath=True, cache=True)
def kernel_mewma(lamb, h, inv_mat, mu, L, shift_vec, scale_vec, seed, max_steps):
    state = seed if seed != 0 else np.uint64(12345)
    p = mu.shape[0]
    Z = np.zeros(p, dtype=np.float64)
    noise = np.zeros(p, dtype=np.float64)

    for t in range(1, max_steps + 1):
        for i in range(0, p, 2):
            n1, n2, state = rand_normal_pair(state)
            noise[i] = n1
            if i + 1 < p: noise[i+1] = n2
        Lx = np.dot(L, noise)
        for i in range(p):
            innovation = shift_vec[i] + scale_vec[i] * Lx[i]
            Z[i] = (1.0 - lamb) * Z[i] + lamb * innovation
        t2 = np.dot(Z, np.dot(inv_mat, Z))
        if t2 > h:
            return t
    return max_steps

@njit(nogil=True, fastmath=True, cache=True)
def simulate_batch(lamb, h, inv_mat, mu, L, shift, scale, start_seed, n_rep, max_steps):
    out = np.empty(n_rep, dtype=np.int64)
    for i in range(n_rep):
        s = np.uint64(start_seed + i)
        out[i] = kernel_mewma(lamb, h, inv_mat, mu, L, shift, scale, s, max_steps)
    return out

# ====================== ENGINE ======================
class Engine:
    def __init__(self, args):
        self.args = args
        self.out_dir = Path(args.out)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_sys = LoggerSetup(self.out_dir)
        self.log = self.log_sys.log
        self.log(f"Engine Started - Shard {args.shard_id} | Reps: {args.n_rep}")

    def load_data(self):
        feats = ["mean_error", "inlier_ratio", "log_cond_H"]
        csv_path = self.args.csv_path
        
        if not os.path.exists(csv_path):
            self.log("CSV not found → Using synthetic")
            Sigma = np.eye(3) + 0.2
            return [{"name": "Synthetic", "Sigma": Sigma, "p": 3}]

        df = pd.read_csv(csv_path)
        X = df[feats].values.astype(np.float64)
        lw = LedoitWolf().fit(X)
        Sigma = lw.covariance_ + np.eye(3) * 1e-2
        cond = np.linalg.cond(Sigma)
        self.log(f"Covariance loaded - condition number: {cond:.2f}")

        return [{"name": "Uploaded_Data", "Sigma": Sigma, "p": 3}]

    def get_scenario(self, name, Sigma):
        p = 3
        invS = np.linalg.inv(Sigma)
        vals, vecs = np.linalg.eigh(Sigma)
        v_worst = vecs[:, -1]

        shift = np.zeros(p)
        scale = np.ones(p)

        def nc(delta):
            return math.sqrt(delta / float(v_worst @ invS @ v_worst)) * v_worst

        if name == "IC": pass
        elif name == "small":     shift = nc(1.0)
        elif name == "moderate":  shift = nc(2.5)
        elif name == "large":     shift = nc(6.0)
        elif name == "cond":      shift[2] += 4.0
        elif name == "inlier":    scale[1] = 0.4

        return shift, scale

    def run_parallel(self, lamb, h, inv_mat, mu, L, scenario, Sigma, n_rep, offset):
        shift, scale = self.get_scenario(scenario, Sigma)
        batch_size = 5000
        n_batches = (n_rep + batch_size - 1) // batch_size
        seeds = [np.uint64(self.args.seed + offset + i * 100000) for i in range(n_batches)]
        sizes = [min(batch_size, n_rep - i*batch_size) for i in range(n_batches)]

        with parallel_backend('threading', n_jobs=2):
            results = Parallel()(
                delayed(simulate_batch)(lamb, h, inv_mat, mu, L, shift, scale, seeds[i], sizes[i], 1000000)
                for i in range(n_batches)
            )
        return np.mean(np.concatenate(results))

    def calibrate_h(self, lamb, inv_mat, mu, L, Sigma):
        target = 370.0
        low, high = 1.0, 100.0
        for _ in range(18):
            mid = (low + high) / 2
            arl = self.run_parallel(lamb, mid, inv_mat, mu, L, "IC", Sigma, 20000, 0)
            self.log(f"  Calibration: h={mid:.4f} → ARL0≈{arl:.1f}")
            if arl > target: high = mid
            else: low = mid
            if high - low < 0.05: break
        return (low + high) / 2

    def execute(self):
        datasets = self.load_data()
        results = []

        for ds in datasets:
            self.log(f"Processing {ds['name']} (p={ds['p']})")
            mu = np.zeros(3)
            L = np.linalg.cholesky(ds['Sigma'] + np.eye(3)*1e-2)
            inv_mat_base = np.linalg.inv(ds['Sigma'])

            for lamb in [0.05, 0.10, 0.20]:
                inv_mat = ((2.0 - lamb) / lamb) * inv_mat_base
                h = self.calibrate_h(lamb, inv_mat, mu, L, ds['Sigma'])
                self.log(f"λ={lamb} → Final h={h:.4f}")

                row = {"Dataset": ds['name'], "Lambda": lamb, "h": round(h,4), "Shard": self.args.shard_id}
                for sc in ["IC", "small", "moderate", "large", "cond", "inlier"]:
                    arl = self.run_parallel(lamb, h, inv_mat, mu, L, sc, ds['Sigma'], self.args.n_rep, 100000)
                    row[sc] = round(arl, 2)
                    self.log(f"  {sc}: {arl:.2f}")
                results.append(row)

        out_file = self.out_dir / f"results_shard_{self.args.shard_id}.csv"
        pd.DataFrame(results).to_csv(out_file, index=False)
        self.log(f"SHARD {self.args.shard_id} SUCCESSFULLY COMPLETED!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="phase2_final_features.csv")
    parser.add_argument("--out", type=str, default="results")
    parser.add_argument("--shard_id", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_rep", type=int, default=25000)
    args = parser.parse_args()
    Engine(args).execute()
