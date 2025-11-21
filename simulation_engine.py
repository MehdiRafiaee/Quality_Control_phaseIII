#!/usr/bin/env python3
"""
improved_simulation_engine.py
نسخهٔ بهبود یافته با اصلاحات منطقی و اجرایی (numba parallel, stable calibration, safe normalization)
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from sklearn.covariance import LedoitWolf

try:
    from numba import njit, prange
except Exception as e:
    sys.exit("CRITICAL: numba is required. Install numba and retry. Error: " + str(e))


# -------------------- RNG: xorshift64* (numba) --------------------
@njit(inline='always')
def xorshift64_star(state: np.uint64) -> np.uint64:
    x = state
    x ^= (x >> np.uint64(12))
    x ^= (x << np.uint64(25))
    x ^= (x >> np.uint64(27))
    return x * np.uint64(2685821657736338717)


@njit(inline='always')
def rand_normal_pair(state: np.uint64):
    state = xorshift64_star(state)
    u1 = np.uint64(state) / 18446744073709551616.0
    # safety for log(0)
    if u1 <= 0.0:
        u1 = 1e-18
    state = xorshift64_star(state)
    u2 = np.uint64(state) / 18446744073709551616.0
    r = math.sqrt(-2.0 * math.log(u1))
    theta = 2.0 * math.pi * u2
    return r * math.cos(theta), r * math.sin(theta), state


# -------------------- MEWMA kernel (numba) --------------------
@njit(nogil=True, fastmath=True, cache=True)
def kernel_mewma(lamb, h, inv_mat, mu, L, shift_vec, scale_vec, seed, max_steps):
    state = seed if seed != 0 else np.uint64(12345)
    p = mu.shape[0]
    Z = np.zeros(p, dtype=np.float64)
    noise = np.zeros(p, dtype=np.float64)

    for t in range(1, max_steps + 1):
        # generate normals in pairs (Box-Muller via xorshift)
        for i in range(0, p, 2):
            n1, n2, state = rand_normal_pair(state)
            noise[i] = n1
            if i + 1 < p:
                noise[i + 1] = n2

        # project with Cholesky
        Lx = np.dot(L, noise)

        # update MEWMA
        for i in range(p):
            innovation = shift_vec[i] + scale_vec[i] * Lx[i]
            Z[i] = (1.0 - lamb) * Z[i] + lamb * innovation

        # Hotelling-like statistic
        t2 = np.dot(Z, np.dot(inv_mat, Z))
        if t2 > h:
            return t
    return max_steps


@njit(nogil=True, fastmath=True, cache=True, parallel=True)
def simulate_batch(lamb, h, inv_mat, mu, L, shift, scale, start_seed, n_rep, max_steps):
    out = np.empty(n_rep, dtype=np.int64)
    for i in prange(n_rep):
        s = np.uint64(start_seed + i)
        out[i] = kernel_mewma(lamb, h, inv_mat, mu, L, shift, scale, s, max_steps)
    return out


# -------------------- Engine --------------------
class Engine:
    def __init__(self, args):
        self.args = args
        self.out_dir = Path(args.out)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log = lambda *a, **k: print(*a, **k)
        self.n_seeds = args.n_seeds
        self.n_rep_cal = args.n_rep_cal
        self.n_rep_final = args.n_rep_final
        self.base_seed = int(args.base_seed)
        self.max_rl = int(args.max_rl)

    def load_data(self):
        csv_path = Path(self.args.csv_path)
        if not csv_path.exists():
            self.log("CSV not found → Using synthetic high-cond data (p=5, cond≈200)")
            Sigma = np.diag([1.0, 1.0, 1.0, 1.0, 215.0])
            p = 5
            name = "Synthetic"
        else:
            df = pd.read_csv(csv_path)
            X = df.select_dtypes(include=np.number).values.astype(np.float64)
            if X.shape[1] < 2:
                raise ValueError("Input CSV must contain at least 2 numeric columns.")
            lw = LedoitWolf()
            lw.fit(X)
            Sigma = lw.covariance_
            # tiny regularization
            Sigma += np.eye(X.shape[1]) * 1e-8
            p = X.shape[1]
            name = "RealData"
        cond = np.linalg.cond(Sigma)
        self.log(f"[DATA] Loaded data – name={name}, p = {p}, condition number = {cond:.2e}")
        return [{"name": name, "Sigma": Sigma, "p": p}]

    def get_scenario(self, name: str, Sigma: np.ndarray):
        p = Sigma.shape[0]
        invS = np.linalg.inv(Sigma)
        vals, vecs = np.linalg.eigh(Sigma)
        # choose principal / worst eigenvectors robustly
        v_easy = vecs[:, -1].astype(np.float64)   # direction with largest variance
        v_hard = vecs[:, 0].astype(np.float64)    # smallest eigenvalue direction

        shift = np.zeros(p, dtype=np.float64)
        scale = np.ones(p, dtype=np.float64)

        if name == "ARL0":
            pass
        elif name == "Small":
            shift = 1.0 * v_easy
            denom = math.sqrt(float(shift @ (invS @ shift)))
            if denom == 0.0:
                denom = 1.0
            shift = shift / denom
        elif name == "Moderate":
            shift = 2.0 * v_easy
            denom = math.sqrt(float(shift @ (invS @ shift)))
            if denom == 0.0:
                denom = 1.0
            shift = shift / denom
        elif name == "Large":
            shift = 3.5 * v_easy
            denom = math.sqrt(float(shift @ (invS @ shift)))
            if denom == 0.0:
                denom = 1.0
            shift = shift / denom
        elif name == "Cond+4":
            # push along smallest-eigenvalue direction (hardest)
            shift = 4.0 * v_hard
            denom = math.sqrt(float(shift @ (invS @ shift)))
            if denom == 0.0:
                denom = 1.0
            shift = shift / denom
        elif name == "Inlier/Err":
            # reduce scale for the smallest-variance direction to model inlier attack
            idx = int(np.argmin(vals))
            scale[idx] = 0.3
        else:
            # unknown scenario: treat as ARL0
            pass

        return shift, scale

    def _get_seed_for_batch(self, batch_index, seed_offset):
        # deterministic seed generator for batches
        return np.uint64(self.base_seed + seed_offset + int(batch_index) * np.uint64(1000000))

    def run_parallel(self, lamb, h, inv_mat, shift_vec, scale_vec, n_rep, seed_offset):
        batch_size = int(self.args.batch_size)
        n_batches = (n_rep + batch_size - 1) // batch_size
        batch_sizes = [min(batch_size, n_rep - i * batch_size) for i in range(n_batches)]
        seeds = [self._get_seed_for_batch(i, seed_offset) for i in range(n_batches)]

        all_rls = []
        # call numba-compiled simulate_batch for each batch (numba will parallelize inside)
        for i in range(n_batches):
            batch_rl = simulate_batch(lamb, h, inv_mat, np.zeros(inv_mat.shape[0]), np.eye(inv_mat.shape[0]),
                                      shift_vec, scale_vec, seeds[i], batch_sizes[i], self.max_rl)
            all_rls.append(batch_rl)

        all_rls = np.concatenate(all_rls)
        mean = float(np.mean(all_rls))
        se = float(np.std(all_rls, ddof=1) / math.sqrt(len(all_rls))) if len(all_rls) > 1 else 0.0
        return mean, se

    def calibrate_h(self, lamb, inv_mat, mu, L, Sigma, seed_offset_base=0):
        target = float(self.args.target_arl0)
        # sensible bracket
        low, high = 1.0, 200.0
        best_h = (low + high) / 2.0
        for iter in range(int(self.args.calib_iters)):
            mid = (low + high) / 2.0
            # use deterministic seed offset per iteration (keeps reproducibility)
            arl, _ = self.run_parallel(lamb, mid, inv_mat, np.zeros_like(mu), np.ones_like(mu),
                                       int(self.n_rep_cal), seed_offset=seed_offset_base + iter * 12345)
            self.log(f"  [Calib iter {iter+1:02d}] h = {mid:.6f} → ARL0 ≈ {arl:.2f}")
            # ARL increases with h; so if arl > target -> h is too large -> decrease high
            if arl > target:
                high = mid
            else:
                low = mid
            if (high - low) < 1e-3:
                break
        best_h = (low + high) / 2.0
        self.log(f"  Calibration complete → final h = {best_h:.8f}")
        return best_h

    def execute(self):
        datasets = self.load_data()
        all_results = []

        for ds in datasets:
            p = ds["p"]
            Sigma = ds["Sigma"].astype(np.float64)
            mu = np.zeros(p, dtype=np.float64)
            # stable cholesky with tiny jitter
            jitter = 1e-8
            L = np.linalg.cholesky(Sigma + np.eye(p) * jitter)
            invS = np.linalg.inv(Sigma)

            # calibrate per seed and take median
            h_per_seed = []
            for seed_idx in range(self.n_seeds):
                seed_offset_base = seed_idx * 1_000_000
                inv_mat_scaled = ((2.0 - 0.1) / 0.1) * invS
                h = self.calibrate_h(0.1, inv_mat_scaled, mu, L, Sigma, seed_offset_base)
                h_per_seed.append(h)
            h_final = float(np.median(np.array(h_per_seed)))
            self.log(f"[CALIB] Median h over {self.n_seeds} seeds = {h_final:.8f}")

            # Evaluate scenarios
            for seed_idx in range(self.n_seeds):
                row = {"Dataset": ds["name"], "Lambda": 0.1, "h": round(h_final, 8), "Seed": seed_idx}
                for sc in ["ARL0", "Small", "Moderate", "Large", "Cond+4", "Inlier/Err"]:
                    shift, scale = self.get_scenario(sc, Sigma)
                    inv_mat_scaled = ((2.0 - 0.1) / 0.1) * invS
                    mean, se = self.run_parallel(0.1, h_final, inv_mat_scaled, shift, scale,
                                                int(self.n_rep_final), seed_offset=seed_idx * 100_000_000)
                    row[sc] = f"{mean:.2f} ± {se:.2f}"
                    self.log(f"  {sc:12s} | Seed {seed_idx:02d} → {mean:.2f} ± {se:.2f}")
                all_results.append(row)

        summary_df = pd.DataFrame(all_results)
        summary_path = self.out_dir / "FINAL_SUMMARY_WITH_SE.csv"
        summary_df.to_csv(summary_path, index=False)
        self.log(f"\n=== ALL DONE – RESULTS saved to {summary_path} ===")


# -------------------- CLI --------------------
def parse_args():
    parser = argparse.ArgumentParser(prog="improved_simulation_engine.py")
    parser.add_argument("--csv_path", type=str, default="phase2_final_features.csv")
    parser.add_argument("--out", type=str, default="results_combined")
    parser.add_argument("--n_seeds", type=int, default=3, help="Number of independent seeds to average over")
    parser.add_argument("--n_rep_cal", type=int, default=20000, help="Reps used for calibration of h (per seed)")
    parser.add_argument("--n_rep_final", type=int, default=50000, help="Reps used for final evaluation (per seed)")
    parser.add_argument("--base_seed", type=int, default=42514251)
    parser.add_argument("--max_rl", type=int, default=5_000_000)
    parser.add_argument("--batch_size", type=int, default=10000, help="Internal batching of simulations")
    parser.add_argument("--calib_iters", type=int, default=25, help="Binary search iterations for h calibration")
    parser.add_argument("--target_arl0", type=float, default=370.0, help="Target in-control ARL0")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Engine(args).execute()
