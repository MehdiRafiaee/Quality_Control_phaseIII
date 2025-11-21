import os
import numpy as np
import pandas as pd
import math
from pathlib import Path
from joblib import Parallel, delayed, parallel_backend
from sklearn.covariance import LedoitWolf
from numba import njit, prange
import sys
import argparse  # اضافه کردن این خط برای رفع خطای argparse

# تنظیمات برای بهینه‌سازی (Threading) در Numba و Joblib
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ------------------- 1. RNG بهینه‌شده با Xorshift64 -------------------
@njit(inline='always')
def xorshift64_star(state: np.uint64) -> np.uint64:
    x = state
    x ^= (x >> np.uint64(12)) 
    x ^= (x << np.uint64(25))
    x ^= (x >> np.uint64(27))
    return (x * np.uint64(2685821657736338717))

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

# ------------------- 2. محاسبات MEWMA (استفاده از Numba) -------------------
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
    for i in prange(n_rep):
        s = np.uint64(start_seed + i)
        out[i] = kernel_mewma(lamb, h, inv_mat, mu, L, shift, scale, s, max_steps)
    return out

# ------------------- 3. کالیبراسیون و پیدا کردن h -------------------
class Engine:
    def __init__(self, args):
        self.args = args
        self.out_dir = Path(args.out)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log = print
        self.n_seeds = args.n_seeds
        self.n_rep_cal = args.n_rep_cal
        self.n_rep_final = args.n_rep_final

    def load_data(self):
        csv_path = Path(self.args.csv_path)
        if not csv_path.exists():
            self.log("CSV not found → Using synthetic high-cond data (p=5, cond≈200)")
            Sigma = np.diag([1.0, 1.0, 1.0, 1.0, 215.0])
            p = 5
        else:
            df = pd.read_csv(csv_path)
            X = df.select_dtypes(include=np.number).values.astype(np.float64)
            lw = LedoitWolf()
            lw.fit(X)
            Sigma = lw.covariance_
            Sigma += np.eye(X.shape[1]) * 1e-6
            p = X.shape[1]
        cond = np.linalg.cond(Sigma)
        self.log(f"Loaded data – p = {p}, condition number = {cond:.2e}")
        return [{"name": "RealData" if csv_path.exists() else "Synthetic", "Sigma": Sigma, "p": p}]

    def get_scenario(self, name: str, Sigma: np.ndarray):
        p = Sigma.shape[0]
        invS = np.linalg.inv(Sigma)
        vals, vecs = np.linalg.eigh(Sigma) 
        v_easy = vecs[:, -1]  
        v_hard = vecs[:, 0]    

        shift = np.zeros(p, dtype=np.float64)
        scale = np.ones(p, dtype=np.float64)

        if name == "ARL0":
            pass  
        elif name == "Small":
            shift = 1.0 * v_easy  
            shift /= np.sqrt(shift @ invS @ shift) or 1.0  
        elif name == "Moderate":
            shift = 2.0 * v_easy
            shift /= np.sqrt(shift @ invS @ shift) or 1.0
        elif name == "Large":
            shift = 3.5 * v_easy
            shift /= np.sqrt(shift @ invS @ shift) or 1.0
        elif name == "Cond+4":
            shift = 4.0 * v_hard  
        elif name == "Inlier/Err":
            scale[np.argmin(vals)] = 0.3  
        return shift, scale

    def calibrate_h(self, lamb, inv_mat, mu, L, Sigma):
        target = 370.0
        low, high = 5.0, 50.0
        best_h = (low + high) / 2
        for iter in range(25):  
            mid = (low + high) / 2
            arl, _ = self.run_parallel(lamb, mid, inv_mat, np.zeros_like(mu), np.ones_like(mu), self.n_rep_cal, seed_offset=1000000)
            self.log(f"  [Calib iter {iter+1:02d}] h = {mid:.6f} → ARL0 ≈ {arl:.2f}")
            if arl > target:
                low = mid
            else:
                high = mid
        best_h = (low + high) / 2
        self.log(f"  Calibration complete → final h = {best_h:.8f}")
        return best_h

    def run_parallel(self, lamb, h, inv_mat, shift_vec, scale_vec, n_rep, seed_offset):
        batch_size = 10000
        n_batches = (n_rep + batch_size - 1) // batch_size
        batch_sizes = [min(batch_size, n_rep - i * batch_size) for i in range(n_batches)]
        seeds = [np.uint64(self.args.base_seed + seed_offset + i * 1000000) for i in range(n_batches)]

        all_rls = []
        for i in range(n_batches):
            batch_rl = simulate_batch(lamb, h, inv_mat, shift_vec, scale_vec, seeds[i], batch_sizes[i], self.args.max_rl)
            all_rls.append(batch_rl)

        all_rls = np.concatenate(all_rls)
        mean = np.mean(all_rls)
        se = np.std(all_rls, ddof=1) / np.sqrt(len(all_rls))
        return mean, se

    def execute(self):
        datasets = self.load_data()
        all_results = []

        for ds in datasets:
            p = ds["p"]
            Sigma = ds["Sigma"]
            mu = np.zeros(p)
            L = np.linalg.cholesky(Sigma)  
            invS = np.linalg.inv(Sigma)

            h_per_seed = []
            for seed_idx in range(self.n_seeds):
                h = self.calibrate_h(0.1, ((2.0 - 0.1)/0.1) * invS, mu, L, Sigma)
                h_per_seed.append(h)
            h_final = float(np.median(h_per_seed))
            self.log(f"Median h over {self.n_seeds} seeds = {h_final:.8f}")

            for seed_idx in range(self.n_seeds):
                row = {"Dataset": ds["name"], "Lambda": 0.1, "h": round(h_final, 8), "Seed": seed_idx}
                for sc in ["ARL0", "Small", "Moderate", "Large", "Cond+4", "Inlier/Err"]:
                    mean, se = self.run_parallel(0.1, h_final, ((2.0 - 0.1)/0.1) * invS, *self.get_scenario(sc, Sigma), self.n_rep_final, seed_offset=seed_idx * 100_000_000)
                    row[sc] = f"{mean:.2f} ± {se:.2f}"
                    self.log(f"  {sc:12s} | Seed {seed_idx:02d} → {mean:.2f} ± {se:.2f}")
                all_results.append(row)

        summary_df = pd.DataFrame(all_results)
        summary_path = self.out_dir / "FINAL_SUMMARY_WITH_SE.csv"
        summary_df.to_csv(summary_path, index=False)

        self.log("\n=== ALL DONE – RESULTS READY ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="phase2_final_features.csv")
    parser.add_argument("--out", type=str, default="results_combined")
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--n_rep_cal", type=int, default=60000)
    parser.add_argument("--n_rep_final", type=int, default=250000)
    parser.add_argument("--base_seed", type=int, default=42514251)
    parser.add_argument("--max_rl", type=int, default=5_000_000)
    args = parser.parse_args()
    Engine(args).execute()
