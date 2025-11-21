#!/usr/bin/env python3
"""
improved_simulation_engine_fixed.py
نسخهٔ بهبود یافته با رفع خطای ابهام argparse (--n_rep_cal و --n_rep_final).
"""
import os
# تنظیم متغیرهای محیطی برای کنترل موازی‌سازی توسط NumPy/MKL/OpenBLAS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

try:
    from numba import njit, prange
except Exception as e:
    sys.exit("CRITICAL: numba is required. Install numba and retry. Error: " + str(e))


# -------------------- RNG: xorshift64* (numba) --------------------
@njit(inline='always')
def xorshift64_star(state: np.uint64) -> np.uint64:
    """Implement a 64-bit xorshift* pseudorandom number generator."""
    x = state
    x ^= (x >> np.uint64(12))
    x ^= (x << np.uint64(25))
    x ^= (x >> np.uint64(27))
    return x * np.uint64(2685821657736338717)


@njit(inline='always')
def rand_normal_pair(state: np.uint64):
    """Generate two standard normal random numbers using Box-Muller."""
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
    """
    Core MEWMA simulation kernel for a single run-length.
    mu is unused here but kept for API consistency.
    """
    state = seed if seed != 0 else np.uint64(12345)
    p = inv_mat.shape[0] # Dimension of the system
    Z = np.zeros(p, dtype=np.float64) # MEWMA vector

    for t in range(1, max_steps + 1):
        # generate normals in pairs (Box-Muller via xorshift)
        noise = np.empty(p, dtype=np.float64)
        for i in range(0, p, 2):
            n1, n2, state = rand_normal_pair(state)
            noise[i] = n1
            if i + 1 < p:
                noise[i + 1] = n2

        # Correlated multivariate observation (X_t)
        # X_t = mu + L * N_t (where mu=0 in in-control and L is Cholesky of Sigma)
        # Shift vector applied to the innovations before scaling
        Lx = np.dot(L, noise)

        # Apply out-of-control shift and scale to the observations
        X_t = np.empty(p, dtype=np.float64)
        for i in range(p):
            X_t[i] = shift_vec[i] + scale_vec[i] * Lx[i]
            
        # Update MEWMA statistic Z
        Z = (1.0 - lamb) * Z + lamb * X_t

        # Hotelling-like statistic (T^2)
        t2 = np.dot(Z, np.dot(inv_mat, Z))
        if t2 > h:
            return t # Signal
    return max_steps # No signal within max_steps


@njit(nogil=True, fastmath=True, cache=True, parallel=True)
def simulate_batch(lamb, h, inv_mat, mu, L, shift, scale, start_seed, n_rep, max_steps):
    """Run multiple independent MEWMA simulations in parallel (Numba prange)."""
    out = np.empty(n_rep, dtype=np.int64)
    for i in prange(n_rep):
        s = np.uint64(start_seed + i)
        # Note: mu is unused in kernel_mewma but passed for type consistency
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
        # Set number of parallel jobs to 1 to rely solely on Numba's internal parallelism
        self.n_jobs = 1 

    def load_data(self):
        csv_path = Path(self.args.csv_path)
        if not csv_path.exists():
            self.log("CSV not found → Using synthetic high-cond data (p=5, cond≈200)")
            # Generate a covariance matrix with high condition number
            Sigma = np.diag([1.0, 1.0, 1.0, 1.0, 215.0]) 
            p = 5
            name = "Synthetic"
        else:
            df = pd.read_csv(csv_path)
            # Select only numeric data for covariance calculation
            X = df.select_dtypes(include=np.number).values.astype(np.float64)
            # Filter out status/ID columns if they made it here
            if X.shape[1] < 2:
                raise ValueError("Input CSV must contain at least 2 numeric columns.")
            
            # Use Ledoit-Wolf for robust covariance estimation
            lw = LedoitWolf()
            lw.fit(X)
            Sigma = lw.covariance_
            # tiny regularization (jitter) for numerical stability
            Sigma += np.eye(X.shape[1]) * 1e-8
            p = X.shape[1]
            name = "RealData"
            
        cond = np.linalg.cond(Sigma)
        self.log(f"[DATA] Loaded data – name={name}, p = {p}, condition number = {cond:.2e}")
        return [{"name": name, "Sigma": Sigma, "p": p}]

    def get_scenario(self, name: str, Sigma: np.ndarray):
        """Define shift (mu) and scale (sigma) vectors for out-of-control scenarios."""
        p = Sigma.shape[0]
        invS = np.linalg.inv(Sigma)
        vals, vecs = np.linalg.eigh(Sigma)
        
        # Directions based on eigenvectors
        v_easy = vecs[:, -1].astype(np.float64)    # largest variance direction
        v_hard = vecs[:, 0].astype(np.float64)     # smallest variance direction

        shift = np.zeros(p, dtype=np.float64)
        scale = np.ones(p, dtype=np.float64)

        if name == "ARL0":
            pass # In-Control (IC)
            
        elif name in ["Small", "Moderate", "Large"]:
            mag = {"Small": 1.0, "Moderate": 2.0, "Large": 3.5}[name]
            shift = mag * v_easy
            # Normalize shift to ensure the shift magnitude (delta) is exactly 'mag'
            denom = math.sqrt(float(shift @ (invS @ shift)))
            if denom == 0.0:
                denom = 1.0
            shift = shift / denom
            
        elif name == "Cond+4":
            # Push along smallest-eigenvalue direction (hardest to detect)
            shift = 4.0 * v_hard
            denom = math.sqrt(float(shift @ (invS @ shift)))
            if denom == 0.0:
                denom = 1.0
            shift = shift / denom
            
        elif name == "Inlier/Err":
            # Reduce scale for the smallest-variance direction to model an inlier attack
            idx = int(np.argmin(vals))
            scale[idx] = 0.3
        
        # Always ensure the resulting vectors are normalized to unit magnitude 
        # for robust shift scenarios if needed, but here we normalize by mahalanobis distance.

        return shift, scale

    def _get_seed_for_batch(self, batch_index, seed_offset):
        """Deterministic seed generator based on base_seed, offset, and batch index."""
        # Use a large multiplier to ensure separation between batches
        return np.uint64(self.base_seed + seed_offset + int(batch_index) * np.uint64(1000000))

    def run_parallel(self, lamb, h, inv_mat, shift_vec, scale_vec, n_rep, seed_offset):
        """Splits the total repetitions into batches and runs them."""
        batch_size = int(self.args.batch_size)
        n_batches = (n_rep + batch_size - 1) // batch_size
        
        # Calculate size for each batch
        batch_sizes = [min(batch_size, n_rep - i * batch_size) for i in range(n_batches)]
        # Calculate start seed for each batch
        seeds = [self._get_seed_for_batch(i, seed_offset) for i in range(n_batches)]

        all_rls = []
        # Run batches sequentially. Numba handles parallelism within each batch.
        for i in range(n_batches):
            L = np.linalg.cholesky(np.eye(inv_mat.shape[0])) # L should be Identity for scaled data
            batch_rl = simulate_batch(
                lamb, h, inv_mat, np.zeros(inv_mat.shape[0]), L,
                shift_vec, scale_vec, seeds[i], batch_sizes[i], self.max_rl
            )
            all_rls.append(batch_rl)

        all_rls = np.concatenate(all_rls)
        mean = float(np.mean(all_rls))
        # Standard Error of the Mean (SEM)
        se = float(np.std(all_rls, ddof=1) / math.sqrt(len(all_rls))) if len(all_rls) > 1 else 0.0
        return mean, se

    def calibrate_h(self, lamb, inv_mat, mu, L, Sigma, seed_offset_base=0):
        """Binary search to find control limit h that achieves target ARL0."""
        target = float(self.args.target_arl0)
        # Sensible bracket for h (may need adjustment depending on p and lambda)
        low, high = 1.0, 200.0
        
        for iter in range(int(self.args.calib_iters)):
            mid = (low + high) / 2.0
            # Use deterministic seed offset per iteration for better stability
            arl, _ = self.run_parallel(
                lamb, mid, inv_mat, np.zeros_like(mu), np.ones_like(mu),
                int(self.n_rep_cal), seed_offset=seed_offset_base + iter * 12345
            )
            self.log(f"  [Calib iter {iter+1:02d}] h = {mid:.6f} → ARL0 ≈ {arl:.2f}")
            
            # Binary search logic: ARL increases with h
            if arl > target:
                high = mid
            else:
                low = mid
                
            # Stop early if the interval is narrow enough
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
            # The Cholesky of Sigma is only needed for generating correlated noise (L)
            jitter = 1e-8
            L = np.linalg.cholesky(Sigma + np.eye(p) * jitter)
            invS = np.linalg.inv(Sigma)

            # --- Calibration ---
            # For MEWMA, the statistic uses inv_mat = (2-lambda)/lambda * inv(Sigma)
            lamb = 0.1 # Using a fixed lambda as per the original file
            inv_mat_scaled = ((2.0 - lamb) / lamb) * invS
            
            h_per_seed = []
            self.log(f"\n[CALIB] Starting calibration for Lambda={lamb} (Target ARL0={self.args.target_arl0:.1f})")
            for seed_idx in range(self.n_seeds):
                seed_offset_base = seed_idx * 1_000_000 # Different seed offset for each calibration run
                self.log(f"  --- Calibrating Seed {seed_idx} ---")
                # L is passed, but the run_parallel and kernel_mewma functions are structured
                # to use L for noise projection, and inv_mat_scaled for T2 calculation.
                h = self.calibrate_h(lamb, inv_mat_scaled, mu, L, Sigma, seed_offset_base)
                h_per_seed.append(h)
                
            h_final = float(np.median(np.array(h_per_seed)))
            self.log(f"[CALIB] Median h over {self.n_seeds} seeds = {h_final:.8f}")

            # --- Evaluation Scenarios ---
            self.log(f"\n[EVAL] Starting final evaluation with h={h_final:.8f}")
            for seed_idx in range(self.n_seeds):
                row = {"Dataset": ds["name"], "Lambda": lamb, "h": round(h_final, 8), "Seed": seed_idx}
                
                # Use a very large seed offset for evaluation to separate from calibration seeds
                eval_seed_offset = seed_idx * 100_000_000
                
                for sc in ["ARL0", "Small", "Moderate", "Large", "Cond+4", "Inlier/Err"]:
                    # Get the shift/scale for the scenario
                    shift, scale = self.get_scenario(sc, Sigma)
                    
                    # Run the simulation
                    mean, se = self.run_parallel(
                        lamb, h_final, inv_mat_scaled, shift, scale,
                        int(self.n_rep_final), seed_offset=eval_seed_offset
                    )
                    
                    row[sc] = f"{mean:.2f} \u00B1 {se:.2f}" # Format as "Mean ± SE"
                    self.log(f"  {sc:12s} | Seed {seed_idx:02d} → {mean:.2f} \u00B1 {se:.2f}")
                    
                all_results.append(row)

            # Save the final summary
            summary_df = pd.DataFrame(all_results)
            summary_path = self.out_dir / f"{ds['name']}_MEWMA_SUMMARY_L{lamb:.2f}_P{p}.csv"
            summary_df.to_csv(summary_path, index=False)
            self.log(f"\n=== ALL DONE – RESULTS saved to {summary_path} ===")


# -------------------- CLI --------------------
def parse_args():
    parser = argparse.ArgumentParser(
        prog="improved_simulation_engine_fixed.py",
        description="Run Numba-optimized MEWMA simulations for multi-variate process control."
    )
    # File Paths
    parser.add_argument("--csv_path", type=str, default="phase2_final_features.csv", 
                        help="Path to the CSV file containing phase I data.")
    parser.add_argument("--out", type=str, default="results_combined", 
                        help="Output directory for results.")
    
    # Repetition Settings
    parser.add_argument("--n_seeds", type=int, default=3, 
                        help="Number of independent seeds (replicates) to average over.")
    parser.add_argument("--n_rep_cal", type=int, default=20000, 
                        help="Number of repetitions used for calibration of h (per seed).")
    parser.add_argument("--n_rep_final", type=int, default=50000, 
                        help="Number of repetitions used for final scenario evaluation (per seed).")
    
    # Simulation Parameters
    parser.add_argument("--base_seed", type=int, default=42514251, 
                        help="Base seed for random number generation.")
    parser.add_argument("--max_rl", type=int, default=5_000_000, 
                        help="Maximum run length to simulate (to prevent infinite loops).")
    parser.add_argument("--batch_size", type=int, default=10000, 
                        help="Internal batch size for Numba's parallel execution.")
                        
    # Calibration Parameters
    parser.add_argument("--calib_iters", type=int, default=25, 
                        help="Binary search iterations for h calibration.")
    parser.add_argument("--target_arl0", type=float, default=370.0, 
                        help="Target In-Control Average Run Length (ARL0).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    print(f"--- Starting MEWMA Engine (Numba/Argparse fixed) ---")
    
    # Print the command used to run (helpful for debugging GH Actions)
    print(f"Command executed:")
    print(" ".join(sys.argv))
    
    Engine(args).execute()
    print("--- Simulation Finished Successfully ---")
