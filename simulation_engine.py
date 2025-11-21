#!/usr/bin/env python3
# simulation_engine.py - FINAL PRODUCTION VERSION (NOV 2025)
# Highly optimized for ARL simulation using Numba, Joblib, and custom RNG.
# Fixed Numba math issues and implemented robust binary search for control limit 'h'.

import os
# CRITICAL: Restrict numpy/scipy threading to 1 to avoid conflicts with joblib/numba parallelism
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
import math # <=== حیاتی برای پایداری numba.math.log و numba.math.sqrt

# Numba Check (Must be present for high-speed simulation)
try:
    from numba import njit, prange
except ImportError:
    sys.exit("CRITICAL: Numba is not installed. Simulation cannot run.")

# ====================== 1. LOGGING SETUP ======================
class LoggerSetup:
    """Setup logger for both console and file output (for GitHub Artifacts)."""
    def __init__(self, output_dir: Path):
        self.logger = logging.getLogger("SimEngine")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        
        # Console Handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(ch)
        
        # File Handler (for log.txt artifact)
        log_path = output_dir / "execution_log.txt"
        fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(fh)
        
    def log(self, msg: str):
        """Standard logging method."""
        self.logger.info(msg)

# ====================== 2. NUMBA KERNELS (High Speed Core) ======================
# Custom Xorshift64* RNG - Fast, non-cryptographic, and Numba-compatible
@njit(inline='always')
def xorshift64_star(state: np.uint64) -> np.uint64:
    """Updates the RNG state using Xorshift64* algorithm."""
    x = state
    x ^= (x >> np.uint64(12)) 
    x ^= (x << np.uint64(25))
    x ^= (x >> np.uint64(27))
    return (x * np.uint64(2685821657736338717))

@njit(inline='always')
def rand_normal_pair(state: np.uint64):
    """Generates two N(0, 1) random numbers using Box-Muller transform."""
    state = xorshift64_star(state)
    # Generate uniform U(0, 1)
    # Scale from max uint64 (2^64)
    u1 = np.uint64(state) / 18446744073709551616.0 
    if u1 <= 0.0: u1 = 1e-18 # Ensure log(u1) is defined

    state = xorshift64_star(state)
    u2 = np.uint64(state) / 18446744073709551616.0
    
    # Box-Muller transformation
    r = math.sqrt(-2.0 * math.log(u1))
    theta = 2.0 * math.pi * u2
    
    # Return two standard normals and the new state
    return r * math.cos(theta), r * math.sin(theta), state

@njit(nogil=True, fastmath=True, cache=True)
def kernel_mewma(lamb, h, inv_mat, mu, L, shift_vec, scale_vec, seed, max_steps):
    """
    Runs a single MEWMA simulation to find the Run Length (RL).
    
    Returns:
        The step number 't' where the chart signaled (T^2 > h), or max_steps.
    """
    state = seed if seed != 0 else np.uint64(12345)
    p = mu.shape[0] # Dimension of data
    Z = np.zeros(p, dtype=np.float64) # MEWMA vector
    noise = np.zeros(p, dtype=np.float64)
    
    for t in range(1, max_steps + 1):
        # 1. Generate N(0, 1) vector
        for i in range(0, p, 2):
            n1, n2, state = rand_normal_pair(state)
            noise[i] = n1
            if i + 1 < p: noise[i+1] = n2
            
        # 2. Generate multivariate normal vector (Lx)
        Lx = np.dot(L, noise)
        
        # 3. Apply OOC shift/scale and MEWMA update
        for i in range(p):
            innovation = shift_vec[i] + scale_vec[i] * Lx[i]
            Z[i] = (1.0 - lamb) * Z[i] + lamb * innovation
            
        # 4. Calculate T^2 statistic: T^2 = Z^T * inv_mat * Z
        # inv_mat = ( (2-lambda)/lambda ) * Sigma_inv
        t2 = np.dot(Z, np.dot(inv_mat, Z))
        
        # 5. Check control limit
        if t2 > h:
            return t # Chart signaled at time t
            
    return max_steps # Did not signal within max_steps

@njit(nogil=True, fastmath=True, cache=True)
def simulate_batch(lamb, h, inv_mat, mu, L, shift, scale, start_seed, n_rep, max_steps):
    """Runs a batch of simulations, highly parallelized by joblib."""
    out = np.empty(n_rep, dtype=np.int64)
    for i in prange(n_rep): # prange is for Numba internal parallelization if used, but mainly for clarity here
        s = np.uint64(start_seed + i)
        out[i] = kernel_mewma(lamb, h, inv_mat, mu, L, shift, scale, s, max_steps)
    return out

# ====================== 3. SIMULATION ENGINE ======================
class Engine:
    """The main control and execution class for the MEWMA simulation."""
    def __init__(self, args):
        self.args = args
        self.out_dir = Path(args.out)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_sys = LoggerSetup(self.out_dir)
        self.log = self.log_sys.log
        self.log(f"Engine Started - Shard {args.shard_id} | Reps: {args.n_rep}")
        
    def load_data(self):
        """Loads data, calculates covariance using Ledoit-Wolf, and prepares the dataset dictionary."""
        feats = ["mean_error", "inlier_ratio", "log_cond_H"]
        csv_path = self.args.csv_path
        
        # Fallback to synthetic data if CSV is not found
        if not os.path.exists(csv_path):
            self.log("CRITICAL: CSV not found. Using safe synthetic data (Identity matrix).")
            Sigma = np.eye(3) + 0.2
            return [{"name": "Synthetic", "Sigma": Sigma, "p": 3}]

        # Load real data and estimate covariance
        df = pd.read_csv(csv_path)
        X = df[feats].values.astype(np.float64)
        
        # Robust Covariance Estimation
        lw = LedoitWolf().fit(X)
        Sigma = lw.covariance_ + np.eye(3) * 1e-2 # Add small jitter for stability
        
        cond = np.linalg.cond(Sigma)
        self.log(f"Covariance loaded - condition number: {cond:.2f} (Using LedoitWolf)")
        
        return [{"name": "Uploaded_Data", "Sigma": Sigma, "p": 3}]

    def get_scenario(self, name: str, Sigma: np.ndarray):
        """
        Defines the Out-of-Control (OOC) shift vector and scale factor 
        based on the scenario name.
        """
        p = 3
        shift = np.zeros(p)
        scale = np.ones(p)
        
        # Worst-Case Eigenvector calculation for standardized shifts
        invS = np.linalg.inv(Sigma)
        vals, vecs = np.linalg.eigh(Sigma)
        v_worst = vecs[:, -1] # Eigenvector corresponding to largest eigenvalue
        
        def nc(delta):
            """Calculates the non-centrality shift vector mu_1."""
            # Formula: mu_1 = sqrt(delta / (v_worst^T * Sigma^-1 * v_worst)) * v_worst
            return math.sqrt(delta / float(v_worst @ invS @ v_worst)) * v_worst
        
        # Scenario definitions
        if name == "IC": 
            pass # In-Control (no shift, scale=1)
        elif name == "small": 
            shift = nc(1.0)
        elif name == "moderate": 
            shift = nc(2.5)
        elif name == "large": 
            shift = nc(6.0)
        elif name == "cond": 
            shift[2] += 4.0 # Specific shift on log_cond_H feature
        elif name == "inlier": 
            scale[1] = 0.4 # Specific decrease in standard deviation of inlier_ratio feature
            
        return shift, scale

    def run_parallel(self, lamb, h, inv_mat, mu, L, scenario, Sigma, n_rep, offset):
        """Executes the simulation in parallel using joblib threading backend."""
        shift, scale = self.get_scenario(scenario, Sigma)
        
        # Divide total reps into small batches for joblib
        batch_size = 5000
        n_batches = (n_rep + batch_size - 1) // batch_size
        
        # Create unique seed for each batch
        seeds = [np.uint64(self.args.seed + offset + i * 100000) for i in range(n_batches)]
        sizes = [min(batch_size, n_rep - i*batch_size) for i in range(n_batches)]
        
        # Use safe threading backend with 2 cores (optimal for GitHub runners)
        with parallel_backend('threading', n_jobs=2): 
            results = Parallel()(
                delayed(simulate_batch)(
                    lamb, h, inv_mat, mu, L, shift, scale, seeds[i], sizes[i], 1000000
                ) for i in range(n_batches)
            )
            
        # Concatenate results and return the Average Run Length (ARL)
        return np.mean(np.concatenate(results))

    def calibrate_h(self, lamb, inv_mat, mu, L, Sigma):
        """
        Performs binary search to find the control limit 'h' 
        such that ARL0 (IC scenario) is close to the target (370).
        """
        target = 370.0
        low, high = 1.0, 100.0 # Initial search range for h
        
        # Binary search for high precision
        for i in range(1, 19): # 18 iterations is sufficient
            mid = (low + high) / 2
            # Use a smaller repetition count (20000) for calibration speed
            arl = self.run_parallel(lamb, mid, inv_mat, mu, L, "IC", Sigma, 20000, 0)
            self.log(f" Calibration Iteration {i:2d}: h={mid:.4f} → ARL0≈{arl:.1f}")
            
            if arl > target: high = mid
            else: low = mid
            
            # Early exit if precision is reached
            if high - low < 0.05: break 
            
        return (low + high) / 2

    def execute(self):
        """Main execution flow for the simulation engine."""
        datasets = self.load_data()
        results = []
        
        for ds in datasets:
            self.log(f"Processing {ds['name']} (p={ds['p']})")
            
            # Pre-calculations
            mu = np.zeros(3) # In-Control mean (standardized)
            # Add small identity matrix to Sigma for stable Cholesky
            L = np.linalg.cholesky(ds['Sigma'] + np.eye(3)*1e-2) 
            inv_mat_base = np.linalg.inv(ds['Sigma'])
            
            # Run for specified lambda values
            for lamb in [0.05, 0.10, 0.20]:
                # Calculate required inverse matrix for MEWMA statistic: ((2-lambda)/lambda) * Sigma_inv
                inv_mat = ((2.0 - lamb) / lamb) * inv_mat_base
                
                # 1. Calibrate h for ARL0 ≈ 370
                h = self.calibrate_h(lamb, inv_mat, mu, L, ds['Sigma'])
                self.log(f"λ={lamb} → Final Calibrated h={h:.4f}")
                
                # Prepare result row
                row = {"Dataset": ds['name'], "Lambda": lamb, "h": round(h, 4), "Shard": self.args.shard_id}
                
                # 2. Run OOC scenarios
                cursor = 100000 # Use a large offset to prevent overlapping seeds with calibration
                for sc in ["IC", "small", "moderate", "large", "cond", "inlier"]:
                    arl = self.run_parallel(lamb, h, inv_mat, mu, L, sc, ds['Sigma'], self.args.n_rep, cursor)
                    cursor += self.args.n_rep # Move cursor for next scenario
                    
                    row[sc] = round(arl, 2)
                    self.log(f" {sc}: {arl:.2f}")
                
                results.append(row)
                
        # Save results CSV file
        out_file = self.out_dir / f"results_shard_{self.args.shard_id}.csv"
        pd.DataFrame(results).to_csv(out_file, index=False)
        self.log(f"SHARD {self.args.shard_id} SUCCESSFULLY COMPLETED!")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MEWMA Control Chart Simulation Engine (GitHub Optimized)")
    parser.add_argument("--csv_path", type=str, default="phase2_final_features.csv", help="Path to the input feature CSV file.")
    parser.add_argument("--out", type=str, default="results", help="Output directory for logs and result CSVs.")
    parser.add_argument("--shard_id", type=int, required=True, help="Unique ID for the parallel shard (used for file naming).")
    parser.add_argument("--seed", type=int, default=42514251, help="Base seed for reproducibility across shards.")
    parser.add_argument("--n_rep", type=int, default=25000, help="Number of repetitions per scenario per lambda value.")
    args = parser.parse_args()
    Engine(args).execute()
