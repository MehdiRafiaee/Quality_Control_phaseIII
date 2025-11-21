#!/usr/bin/env python3
# simulation_engine.py
# -----------------------------------------------------------------------------
# GITHUB ACTIONS EDITION (CPU OPTIMIZED)
# -----------------------------------------------------------------------------
# - Adapted for 'phase2_final_features.csv' columns
# - Optimized for 2-Core GitHub Runners
# - File-based logging for Artifact upload
# -----------------------------------------------------------------------------

import os
# Force single thread per numpy op to avoid conflict with joblib
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import time
import math
import json
import argparse
import logging
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from sklearn.covariance import LedoitWolf
from datetime import datetime
from pathlib import Path

# Numba Check
try:
    from numba import njit
    import numba
except ImportError:
    sys.exit("CRITICAL: Numba missing.")

# -----------------------------------------------------------------------------
# 1. LOGGING SETUP (Crucial for Debugging in GitHub)
# -----------------------------------------------------------------------------
class LoggerSetup:
    def __init__(self, output_dir):
        self.logger = logging.getLogger("SimEngine")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        
        # Console Handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(ch)
        
        # File Handler (Save this to upload as artifact)
        self.log_path = output_dir / "execution_log.txt"
        fh = logging.FileHandler(self.log_path, mode='a', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(fh)

    def log(self, msg):
        self.logger.info(msg)

# -----------------------------------------------------------------------------
# 2. NUMBA KERNELS (Fast CPU Logic)
# -----------------------------------------------------------------------------
@njit(inline='always')
def xorshift64_star(state):
    x = state
    x ^= (x >> 12) & 0xFFFFFFFFFFFFFFFF
    x ^= (x << 25) & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 27) & 0xFFFFFFFFFFFFFFFF
    return (x * 2685821657736338717) & 0xFFFFFFFFFFFFFFFF

@njit(inline='always')
def rand_normal_pair(state):
    state = xorshift64_star(state)
    u1 = (state & 0xFFFFFFFFFFFFFFFF) / 18446744073709551616.0
    if u1 == 0.0: u1 = 1e-18
    state = xorshift64_star(state)
    u2 = (state & 0xFFFFFFFFFFFFFFFF) / 18446744073709551616.0
    r = math.sqrt(-2.0 * math.log(u1))
    theta = 2.0 * math.pi * u2
    return r * math.cos(theta), r * math.sin(theta), state

@njit(nogil=True, fastmath=True, cache=True)
def kernel_mewma(lamb, h, inv_mat, mu, L, shift_vec, scale_vec, seed, max_steps):
    state = seed if seed != 0 else 12345
    p = mu.shape[0]
    Z = np.zeros(p, dtype=np.float64)
    noise = np.zeros(p, dtype=np.float64)
    
    for t in range(1, max_steps + 1):
        # Generate Noise
        for i in range(0, p, 2):
            n1, n2, state = rand_normal_pair(state)
            noise[i] = n1
            if i + 1 < p: noise[i+1] = n2
            
        Lx = np.dot(L, noise)
        
        # Update EWMA
        for i in range(p):
            # X_t = mu + shift + scale * Lx
            innovation = shift_vec[i] + scale_vec[i] * Lx[i]
            Z[i] = (1.0 - lamb) * Z[i] + lamb * innovation
            
        # T2 Statistic
        # temp = inv_mat @ Z
        temp = np.dot(inv_mat, Z)
        t2 = np.dot(Z, temp)
        
        if t2 > h:
            return t
    return max_steps

@njit(nogil=True, fastmath=True, cache=True)
def simulate_batch(lamb, h, inv_mat, mu, L, shift, scale, start_seed, n_rep, max_steps):
    out = np.empty(n_rep, dtype=np.int64)
    for i in range(n_rep):
        s = int((start_seed + i) & 0xFFFFFFFFFFFFFFFF)
        out[i] = kernel_mewma(lamb, h, inv_mat, mu, L, shift, scale, s, max_steps)
    return out

# -----------------------------------------------------------------------------
# 3. ENGINE CLASS
# -----------------------------------------------------------------------------
class Engine:
    def __init__(self, args):
        self.args = args
        self.out_dir = Path(args.out)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_sys = LoggerSetup(self.out_dir)
        self.log = self.log_sys.log
        self.n_cpus = 2 # GitHub Runners usually have 2 vCPUs
        self.log(f"Engine Started. Shard ID: {args.shard_id}")

    def load_data(self):
        # Specific column names based on your uploaded CSV
        feats = ["mean_error", "median_error", "p90_error", "inlier_ratio", "log_cond_H"]
        
        datasets = []
        if self.args.csv_path and os.path.exists(self.args.csv_path):
            self.log(f"Loading CSV: {self.args.csv_path}")
            try:
                df = pd.read_csv(self.args.csv_path)
                # Check columns
                missing = [c for c in feats if c not in df.columns]
                if missing:
                    self.log(f"ERROR: Missing columns in CSV: {missing}")
                else:
                    X = df[feats].values.astype(np.float64)
                    # Robust Covariance
                    try:
                        lw = LedoitWolf().fit(X)
                        Sigma = lw.covariance_
                    except:
                        Sigma = np.cov(X, rowvar=False) + np.eye(len(feats))*1e-8
                    
                    datasets.append({
                        "name": "Uploaded_Data",
                        "Sigma": Sigma,
                        "p": len(feats)
                    })
            except Exception as e:
                self.log(f"Failed to read CSV: {e}")
        else:
            self.log("No CSV found or provided. Using Synthetic Data.")
            # Fallback Synthetic
            p = 5
            rng = np.random.RandomState(42)
            A = rng.randn(p, p)
            Sigma = np.dot(A, A.T)
            datasets.append({"name": "Synthetic", "Sigma": Sigma, "p": p})
            
        return datasets

    def get_scenario(self, name, Sigma):
        p = Sigma.shape[0]
        invS = np.linalg.inv(Sigma)
        vals, vecs = np.linalg.eigh(Sigma)
        v_worst = vecs[:, -1]
        
        shift = np.zeros(p)
        scale = np.ones(p)
        
        def calc_shift(delta):
            denom = float(v_worst.T @ invS @ v_worst)
            alpha = math.sqrt(delta / denom)
            return alpha * v_worst
            
        if name == "IC": pass
        elif name == "small": shift = calc_shift(0.5)
        elif name == "moderate": shift = calc_shift(1.0)
        elif name == "large": shift = calc_shift(4.0)
        elif name == "cond": shift[-1] += 4.0 * np.sqrt(np.diag(Sigma)[-1])
        elif name == "inlier":
            if p >= 4:
                scale[3] = 0.5
                scale[0:3] = 1.4
        
        return shift, scale

    def run_parallel(self, lamb, h, inv_mat, mu, L, sc_name, Sigma, n_rep, offset_seed):
        shift, scale = self.get_scenario(sc_name, Sigma)
        batch_size = 2500
        n_batches = (n_rep + batch_size - 1) // batch_size
        
        seeds = [self.args.seed + offset_seed + (i * batch_size) for i in range(n_batches)]
        sizes = [min(batch_size, n_rep - i*batch_size) for i in range(n_batches)]
        
        with parallel_backend('threading', n_jobs=self.n_cpus):
            results = Parallel()(
                delayed(simulate_batch)(lamb, h, inv_mat, mu, L, shift, scale, s, sz, 1000000)
                for s, sz in zip(seeds, sizes)
            )
            
        flat = np.concatenate(results)
        return np.mean(flat), len(flat)

    def calibrate(self, lamb, inv_mat, mu, L, Sigma):
        # Simplified fast calibration for GitHub Actions time limits
        target = 370.0
        low, high = 1.0, 50.0
        
        # Quick binary search
        for i in range(12):
            mid = (low + high) / 2.0
            res, _ = self.run_parallel(lamb, mid, inv_mat, mu, L, "IC", Sigma, 10000, 0)
            if res > target: high = mid
            else: low = mid
        return (low + high) / 2.0

    def execute(self):
        datasets = self.load_data()
        results = []
        
        for ds in datasets:
            self.log(f"Processing: {ds['name']}")
            mu = np.zeros(ds['p'])
            L = np.linalg.cholesky(ds['Sigma'] + np.eye(ds['p'])*1e-6)
            invS = np.linalg.inv(ds['Sigma'])
            
            for lamb in [0.05, 0.10, 0.20]:
                # Using AE-MEWMA (UltraSafe) logic
                scale_fac = (2.0 - lamb) / lamb
                inv_mat = scale_fac * invS
                
                # Calibrate (Small run)
                h = self.calibrate(lamb, inv_mat, mu, L, ds['Sigma'])
                self.log(f"Lambda={lamb}, Calibrated h={h:.4f}")
                
                # Scenarios
                scenarios = ["IC", "small", "moderate", "large", "cond", "inlier"]
                row = {"Dataset": ds['name'], "Lambda": lamb, "h": h, "Shard": self.args.shard_id}
                
                cursor = 0
                for sc in scenarios:
                    self.log(f"  Running {sc}...")
                    mean_rl, n = self.run_parallel(
                        lamb, h, inv_mat, mu, L, sc, ds['Sigma'], 
                        self.args.n_rep, cursor
                    )
                    cursor += self.args.n_rep
                    row[sc] = mean_rl
                
                results.append(row)
                
        # Save
        df = pd.DataFrame(results)
        out_file = self.out_dir / f"results_shard_{self.args.shard_id}.csv"
        df.to_csv(out_file, index=False)
        self.log(f"Saved results to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="phase2_final_features.csv")
    parser.add_argument("--out", type=str, default="results")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_rep", type=int, default=10000)
    args = parser.parse_args()
    
    Engine(args).execute()
