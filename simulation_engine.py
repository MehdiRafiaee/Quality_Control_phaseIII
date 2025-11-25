import argparse
import numpy as np
import pandas as pd
from numba import njit, prange
import os

# -----------------------------------------
# Efficient batch-based Gaussian sampling
# -----------------------------------------
@njit(fastmath=True, nogil=True)
def generate_normal_batch(n, p, seed):
    """Generates a batch of standard normal random variables."""
    # اصلاح شده: استفاده از seed مستقیم به جای RandomState
    np.random.seed(seed)
    return np.random.normal(0.0, 1.0, (n, p))

# -----------------------------------------
# Fast MEWMA simulation kernel
# -----------------------------------------
@njit(fastmath=True, nogil=True, parallel=True)
def simulate_batch(lamb, h, chol_mat, inv_mat, shift, scale, seeds, max_steps):
    """
    Performs the MEWMA simulation using Cholesky decomposition for correlation.
    """
    n = seeds.shape[0]
    p = shift.shape[0]
    results = np.empty(n, dtype=np.int32)

    for i in prange(n):
        # 🛠️ اصلاح مهم: حذف RandomState و استفاده از توابع استاندارد Numba
        # تنظیم Seed برای هر شبیه‌سازی برای اطمینان از تکرارپذیری
        np.random.seed(seeds[i])
        
        Z = np.zeros(p)
        
        # Flag for early stopping
        finished_at = max_steps
        
        for t in range(1, max_steps + 1):
            # استفاده مستقیم از np.random.normal (پشتیبانی شده توسط Numba)
            noise = np.random.normal(0.0, 1.0, p)
            
            # تولید داده همبسته: X = shift + Chol @ noise
            correlated_noise = chol_mat.dot(noise)
            X = shift + scale * correlated_noise
            
            # MEWMA update
            Z = (1 - lamb) * Z + lamb * X
            
            # T-squared Statistic: Z.T @ Sigma^-1 @ Z
            stat = Z @ inv_mat @ Z
            
            if stat > h:
                finished_at = t
                break
        
        results[i] = finished_at
        
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

    # ---------------------------------------------------------
    # 1. بارگذاری و تمیزکاری داده
    # ---------------------------------------------------------
    try:
        df = pd.read_csv("phase2_final_features.csv")
    except FileNotFoundError:
        print("❌ Error: 'phase2_final_features.csv' not found.")
        return

    # استخراج ستون‌های عددی
    df_numeric = df.select_dtypes(include=[np.number])

    if df_numeric.empty:
        print("❌ Error: No numeric columns found in input CSV.")
        return

    # تبدیل به ماتریس NumPy و تمیزکاری
    data = df_numeric.values.astype(np.float64)
    data = data[~np.isnan(data).any(axis=1)]

    print(f"✅ Data loaded. Shape: {data.shape}")

    # ---------------------------------------------------------
    # 2. محاسبات ماتریسی
    # ---------------------------------------------------------
    # ترانهاده برای np.cov (Features x Samples)
    cov_mat = np.cov(data.T)
    p = cov_mat.shape[0]

    # محاسبه معکوس ماتریس کوواریانس
    try:
        inv_mat = np.linalg.inv(cov_mat)
    except np.linalg.LinAlgError:
        print("⚠️ Singular matrix! Using pseudo-inverse.")
        inv_mat = np.linalg.pinv(cov_mat)

    # محاسبه تجزیه چولسکی
    try:
        chol_mat = np.linalg.cholesky(cov_mat)
    except np.linalg.LinAlgError:
        print("⚠️ Matrix not positive definite. Adding epsilon jitter.")
        jitter = 1e-6 * np.eye(p)
        chol_mat = np.linalg.cholesky(cov_mat + jitter)
        inv_mat = np.linalg.inv(cov_mat + jitter)

    # اطمینان از پیوسته بودن حافظه برای Numba (خیلی مهم برای سرعت و جلوگیری از خطا)
    chol_mat = np.ascontiguousarray(chol_mat)
    inv_mat = np.ascontiguousarray(inv_mat)

    # ---------------------------------------------------------
    # 3. پیکربندی شبیه‌سازی
    # ---------------------------------------------------------
    scenarios = {
        "IC":       np.zeros(p),
        "small":    np.ones(p) * 0.1,
        "moderate": np.ones(p) * 0.5,
        "large":    np.ones(p) * 1.0,
    }

    h = 12.0 

    # تولید بذرها در پایتون اصلی (Numba با لیست بذرهای uint32 مشکلی ندارد)
    rng_main = np.random.RandomState(args.base_seed)
    seeds = rng_main.randint(0, 2**32, size=args.n_sim, dtype=np.uint32)

    records = []
    print(f"🚀 Starting simulation for lambda={lamb} with {args.n_sim} runs...")

    for name, shift in scenarios.items():
        # شیفت را هم باید contiguous کنیم تا numba سریعتر کار کند
        shift_arr = np.ascontiguousarray(shift)
        
        res = simulate_batch(
            lamb, h, chol_mat, inv_mat, shift_arr,
            scale=1.0,
            seeds=seeds,
            max_steps=10000 
        )
        
        arl = res.mean()
        sdrling = res.std()
        
        records.append({
            "Lambda": lamb,
            "Scenario": name,
            "ARL": arl,
            "SDRL": sdrling,
            "N_Sim": args.n_sim
        })
        print(f"   -> {name}: ARL={arl:.2f}")

    # ---------------------------------------------------------
    # 4. ذخیره خروجی
    # ---------------------------------------------------------
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    out_df = pd.DataFrame(records)
    out_file = os.path.join(args.out, f"results_lambda_{lamb}_{args.base_seed}.csv")
    out_df.to_csv(out_file, index=False)
    print(f"💾 Results saved to: {out_file}")

if __name__ == "__main__":
    main()
