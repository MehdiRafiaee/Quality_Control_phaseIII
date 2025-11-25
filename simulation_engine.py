import argparse
import numpy as np
import pandas as pd
from numba import njit, prange
import os

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

    # Pre-allocate reuseable arrays helps slightly with cache locality in some cases,
    # but inside parallel loop, we declare local variables.
    
    for i in prange(n):
        rng = np.random.RandomState(seeds[i])
        Z = np.zeros(p)
        
        # Flag for early stopping if needed, but 'break' works in numba loops
        finished_at = max_steps
        
        for t in range(1, max_steps + 1):
            noise = rng.normal(0.0, 1.0, p)
            
            # 💡 اصلاح ریاضی: تولید داده همبسته با استفاده از ماتریس چولسکی
            # X ~ N(shift, Sigma) => X = shift + Chol @ noise
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
    # 1. بارگذاری و تمیزکاری داده (بهینه شده)
    # ---------------------------------------------------------
    try:
        # خواندن فایل (فرض بر این است که هدر دارد، اگر ندارد header=None اضافه شود)
        # استفاده از engine='c' برای سرعت بیشتر
        df = pd.read_csv("phase2_final_features.csv")
    except FileNotFoundError:
        print("❌ Error: 'phase2_final_features.csv' not found.")
        return

    # فقط ستون‌های عددی را نگه می‌داریم (این کار خودکار ID، Timestamp و رشته‌ها را حذف می‌کند)
    df_numeric = df.select_dtypes(include=[np.number])

    if df_numeric.empty:
        print("❌ Error: No numeric columns found in input CSV.")
        return

    # تبدیل به ماتریس NumPy و حذف ردیف‌های دارای NaN (پاکسازی نهایی)
    data = df_numeric.values.astype(np.float64)
    data = data[~np.isnan(data).any(axis=1)] # حذف نمونه‌های ناقص

    print(f"✅ Data loaded. Shape: {data.shape}")

    # ---------------------------------------------------------
    # 2. محاسبات ماتریسی (کوواریانس، چولسکی، معکوس)
    # ---------------------------------------------------------
    # np.cov انتظار دارد: سطر=ویژگی، ستون=نمونه. بنابراین Transpose می‌گیریم.
    # داده‌های ما: (Samples x Features) -> Transpose -> (Features x Samples)
    cov_mat = np.cov(data.T)
    p = cov_mat.shape[0]

    # محاسبه معکوس ماتریس کوواریانس برای آماره T2
    try:
        inv_mat = np.linalg.inv(cov_mat)
    except np.linalg.LinAlgError:
        print("⚠️ Singular matrix! Using pseudo-inverse.")
        inv_mat = np.linalg.pinv(cov_mat)

    # محاسبه تجزیه چولسکی برای شبیه‌سازی (Sigma = L @ L.T)
    # این ماتریس برای تولید نویز همبسته استفاده می‌شود
    try:
        chol_mat = np.linalg.cholesky(cov_mat)
    except np.linalg.LinAlgError:
        print("⚠️ Matrix not positive definite. Adding epsilon jitter.")
        # افزودن مقدار بسیار کم به قطر اصلی برای حل مشکل مثبت معین نبودن
        jitter = 1e-6 * np.eye(p)
        chol_mat = np.linalg.cholesky(cov_mat + jitter)
        inv_mat = np.linalg.inv(cov_mat + jitter)

    # ---------------------------------------------------------
    # 3. پیکربندی شبیه‌سازی
    # ---------------------------------------------------------
    scenarios = {
        "IC":       np.zeros(p),
        "small":    np.ones(p) * 0.1,  # شیفت کوچک (قدرت واقعی MEWMA)
        "moderate": np.ones(p) * 0.5,
        "large":    np.ones(p) * 1.0,
    }

    # حد کنترل (h) باید کالیبره شود. فعلا مقدار ثابت:
    h = 12.0 # مثال: برای p=10 معمولا حدود 10-15 است

    # تولید بذرهای مستقل برای این Shard
    rng_main = np.random.RandomState(args.base_seed)
    seeds = rng_main.randint(0, 2**32, size=args.n_sim, dtype=np.uint32)

    records = []
    print(f"🚀 Starting simulation for lambda={lamb} with {args.n_sim} runs...")

    for name, shift in scenarios.items():
        # فراخوانی هسته Numba
        # توجه: ما chol_mat را پاس می‌دهیم، نه داده خام را
        res = simulate_batch(
            lamb, h, chol_mat, inv_mat, shift,
            scale=1.0,
            seeds=seeds,
            max_steps=10000 # کاهش max_steps برای تست سریع‌تر (قابل تغییر)
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
