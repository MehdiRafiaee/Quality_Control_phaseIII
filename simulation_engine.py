import argparse
import numpy as np
import pandas as pd
from numba import njit, prange

# -----------------------------------------
# Efficient batch-based Gaussian sampling
# -----------------------------------------
@njit(fastmath=True, nogil=True)
def generate_normal_batch(n, p, seed):
    """Generates a batch of standard normal random variables."""
    rng = np.random.RandomState(seed)
    return rng.normal(0.0, 1.0, (n, p))

# -----------------------------------------
# Fast MEWMA simulation kernel
# -----------------------------------------
@njit(fastmath=True, nogil=True, parallel=True)
def simulate_batch(lamb, h, L, inv_mat, shift, scale, seeds, max_steps):
    """Performs the MEWMA simulation for a batch of seeds."""
    n = seeds.shape[0]
    p = shift.shape[0]
    results = np.empty(n, dtype=np.int32)

    for i in prange(n):
        rng = np.random.RandomState(seeds[i])
        Z = np.zeros(p)
        for t in range(1, max_steps + 1):
            noise = rng.normal(0.0, 1.0, p)
            # L.dot(noise) - اینجا L ماتریس مثلثی پایین است (Cholesky decomposition)
            X = shift + scale * L.dot(noise)
            Z = (1 - lamb) * Z + lamb * X
            
            # محاسبه آماره T-squared: Z^T * inv_mat * Z
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

    ## 🛠️ رفع خطا: اطمینان از عددی بودن L و ساختار صحیح برای np.cov
    
    # 1. بارگذاری ماتریس ویژگی‌های فاز 2
    # فرض می‌کنیم فایل 'phase2_final_features.csv' دارای داده‌های Sample x Feature است.
    # برای جلوگیری از خواندن ستون‌های متنی (مثل ایندکس یا هدر)، از پارامترهای زیر استفاده می‌کنیم:
    df = pd.read_csv(
        "phase2_final_features.csv",
        header=None, # فرض می‌کنیم فایل فاقد ردیف هدر متنی است
        skiprows=1 if pd.read_csv("phase2_final_features.csv").iloc[0].dtype == object else 0 # اگر ردیف اول رشته بود، آن را حذف کن
    )
    
    # L_raw شامل داده‌ها به همراه ستون‌های احتمالی غیر ویژگی (مثل ID یا ایندکس) است.
    # ستون اول (Index 0) را به عنوان ستون غیر ویژگی حذف می‌کنیم و باقی ستون‌ها را می‌گیریم.
    L_features = df.iloc[:, 1:].values 
    
    # 2. تبدیل اجباری به نوع float برای رفع TypeError
    # این گام اطمینان می‌دهد که هیچ رشته‌ای در آرایه باقی نمانده است.
    try:
        L_numeric = L_features.astype(np.float64)
    except ValueError as e:
        print(f"خطا در تبدیل داده‌ها به عدد. بررسی کنید که 'phase2_final_features.csv' فقط شامل اعداد است: {e}")
        return
        
    # 3. آماده‌سازی برای np.cov
    # np.cov انتظار دارد Features در سطرها و Samples در ستون‌ها باشند.
    # اگر L_numeric به صورت (Samples x Features) است، باید ترانهاده شود.
    L = L_numeric.T 
    
    p = L.shape[0] # p: تعداد ویژگی‌ها

    # 4. محاسبه ماتریس کوواریانس و معکوس آن (حالا بدون TypeError)
    inv_mat = np.linalg.inv(np.cov(L))
    
    # ----------------------------------------------------
    
    # Example shift vectors (IC, small, moderate ...)
    scenarios = {
        "IC":       np.zeros(p),
        "small":    np.ones(p) * 0.1,
        "moderate": np.ones(p) * 0.3,
        "large":    np.ones(p) * 0.6,
    }

    # Calibrated h (placeholder: set per-lambda)
    # 💡 توجه: مقدار h معمولاً باید بر اساس ARL0 مورد نظر در حالت IC کالیبره شود.
    h = 8.5 

    # تولید بذرها (seeds) برای شبیه‌سازی‌های موازی
    rng_main = np.random.RandomState(args.base_seed)
    seeds = rng_main.randint(0, 2**32, size=args.n_sim)

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

    # 5. ذخیره نتایج
    out_df = pd.DataFrame(records, columns=["Lambda", "Scenario", "ARL"])
    out_df.to_csv(f"{args.out}/results_lambda_{lamb}.csv", index=False)

if __name__ == "__main__":
    main()
