import argparse
import pandas as pd
import os
import glob
import sys

def merge_results(input_dir, output_file):
    print(f"--- شروع عملیات ادغام ---")
    print(f"📂 جستجو در پوشه: {input_dir}")
    
    # بررسی وجود پوشه
    if not os.path.exists(input_dir):
        print(f"❌ خطا: پوشه ورودی '{input_dir}' وجود ندارد!")
        sys.exit(1)

    # لیست کردن تمام فایل‌های موجود در پوشه برای دیباگ
    all_files_in_dir = os.listdir(input_dir)
    print(f"📄 فایل‌های موجود در پوشه ({len(all_files_in_dir)} عدد):")
    for f in all_files_in_dir:
        print(f"   - {f}")

    # جستجوی فایل‌های CSV
    # الگوی جستجو را با فایل‌های تولید شده در simulation_engine هماهنگ کنید
    # فایل‌ها معمولاً با results_ شروع می‌شوند
    search_pattern = os.path.join(input_dir, "*.csv") 
    csv_files = glob.glob(search_pattern)
    
    print(f"🔍 تعداد فایل‌های CSV پیدا شده: {len(csv_files)}")

    if not csv_files:
        print("❌ هیچ فایل CSV برای ادغام پیدا نشد.")
        print("💡 نکته: بررسی کنید که مرحله 'simulate' با موفقیت تمام شده و فایل‌ها را تولید کرده است.")
        # اگر می‌خواهید ورک‌فلو قرمز نشود، sys.exit(0) بگذارید، ولی بهتر است خطا دهد:
        sys.exit(1) 

    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ هشدار: خطا در خواندن فایل {file}: {e}")

    if not dfs:
        print("❌ هیچ دیتافریمی بارگذاری نشد.")
        sys.exit(1)

    # ادغام نتایج
    print("🔄 در حال ادغام...")
    final_df = pd.concat(dfs, ignore_index=True)
    
    # ذخیره نتیجه نهایی
    final_df.to_csv(output_file, index=False)
    print(f"✅ با موفقیت ذخیره شد در: {output_file}")
    print(f"📊 ابعاد نهایی دیتاست: {final_df.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    merge_results(args.input_dir, args.output)
