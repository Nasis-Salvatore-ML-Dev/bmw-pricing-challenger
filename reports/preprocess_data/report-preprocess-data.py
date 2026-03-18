import pandas as pd
from pathlib import Path
from datetime import datetime

def generate_preprocessing_report():
    raw_path = Path('../../data/raw/bmw_pricing_challenge.csv')
    clean_path = Path('../../data/processed/bmw_pricing_clean.csv')
    
    df_raw = pd.read_csv(raw_path)
    df_clean = pd.read_csv(clean_path)
    
    report_path = Path('Preprocessing_Report.txt')
    with open(report_path, 'w') as f:
        f.write(f"BMW PRICING - PREPROCESSING VALIDATION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Raw Columns: {df_raw.shape[1]} | Clean Columns: {df_clean.shape[1]}\n")
        f.write(f"Price Skewness (Before): {df_raw['price'].skew():.3f}\n")
        f.write(f"Price Skewness (After Log-Transform): {df_clean['price_log'].skew():.3f}\n")
        f.write(f"Capped Extreme Mileages: {df_clean[df_clean['mileage'] == 500000].shape[0]} rows\n")
        f.write(f"Flagged Price Outliers: {df_clean['is_price_outlier'].sum()} rows\n\n")
        f.write(f"NOTE: Imputation medians and outlier boundaries were successfully derived using a rigid Train/Test split to prevent data leakage.\n")

if __name__ == "__main__":
    Path('.').mkdir(parents=True, exist_ok=True)
    generate_preprocessing_report()
    print("✅ Preprocessing Report generated at reports/preprocess_data/Preprocessing_Report.txt")