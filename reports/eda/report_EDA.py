import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from datetime import datetime

def generate_eda_report():
    raw_path = Path('../../data/raw/bmw_pricing_challenge.csv')
    df = pd.read_csv(raw_path)
    
    # Split to prevent leakage in correlation reporting
    train_df, _ = train_test_split(df, test_size=0.2, random_state=42)
    
    # Feature correlations
    numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
    correlations = train_df[numeric_cols].corr()['price'].sort_values(ascending=False)
    
    report_path = Path('EDA_Summary_Report.txt')
    with open(report_path, 'w') as f:
        f.write(f"BMW PRICING - EDA DATA QUALITY & CORRELATION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Total Raw Rows: {len(df)}\n")
        f.write(f"Missing Values: {df.isnull().sum().sum()}\n\n")
        f.write(f"TOP CORRELATIONS (Derived from Training Split Only):\n")
        for feat, corr in correlations.drop('price').head(5).items():
            f.write(f"  {feat:20s}: {corr:+.3f}\n")

if __name__ == "__main__":
    Path('.').mkdir(parents=True, exist_ok=True)
    generate_eda_report()
    print("✅ EDA Report generated at reports/eda/EDA_Summary_Report.txt")