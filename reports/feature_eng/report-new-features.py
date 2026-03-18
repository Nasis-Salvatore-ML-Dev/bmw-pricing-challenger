import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from pathlib import Path

def generate_feature_report():
    df = pd.read_csv('../../data/raw/bmw_pricing_challenge.csv')
    train_df, _ = train_test_split(df, test_size=0.2, random_state=42)
    
    report_path = Path('Feature_Engineering_Report.txt')
    with open(report_path, 'w') as f:
        f.write("BMW PRICING - FEATURE ENGINEERING ANOVA REPORT\n")
        f.write("="*60 + "\n")
        
        for cat_col in ['fuel', 'car_type', 'paint_color']:
            groups = [train_df[train_df[cat_col] == cat]['price'].values 
                      for cat in train_df[cat_col].unique() if pd.notna(cat)]
            f_stat, p_value = stats.f_oneway(*groups)
            sig = "HIGHLY SIGNIFICANT" if p_value < 0.001 else "NOT SIGNIFICANT"
            f.write(f"Feature: {cat_col.upper()}\n  F-statistic: {f_stat:.2f} | P-value: {p_value:.4f} | {sig}\n\n")

if __name__ == "__main__":
    Path('.').mkdir(parents=True, exist_ok=True)
    generate_feature_report()
    print("✅ Feature Report generated at reports/feature_eng/Feature_Engineering_Report.txt")