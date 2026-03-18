"""
Quick diagnostic script to check for infinity/NaN values in processed data
"""

import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/processed/bmw_pricing_clean.csv')

print("="*80)
print("DATA QUALITY CHECK - Infinity & NaN Values")
print("="*80)

print(f"\nDataset shape: {df.shape}")

# Check each numeric column
numeric_cols = df.select_dtypes(include=[np.number]).columns

print("\n" + "="*80)
print("INFINITY VALUES CHECK")
print("="*80)

inf_issues = {}
for col in numeric_cols:
    inf_count = np.isinf(df[col]).sum()
    if inf_count > 0:
        inf_issues[col] = inf_count
        print(f"❌ {col:30s}: {inf_count:,} infinity values")

if not inf_issues:
    print("✅ No infinity values found!")
else:
    print(f"\n🚨 FOUND {len(inf_issues)} columns with infinity values")

print("\n" + "="*80)
print("NaN VALUES CHECK")
print("="*80)

nan_issues = {}
for col in df.columns:
    nan_count = df[col].isna().sum()
    if nan_count > 0:
        nan_issues[col] = nan_count
        print(f"⚠️  {col:30s}: {nan_count:,} NaN values ({nan_count/len(df)*100:.1f}%)")

if not nan_issues:
    print("✅ No NaN values found!")
else:
    print(f"\n⚠️  FOUND {len(nan_issues)} columns with NaN values")

# Check specific problematic columns
print("\n" + "="*80)
print("SPECIFIC CHECKS")
print("="*80)

if 'mileage_per_power' in df.columns:
    print(f"\n'mileage_per_power' statistics:")
    print(f"  Min:  {df['mileage_per_power'].min():.2f}")
    print(f"  Max:  {df['mileage_per_power'].max():.2f}")
    print(f"  Mean: {df['mileage_per_power'].mean():.2f}")
    
    # Check for division by zero
    if 'engine_power' in df.columns:
        zero_power = (df['engine_power'] == 0).sum()
        if zero_power > 0:
            print(f"  🚨 Found {zero_power} rows with engine_power = 0 (causes inf in division)")

if 'annual_mileage' in df.columns:
    print(f"\n'annual_mileage' statistics:")
    print(f"  Min:  {df['annual_mileage'].min():.2f}")
    print(f"  Max:  {df['annual_mileage'].max():.2f}")
    print(f"  Mean: {df['annual_mileage'].mean():.2f}")
    
    # Check for division by zero
    if 'car_age_years' in df.columns:
        zero_age = (df['car_age_years'] == 0).sum()
        if zero_age > 0:
            print(f"  🚨 Found {zero_age} rows with car_age_years = 0 (causes inf in division)")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

if inf_issues or nan_issues:
    print("\n🔧 FIX REQUIRED:")
    print("   1. Update preprocess_data.py to handle division by zero")
    print("   2. Add: df.replace([np.inf, -np.inf], np.nan, inplace=True)")
    print("   3. Fill NaN with median: df.fillna(df.median(), inplace=True)")
    print("   4. Re-run: python scripts/data/preprocess_data.py")
else:
    print("\n✅ Data is clean - ready for training!")

print("="*80)