"""
BMW Pricing - CORRECTED Advanced Data Preprocessing (NO DATA LEAKAGE)

CRITICAL FIXES:
- ❌ REMOVED depreciation_rate (was price / age) - DATA LEAKAGE!
- ❌ REMOVED price_per_km (was price / mileage) - DATA LEAKAGE!
- ❌ REMOVED price_segment (was bins of price) - DATA LEAKAGE!
- ✅ ADDED legitimate segment features based on BMW series
- ✅ ADDED legitimate quality proxies (power-to-weight, luxury indicators)

This version has NO leakage - all features can be calculated at prediction time.

Usage:
    python scripts/data/preprocess_data.py

Input:  data/raw/bmw_pricing_challenge.csv
Output: data/processed/bmw_pricing_clean.csv (with 'data_split' column)
        data/processed/target_encodings.json   (mappings for inference)
"""

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw BMW pricing data"""
    logger.info(f"Loading raw data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def split_raw_data(
    df: pd.DataFrame,
    train_size: float = 0.6,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Split raw data FIRST before any transformations
    """
    logger.info("=" * 80)
    logger.info("SPLITTING RAW DATA (BEFORE TRANSFORMATIONS)")
    logger.info("=" * 80)

    assert abs(train_size + val_size + test_size - 1.0) < 0.001, "Split sizes must sum to 1.0"

    train_df, temp_df = train_test_split(
        df, train_size=train_size, random_state=random_state, shuffle=True
    )
    val_ratio = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df, train_size=val_ratio, random_state=random_state, shuffle=True
    )

    train_df["data_split"] = "train"
    val_df["data_split"] = "val"
    test_df["data_split"] = "test"

    df_with_splits = pd.concat([train_df, val_df, test_df], axis=0)

    logger.info(f"  Training:   {len(train_df):,} rows ({len(train_df)/len(df):.1%})")
    logger.info(f"  Validation: {len(val_df):,} rows ({len(val_df)/len(df):.1%})")
    logger.info(f"  Test:       {len(test_df):,} rows ({len(test_df)/len(df):.1%})")
    logger.info("=" * 80)

    return df_with_splits


def extract_bmw_series(model_key: str) -> str:
    """Extract BMW series from model_key"""
    model_key = str(model_key).upper()

    # X-series (SUVs)
    if "X1" in model_key:
        return "X1"
    elif "X2" in model_key:
        return "X2"
    elif "X3" in model_key:
        return "X3"
    elif "X4" in model_key:
        return "X4"
    elif "X5" in model_key:
        return "X5"
    elif "X6" in model_key:
        return "X6"
    elif "X7" in model_key:
        return "X7"
    elif "1" in model_key and any(
        x in model_key for x in ["114", "116", "118", "120", "125", "128", "130", "135", "140"]
    ):
        return "1_series"
    elif "2" in model_key and any(
        x in model_key for x in ["214", "216", "218", "220", "225", "228", "230", "235", "240"]
    ):
        return "2_series"
    elif "3" in model_key and any(
        x in model_key for x in ["316", "318", "320", "325", "328", "330", "335", "340"]
    ):
        return "3_series"
    elif "4" in model_key and any(
        x in model_key for x in ["418", "420", "425", "428", "430", "435", "440"]
    ):
        return "4_series"
    elif "5" in model_key and any(
        x in model_key for x in ["518", "520", "523", "525", "528", "530", "535", "540", "550"]
    ):
        return "5_series"
    elif "6" in model_key and any(x in model_key for x in ["620", "630", "640", "650"]):
        return "6_series"
    elif "7" in model_key and any(
        x in model_key for x in ["730", "735", "740", "745", "750", "760"]
    ):
        return "7_series"
    elif "8" in model_key:
        return "8_series"
    elif "M2" in model_key or "M3" in model_key or "M4" in model_key or "M5" in model_key:
        return "M_series"
    elif "I3" in model_key or "I8" in model_key:
        return "i_series"
    elif "Z4" in model_key:
        return "Z4"
    else:
        return "other_series"


def create_bmw_segment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create BMW segment features based on SERIES, not price

    NO LEAKAGE: Uses bmw_series which is observable at prediction time
    """
    logger.info("Creating BMW segment features (based on series, NOT price)...")

    # Extract BMW series
    df["bmw_series"] = df["model_key"].apply(extract_bmw_series)

    # Define luxury level based on BMW series (observable fact, not price!)
    luxury_mapping = {
        "1_series": 1,  # Entry level
        "2_series": 1,
        "X1": 1,
        "X2": 1,
        "3_series": 2,  # Mid-range
        "4_series": 2,
        "X3": 2,
        "5_series": 3,  # Premium
        "X4": 3,
        "X5": 3,
        "6_series": 4,  # Luxury
        "7_series": 4,
        "8_series": 4,
        "X6": 4,
        "X7": 4,
        "M_series": 4,  # Performance luxury
        "i_series": 3,  # Premium electric
        "Z4": 3,
        "other_series": 2,
    }

    df["luxury_tier"] = df["bmw_series"].map(luxury_mapping).fillna(2).astype(int)

    # Is luxury car? (series 5+, X5+, M-series)
    df["is_luxury"] = (df["luxury_tier"] >= 3).astype(int)

    # Is performance model? (M-series or high trim)
    df["is_performance"] = df["bmw_series"].isin(["M_series"]).astype(int)

    series_counts = df["bmw_series"].value_counts()
    logger.info(f"  ✅ Created {len(series_counts)} BMW series categories")
    logger.info(f"  ✅ Luxury tier distribution: {df['luxury_tier'].value_counts().to_dict()}")

    return df


def create_temporal_features(df: pd.DataFrame, sold_date_col: str = "sold_at") -> pd.DataFrame:
    """Extract temporal features and calculate car_age_years"""
    logger.info("Creating temporal features from registration_date...")

    df["registration_datetime"] = pd.to_datetime(df["registration_date"], errors="coerce")
    df["registration_year"] = df["registration_datetime"].dt.year
    df["registration_month"] = df["registration_datetime"].dt.month
    df["registration_quarter"] = df["registration_datetime"].dt.quarter

    df["is_summer_registration"] = df["registration_month"].isin([6, 7, 8]).astype(int)
    df["is_year_end_registration"] = df["registration_month"].isin([11, 12]).astype(int)

    # Calculate car_age_years if missing
    if "car_age_years" not in df.columns:
        logger.info(
            "  ℹ️  car_age_years not found, calculating from registration_date and sold_at..."
        )
        df["sold_datetime"] = pd.to_datetime(df[sold_date_col], errors="coerce")
        df["car_age_years"] = (df["sold_datetime"] - df["registration_datetime"]).dt.days / 365.25
        df["car_age_years"] = df["car_age_years"].clip(lower=0)
        df["car_age_years"] = df["car_age_years"].fillna(df["car_age_years"].median())
        logger.info(f"  ✅ Calculated car_age_years: mean={df['car_age_years'].mean():.1f} years")
        df.drop("sold_datetime", axis=1, inplace=True)

    df.drop("registration_datetime", axis=1, inplace=True)

    logger.info("  ✅ Added: registration_year, registration_month, registration_quarter")
    logger.info("  ✅ Added: is_summer_registration, is_year_end_registration")

    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features - NO LEAKAGE

    All features use only observable data (mileage, age, power, series)
    """
    logger.info("Creating interaction features (NO price leakage)...")

    # 1. Age × Mileage interaction
    df["age_mileage_interaction"] = df["car_age_years"] * df["mileage"] / 10000

    # 2. Mileage per power (efficiency)
    df["mileage_per_power"] = df["mileage"] / (df["engine_power"] + 1)

    # 3. Annual mileage (usage intensity)
    df["annual_mileage"] = df["mileage"] / (df["car_age_years"] + 1)

    # 4. Power-to-age ratio (performance retention)
    df["power_age_ratio"] = df["engine_power"] / (df["car_age_years"] + 1)

    # 5. Luxury × Mileage interaction (luxury cars hurt more by mileage)
    df["luxury_mileage_interaction"] = df["is_luxury"] * df["mileage"] / 10000

    # 6. Luxury × Age interaction
    df["luxury_age_interaction"] = df["is_luxury"] * df["car_age_years"]

    # 7. High mileage indicator
    df["is_high_mileage"] = (df["annual_mileage"] > 25000).astype(int)

    # 8. Is old car?
    df["is_old_car"] = (df["car_age_years"] > 8).astype(int)

    logger.info("  ✅ Added: age_mileage_interaction, mileage_per_power, annual_mileage")
    logger.info("  ✅ Added: power_age_ratio, luxury interactions, condition flags")

    return df


def learn_processing_parameters(train_df: pd.DataFrame) -> dict:
    """Learn ALL processing parameters from TRAINING set only"""
    logger.info("=" * 80)
    logger.info("LEARNING PROCESSING PARAMETERS (FROM TRAINING SET ONLY)")
    logger.info("=" * 80)

    params = {}

    # Fuel type encoding
    fuel_means = train_df.groupby("fuel")["price"].mean()
    params["fuel_target_encoding"] = fuel_means.to_dict()
    logger.info(f"  Fuel target encoding: {len(fuel_means)} categories")
    for fuel, mean_price in fuel_means.items():
        logger.info(f"    {fuel:20s}: €{mean_price:,.0f}")

    # Paint color encoding
    color_means = train_df.groupby("paint_color")["price"].mean()
    params["color_target_encoding"] = color_means.to_dict()
    logger.info(f"  Paint color target encoding: {len(color_means)} categories")

    # Car type encoding
    car_type_means = train_df.groupby("car_type")["price"].mean()
    params["car_type_target_encoding"] = car_type_means.to_dict()
    logger.info(f"  Car type target encoding: {len(car_type_means)} categories")

    # Mileage outliers
    params["mileage_cap"] = train_df["mileage"].quantile(0.99)
    logger.info(f"  Mileage cap (99th percentile): {params['mileage_cap']:,.0f} km")

    # Price outliers
    params["price_floor"] = train_df["price"].quantile(0.01)
    params["price_cap"] = train_df["price"].quantile(0.99)
    logger.info(f"  Price floor (1st percentile): €{params['price_floor']:,.0f}")
    logger.info(f"  Price cap (99th percentile): €{params['price_cap']:,.0f}")

    # Median values for imputation
    params["median_mileage"] = train_df["mileage"].median()
    params["median_engine_power"] = train_df["engine_power"].median()
    logger.info(f"  Median mileage: {params['median_mileage']:,.0f} km")
    logger.info(f"  Median engine power: {params['median_engine_power']:.0f} HP")

    logger.info("=" * 80)

    return params


def apply_transformations(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Apply transformations using learned parameters"""
    logger.info(f"Applying transformations to {len(df):,} rows...")

    # Handle missing values
    df["mileage"] = df["mileage"].fillna(params["median_mileage"])
    df["engine_power"] = df["engine_power"].fillna(params["median_engine_power"])

    # Fix data errors
    df.loc[df["mileage"] < 0, "mileage"] = 0
    df.loc[df["engine_power"] < 0, "engine_power"] = params["median_engine_power"]

    # Cap outliers
    df.loc[df["mileage"] > params["mileage_cap"], "mileage"] = params["mileage_cap"]
    df.loc[df["price"] < params["price_floor"], "price"] = params["price_floor"]
    df.loc[df["price"] > params["price_cap"], "price"] = params["price_cap"]

    # Target encoding for categorical variables
    df["fuel_encoded"] = df["fuel"].map(params["fuel_target_encoding"])
    df["fuel_encoded"] = df["fuel_encoded"].fillna(df["fuel_encoded"].mean())

    df["color_encoded"] = df["paint_color"].map(params["color_target_encoding"])
    df["color_encoded"] = df["color_encoded"].fillna(df["color_encoded"].mean())

    df["car_type_encoded"] = df["car_type"].map(params["car_type_target_encoding"])
    df["car_type_encoded"] = df["car_type_encoded"].fillna(df["car_type_encoded"].mean())

    # Log transform price
    df["price_log"] = np.log1p(df["price"])

    logger.info("  ✅ Transformations applied")

    return df


def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Orchestrate all advanced feature creation - NO LEAKAGE"""
    logger.info("=" * 80)
    logger.info("CREATING ADVANCED FEATURES (NO DATA LEAKAGE)")
    logger.info("=" * 80)

    # 1. BMW segment features (based on series, NOT price)
    df = create_bmw_segment_features(df)

    # 2. Temporal features
    df = create_temporal_features(df)

    # 3. Interaction features
    df = create_interaction_features(df)

    logger.info("=" * 80)
    logger.info(f"✅ Total features created: {len(df.columns)}")
    logger.info("=" * 80)

    return df


def main():
    """Main preprocessing pipeline"""

    logger.info("=" * 80)
    logger.info("BMW PRICING - CORRECTED PREPROCESSING (NO DATA LEAKAGE)")
    logger.info("=" * 80)
    logger.info("CRITICAL FIX: Removed all price-dependent features")
    logger.info("=" * 80)

    # Paths
    raw_path = Path("data/raw/bmw_pricing_challenge.csv")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load raw data
    df = load_raw_data(raw_path)

    # 1b. Check required columns
    logger.info("Checking for required columns...")
    required_cols = {
        "maker_key": None,
        "model_key": None,
        "registration_date": None,
        "fuel": None,
        "paint_color": None,
        "car_type": None,
        "mileage": None,
        "engine_power": "power",
        "price": None,
        "sold_at": None,
    }

    missing_cols = []
    for col, alternative in required_cols.items():
        if col not in df.columns:
            if alternative and alternative in df.columns:
                logger.info(f"  ℹ️  '{col}' not found, using '{alternative}' instead")
                df[col] = df[alternative]
            else:
                missing_cols.append(col)

    if missing_cols:
        logger.error(f"❌ Missing required columns: {missing_cols}")
        raise ValueError(f"Missing required columns: {missing_cols}")

    logger.info("✅ All required columns present")

    # 2. Split data FIRST
    df = split_raw_data(df, train_size=0.6, val_size=0.2, test_size=0.2)

    # 3. Get splits
    train_mask = df["data_split"] == "train"
    val_mask = df["data_split"] == "val"
    test_mask = df["data_split"] == "test"

    # 4. Learn parameters from TRAINING set only
    params = learn_processing_parameters(df[train_mask])

    # 5. Save target encoding mappings for later inference
    target_encodings = {
        "fuel": params["fuel_target_encoding"],
        "color": params["color_target_encoding"],
        "car_type": params["car_type_target_encoding"],
    }
    encodings_path = output_dir / "target_encodings.json"
    with open(encodings_path, "w") as f:
        json.dump(target_encodings, f, indent=2)
    logger.info(f"💾 Target encodings saved to: {encodings_path}")

    # 6. Create advanced features BEFORE transformations
    df = create_advanced_features(df)

    # 7. Apply transformations to all splits
    df_train = apply_transformations(df[train_mask].copy(), params)
    df_val = apply_transformations(df[val_mask].copy(), params)
    df_test = apply_transformations(df[test_mask].copy(), params)

    # 8. Combine splits
    df_clean = pd.concat([df_train, df_val, df_test], axis=0)

    # 9. Save processed data
    output_path = output_dir / "bmw_pricing_clean.csv"
    df_clean.to_csv(output_path, index=False)
    logger.info(f"💾 Processed data saved to: {output_path}")
    logger.info(f"   Shape: {df_clean.shape}")
    logger.info(f"   Features: {len(df_clean.columns)}")

    # 10. Save metadata
    metadata_path = output_dir / "bmw_pricing_clean_metadata.txt"
    with open(metadata_path, "w") as f:
        f.write("BMW Pricing - CORRECTED Preprocessing (NO DATA LEAKAGE)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total samples: {len(df_clean):,}\n")
        f.write(f"Total features: {len(df_clean.columns)}\n\n")
        f.write("Split distribution:\n")
        f.write(f"  Training:   {(df_clean['data_split'] == 'train').sum():,}\n")
        f.write(f"  Validation: {(df_clean['data_split'] == 'val').sum():,}\n")
        f.write(f"  Test:       {(df_clean['data_split'] == 'test').sum():,}\n\n")
        f.write("CRITICAL: All features are observable at prediction time\n")
        f.write("NO price-dependent features (no leakage)\n")

    logger.info(f"📄 Metadata saved to: {metadata_path}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ PREPROCESSING COMPLETE (NO DATA LEAKAGE)")
    logger.info("=" * 80)
    logger.info(f"Output file: {output_path}")
    logger.info(f"Features: {len(df_clean.columns)}")
    logger.info("\n🚀 Next steps:")
    logger.info("   1. Retune: jupyter notebook notebooks/experiments/tuning.ipynb")
    logger.info("   2. Train: python src/models/train.py")
    logger.info("\n Expected: R² 0.80-0.85 (NOT 0.99!), TR 45-55%")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
