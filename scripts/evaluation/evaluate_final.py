"""
Final Test Set Evaluation

⚠️  WARNING: This script should be run ONLY ONCE at the end of development.
    Running it multiple times defeats the purpose of having a test set.

Purpose:
    - Load trained model
    - Evaluate on held-out test set
    - Generate final performance report
    - Compare against business targets

Usage:
    python scripts/evaluation/evaluate_final.py --model-path data/models/checkpoints/rand_forest_v1.pkl
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from src.evaluation.metrics import calculate_all_metrics, check_targets_met

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model_package(model_path: str):
    """
    Load trained model + encoders package

    Args:
        model_path: Path to .pkl file containing model + encoders

    Returns:
        model, encoders dictionary
    """
    logger.info(f"Loading model package from: {model_path}")

    package = joblib.load(model_path)

    # Handle both old format (just model) and new format (dict with model + encoders)
    if isinstance(package, dict):
        model = package["model"]
        encoders = package["encoders"]
        logger.info(f"✅ Loaded model + {len(encoders)} encoders")
    else:
        model = package
        encoders = {}
        logger.warning("⚠️  Old model format - no encoders found")

    return model, encoders


def load_test_data(data_path: str):
    """
    Load ONLY the test set from processed data

    Args:
        data_path: Path to bmw_pricing_clean.csv

    Returns:
        X_test, y_test
    """
    logger.info(f"Loading test data from: {data_path}")

    df = pd.read_csv(data_path)

    # Extract ONLY test set
    if "data_split" not in df.columns:
        raise ValueError("❌ 'data_split' column not found! Did you run preprocess_data.py?")

    test_df = df[df["data_split"] == "test"].copy()

    if len(test_df) == 0:
        raise ValueError("❌ No test samples found!")

    logger.info(f"✅ Loaded {len(test_df):,} test samples")

    # Separate features and target
    X_test = test_df.drop(["price", "price_log", "data_split"], axis=1)
    y_test = test_df["price_log"]

    return X_test, y_test


def apply_encoders(X, encoders):
    """Apply fitted label encoders to dataset"""
    X_encoded = X.copy()

    for col, encoder in encoders.items():
        if col in X_encoded.columns:
            X_encoded[col] = (
                X_encoded[col]
                .astype(str)
                .map(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)
            )

    return X_encoded


def evaluate_final_test(model, encoders, X_test, y_test, model_name: str):
    """
    Perform final evaluation on test set

    Args:
        model: Trained model
        encoders: Fitted label encoders
        X_test: Test features
        y_test: Test targets
        model_name: Model identifier

    Returns:
        Dictionary of test metrics
    """
    logger.info("=" * 80)
    logger.info("🚨 FINAL TEST SET EVALUATION")
    logger.info("=" * 80)
    logger.warning("⚠️  This evaluation should be performed ONLY ONCE!")
    logger.warning("⚠️  Running multiple times invalidates the test set purpose.")
    logger.info("=" * 80)

    # Encode test data
    X_test_encoded = apply_encoders(X_test, encoders)

    # Make predictions
    y_test_pred = model.predict(X_test_encoded)

    # Calculate all metrics
    metrics = calculate_all_metrics(y_test, y_test_pred)
    targets_met = check_targets_met(metrics)

    # Log results with prominent formatting
    logger.info("\n" + "=" * 80)
    logger.info("📊 FINAL TEST SET METRICS")
    logger.info("=" * 80)
    logger.info(f"  MAE:   €{metrics['mae']:,.0f}")
    logger.info(f"  RMSE:  €{metrics['rmse']:,.0f}")
    logger.info(f"  R²:    {metrics['r2']:.3f}")
    logger.info(f"  MAPE:  {metrics['mape']:.1f}%")
    logger.info(f"  TR:    {metrics['tail_rate']:.1f}%")
    logger.info(f"  TC-APE:{metrics['tc_ape']:.1f}%")
    logger.info("=" * 80)

    # Check targets
    logger.info("\n" + "=" * 80)
    logger.info("🎯 TARGET COMPARISON (FINAL)")
    logger.info("=" * 80)
    logger.info(f"  MAE < €2,500:     {'✅' if targets_met['mae_ok'] else '❌'}")
    logger.info(f"  RMSE < €3,000:    {'✅' if targets_met['rmse_ok'] else '❌'}")
    logger.info(f"  R² > 0.85:        {'✅' if targets_met['r2_ok'] else '❌'}")
    logger.info(f"  MAPE < 4.5%:      {'✅' if targets_met['mape_ok'] else '❌'}")
    logger.info(f"  TR < 15%:         {'✅' if targets_met['tr_ok'] else '❌'}")
    logger.info(f"  TC-APE < 6.5%:    {'✅' if targets_met['tc_ape_ok'] else '❌'}")
    logger.info("=" * 80)

    # Overall assessment
    all_targets_met = all(targets_met.values())

    if all_targets_met:
        logger.info("\n🎉 SUCCESS: All business targets met!")
        logger.info("Model is ready for production deployment.")
    else:
        logger.warning("\n⚠️  WARNING: Some targets not met.")
        logger.warning("Consider additional tuning or feature engineering.")

    # Save final report
    save_final_report(metrics, targets_met, model_name, len(X_test))

    return metrics


def save_final_report(metrics, targets_met, model_name, test_size):
    """
    Save final test evaluation report

    This report is marked as FINAL to distinguish from validation reports
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report = {
        "evaluation_type": "FINAL_TEST_SET",
        "timestamp": timestamp,
        "model_name": model_name,
        "test_samples": test_size,
        "warning": "This is a FINAL evaluation on held-out test set. Should be run only ONCE.",
        "metrics": {
            "mae": float(metrics["mae"]),
            "rmse": float(metrics["rmse"]),
            "r2": float(metrics["r2"]),
            "mape": float(metrics["mape"]),
            "tail_rate": float(metrics["tail_rate"]),
            "tc_ape": float(metrics["tc_ape"]),
        },
        "targets": {
            "mae_target": 2500,
            "rmse_target": 3000,
            "r2_target": 0.85,
            "mape_target": 4.5,
            "tr_target": 15.0,
            "tc_ape_target": 6.5,
        },
        "targets_met": targets_met,
        "all_targets_met": all(targets_met.values()),
        "production_ready": all(targets_met.values()),
    }

    # Save to special location
    output_dir = Path("src/evaluation/reports/final")
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / f"FINAL_TEST_{model_name}_{timestamp}.json"

    with open(filepath, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\n📄 Final test report saved to: {filepath}")

    return str(filepath)


def main():
    """Main evaluation pipeline"""

    # Paths
    model_path = "data/models/checkpoints/rand_forest_v1.pkl"
    data_path = "data/processed/bmw_pricing_clean.csv"

    logger.info("\n" + "=" * 80)
    logger.info("🚨 FINAL TEST SET EVALUATION SCRIPT")
    logger.info("=" * 80)
    logger.warning("\n⚠️  CRITICAL WARNING:")
    logger.warning("    This script evaluates the model on the held-out test set.")
    logger.warning("    It should be run ONLY ONCE at the end of development.")
    logger.warning("    Running it multiple times defeats the purpose of having a test set.\n")

    # Confirmation prompt
    response = input("Do you want to continue with FINAL test evaluation? (yes/no): ")

    if response.lower() != "yes":
        logger.info("❌ Evaluation cancelled by user.")
        return

    # 1. Load model
    model, encoders = load_model_package(model_path)

    # 2. Load test data
    X_test, y_test = load_test_data(data_path)

    # 3. Evaluate on test set

    logger.info("\n" + "=" * 80)
    logger.info("✅ FINAL TEST EVALUATION COMPLETE")
    logger.info("=" * 80)
    logger.info("\nNext steps:")
    logger.info("  - Review final report in: src/evaluation/reports/final/")
    logger.info("  - If targets met → Deploy to production")
    logger.info("  - If targets not met → Return to tuning/feature engineering")
    logger.info("\n⚠️  DO NOT run this script again unless you retrain from scratch!")


if __name__ == "__main__":
    main()
