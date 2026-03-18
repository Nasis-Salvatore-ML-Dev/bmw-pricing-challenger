"""
Model Comparison Script

Purpose: Compare Random Forest vs XGBoost performance
Automates the comparison workflow for CI/CD integration

Location: scripts/training/compare_models.py

Usage:
    python scripts/training/compare_models.py
"""

import numpy as np
import pandas as pd
import time
import logging
from pathlib import Path
from datetime import datetime
import json
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.models.train import (
    load_data,
    split_data_by_marker,
    train_model,
    create_label_encoders,
    apply_encoders,
    load_best_hyperparameters
)
from src.models.xgboost_trainer import (
    train_xgboost_model,
    load_best_xgboost_params
)
from src.evaluation.metrics import calculate_all_metrics, check_targets_met

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_and_evaluate_random_forest(X_train, y_train, X_val, y_val, encoders):
    """
    Train and evaluate Random Forest model
    
    Returns:
        metrics dict, training time
    """
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING RANDOM FOREST")
    logger.info("=" * 80)
    
    # Load best hyperparameters
    rf_params = load_best_hyperparameters("rand_forest")
    
    # Train
    start_time = time.time()
    rf_model, _ = train_model(X_train, y_train, hyperparameters=rf_params)
    training_time = time.time() - start_time
    
    # Evaluate
    X_val_enc = apply_encoders(X_val, encoders)
    y_val_pred = rf_model.predict(X_val_enc)
    metrics = calculate_all_metrics(y_val, y_val_pred)
    
    logger.info(f"✅ Random Forest trained in {training_time:.1f}s")
    logger.info(f"   MAE: €{metrics['mae']:,.0f}")
    logger.info(f"   R²:  {metrics['r2']:.3f}")
    
    return metrics, training_time


def train_and_evaluate_xgboost(X_train, y_train, X_val, y_val, encoders):
    """
    Train and evaluate XGBoost model
    
    Returns:
        metrics dict, training time
    """
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING XGBOOST")
    logger.info("=" * 80)
    
    # Load best hyperparameters
    xgb_params = load_best_xgboost_params("xgboost")
    
    # Train
    start_time = time.time()
    xgb_model = train_xgboost_model(
        X_train, y_train,
        X_val, y_val,
        encoders,
        hyperparameters=xgb_params,
        early_stopping_rounds=20
    )
    training_time = time.time() - start_time
    
    # Evaluate
    X_val_enc = apply_encoders(X_val, encoders)
    y_val_pred = xgb_model.predict(X_val_enc)
    metrics = calculate_all_metrics(y_val, y_val_pred)
    
    logger.info(f"✅ XGBoost trained in {training_time:.1f}s")
    logger.info(f"   MAE: €{metrics['mae']:,.0f}")
    logger.info(f"   R²:  {metrics['r2']:.3f}")
    
    return metrics, training_time


def compare_models():
    """
    Main comparison function
    """
    logger.info("=" * 80)
    logger.info("BMW PRICING - MODEL COMPARISON")
    logger.info("=" * 80)
    logger.info("Comparing Random Forest vs XGBoost on validation set")
    
    # 1. Load data
    df_clean, X, y = load_data("data/processed/bmw_pricing_clean.csv")
    
    # 2. Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_by_marker(df_clean, X, y)
    
    # 3. Create encoders (shared between models)
    encoders = create_label_encoders(X_train)
    
    # 4. Train and evaluate both models
    rf_metrics, rf_time = train_and_evaluate_random_forest(
        X_train, y_train, X_val, y_val, encoders
    )
    
    xgb_metrics, xgb_time = train_and_evaluate_xgboost(
        X_train, y_train, X_val, y_val, encoders
    )
    
    # 5. Generate comparison report
    generate_comparison_report(rf_metrics, rf_time, xgb_metrics, xgb_time)
    
    # 6. Determine winner
    determine_winner(rf_metrics, xgb_metrics)


def generate_comparison_report(rf_metrics, rf_time, xgb_metrics, xgb_time):
    """
    Generate detailed comparison report
    """
    logger.info("\n" + "=" * 80)
    logger.info("MODEL PERFORMANCE COMPARISON")
    logger.info("=" * 80)
    
    # Create comparison table
    comparison = {
        'Random Forest': {
            'MAE': f"€{rf_metrics['mae']:,.0f}",
            'RMSE': f"€{rf_metrics['rmse']:,.0f}",
            'R²': f"{rf_metrics['r2']:.3f}",
            'MAPE': f"{rf_metrics['mape']:.1f}%",
            'TR': f"{rf_metrics['tail_rate']:.1f}%",
            'TC-APE': f"{rf_metrics['tc_ape']:.1f}%",
            'Training Time': f"{rf_time:.1f}s"
        },
        'XGBoost': {
            'MAE': f"€{xgb_metrics['mae']:,.0f}",
            'RMSE': f"€{xgb_metrics['rmse']:,.0f}",
            'R²': f"{xgb_metrics['r2']:.3f}",
            'MAPE': f"{xgb_metrics['mape']:.1f}%",
            'TR': f"{xgb_metrics['tail_rate']:.1f}%",
            'TC-APE': f"{xgb_metrics['tc_ape']:.1f}%",
            'Training Time': f"{xgb_time:.1f}s"
        }
    }
    
    # Print formatted table
    metrics_order = ['MAE', 'RMSE', 'R²', 'MAPE', 'TR', 'TC-APE', 'Training Time']
    
    # Header
    logger.info(f"{'Metric':<20} {'Random Forest':>20} {'XGBoost':>20}")
    logger.info("-" * 80)
    
    # Rows
    for metric in metrics_order:
        rf_val = comparison['Random Forest'][metric]
        xgb_val = comparison['XGBoost'][metric]
        logger.info(f"{metric:<20} {rf_val:>20} {xgb_val:>20}")
    
    logger.info("=" * 80)
    
    # Calculate improvements
    mae_improvement = ((rf_metrics['mae'] - xgb_metrics['mae']) / rf_metrics['mae']) * 100
    r2_improvement = ((xgb_metrics['r2'] - rf_metrics['r2']) / rf_metrics['r2']) * 100
    tr_improvement = ((rf_metrics['tail_rate'] - xgb_metrics['tail_rate']) / rf_metrics['tail_rate']) * 100
    
    logger.info("\n" + "=" * 80)
    logger.info("KEY IMPROVEMENTS (XGBoost vs Random Forest)")
    logger.info("=" * 80)
    logger.info(f"  MAE:        {mae_improvement:+.1f}% change")
    logger.info(f"  R²:         {r2_improvement:+.1f}% change")
    logger.info(f"  Tail Rate:  {tr_improvement:+.1f}% change")
    logger.info(f"  Training:   {(xgb_time - rf_time):+.1f}s difference")
    
    # Save comparison to JSON
    save_comparison_json(rf_metrics, rf_time, xgb_metrics, xgb_time)


def determine_winner(rf_metrics, xgb_metrics):
    """
    Determine which model is better based on multiple criteria
    """
    logger.info("\n" + "=" * 80)
    logger.info("WINNER DETERMINATION")
    logger.info("=" * 80)
    
    # Score based on multiple metrics
    rf_score = 0
    xgb_score = 0
    
    # MAE (lower is better)
    if xgb_metrics['mae'] < rf_metrics['mae']:
        xgb_score += 2
        logger.info("  MAE:       XGBoost wins ✅")
    else:
        rf_score += 2
        logger.info("  MAE:       Random Forest wins ✅")
    
    # R² (higher is better)
    if xgb_metrics['r2'] > rf_metrics['r2']:
        xgb_score += 2
        logger.info("  R²:        XGBoost wins ✅")
    else:
        rf_score += 2
        logger.info("  R²:        Random Forest wins ✅")
    
    # Tail Rate (lower is better) - most important for business
    if xgb_metrics['tail_rate'] < rf_metrics['tail_rate']:
        xgb_score += 3  # Higher weight
        logger.info("  Tail Rate: XGBoost wins ✅ (3 points)")
    else:
        rf_score += 3
        logger.info("  Tail Rate: Random Forest wins ✅ (3 points)")
    
    # MAPE (lower is better)
    if xgb_metrics['mape'] < rf_metrics['mape']:
        xgb_score += 1
        logger.info("  MAPE:      XGBoost wins ✅")
    else:
        rf_score += 1
        logger.info("  MAPE:      Random Forest wins ✅")
    
    logger.info("\n" + "-" * 80)
    logger.info(f"Final Score: Random Forest {rf_score} - XGBoost {xgb_score}")
    logger.info("-" * 80)
    
    if xgb_score > rf_score:
        logger.info("\n🏆 WINNER: XGBoost")
        logger.info(f"   XGBoost is the better model for BMW pricing")
        logger.info(f"   MAE improvement: {((rf_metrics['mae'] - xgb_metrics['mae']) / rf_metrics['mae']) * 100:.1f}%")
    elif rf_score > xgb_score:
        logger.info("\n🏆 WINNER: Random Forest")
        logger.info(f"   Random Forest is the better model for BMW pricing")
    else:
        logger.info("\n🤝 TIE: Both models perform equally")
        logger.info(f"   Consider ensemble or model selection based on deployment constraints")
    
    logger.info("=" * 80)


def save_comparison_json(rf_metrics, rf_time, xgb_metrics, xgb_time):
    """
    Save comparison results to JSON file
    """
    comparison_data = {
        "timestamp": datetime.now().isoformat(),
        "random_forest": {
            "mae": float(rf_metrics['mae']),
            "rmse": float(rf_metrics['rmse']),
            "r2": float(rf_metrics['r2']),
            "mape": float(rf_metrics['mape']),
            "tail_rate": float(rf_metrics['tail_rate']),
            "tc_ape": float(rf_metrics['tc_ape']),
            "training_time_seconds": float(rf_time)
        },
        "xgboost": {
            "mae": float(xgb_metrics['mae']),
            "rmse": float(xgb_metrics['rmse']),
            "r2": float(xgb_metrics['r2']),
            "mape": float(xgb_metrics['mape']),
            "tail_rate": float(xgb_metrics['tail_rate']),
            "tc_ape": float(xgb_metrics['tc_ape']),
            "training_time_seconds": float(xgb_time)
        },
        "improvements": {
            "mae_improvement_pct": float(((rf_metrics['mae'] - xgb_metrics['mae']) / rf_metrics['mae']) * 100),
            "r2_improvement_pct": float(((xgb_metrics['r2'] - rf_metrics['r2']) / rf_metrics['r2']) * 100),
            "tail_rate_improvement_pct": float(((rf_metrics['tail_rate'] - xgb_metrics['tail_rate']) / rf_metrics['tail_rate']) * 100)
        }
    }
    
    # Create directory
    output_dir = Path("reports/model_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = output_dir / f"rf_vs_xgboost_{timestamp}.json"
    
    with open(filepath, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    logger.info(f"\n📊 Comparison report saved to: {filepath}")


if __name__ == "__main__":
    compare_models()