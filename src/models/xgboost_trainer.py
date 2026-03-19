"""
XGBoost Training Module

Purpose: Train XGBoost model for BMW pricing prediction
Gradient boosting typically outperforms Random Forest for regression tasks

Location: src/models/xgboost_trainer.py
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ML Libraries
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from config.visualization_config import IMAGES_PATH, save_fig

# Internal modules
from src.evaluation.metrics import calculate_all_metrics, check_targets_met

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_best_xgboost_params(model_name: str = "xgboost") -> dict:
    """
    Load best XGBoost hyperparameters from experimentation

    Args:
        model_name: Name of the model configuration file

    Returns:
        Dictionary of hyperparameters
    """
    default_params = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0,
        "reg_alpha": 0,
        "reg_lambda": 1,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",  # Faster training
        "objective": "reg:squarederror",
    }

    try:
        config_path = (
            Path(__file__).parent.parent.parent
            / "config"
            / "models"
            / f"best_{model_name}_params.json"
        )

        with open(config_path) as f:
            params = json.load(f)

        # Ensure required params are set
        if "n_jobs" not in params:
            params["n_jobs"] = -1
        if "tree_method" not in params:
            params["tree_method"] = "hist"
        if "objective" not in params:
            params["objective"] = "reg:squarederror"

        logger.info(f"✅ Loaded hyperparameters from {config_path}")
        return params

    except FileNotFoundError:
        logger.warning("⚠️  Config file not found. Using defaults.")
        return default_params


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    encoders: dict[str, LabelEncoder],
    hyperparameters: dict | None = None,
    early_stopping_rounds: int = 20,
) -> xgb.XGBRegressor:
    """
    Train XGBoost model with early stopping on validation set

    Args:
        X_train: Training features
        y_train: Training targets (price_log)
        X_val: Validation features
        y_val: Validation targets
        encoders: Fitted label encoders
        hyperparameters: Model hyperparameters
        early_stopping_rounds: Stop if no improvement for N rounds

    Returns:
        Trained XGBoost model
    """
    if hyperparameters is None:
        hyperparameters = load_best_xgboost_params()

    # Import encoding functions
    from src.models.train import apply_encoders

    # Encode features
    X_train_enc = apply_encoders(X_train, encoders)
    X_val_enc = apply_encoders(X_val, encoders)

    logger.info("=" * 50)
    logger.info("TRAINING XGBOOST MODEL")
    logger.info("=" * 50)
    logger.info(f"Hyperparameters: {hyperparameters}")

    # Initialize model
    model = xgb.XGBRegressor(**hyperparameters)

    # Train with early stopping
    logger.info(f"Training on {len(X_train):,} samples with early stopping...")

    model.fit(X_train_enc, y_train, eval_set=[(X_val_enc, y_val)], verbose=False)

    # Get best iteration
    best_iteration = (
        model.best_iteration if hasattr(model, "best_iteration") else model.n_estimators
    )

    logger.info(f"✅ Model trained on {len(X_train):,} samples")
    logger.info(f"   Best iteration: {best_iteration}")
    logger.info(f"   Total trees: {model.n_estimators}")

    return model


def evaluate_xgboost_model(
    model: xgb.XGBRegressor,
    encoders: dict[str, LabelEncoder],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    model_name: str = "xgboost",
    save_report: bool = True,
) -> dict[str, float]:
    """
    Evaluate XGBoost model on validation set

    Args:
        model: Trained XGBoost model
        encoders: Fitted label encoders
        X_val: Validation features
        y_val: Validation targets
        X_train: Training features (for reporting)
        X_test: Test features (for reporting)
        model_name: Model identifier
        save_report: Whether to save JSON report

    Returns:
        Dictionary of all metrics
    """
    from src.models.train import apply_encoders

    # Encode validation data
    X_val_enc = apply_encoders(X_val, encoders)

    # Make predictions
    y_val_pred = model.predict(X_val_enc)

    # Calculate all metrics
    metrics = calculate_all_metrics(y_val, y_val_pred)
    targets_met = check_targets_met(metrics)

    # Log results
    logger.info("=" * 50)
    logger.info("VALIDATION METRICS")
    logger.info("=" * 50)
    logger.info(f"  MAE:   €{metrics['mae']:,.0f}")
    logger.info(f"  RMSE:  €{metrics['rmse']:,.0f}")
    logger.info(f"  R²:    {metrics['r2']:.3f}")
    logger.info(f"  MAPE:  {metrics['mape']:.1f}%")
    logger.info(f"  TR:    {metrics['tail_rate']:.1f}%")
    logger.info(f"  TC-APE:{metrics['tc_ape']:.1f}%")
    logger.info("=" * 50)

    # Check targets
    logger.info("TARGET COMPARISON")
    logger.info("=" * 50)
    logger.info(f"  MAE < €2,500:     {'✅' if targets_met['mae_ok'] else '❌'}")
    logger.info(f"  RMSE < €3,000:    {'✅' if targets_met['rmse_ok'] else '❌'}")
    logger.info(f"  R² > 0.85:        {'✅' if targets_met['r2_ok'] else '❌'}")
    logger.info(f"  MAPE < 4.5%:      {'✅' if targets_met['mape_ok'] else '❌'}")
    logger.info(f"  TR < 15%:         {'✅' if targets_met['tr_ok'] else '❌'}")
    logger.info(f"  TC-APE < 6.5%:    {'✅' if targets_met['tc_ape_ok'] else '❌'}")
    logger.info("=" * 50)

    # Save report
    if save_report:
        save_xgboost_metrics_report(
            metrics=metrics,
            targets_met=targets_met,
            model=model,
            model_name=model_name,
            dataset_sizes={"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        )

    return metrics


def save_xgboost_metrics_report(
    metrics: dict[str, float],
    targets_met: dict[str, bool],
    model: xgb.XGBRegressor,
    model_name: str,
    dataset_sizes: dict[str, int],
) -> str:
    """
    Save XGBoost metrics report to JSON

    Args:
        metrics: Calculated metrics
        targets_met: Target achievement flags
        model: Trained XGBoost model
        model_name: Model identifier
        dataset_sizes: Dictionary of dataset sizes

    Returns:
        Path to saved report
    """
    # Get model parameters
    model_params = model.get_params()

    report = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "model_type": "XGBoost",
        "dataset": {
            "training_samples": dataset_sizes["train"],
            "validation_samples": dataset_sizes["val"],
            "test_samples": dataset_sizes["test"],
        },
        "model_parameters": {
            "n_estimators": model_params["n_estimators"],
            "max_depth": model_params["max_depth"],
            "learning_rate": model_params["learning_rate"],
            "subsample": model_params["subsample"],
            "colsample_bytree": model_params["colsample_bytree"],
            "gamma": model_params["gamma"],
            "reg_alpha": model_params["reg_alpha"],
            "reg_lambda": model_params["reg_lambda"],
            "random_state": model_params["random_state"],
        },
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
        "targets_met": {k: bool(v) for k, v in targets_met.items()},
    }

    # Create directory
    os.makedirs("src/evaluation/reports/metrics", exist_ok=True)

    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"src/evaluation/reports/metrics/{model_name}_{timestamp}.json"

    with open(filepath, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"📊 Metrics report saved to: {filepath}")

    return filepath


def get_xgboost_feature_importance(
    model: xgb.XGBRegressor,
    feature_names: list,
    importance_type: str = "weight",
    top_n: int = 10,
    save_plot: bool = True,
) -> dict[str, float]:
    """
    Extract and visualize XGBoost feature importances

    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        importance_type: 'weight' (default), 'gain', or 'cover'
        top_n: Number of top features to return
        save_plot: Whether to save visualization

    Returns:
        Dictionary of {feature_name: importance}
    """
    # Get importances
    if importance_type == "weight":
        importances = model.feature_importances_
    else:
        importance_dict = model.get_booster().get_score(importance_type=importance_type)
        # Map to feature names and fill missing with 0
        importances = np.array([importance_dict.get(f"f{i}", 0) for i in range(len(feature_names))])

    # Create sorted list
    feature_importance = list(zip(feature_names, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    # Get top N
    top_features = feature_importance[:top_n]

    # Log results
    logger.info("=" * 50)
    logger.info(f"TOP {top_n} FEATURES BY IMPORTANCE ({importance_type.upper()})")
    logger.info("=" * 50)
    for i, (feature, importance) in enumerate(top_features, 1):
        logger.info(f"{i:2d}. {feature:25s} {importance:.4f}")
    logger.info("=" * 50)

    # Convert to dict
    importance_dict = {feature: float(importance) for feature, importance in top_features}

    # Save to JSON
    os.makedirs("reports/features", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f"reports/features/xgboost_feature_importance_{timestamp}.json"

    with open(json_path, "w") as f:
        json.dump(importance_dict, f, indent=2)

    logger.info(f"💾 Feature importance saved to: {json_path}")

    # Create plot
    if save_plot:
        create_xgboost_feature_plot(top_features, top_n, importance_type)

    return importance_dict


def create_xgboost_feature_plot(top_features: list, top_n: int, importance_type: str):
    """
    Create horizontal bar plot of XGBoost feature importances

    Args:
        top_features: List of (feature_name, importance) tuples
        top_n: Number of features
        importance_type: Type of importance metric
    """
    features = [f[0] for f in top_features]
    importances = [f[1] for f in top_features]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Horizontal bars with color gradient
    bars = ax.barh(range(len(features)), importances, align="center")

    # Color gradient (XGBoost uses orange theme)
    for i, (bar, imp) in enumerate(zip(bars, importances)):
        bar.set_color(plt.cm.Oranges(0.4 + 0.6 * (imp / max(importances))))  # pylint: disable=no-member

    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel(f"Importance Score ({importance_type})")
    ax.set_title(f"XGBoost: Top {top_n} Most Important Features")
    ax.invert_yaxis()

    # Value labels
    for i, (bar, imp) in enumerate(zip(bars, importances)):
        ax.text(
            imp + 0.005, bar.get_y() + bar.get_height() / 2, f"{imp:.3f}", va="center", fontsize=10
        )

    ax.grid(axis="x", alpha=0.3)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_fig(f"xgboost_feature_importance_top{top_n}_{timestamp}")
    plt.close()

    logger.info(f"📊 Feature importance plot saved to: {IMAGES_PATH}")


def save_xgboost_model(
    model: xgb.XGBRegressor,
    encoders: dict[str, LabelEncoder],
    model_name: str = "xgboost",
    version: str = "v1",
) -> str:
    """
    Save trained XGBoost model and encoders together

    Args:
        model: Trained XGBoost model
        encoders: Fitted label encoders
        model_name: Model identifier
        version: Version string

    Returns:
        Path to saved model file
    """
    save_dir = Path("data/models/checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save both model and encoders together
    model_package = {
        "model": model,
        "encoders": encoders,
        "model_type": "XGBoost",
        "timestamp": datetime.now().isoformat(),
        "version": version,
    }

    filepath = save_dir / f"{model_name}_{version}.pkl"
    joblib.dump(model_package, filepath)

    logger.info(f"💾 XGBoost model package saved to: {filepath}")
    logger.info(f"   Contains: model + {len(encoders)} encoders")

    return str(filepath)
