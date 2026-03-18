"""
Model Training Module

Purpose: Train Random Forest model for BMW pricing prediction
Follows Carmack's principles: incremental progress, immediate feedback, reusable components
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional
import logging
import os
import joblib
import json
from datetime import datetime
from pathlib import Path

# ML Libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Internal modules
from src.evaluation.metrics import calculate_all_metrics, check_targets_met
from config.visualization_config import save_fig, IMAGES_PATH
from src.evaluation.residual_analysis import perform_residual_analysis

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(input_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load preprocessed data and respect train/test split markers

    Args:
        input_path: Path to bmw_pricing_clean.csv

    Returns:
        - Full dataframe
        - Feature matrix X
        - Target vector y (price_log)
    """
    logger.info(f"Loading clean dataset from {input_path}")
    df = pd.read_csv(input_path)

    # Check if data_split column exists (from preprocessing)
    if 'data_split' not in df.columns:
        logger.warning("⚠️  'data_split' column missing. Data may not be properly split.")

    # Separate features and target
    X = df.drop(["price", "price_log"], axis=1)
    y = df["price_log"]

    logger.info(f"✅ Loaded {len(df):,} rows, {X.shape[1]} features")

    return df, X, y


def split_data_by_marker(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data using existing 'data_split' column to prevent data leakage

    Args:
        df: Full dataframe with 'data_split' column
        X: Feature matrix
        y: Target vector

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    if 'data_split' in df.columns:
        logger.info("Using existing data_split markers from preprocessing")

        # Use existing train/test split
        train_mask = df['data_split'] == 'train'
        test_mask = df['data_split'] == 'test'

        # Further split train into train/val (80/20 of training data)
        train_indices = df[train_mask].index
        val_size = int(len(train_indices) * 0.2)

        # Take last 20% of training data as validation
        val_indices = train_indices[-val_size:]
        train_indices = train_indices[:-val_size]

        X_train = X.loc[train_indices].drop('data_split', axis=1, errors='ignore')
        X_val = X.loc[val_indices].drop('data_split', axis=1, errors='ignore')
        X_test = X.loc[test_mask].drop('data_split', axis=1, errors='ignore')

        y_train = y.loc[train_indices]
        y_val = y.loc[val_indices]
        y_test = y.loc[test_mask]

    else:
        # Fallback: manual split (should not happen if preprocessing is correct)
        logger.warning("⚠️  Using manual split - preprocessing may be incomplete")
        from sklearn.model_selection import train_test_split

        # Drop data_split if it exists
        X_clean = X.drop('data_split', axis=1, errors='ignore')

        X_train, X_temp, y_train, y_temp = train_test_split(
            X_clean, y, train_size=0.6, random_state=42, shuffle=True
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, train_size=0.5, random_state=42, shuffle=True
        )

    logger.info("=" * 50)
    logger.info("DATA SPLIT RESULTS")
    logger.info("=" * 50)
    logger.info(f"Training set:    {len(X_train):5d} rows ({len(X_train)/len(X):.1%})")
    logger.info(f"Validation set:  {len(X_val):5d} rows ({len(X_val)/len(X):.1%})")
    logger.info(f"Test set:        {len(X_test):5d} rows ({len(X_test)/len(X):.1%})")
    logger.info("=" * 50)

    return X_train, X_val, X_test, y_train, y_val, y_test


def create_label_encoders(X_train: pd.DataFrame) -> Dict[str, LabelEncoder]:
    """
    Create and fit label encoders for categorical columns ONLY on training data

    This prevents data leakage by learning encoding mappings only from training set

    Args:
        X_train: Training feature matrix

    Returns:
        Dictionary mapping column names to fitted LabelEncoders
    """
    encoders = {}
    categorical_cols = X_train.select_dtypes(include=['object']).columns

    if len(categorical_cols) > 0:
        logger.info(f"Creating encoders for: {list(categorical_cols)}")

        for col in categorical_cols:
            encoder = LabelEncoder()

            # Encoding the categorical values
            encoder.fit(X_train[col].astype(str))
            encoders[col] = encoder
            logger.info(f"  ✅ {col}: {len(encoder.classes_)} categories")

    return encoders


def apply_encoders(
    X: pd.DataFrame,
    encoders: Dict[str, LabelEncoder]
) -> pd.DataFrame:
    """
    Purpose:
        To encode the categorical values into integers

    Args:
        X: Feature matrix to encode
        encoders: Dictionary of fitted LabelEncoders

    Returns:
        Encoded feature matrix
    """
    X_encoded = X.copy()
    for col, encoder in encoders.items():
        if col in X_encoded.columns:
            # Handle unseen categories by mapping to -1
            X_encoded[col] = X_encoded[col].astype(str).map(
                lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
            )

    return X_encoded


def load_best_hyperparameters(model_name: str = "rand_forest") -> dict:
    """
    Load best hyperparameters found during experimentation

    Args:
        model_name: Name of the model configuration file

    Returns:
        Dictionary of hyperparameters
    """
    default_params = {
        'n_estimators': 100,
        'max_depth': 20,
        'min_samples_split': 5,
        'random_state': 42,
        'n_jobs': -1
    }

    try:
        config_path = Path(__file__).parent.parent.parent / "config" / "models" / f"best_{model_name}_params.json"

        with open(config_path, 'r') as f:
            params = json.load(f)

        # Ensure n_jobs is set
        if 'n_jobs' not in params:
            params['n_jobs'] = -1

        logger.info(f"✅ Loaded hyperparameters from {config_path}")
        return params

    except FileNotFoundError:
        logger.warning(f"⚠️  {config_path} not found. Using defaults.")
        return default_params


def load_target_encodings() -> Dict[str, Dict[str, float]]:
    """
    Load target encoding mappings saved during preprocessing
    """
    encodings_path = Path("data/processed/target_encodings.json")
    if not encodings_path.exists():
        logger.warning("⚠️  Target encodings file not found. Inference may fail.")
        return {}
    with open(encodings_path, 'r') as f:
        return json.load(f)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    hyperparameters: Optional[dict] = None
) -> Tuple[RandomForestRegressor, Dict[str, LabelEncoder], Dict[str, Dict[str, float]]]:
    """
    Train Random Forest model with categorical encoding

    Args:
        X_train: Training features
        y_train: Training targets (price_log)
        hyperparameters: Model hyperparameters (optional)

    Returns:
        - Trained model
        - Dictionary of fitted encoders (for use on val/test sets)
        - Target encoding mappings (for inference)
    """
    if hyperparameters is None:
        hyperparameters = load_best_hyperparameters()

    # Create and fit encoders on training data ONLY
    encoders = create_label_encoders(X_train)

    # Load target encodings (these were already used during preprocessing)
    target_encodings = load_target_encodings()

    # Encode training data
    X_train_encoded = apply_encoders(X_train, encoders)

    # Initialize and train model
    logger.info("=" * 50)
    logger.info("TRAINING RANDOM FOREST MODEL")
    logger.info("=" * 50)
    logger.info(f"Hyperparameters: {hyperparameters}")

    model = RandomForestRegressor(**hyperparameters)
    model.fit(X_train_encoded, y_train)

    logger.info(f"✅ Model trained on {len(X_train):,} samples")

    return model, encoders, target_encodings


def evaluate_model(
    model: RandomForestRegressor,
    encoders: Dict[str, LabelEncoder],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    model_name: str = "rand_forest",
    save_report: bool = True
) -> Dict[str, float]:
    """
    Evaluate model on validation set and generate comprehensive report

    Args:
        model: Trained model
        encoders: Fitted label encoders
        X_val: Validation features
        y_val: Validation targets
        X_train: Training features (for reporting only)
        X_test: Test features (for reporting only)
        model_name: Model identifier
        save_report: Whether to save JSON report

    Returns:
        Dictionary of all metrics
    """
    # Encode validation data using training encoders
    X_val_encoded = apply_encoders(X_val, encoders)

    # Make predictions
    y_val_pred = model.predict(X_val_encoded)

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

    # Save comprehensive report
    if save_report:
        save_metrics_report(
            metrics=metrics,
            targets_met=targets_met,
            model=model,
            model_name=model_name,
            dataset_sizes={
                'train': len(X_train),
                'val': len(X_val),
                'test': len(X_test)
            }
        )

    return metrics


def save_metrics_report(
    metrics: Dict[str, float],
    targets_met: Dict[str, bool],
    model: RandomForestRegressor,
    model_name: str,
    dataset_sizes: Dict[str, int]
) -> str:
    """
    Save comprehensive metrics report to JSON

    Args:
        metrics: Calculated metrics
        targets_met: Target achievement flags
        model: Trained model
        model_name: Model identifier
        dataset_sizes: Dictionary of dataset sizes

    Returns:
        Path to saved report
    """
    model_params = model.get_params()

    report = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "dataset": {
            "training_samples": dataset_sizes['train'],
            "validation_samples": dataset_sizes['val'],
            "test_samples": dataset_sizes['test']
        },
        "model_parameters": {
            "n_estimators": model_params["n_estimators"],
            "max_depth": model_params["max_depth"],
            "min_samples_split": model_params["min_samples_split"],
            "random_state": model_params["random_state"]
        },
        "metrics": {
            "mae": float(metrics['mae']),
            "rmse": float(metrics['rmse']),
            "r2": float(metrics['r2']),
            "mape": float(metrics['mape']),
            "tail_rate": float(metrics['tail_rate']),
            "tc_ape": float(metrics['tc_ape'])
        },
        "targets": {
            "mae_target": 2500,
            "rmse_target": 3000,
            "r2_target": 0.85,
            "mape_target": 4.5,
            "tr_target": 15.0,
            "tc_ape_target": 6.5
        },
        "targets_met": {k: bool(v) for k, v in targets_met.items()}  # Convert numpy bool_ to Python bool
    }

    # Create directory
    os.makedirs("src/evaluation/reports/metrics", exist_ok=True)

    # Save report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = f"src/evaluation/reports/metrics/{model_name}_{timestamp}.json"

    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"📊 Metrics report saved to: {filepath}")

    return filepath


def get_feature_importance(
    model: RandomForestRegressor,
    feature_names: list,
    top_n: int = 10,
    save_plot: bool = True
) -> Dict[str, float]:
    """
    Extract and visualize feature importances

    Args:
        model: Trained Random Forest model
        feature_names: List of feature names
        top_n: Number of top features to return
        save_plot: Whether to save visualization

    Returns:
        Dictionary of {feature_name: importance}
    """
    importances = model.feature_importances_

    # Create sorted list of (feature, importance)
    feature_importance = list(zip(feature_names, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    # Get top N
    top_features = feature_importance[:top_n]

    # Log results
    logger.info("=" * 50)
    logger.info(f"TOP {top_n} FEATURES BY IMPORTANCE")
    logger.info("=" * 50)
    for i, (feature, importance) in enumerate(top_features, 1):
        logger.info(f"{i:2d}. {feature:25s} {importance:.4f}")
    logger.info("=" * 50)

    # Convert to dict
    importance_dict = {feature: float(importance) for feature, importance in top_features}

    # Save to JSON
    os.makedirs("reports/features", exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = f"reports/features/feature_importance_{timestamp}.json"

    with open(json_path, 'w') as f:
        json.dump(importance_dict, f, indent=2)

    logger.info(f"💾 Feature importance saved to: {json_path}")

    # Create plot
    if save_plot:
        create_feature_importance_plot(top_features, top_n)

    return importance_dict


def create_feature_importance_plot(top_features: list, top_n: int):
    """
    Create horizontal bar plot of feature importances

    Args:
        top_features: List of (feature_name, importance) tuples
        top_n: Number of features
    """
    features = [f[0] for f in top_features]
    importances = [f[1] for f in top_features]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Horizontal bars with color gradient
    bars = ax.barh(range(len(features)), importances, align='center')

    for i, (bar, imp) in enumerate(zip(bars, importances)):
        bar.set_color(plt.cm.Blues(0.3 + 0.7 * (imp / max(importances))))

    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top {top_n} Most Important Features')
    ax.invert_yaxis()

    # Value labels
    for i, (bar, imp) in enumerate(zip(bars, importances)):
        ax.text(imp + 0.005, bar.get_y() + bar.get_height()/2,
                f'{imp:.3f}', va='center', fontsize=10)

    ax.grid(axis='x', alpha=0.3)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_fig(f'feature_importance_top{top_n}_{timestamp}')
    plt.close()

    logger.info(f"📊 Feature importance plot saved to: {IMAGES_PATH}")


def save_model(
    model: RandomForestRegressor,
    encoders: Dict[str, LabelEncoder],
    target_encodings: Dict[str, Dict[str, float]],
    model_name: str = "rand_forest",
    version: str = "v1"
) -> str:
    """
    Save trained model AND encoders AND target encodings together

    This ensures the same encoding is used in production

    Args:
        model: Trained model
        encoders: Fitted label encoders
        target_encodings: Target encoding mappings for fuel, color, car_type
        model_name: Model identifier
        version: Version string

    Returns:
        Path to saved model file
    """
    save_dir = Path("data/models/checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save both model and encoders together
    model_package = {
        'model': model,
        'encoders': encoders,
        'target_encodings': target_encodings,
        'timestamp': datetime.now().isoformat(),
        'version': version
    }

    filepath = save_dir / f"{model_name}_{version}.pkl"
    joblib.dump(model_package, filepath)

    logger.info(f"💾 Model package saved to: {filepath}")
    logger.info(f"   Contains: model + {len(encoders)} encoders + target encodings")

    return str(filepath)


if __name__ == "__main__":
    # 1. Load data
    df_clean, X, y = load_data("data/processed/bmw_pricing_clean.csv")

    # 2. Split data using existing markers
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_by_marker(df_clean, X, y)

    # 3. Load best hyperparameters from experimentation
    best_params = load_best_hyperparameters("rand_forest")

    # 4. Train model (returns model, encoders, and target encodings)
    trained_model, encoders, target_encodings = train_model(X_train, y_train, hyperparameters=best_params)

    # 5. Evaluate on validation set
    metrics = evaluate_model(
        model=trained_model,
        encoders=encoders,
        X_val=X_val,
        y_val=y_val,
        X_train=X_train,
        X_test=X_test,
        model_name="rand_forest",
        save_report=True
    )

    # Perform Residual Analysis (diagnose high tail rate)
    logger.info("\n" + "=" * 50)
    logger.info("PERFORMING RESIDUAL ANALYSIS")
    logger.info("=" * 50)

    # Make predictions on validation set for analysis
    X_val_encoded = apply_encoders(X_val, encoders)
    y_val_pred = trained_model.predict(X_val_encoded)

    # Run residual analysis
    residual_df = perform_residual_analysis(
        y_true=y_val.values,
        y_pred=y_val_pred,
        X_features=X_val,
        model_name="rand_forest_validation",
        save_plots=True
    )

    logger.info("✅ Residual analysis complete")
    logger.info("   Check: reports/images/residuals/")
    logger.info("   Check: reports/residuals/")

    # 6. Extract feature importances
    feature_names = X_train.columns.tolist()
    top_features = get_feature_importance(trained_model, feature_names, top_n=10)

    # 7. Save model package (model + encoders + target encodings)
    model_path = save_model(trained_model, encoders, target_encodings, model_name="rand_forest", version="v1")

    logger.info("=" * 50)
    logger.info("✅ TRAINING PIPELINE COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Validation MAE: €{metrics['mae']:,.0f}")
    logger.info(f"All targets met: {all(check_targets_met(metrics).values())}")