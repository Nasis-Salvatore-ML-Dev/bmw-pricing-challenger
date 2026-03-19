"""
Model hyperparameters configuration.
Separating config from code for better visibility and maintainability.
"""

# Random Forest configurations
RANDOM_FOREST_CONFIGS = {
    "baseline": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1,
        "description": "Baseline Random Forest - conservative",
    },
    "default": {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42,
        "n_jobs": -1,
        "description": "Default scikit-learn Random Forest",
    },
    "deep": {
        "n_estimators": 200,
        "max_depth": 30,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42,
        "n_jobs": -1,
        "description": "Deep trees - may overfit",
    },
    "shallow": {
        "n_estimators": 100,
        "max_depth": 5,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "random_state": 42,
        "n_jobs": -1,
        "description": "Shallow trees - prevents overfitting",
    },
    "balanced": {
        "n_estimators": 150,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1,
        "description": "Balanced configuration",
    },
}

# XGBoost configurations (for future use)
XGBOOST_CONFIGS = {
    "baseline": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "description": "Baseline XGBoost",
    },
    "conservative": {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "random_state": 42,
        "description": "Conservative - prevents overfitting",
    },
}

# Model selection mapping
MODEL_CONFIGS = {"random_forest": RANDOM_FOREST_CONFIGS, "xgboost": XGBOOST_CONFIGS}


def get_model_config(model_type: str = "random_forest", config_name: str = "baseline") -> dict:
    """
    Get model configuration by type and name.

    Args:
        model_type: "random_forest" or "xgboost"
        config_name: Configuration name (baseline, default, deep, etc.)

    Returns:
        Dictionary of hyperparameters
    """
    try:
        return MODEL_CONFIGS[model_type][config_name]
    except KeyError:
        raise ValueError(f"Configuration {config_name} not found for {model_type}")
