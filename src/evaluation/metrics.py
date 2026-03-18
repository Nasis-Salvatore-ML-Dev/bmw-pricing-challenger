"""
Evaluation Metrics Module

Purpose: Reusable metric calculations aligned with business KPIs
All functions are pure: they take predictions and targets, return metrics.
No side effects, no logging in metric calculations.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error
    
    Args:
        y_true: Actual target values (log-transformed prices)
        y_pred: Predicted values (log-transformed prices)
    
    Returns:
        MAE in euros (after inverse transform)
    """
    # Convert log prices back to original scale for interpretable MAE
    y_true_euro = np.exp(y_true)
    y_pred_euro = np.exp(y_pred)
    
    return mean_absolute_error(y_true_euro, y_pred_euro)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error
    
    Args:
        y_true: Actual target values (log-transformed prices)
        y_pred: Predicted values (log-transformed prices)
    
    Returns:
        RMSE in euros (after inverse transform)
    """
    # Convert log prices back to original scale
    y_true_euro = np.exp(y_true)
    y_pred_euro = np.exp(y_pred)
    
    mse = mean_squared_error(y_true_euro, y_pred_euro)
    return np.sqrt(mse)


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R² (Coefficient of Determination)
    
    Args:
        y_true: Actual target values (log-transformed prices)
        y_pred: Predicted values (log-transformed prices)
    
    Returns:
        R² score (0 to 1, higher is better)
    """
    return r2_score(y_true, y_pred)


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error
    
    Args:
        y_true: Actual target values (log-transformed prices)
        y_pred: Predicted values (log-transformed prices)
    
    Returns:
        MAPE as percentage (0-100)
    """
    # Convert to original price scale
    y_true_euro = np.exp(y_true)
    y_pred_euro = np.exp(y_pred)
    
    # Calculate percentage errors
    ape = np.abs((y_true_euro - y_pred_euro) / y_true_euro) * 100
    
    return np.mean(ape)


def calculate_business_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate business-specific metrics: Tail Rate (TR) and TC-APE
    
    Business Context:
    - TR (Tail Rate): % of cars with pricing error > 5%
    - TC-APE (Tail-Conditional APE): Average error among mispriced cars
    
    Target: TR < 15%, TC-APE < 6.5%
    
    Args:
        y_true: Actual target values (log-transformed prices)
        y_pred: Predicted values (log-transformed prices)
    
    Returns:
        Dictionary with:
            - tail_rate: Percentage of cars with >5% error
            - tc_ape: Average percentage error among tail samples
            - total_samples: Total number of predictions
            - tail_samples: Number of samples in tail (error > 5%)
    """
    # Convert to original price scale
    y_true_euro = np.exp(y_true)
    y_pred_euro = np.exp(y_pred)
    
    # Calculate Absolute Percentage Error for each prediction
    ape = np.abs((y_true_euro - y_pred_euro) / y_true_euro) * 100
    
    # Tail Rate: percentage of samples with error > 5%
    tail_mask = ape > 5.0
    tail_rate = np.mean(tail_mask) * 100
    
    # TC-APE: average error among tail samples only
    tail_errors = ape[tail_mask]
    tc_ape = np.mean(tail_errors) if len(tail_errors) > 0 else 0.0
    
    return {
        'tail_rate': float(tail_rate),
        'tc_ape': float(tc_ape),
        'total_samples': int(len(y_true)),
        'tail_samples': int(np.sum(tail_mask))
    }


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Convenience function to calculate ALL metrics at once
    
    Args:
        y_true: Actual target values (log-transformed prices)
        y_pred: Predicted values (log-transformed prices)
    
    Returns:
        Dictionary containing all metric values
    """
    business_metrics = calculate_business_metrics(y_true, y_pred)
    
    return {
        'mae': calculate_mae(y_true, y_pred),
        'rmse': calculate_rmse(y_true, y_pred),
        'r2': calculate_r2(y_true, y_pred),
        'mape': calculate_mape(y_true, y_pred),
        'tail_rate': business_metrics['tail_rate'],
        'tc_ape': business_metrics['tc_ape'],
        'total_samples': business_metrics['total_samples'],
        'tail_samples': business_metrics['tail_samples']
    }


def check_targets_met(metrics: Dict[str, float]) -> Dict[str, bool]:
    """
    Check if model metrics meet business targets
    
    Targets (from roadmap):
    - MAE < €2,500
    - RMSE < €3,000
    - R² > 0.85
    - MAPE < 4.5%
    - TR < 15%
    - TC-APE < 6.5%
    
    Args:
        metrics: Dictionary of calculated metrics
    
    Returns:
        Dictionary of {metric_name: target_met (bool)}
    """
    return {
        'mae_ok': bool(metrics['mae'] < 2500),
        'rmse_ok': bool(metrics['rmse'] < 3000),
        'r2_ok': bool(metrics['r2'] > 0.85),
        'mape_ok': bool(metrics['mape'] < 4.5),
        'tr_ok': bool(metrics['tail_rate'] < 15.0),
        'tc_ape_ok': bool(metrics['tc_ape'] < 6.5)
    }
