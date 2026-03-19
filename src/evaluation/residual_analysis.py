"""
Residual Analysis Module

Purpose: Diagnose model errors and identify patterns in high-error predictions
This helps answer: "Why is our tail rate 72.8% when target is 15%?"

Location: src/evaluation/residual_analysis.py
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def perform_residual_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X_features: pd.DataFrame,
    model_name: str = "model",
    save_plots: bool = True,
) -> pd.DataFrame:
    """
    Production-grade residual analysis for BMW pricing model

    CRITICAL: Assumes y_true and y_pred are in LOG scale (price_log)
    Automatically back-transforms to Euro for business interpretation

    Args:
        y_true: Actual log-transformed prices (from model)
        y_pred: Predicted log-transformed prices (from model)
        X_features: Feature DataFrame (must have same indices as y_true)
        model_name: Name for saving outputs
        save_plots: Whether to save visualizations

    Returns:
        DataFrame with residuals and features for further analysis
    """

    logger.info("=" * 80)
    logger.info("RESIDUAL ANALYSIS")
    logger.info("=" * 80)

    # 1. Back-transform from log scale to Euro
    y_true_euro = np.exp(y_true)
    y_pred_euro = np.exp(y_pred)

    logger.info(f"Analyzing {len(y_true):,} predictions")
    logger.info(f"Price range: €{y_true_euro.min():,.0f} - €{y_true_euro.max():,.0f}")

    # 2. Calculate residuals in Euro (business interpretable)
    residuals_euro = y_true_euro - y_pred_euro
    abs_residuals_euro = np.abs(residuals_euro)
    ape = np.abs((y_true_euro - y_pred_euro) / y_true_euro) * 100

    # 3. Create results DataFrame
    results = pd.DataFrame(
        {
            "actual_euro": y_true_euro,
            "predicted_euro": y_pred_euro,
            "residual_euro": residuals_euro,
            "abs_residual_euro": abs_residuals_euro,
            "ape": ape,
            "in_tail": ape > 5.0,  # Your 5% threshold
        }
    )

    # 4. Merge with features (ensure indices match)
    X_features_reset = X_features.reset_index(drop=True)
    results_reset = results.reset_index(drop=True)
    analysis_df = pd.concat([results_reset, X_features_reset], axis=1)

    # 5. Basic Statistics
    logger.info("\n" + "=" * 80)
    logger.info("RESIDUAL STATISTICS (in Euro)")
    logger.info("=" * 80)
    logger.info(f"  Mean residual:     €{residuals_euro.mean():,.0f}")
    logger.info(f"  Std residual:      €{residuals_euro.std():,.0f}")
    logger.info(f"  Median abs error:  €{np.median(abs_residuals_euro):,.0f}")
    logger.info(f"  Mean APE:          {ape.mean():.1f}%")
    logger.info(f"  Tail rate (>5%):   {(ape > 5.0).mean() * 100:.1f}%")

    # 6. Identify the Tail Problem
    tail_samples = analysis_df[analysis_df["in_tail"]]
    normal_samples = analysis_df[~analysis_df["in_tail"]]

    logger.info("\n" + "=" * 80)
    logger.info("TAIL ANALYSIS (Errors > 5%)")
    logger.info("=" * 80)
    logger.info(
        f"  Tail samples: {len(tail_samples):,} ({len(tail_samples)/len(analysis_df)*100:.1f}%)"
    )
    logger.info(
        f"  Normal samples: {len(normal_samples):,} ({len(normal_samples)/len(analysis_df)*100:.1f}%)"
    )
    logger.info(f"  Tail avg error: {tail_samples['ape'].mean():.1f}%")
    logger.info(f"  Normal avg error: {normal_samples['ape'].mean():.1f}%")

    # 7. Price Segmentation Analysis
    analyze_by_price_segment(analysis_df)

    # 8. Feature-based Analysis
    analyze_by_features(analysis_df)

    # 9. Generate Visualizations
    if save_plots:
        create_residual_plots(analysis_df, model_name)

    # 10. Save detailed tail analysis
    save_tail_analysis(tail_samples, model_name)

    return analysis_df


def analyze_by_price_segment(df: pd.DataFrame):
    """
    Analyze errors by BMW price segments

    This reveals if model performs poorly on economy vs luxury cars
    """
    logger.info("\n" + "=" * 80)
    logger.info("ERROR ANALYSIS BY PRICE SEGMENT")
    logger.info("=" * 80)

    # Define BMW segments (based on your roadmap)
    df["segment"] = pd.cut(
        df["actual_euro"],
        bins=[0, 20000, 35000, 50000, np.inf],
        labels=["Economy (<€20K)", "Mid-Range (€20-35K)", "Premium (€35-50K)", "Luxury (>€50K)"],
    )

    segment_stats = (
        df.groupby("segment")
        .agg(
            {
                "ape": ["mean", "median", "std"],
                "in_tail": "mean",  # % in tail
                "actual_euro": "count",
            }
        )
        .round(2)
    )

    segment_stats.columns = ["Mean APE (%)", "Median APE (%)", "Std APE", "Tail Rate (%)", "Count"]
    segment_stats["Tail Rate (%)"] = segment_stats["Tail Rate (%)"] * 100

    logger.info("\n" + str(segment_stats))

    # Identify problematic segment
    worst_segment = segment_stats["Tail Rate (%)"].idxmax()
    logger.info(f"\n⚠️  WORST SEGMENT: {worst_segment}")
    logger.info(f"   Tail Rate: {segment_stats.loc[worst_segment, 'Tail Rate (%)']:.1f}%")
    logger.info("   This segment needs feature engineering improvements")


def analyze_by_features(df: pd.DataFrame):
    """
    Analyze which feature values correlate with high errors
    """
    logger.info("\n" + "=" * 80)
    logger.info("ERROR ANALYSIS BY KEY FEATURES")
    logger.info("=" * 80)

    # Analyze by categorical features
    categorical_features = ["fuel", "car_type", "paint_color"]

    for feature in categorical_features:
        if feature in df.columns:
            logger.info(f"\n--- {feature.upper()} ---")

            feature_stats = (
                df.groupby(feature)
                .agg({"ape": "mean", "in_tail": "mean", "actual_euro": "count"})
                .round(2)
            )

            feature_stats.columns = ["Avg APE (%)", "Tail Rate (%)", "Count"]
            feature_stats["Tail Rate (%)"] = feature_stats["Tail Rate (%)"] * 100
            feature_stats = feature_stats.sort_values("Tail Rate (%)", ascending=False)

            logger.info("\n" + str(feature_stats))

            # Flag problematic categories
            problematic = feature_stats[feature_stats["Tail Rate (%)"] > 50]
            if len(problematic) > 0:
                logger.warning(f"⚠️  Problematic {feature} values (>50% tail rate):")
                for cat, row in problematic.iterrows():
                    logger.warning(f"   {cat}: {row['Tail Rate (%)']:.1f}% tail rate")

    # Analyze continuous features
    logger.info("\n--- CONTINUOUS FEATURES CORRELATION WITH ERROR ---")
    continuous_features = ["mileage", "car_age_years", "engine_power", "annual_mileage"]

    for feature in continuous_features:
        if feature in df.columns:
            correlation = df[feature].corr(df["ape"])
            logger.info(f"  {feature:20s}: {correlation:+.3f} correlation with APE")


def create_residual_plots(df: pd.DataFrame, model_name: str):
    """
    Create diagnostic plots for residual analysis
    """
    output_dir = Path("reports/images/residuals")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Residuals vs Predicted (Gold Standard Plot)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Residuals vs Predicted
    axes[0, 0].scatter(df["predicted_euro"], df["residual_euro"], alpha=0.5, s=10)
    axes[0, 0].axhline(0, color="red", linestyle="--", linewidth=2)
    axes[0, 0].set_xlabel("Predicted Price (€)")
    axes[0, 0].set_ylabel("Residual (€)")
    axes[0, 0].set_title(
        "Residuals vs Predicted Price\n(Looking for funnel shape = heteroscedasticity)"
    )
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Distribution of Residuals
    axes[0, 1].hist(df["residual_euro"], bins=50, edgecolor="black", alpha=0.7)
    axes[0, 1].axvline(0, color="red", linestyle="--", linewidth=2)
    axes[0, 1].set_xlabel("Residual (€)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Distribution of Residuals\n(Should be bell-shaped, centered at 0)")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: APE Distribution with Tail Threshold
    axes[1, 0].hist(df["ape"], bins=50, edgecolor="black", alpha=0.7)
    axes[1, 0].axvline(5.0, color="red", linestyle="--", linewidth=2, label="5% Threshold (Tail)")
    axes[1, 0].set_xlabel("Absolute Percentage Error (%)")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title(f'APE Distribution\nTail Rate: {(df["ape"] > 5.0).mean() * 100:.1f}%')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Actual vs Predicted
    axes[1, 1].scatter(df["actual_euro"], df["predicted_euro"], alpha=0.5, s=10)
    axes[1, 1].plot(
        [df["actual_euro"].min(), df["actual_euro"].max()],
        [df["actual_euro"].min(), df["actual_euro"].max()],
        "r--",
        linewidth=2,
        label="Perfect Prediction",
    )
    axes[1, 1].set_xlabel("Actual Price (€)")
    axes[1, 1].set_ylabel("Predicted Price (€)")
    axes[1, 1].set_title("Actual vs Predicted Price")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    filepath = output_dir / f"{model_name}_residual_analysis.png"
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"📊 Residual plots saved to: {filepath}")


def save_tail_analysis(tail_df: pd.DataFrame, model_name: str):
    """
    Save detailed analysis of high-error samples (tail)
    """
    output_dir = Path("reports/residuals")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort by worst errors
    tail_sorted = tail_df.sort_values("ape", ascending=False)

    # Save top 100 worst predictions
    filepath = output_dir / f"{model_name}_tail_analysis.csv"
    tail_sorted.head(100).to_csv(filepath, index=False)

    logger.info(f"💾 Tail analysis saved to: {filepath}")
    logger.info("   Contains top 100 worst predictions for investigation")

    # Generate summary report
    report_path = output_dir / f"{model_name}_tail_summary.txt"

    with open(report_path, "w") as f:
        f.write("BMW PRICING MODEL - TAIL ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total samples in tail (>5% error): {len(tail_df):,}\n")
        f.write(f"Tail rate: {len(tail_df) / (len(tail_df) + len(tail_sorted)) * 100:.1f}%\n\n")

        f.write("TOP 10 WORST PREDICTIONS:\n")
        f.write("-" * 80 + "\n")

        for idx, row in tail_sorted.head(10).iterrows():
            f.write(f"\nPrediction #{idx + 1}:\n")
            f.write(f"  Actual:    €{row['actual_euro']:,.0f}\n")
            f.write(f"  Predicted: €{row['predicted_euro']:,.0f}\n")
            f.write(f"  Error:     €{row['abs_residual_euro']:,.0f} ({row['ape']:.1f}%)\n")

            # Include key features if available
            if "car_type" in row:
                f.write(f"  Car Type:  {row['car_type']}\n")
            if "mileage" in row:
                f.write(f"  Mileage:   {row['mileage']:,.0f} km\n")
            if "car_age_years" in row:
                f.write(f"  Age:       {row['car_age_years']:.1f} years\n")

    logger.info(f"📄 Tail summary saved to: {report_path}")


def generate_recommendations(analysis_df: pd.DataFrame) -> dict[str, str]:
    """
    Generate actionable recommendations based on residual analysis
    """
    recommendations = {}

    # Check for heteroscedasticity (funnel shape)
    correlation = analysis_df["predicted_euro"].corr(analysis_df["abs_residual_euro"])
    if abs(correlation) > 0.3:
        recommendations["heteroscedasticity"] = (
            f"⚠️  Detected heteroscedasticity (correlation={correlation:.3f})\n"
            f"   → Errors increase with price → Try log transformation or weighted loss"
        )

    # Check tail rate by segment
    df_with_segment = analysis_df.copy()
    df_with_segment["segment"] = pd.cut(
        df_with_segment["actual_euro"],
        bins=[0, 20000, 35000, 50000, np.inf],
        labels=["Economy", "Mid-Range", "Premium", "Luxury"],
    )

    segment_tail_rate = df_with_segment.groupby("segment")["in_tail"].mean() * 100
    worst_segment = segment_tail_rate.idxmax()

    if segment_tail_rate[worst_segment] > 50:
        recommendations["segment_issue"] = (
            f"⚠️  {worst_segment} segment has {segment_tail_rate[worst_segment]:.1f}% tail rate\n"
            f"   → Need segment-specific features or separate models"
        )

    # Overall tail rate issue
    overall_tail_rate = analysis_df["in_tail"].mean() * 100
    if overall_tail_rate > 50:
        recommendations["high_tail_rate"] = (
            f"🚨 CRITICAL: Overall tail rate is {overall_tail_rate:.1f}% (target: <15%)\n"
            f"   → This indicates fundamental model limitations\n"
            f"   → Consider: (1) More features (2) Different algorithm (3) Ensemble methods"
        )

    return recommendations
