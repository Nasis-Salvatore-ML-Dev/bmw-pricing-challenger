"""
Model Evaluation Gate
---------------------
Loads the trained model and evaluates it against a fixed golden dataset.
Exits with code 1 if any metric fails its threshold — blocking deployment.

Usage:
    python scripts/evaluate_model.py \
        --model-path models/random_forest.pkl \
        --data-path data/golden/golden_eval.csv \
        --thresholds-path config/evaluation/thresholds.json
"""

import argparse
import json
import pickle
import sys

import numpy as np
import pandas as pd

# ── Metric functions (inline to avoid import path issues in CI) ──────────────


def calculate_mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def calculate_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def calculate_r2(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot)


def calculate_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def calculate_tail_rate(y_true, y_pred, threshold=0.05):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ape = np.abs((y_true - y_pred) / y_true)
    return float(np.mean(ape > threshold) * 100)


# ── Main evaluation logic ────────────────────────────────────────────────────


def evaluate(model_path: str, data_path: str, thresholds_path: str) -> bool:
    """
    Returns True if all thresholds pass, False otherwise.
    """

    print("\n" + "=" * 60)
    print("MODEL EVALUATION GATE")
    print("=" * 60)

    # Load model
    print(f"\n📦 Loading model from: {model_path}")
    with open(model_path, "rb") as f:
        artefact = pickle.load(f)

    # Support both raw model and dict-wrapped artefact
    # (train.py may save {"model": ..., "encoders": ...})
    if isinstance(artefact, dict):
        model = artefact["model"]
        encoders = artefact.get("encoders", {})
    else:
        model = artefact
        encoders = {}

    print(f"✅ Model loaded: {type(model).__name__}")

    # Load golden dataset
    print(f"\n📂 Loading golden dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✅ Golden dataset: {len(df)} rows")

    # Separate features and target
    # Convention: golden CSV has a 'price' column as target
    if "price" not in df.columns:
        print("❌ Golden dataset must contain a 'price' column.")
        return False

    y_true = df["price"].values
    X = df.drop(columns=["price"])

    # Apply encoders if present
    if encoders:
        for col, encoder in encoders.items():
            if col in X.columns:
                # Handle unseen categories gracefully
                X[col] = X[col].map(
                    lambda v, enc=encoder: enc.transform([v])[0] if v in enc.classes_ else -1
                )

    # Generate predictions
    y_pred = model.predict(X)

    # Compute metrics
    metrics = {
        "mae": calculate_mae(y_true, y_pred),
        "rmse": calculate_rmse(y_true, y_pred),
        "r2": calculate_r2(y_true, y_pred),
        "mape": calculate_mape(y_true, y_pred),
        "tail_rate": calculate_tail_rate(y_true, y_pred),
    }

    print("\n📊 Evaluation Results:")
    print(f"{'Metric':<12} {'Value':>12} {'Threshold':>12} {'Status':>8}")
    print("─" * 50)

    # Load thresholds
    with open(thresholds_path, "r") as f:
        thresholds = json.load(f)

    # Evaluate each metric against its threshold
    all_passed = True
    for metric, value in metrics.items():
        if metric not in thresholds:
            continue

        threshold = thresholds[metric]["max_acceptable"]
        higher_is_better = thresholds[metric].get("higher_is_better", False)

        if higher_is_better:
            passed = value >= threshold
        else:
            passed = value <= threshold

        status = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_passed = False

        print(f"{metric:<12} {value:>11.2f}  {threshold:>11.2f}  {status}")

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL THRESHOLDS PASSED — deployment approved")
    else:
        print("❌ EVALUATION GATE FAILED — deployment blocked")
        print("   Fix the model before redeploying.")
    print("=" * 60 + "\n")

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--thresholds-path", required=True)
    args = parser.parse_args()

    passed = evaluate(args.model_path, args.data_path, args.thresholds_path)
    sys.exit(0 if passed else 1)  # exit code 1 blocks GitHub Actions
