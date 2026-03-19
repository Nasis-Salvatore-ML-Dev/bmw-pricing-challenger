# pylint: disable=import-error,no-name-in-module

import argparse

import joblib
import pandas as pd

from src.common.logging import get_logger
from src.models.train import train_pipeline

logger = get_logger(__name__)


def run_prediction(mileage: int, engine_power: int, model_key: str):
    """Example of loading the saved pipeline for a quick CLI prediction."""
    try:
        pipeline = joblib.load("data/models/xgboost_pipeline.pkl")
        # Create a tiny dataframe matching the raw input format
        input_data = pd.DataFrame(
            [
                {
                    "mileage": mileage,
                    "engine_power": engine_power,
                    "model_key": model_key,
                    "registration_date": "2024-01-01",  # Mock date for simple CLI test
                    "feature_1": True,
                }
            ]
        )

        prediction = pipeline.predict(input_data)[0]
        print(f"\n🚀 Estimated Price for BMW {model_key}: €{prediction:,.2f}")
    except FileNotFoundError:
        logger.error("Model not found. Please run 'python src/main.py --mode train' first.")


def main():
    parser = argparse.ArgumentParser(description="BMW Pricing Challenger CLI")
    parser.add_argument(
        "--mode",
        choices=["train", "predict"],
        required=True,
        help="Choose 'train' to build the model or 'predict' for a single car.",
    )

    # Prediction arguments
    parser.add_argument("--model_key", type=str, default="320", help="BMW Model (e.g., 320, 520)")
    parser.add_argument("--mileage", type=int, default=50000)
    parser.add_argument("--power", type=int, default=135, help="Engine power in kW")

    args = parser.parse_args()

    if args.mode == "train":
        logger.info("Starting Project Pipeline: Training Mode")
        train_pipeline()
    elif args.mode == "predict":
        run_prediction(args.mileage, args.power, args.model_key)


if __name__ == "__main__":
    main()
