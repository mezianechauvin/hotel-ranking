"""
Training script for the hotel ranking system.

This script trains a ranking model using the generated data.
"""

import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

from src.modeling.feature_engineering import prepare_features
from src.modeling.model import train_ranking_model, train_pairwise_ranking_model, save_model
from src.modeling.evaluation import evaluate_model, generate_evaluation_report


def train_and_evaluate(data_dir="data", model_dir="models", report_dir="reports", 
                      model_type="pointwise", random_seed=42):
    """
    Train and evaluate a ranking model.
    
    Parameters:
    -----------
    data_dir : str, optional
        Directory containing the data
    model_dir : str, optional
        Directory to save the model
    report_dir : str, optional
        Directory to save the evaluation report
    model_type : str, optional
        Type of model to train ('pointwise' or 'pairwise')
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (model, feature_cols, metrics) containing the trained model,
        feature columns, and evaluation metrics
    """
    print(f"Training {model_type} ranking model with random seed {random_seed}")
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv(os.path.join(data_dir, "interactions_train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "interactions_test.csv"))
    venues_df = pd.read_csv(os.path.join(data_dir, "venues.csv"))
    users_df = pd.read_csv(os.path.join(data_dir, "users_processed.csv"))
    seasonal_df = pd.read_csv(os.path.join(data_dir, "seasonal.csv"))
    
    # Load weather data if available
    weather_path = os.path.join(data_dir, "weather.csv")
    if os.path.exists(weather_path):
        weather_df = pd.read_csv(weather_path)
        # Convert date column to datetime
        if "date" in weather_df.columns:
            weather_df["date"] = pd.to_datetime(weather_df["date"])
    else:
        weather_df = None
    
    # Prepare features for training
    print("Preparing features...")
    train_features_df = prepare_features(train_df, venues_df, users_df, seasonal_df, weather_df)
    
    # Train model
    print(f"Training {model_type} model...")
    if model_type == "pairwise":
        model, feature_cols = train_pairwise_ranking_model(train_features_df)
    else:
        model, feature_cols = train_ranking_model(train_features_df)
    
    # Save model
    save_model(model, feature_cols, model_dir)
    
    # Evaluate model
    print("Evaluating model...")
    metrics, results_df = evaluate_model(
        model, test_df, venues_df, users_df, seasonal_df, weather_df, feature_cols
    )
    
    # Print metrics
    print("\nEvaluation Metrics:")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Average Precision: {metrics['average_precision']:.4f}")
    print(f"NDCG@5: {metrics['ndcg@5']:.4f}")
    print(f"NDCG@10: {metrics['ndcg@10']:.4f}")
    print(f"CTR@1: {metrics['ctr@1']:.4f}")
    print(f"CTR@5: {metrics['ctr@5']:.4f}")
    
    # Generate evaluation report
    generate_evaluation_report(metrics, results_df, report_dir)
    
    return model, feature_cols, metrics


def main():
    parser = argparse.ArgumentParser(description="Train a hotel ranking model")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing the data")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to save the model")
    parser.add_argument("--report-dir", type=str, default="reports", help="Directory to save the evaluation report")
    parser.add_argument("--model-type", type=str, default="pointwise", choices=["pointwise", "pairwise"], 
                      help="Type of model to train")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--sample", action="store_true", help="Use sample data")
    
    args = parser.parse_args()
    
    # Use sample data if requested
    data_dir = args.data_dir
    if args.sample:
        data_dir = os.path.join(data_dir, "sample")
        
        # Check if sample data exists
        if not os.path.exists(data_dir):
            print("Sample data not found. Generating sample data...")
            from src.data_generation.main import generate_sample_data
            generate_sample_data()
    
    # Train and evaluate model
    train_and_evaluate(
        data_dir=data_dir,
        model_dir=args.model_dir,
        report_dir=args.report_dir,
        model_type=args.model_type,
        random_seed=args.seed
    )


if __name__ == "__main__":
    main()
