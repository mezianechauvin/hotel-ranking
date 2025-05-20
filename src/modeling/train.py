"""
Training script for the hotel ranking system.

This script trains a ranking model using the generated data.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.modeling.feature_engineering import prepare_features
from src.modeling.model import train_ranking_model, train_pairwise_ranking_model, save_model
from src.modeling.evaluation import evaluate_model, generate_evaluation_report


def train_and_evaluate(data_dir="data", model_dir="models", report_dir="reports", 
                      model_type="pointwise", random_seed=42, year_month=None):
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
    year_month : str, optional
        Specific year_month to evaluate (format: "YYYY_MM")
        If None, will use the base train/test split
        
    Returns:
    --------
    tuple
        (model, feature_cols, metrics) containing the trained model,
        feature columns, and evaluation metrics
    """
    print(f"Training {model_type} ranking model with random seed {random_seed}")
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Load common data
    print("Loading common data...")
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
    
    # Determine which train/eval files to use
    if year_month:
        # Use specific month's train/eval split
        train_file = f"interactions_train_{year_month}.csv"
        eval_file = f"interactions_eval_{year_month}.csv"
        model_suffix = f"_{year_month}"
        report_suffix = f"_{year_month}"
    else:
        # Use base train/test split
        train_file = "interactions_base_train.csv"
        # Check if we have a full evaluation year file
        if os.path.exists(os.path.join(data_dir, "interactions_eval_year.csv")):
            eval_file = "interactions_eval_year.csv"
        else:
            # Fall back to the first month of evaluation data
            # List all eval files
            eval_files = [f for f in os.listdir(data_dir) if f.startswith("interactions_eval_") and f.endswith(".csv")]
            if eval_files:
                eval_files.sort()
                eval_file = eval_files[0]
            else:
                raise FileNotFoundError("No evaluation data files found")
        
        model_suffix = ""
        report_suffix = ""
    
    print(f"Using train file: {train_file}")
    print(f"Using evaluation file: {eval_file}")
    
    # Load train and evaluation data
    train_df = pd.read_csv(os.path.join(data_dir, train_file))
    eval_df = pd.read_csv(os.path.join(data_dir, eval_file))
    
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
    model_path = os.path.join(model_dir, f"ranking_model{model_suffix}.json")
    save_model(model, feature_cols, model_dir, model_name=f"ranking_model{model_suffix}")
    
    # Evaluate model
    print("Evaluating model...")
    metrics, results_df = evaluate_model(
        model, eval_df, venues_df, users_df, seasonal_df, weather_df, feature_cols
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
    report_path = os.path.join(report_dir, f"evaluation_report{report_suffix}.txt")
    plots_path = os.path.join(report_dir, f"evaluation_plots{report_suffix}.png")
    generate_evaluation_report(metrics, results_df, report_dir, 
                              report_name=f"evaluation_report{report_suffix}",
                              plots_name=f"evaluation_plots{report_suffix}")
    
    return model, feature_cols, metrics


def train_and_evaluate_all_months(data_dir="data", model_dir="models", report_dir="reports",
                                model_type="pointwise", random_seed=42):
    """
    Train and evaluate models for all available monthly evaluation sets.
    
    Parameters:
    -----------
    data_dir : str, optional
        Directory containing the data
    model_dir : str, optional
        Directory to save the models
    report_dir : str, optional
        Directory to save the evaluation reports
    model_type : str, optional
        Type of model to train ('pointwise' or 'pairwise')
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary with year_month keys and (model, feature_cols, metrics) values
    """
    # Find all available evaluation months
    eval_files = [f for f in os.listdir(data_dir) if f.startswith("interactions_eval_") and f.endswith(".csv")]
    year_months = [f.replace("interactions_eval_", "").replace(".csv", "") for f in eval_files]
    year_months.sort()
    
    if not year_months:
        print("No monthly evaluation files found.")
        return {}
    
    print(f"Found {len(year_months)} monthly evaluation sets: {year_months}")
    
    # Train and evaluate for each month
    results = {}
    for year_month in year_months:
        print(f"\n=== Processing month: {year_month} ===\n")
        model, feature_cols, metrics = train_and_evaluate(
            data_dir=data_dir,
            model_dir=model_dir,
            report_dir=report_dir,
            model_type=model_type,
            random_seed=random_seed,
            year_month=year_month
        )
        results[year_month] = (model, feature_cols, metrics)
    
    # Generate summary report across all months
    generate_monthly_summary_report(results, report_dir)
    
    return results


def generate_monthly_summary_report(results, report_dir):
    """
    Generate a summary report comparing metrics across all months.
    
    Parameters:
    -----------
    results : dict
        Dictionary with year_month keys and (model, feature_cols, metrics) values
    report_dir : str
        Directory to save the report
    """
    # Create output directory if it doesn't exist
    os.makedirs(report_dir, exist_ok=True)
    
    # Extract metrics for each month
    months = []
    auc_values = []
    ndcg5_values = []
    ctr1_values = []
    ctr5_values = []
    
    for year_month, (_, _, metrics) in results.items():
        months.append(year_month)
        auc_values.append(metrics["auc"])
        ndcg5_values.append(metrics["ndcg@5"])
        ctr1_values.append(metrics["ctr@1"])
        ctr5_values.append(metrics["ctr@5"])
    
    # Create summary report file
    report_path = os.path.join(report_dir, "monthly_summary_report.txt")
    with open(report_path, "w") as f:
        f.write("# Hotel Ranking Model - Monthly Evaluation Summary\n\n")
        
        f.write("## Performance Metrics by Month\n\n")
        f.write("| Month | AUC | NDCG@5 | CTR@1 | CTR@5 |\n")
        f.write("|-------|-----|--------|-------|-------|\n")
        
        for i, month in enumerate(months):
            f.write(f"| {month} | {auc_values[i]:.4f} | {ndcg5_values[i]:.4f} | {ctr1_values[i]:.4f} | {ctr5_values[i]:.4f} |\n")
        
        # Calculate averages
        avg_auc = sum(auc_values) / len(auc_values)
        avg_ndcg5 = sum(ndcg5_values) / len(ndcg5_values)
        avg_ctr1 = sum(ctr1_values) / len(ctr1_values)
        avg_ctr5 = sum(ctr5_values) / len(ctr5_values)
        
        f.write(f"| **Average** | **{avg_auc:.4f}** | **{avg_ndcg5:.4f}** | **{avg_ctr1:.4f}** | **{avg_ctr5:.4f}** |\n\n")
    
    # Create summary plots
    plt.figure(figsize=(12, 10))
    
    # AUC plot
    plt.subplot(2, 2, 1)
    plt.plot(months, auc_values, marker='o')
    plt.title('AUC by Month')
    plt.xlabel('Month')
    plt.ylabel('AUC')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    # NDCG@5 plot
    plt.subplot(2, 2, 2)
    plt.plot(months, ndcg5_values, marker='o', color='orange')
    plt.title('NDCG@5 by Month')
    plt.xlabel('Month')
    plt.ylabel('NDCG@5')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    # CTR@1 plot
    plt.subplot(2, 2, 3)
    plt.plot(months, ctr1_values, marker='o', color='green')
    plt.title('CTR@1 by Month')
    plt.xlabel('Month')
    plt.ylabel('CTR@1')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    # CTR@5 plot
    plt.subplot(2, 2, 4)
    plt.plot(months, ctr5_values, marker='o', color='red')
    plt.title('CTR@5 by Month')
    plt.xlabel('Month')
    plt.ylabel('CTR@5')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save plots
    plots_path = os.path.join(report_dir, "monthly_summary_plots.png")
    plt.savefig(plots_path)
    
    print(f"Monthly summary report saved to {report_path}")
    print(f"Monthly summary plots saved to {plots_path}")


def main():
    parser = argparse.ArgumentParser(description="Train a hotel ranking model")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing the data")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to save the model")
    parser.add_argument("--report-dir", type=str, default="reports", help="Directory to save the evaluation report")
    parser.add_argument("--model-type", type=str, default="pointwise", choices=["pointwise", "pairwise"], 
                      help="Type of model to train")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--sample", action="store_true", help="Use sample data")
    parser.add_argument("--monthly-eval", action="store_true", 
                      help="Run monthly evaluation (train and evaluate for each month)")
    parser.add_argument("--year-month", type=str, help="Specific year_month to evaluate (format: 'YYYY_MM')")
    
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
    
    # Determine which evaluation approach to use
    if args.monthly_eval:
        # Run monthly evaluation
        train_and_evaluate_all_months(
            data_dir=data_dir,
            model_dir=args.model_dir,
            report_dir=args.report_dir,
            model_type=args.model_type,
            random_seed=args.seed
        )
    elif args.year_month:
        # Evaluate specific month
        train_and_evaluate(
            data_dir=data_dir,
            model_dir=args.model_dir,
            report_dir=args.report_dir,
            model_type=args.model_type,
            random_seed=args.seed,
            year_month=args.year_month
        )
    else:
        # Standard evaluation
        train_and_evaluate(
            data_dir=data_dir,
            model_dir=args.model_dir,
            report_dir=args.report_dir,
            model_type=args.model_type,
            random_seed=args.seed
        )


if __name__ == "__main__":
    main()
