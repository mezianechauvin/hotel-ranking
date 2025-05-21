"""
Evaluation module for the hotel ranking system.

This module provides functions to evaluate the ranking model performance.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

from src.modeling.feature_engineering import prepare_features
from src.modeling.model import predict_rankings


def calculate_ndcg(y_true, y_score, k=10):
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG) at k.
    
    Parameters:
    -----------
    y_true : array-like
        Binary relevance labels
    y_score : array-like
        Predicted scores
    k : int, optional
        Number of top items to consider
        
    Returns:
    --------
    float
        NDCG@k score
    """
    # Sort by predicted scores
    sorted_indices = np.argsort(y_score)[::-1]
    sorted_y_true = np.array(y_true)[sorted_indices]
    
    # Calculate DCG@k
    dcg = 0
    for i in range(min(k, len(sorted_y_true))):
        dcg += sorted_y_true[i] / np.log2(i + 2)  # i+2 because i starts from 0
    
    # Calculate ideal DCG@k
    ideal_sorted_y_true = np.sort(y_true)[::-1]
    idcg = 0
    for i in range(min(k, len(ideal_sorted_y_true))):
        idcg += ideal_sorted_y_true[i] / np.log2(i + 2)
    
    # Calculate NDCG@k
    if idcg == 0:
        return 0.0
    else:
        return dcg / idcg


def calculate_ndcg_by_group(df, group_col, score_col, label_col, k=10):
    """
    Calculate NDCG@k for each group.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing predictions and labels
    group_col : str
        Column name for grouping
    score_col : str
        Column name for predicted scores
    label_col : str
        Column name for true labels
    k : int, optional
        Number of top items to consider
        
    Returns:
    --------
    dict
        Dictionary with group IDs as keys and NDCG@k scores as values
    """
    ndcg_scores = {}
    
    for group_id, group_df in df.groupby(group_col):
        y_true = group_df[label_col].values
        y_score = group_df[score_col].values
        
        # Only calculate NDCG if there are positive examples
        if sum(y_true) > 0:
            ndcg = calculate_ndcg(y_true, y_score, k)
            ndcg_scores[group_id] = ndcg
    
    return ndcg_scores


def evaluate_model(model, test_df, venues_df, users_df, seasonal_df, weather_df=None, feature_cols=None):
    """
    Evaluate a trained ranking model on test data.
    
    Parameters:
    -----------
    model : xgboost.Booster
        Trained XGBoost model
    test_df : pandas.DataFrame
        DataFrame containing test data
    venues_df : pandas.DataFrame
        DataFrame containing venue data
    users_df : pandas.DataFrame
        DataFrame containing user data
    seasonal_df : pandas.DataFrame
        DataFrame containing seasonal venue data
    weather_df : pandas.DataFrame, optional
        DataFrame containing weather data
    feature_cols : list, optional
        List of feature column names
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    # Prepare features
    features_df = prepare_features(test_df, venues_df, users_df, seasonal_df, weather_df)
    
    # Predict rankings
    results_df = predict_rankings(model, features_df, feature_cols)
    
    # Calculate AUC
    auc = roc_auc_score(results_df["clicked"], results_df["predicted_score"])
    
    # Calculate Average Precision
    ap = average_precision_score(results_df["clicked"], results_df["predicted_score"])
    
    # Calculate NDCG@k for different k values
    ndcg_scores = {}
    for k in [5, 10]:
        ndcg_by_session = calculate_ndcg_by_group(
            results_df, "session_id", "predicted_score", "clicked", k
        )
        ndcg_scores[f"ndcg@{k}"] = np.mean(list(ndcg_by_session.values()))
    
    # Calculate CTR@k for different k values
    ctr_at_k = {}
    for k in [1, 3, 5, 10]:
        # For each session, get top k venues by predicted score
        top_k_df = results_df.loc[results_df.groupby("session_id")["predicted_score"].nlargest(k).index.get_level_values(1)]
        ctr_at_k[f"ctr@{k}"] = top_k_df["clicked"].mean()
    
    # Calculate metrics by season
    metrics_by_season = {}
    for season, season_df in results_df.groupby("season"):
        metrics_by_season[season] = {
            "auc": roc_auc_score(season_df["clicked"], season_df["predicted_score"]),
            "ctr@5": season_df.loc[season_df.groupby("session_id")["predicted_score"].nlargest(5).index.get_level_values(1)]["clicked"].mean()
        }
    
    # Calculate metrics by weather quality
    metrics_by_weather = {}
    if "weather_quality" in results_df.columns:
        for weather, weather_df in results_df.groupby("weather_quality"):
            metrics_by_weather[weather] = {
                "auc": roc_auc_score(weather_df["clicked"], weather_df["predicted_score"]),
                "ctr@5": weather_df.loc[weather_df.groupby("session_id")["predicted_score"].nlargest(5).index.get_level_values(1)]["clicked"].mean()
            }
    
    # Combine all metrics
    metrics = {
        "auc": auc,
        "average_precision": ap,
        **ndcg_scores,
        **ctr_at_k,
        "by_season": metrics_by_season,
        "by_weather": metrics_by_weather
    }
    
    return metrics, results_df


def plot_precision_recall_curve(y_true, y_score, ax=None):
    """
    Plot precision-recall curve.
    
    Parameters:
    -----------
    y_true : array-like
        Binary relevance labels
    y_score : array-like
        Predicted scores
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
        
    Returns:
    --------
    matplotlib.axes.Axes
        Axes with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    
    ax.plot(recall, precision, lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve (AP = {ap:.4f})")
    ax.grid(True)
    
    return ax


def plot_score_distribution(results_df, ax=None):
    """
    Plot distribution of predicted scores by clicked status.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing predictions and labels
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
        
    Returns:
    --------
    matplotlib.axes.Axes
        Axes with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.histplot(
        data=results_df,
        x="predicted_score",
        hue="clicked",
        bins=30,
        alpha=0.6,
        ax=ax
    )
    
    ax.set_xlabel("Predicted Score")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Predicted Scores by Clicked Status")
    
    return ax


def plot_metrics_by_category(metrics_dict, category_name, metric_name, ax=None):
    """
    Plot metrics by category.
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary containing metrics by category
    category_name : str
        Name of the category (e.g., 'season', 'weather')
    metric_name : str
        Name of the metric to plot
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
        
    Returns:
    --------
    matplotlib.axes.Axes
        Axes with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    categories = list(metrics_dict.keys())
    values = [metrics_dict[cat][metric_name] for cat in categories]
    
    sns.barplot(x=categories, y=values, ax=ax)
    
    ax.set_xlabel(category_name.capitalize())
    ax.set_ylabel(metric_name.upper())
    ax.set_title(f"{metric_name.upper()} by {category_name.capitalize()}")
    
    # Add value labels on bars
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.4f}", ha="center")
    
    return ax


def generate_evaluation_report(metrics, results_df, output_dir="reports", 
                             report_name="evaluation_report", plots_name="evaluation_plots"):
    """
    Generate an evaluation report with metrics and plots.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing evaluation metrics
    results_df : pandas.DataFrame
        DataFrame containing predictions and labels
    output_dir : str, optional
        Directory to save the report
    report_name : str, optional
        Base name for the report file (without extension)
    plots_name : str, optional
        Base name for the plots file (without extension)
        
    Returns:
    --------
    str
        Path to the saved report
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create report file
    report_path = os.path.join(output_dir, f"{report_name}.txt")
    with open(report_path, "w") as f:
        f.write("# Hotel Ranking Model Evaluation Report\n\n")
        
        f.write("## Overall Metrics\n\n")
        f.write(f"AUC: {metrics['auc']:.4f}\n")
        f.write(f"Average Precision: {metrics['average_precision']:.4f}\n")
        
        f.write("\n## Ranking Metrics\n\n")
        for k in [5, 10]:
            f.write(f"NDCG@{k}: {metrics[f'ndcg@{k}']:.4f}\n")
        
        f.write("\n## Click-Through Rate at k\n\n")
        for k in [1, 3, 5, 10]:
            f.write(f"CTR@{k}: {metrics[f'ctr@{k}']:.4f}\n")
        
        f.write("\n## Metrics by Season\n\n")
        for season, season_metrics in metrics["by_season"].items():
            f.write(f"### {season}\n")
            f.write(f"AUC: {season_metrics['auc']:.4f}\n")
            f.write(f"CTR@5: {season_metrics['ctr@5']:.4f}\n\n")
        
        if "by_weather" in metrics and metrics["by_weather"]:
            f.write("\n## Metrics by Weather Quality\n\n")
            for weather, weather_metrics in metrics["by_weather"].items():
                f.write(f"### {weather}\n")
                f.write(f"AUC: {weather_metrics['auc']:.4f}\n")
                f.write(f"CTR@5: {weather_metrics['ctr@5']:.4f}\n\n")
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Precision-recall curve
    plot_precision_recall_curve(results_df["clicked"], results_df["predicted_score"], axes[0, 0])
    
    # Score distribution
    plot_score_distribution(results_df, axes[0, 1])
    
    # Metrics by season
    plot_metrics_by_category(metrics["by_season"], "season", "auc", axes[1, 0])
    
    # Metrics by weather quality
    if "by_weather" in metrics and metrics["by_weather"]:
        plot_metrics_by_category(metrics["by_weather"], "weather quality", "auc", axes[1, 1])
    
    plt.tight_layout()
    
    # Save plots
    plots_path = os.path.join(output_dir, f"{plots_name}.png")
    plt.savefig(plots_path)
    
    print(f"Evaluation report saved to {report_path}")
    print(f"Evaluation plots saved to {plots_path}")
    
    return report_path


if __name__ == "__main__":
    # For testing, load a trained model and test data
    import os
    from src.modeling.model import load_model
    
    # Check if model exists
    model_dir = "models"
    if not os.path.exists(os.path.join(model_dir, "ranking_model.json")):
        print("Model not found. Training a new model...")
        from src.modeling.model import train_ranking_model
        
        # Check if sample data exists
        sample_dir = "data/sample"
        if not os.path.exists(sample_dir):
            print("Sample data not found. Generating sample data...")
            from src.data_generation.main import generate_sample_data
            data = generate_sample_data()
        else:
            # Load sample data
            print("Loading sample data...")
            train_df = pd.read_csv(os.path.join(sample_dir, "interactions_train.csv"))
            venues_df = pd.read_csv(os.path.join(sample_dir, "venues.csv"))
            users_df = pd.read_csv(os.path.join(sample_dir, "users_processed.csv"))
            seasonal_df = pd.read_csv(os.path.join(sample_dir, "seasonal.csv"))
            weather_df = pd.read_csv(os.path.join(sample_dir, "weather.csv"))
            
            # Convert date columns to datetime
            if "date" in weather_df.columns:
                weather_df["date"] = pd.to_datetime(weather_df["date"])
            
            # Prepare features
            from src.modeling.feature_engineering import prepare_features
            features_df = prepare_features(train_df, venues_df, users_df, seasonal_df, weather_df)
            
            # Train model
            model, feature_cols = train_ranking_model(features_df)
            
            # Save model
            from src.modeling.model import save_model
            save_model(model, feature_cols)
    
    # Load model
    model, feature_cols = load_model()
    
    # Load test data
    test_df = pd.read_csv(os.path.join("data/sample", "interactions_test.csv"))
    venues_df = pd.read_csv(os.path.join("data/sample", "venues.csv"))
    users_df = pd.read_csv(os.path.join("data/sample", "users_processed.csv"))
    seasonal_df = pd.read_csv(os.path.join("data/sample", "seasonal.csv"))
    weather_df = pd.read_csv(os.path.join("data/sample", "weather.csv"))
    
    # Convert date columns to datetime
    if "date" in weather_df.columns:
        weather_df["date"] = pd.to_datetime(weather_df["date"])
    
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
    generate_evaluation_report(metrics, results_df)
