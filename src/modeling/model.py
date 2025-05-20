"""
Model implementation module for the hotel ranking system.

This module provides functions to train and use the XGBoost ranking model.
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

from src.modeling.feature_engineering import prepare_features, select_features


def train_ranking_model(features_df, group_col="session_id", target_col="clicked", 
                       params=None, num_boost_round=100, early_stopping_rounds=10):
    """
    Train an XGBoost ranking model.
    
    Parameters:
    -----------
    features_df : pandas.DataFrame
        DataFrame containing features and target
    group_col : str, optional
        Column name to group by for ranking
    target_col : str, optional
        Column name of the target variable
    params : dict, optional
        XGBoost parameters
    num_boost_round : int, optional
        Number of boosting rounds
    early_stopping_rounds : int, optional
        Number of rounds for early stopping
        
    Returns:
    --------
    xgboost.Booster
        Trained XGBoost model
    """
    # Default XGBoost parameters
    if params is None:
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'eta': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
    
    # Select features
    feature_cols = select_features(features_df)
    features_df = features_df.sort_values(by=group_col)
    
    # Prepare data
    X = features_df[feature_cols]
    y = features_df[target_col].astype(int)
    groups = features_df[group_col]
    
    # Initialize GroupKFold for cross-validation
    gkf = GroupKFold(n_splits=5)
    
    # Train-test split using GroupKFold
    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]
        break  # Just use the first fold for simplicity
    
    # Prepare DMatrix for XGBoost
    dtrain = xgb.DMatrix(np.array(X_train), label=np.array(y_train))
    dtest = xgb.DMatrix(np.array(X_test), label=np.array(y_test))
    
    # Group data by session_id for ranking
    train_groups = groups_train.value_counts().to_list()
    test_groups = groups_test.value_counts().to_list()
    
    dtrain.set_group(train_groups)
    dtest.set_group(test_groups)
    
    # Train model
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=10
    )
    
    # Evaluate model
    y_pred = model.predict(dtest)
    auc = roc_auc_score(y_test, y_pred)
    print(f"Test AUC: {auc:.4f}")
    
    # Feature importance
    importance = model.get_score(importance_type='gain')
    importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 10 features by importance:")
    for feature, score in importance[:10]:
        print(f"{feature}: {score:.4f}")
    
    return model, feature_cols


def train_pairwise_ranking_model(features_df, group_col="session_id", target_col="clicked", 
                               params=None, num_boost_round=100, early_stopping_rounds=10):
    """
    Train an XGBoost pairwise ranking model.
    
    Parameters:
    -----------
    features_df : pandas.DataFrame
        DataFrame containing features and target
    group_col : str, optional
        Column name to group by for ranking
    target_col : str, optional
        Column name of the target variable
    params : dict, optional
        XGBoost parameters
    num_boost_round : int, optional
        Number of boosting rounds
    early_stopping_rounds : int, optional
        Number of rounds for early stopping
        
    Returns:
    --------
    xgboost.Booster
        Trained XGBoost model
    """
    # Default XGBoost parameters for pairwise ranking
    if params is None:
        params = {
            'objective': 'rank:pairwise',
            'eval_metric': ['ndcg@5', 'ndcg@10'],
            'eta': 0.1,
            'gamma': 1.0,
            'min_child_weight': 0.1,
            'max_depth': 6
        }
    
    # Select features
    feature_cols = select_features(features_df)
    
    # Prepare data
    X = features_df[feature_cols]
    y = features_df[target_col].astype(int)
    groups = features_df[group_col]
    
    # Initialize GroupKFold for cross-validation
    gkf = GroupKFold(n_splits=5)
    
    # Train-test split using GroupKFold
    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]
        break  # Just use the first fold for simplicity
    
    # Prepare DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Group data by session_id for ranking
    train_groups = X_train.reset_index().groupby(groups_train).size().values
    test_groups = X_test.reset_index().groupby(groups_test).size().values
    
    dtrain.set_group(train_groups)
    dtest.set_group(test_groups)
    
    # Train model
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=10
    )
    
    # Feature importance
    importance = model.get_score(importance_type='gain')
    importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 10 features by importance:")
    for feature, score in importance[:10]:
        print(f"{feature}: {score:.4f}")
    
    return model, feature_cols


def predict_rankings(model, features_df, feature_cols=None):
    """
    Predict rankings for venues.
    
    Parameters:
    -----------
    model : xgboost.Booster
        Trained XGBoost model
    features_df : pandas.DataFrame
        DataFrame containing features
    feature_cols : list, optional
        List of feature column names
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with predicted scores and rankings
    """
    # If feature columns not provided, select them
    if feature_cols is None:
        feature_cols = select_features(features_df)
    
    # Ensure all feature columns exist
    missing_cols = [col for col in feature_cols if col not in features_df.columns]
    if missing_cols:
        print(f"Warning: Missing feature columns: {missing_cols}")
        feature_cols = [col for col in feature_cols if col in features_df.columns]
    
    # Prepare data
    X = features_df[feature_cols]
    dmatrix = xgb.DMatrix(np.array(X))
    
    # Predict scores
    scores = model.predict(dmatrix)
    
    # Add scores to DataFrame
    results_df = features_df.copy()
    results_df["predicted_score"] = scores
    
    # Rank venues within each session
    results_df["predicted_rank"] = results_df.groupby("session_id")["predicted_score"].rank(ascending=False)
    
    return results_df


def save_model(model, feature_cols, model_dir="models"):
    """
    Save the trained model and feature columns.
    
    Parameters:
    -----------
    model : xgboost.Booster
        Trained XGBoost model
    feature_cols : list
        List of feature column names
    model_dir : str, optional
        Directory to save the model
        
    Returns:
    --------
    str
        Path to the saved model
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(model_dir, "ranking_model.json")
    model.save_model(model_path)
    
    # Save feature columns
    feature_path = os.path.join(model_dir, "feature_cols.txt")
    with open(feature_path, "w") as f:
        for col in feature_cols:
            f.write(f"{col}\n")
    
    print(f"Model saved to {model_path}")
    print(f"Feature columns saved to {feature_path}")
    
    return model_path


def load_model(model_dir="models"):
    """
    Load a trained model and feature columns.
    
    Parameters:
    -----------
    model_dir : str, optional
        Directory containing the model
        
    Returns:
    --------
    tuple
        (xgboost.Booster, list) containing the model and feature columns
    """
    # Load model
    model_path = os.path.join(model_dir, "ranking_model.json")
    model = xgb.Booster()
    model.load_model(model_path)
    
    # Load feature columns
    feature_path = os.path.join(model_dir, "feature_cols.txt")
    with open(feature_path, "r") as f:
        feature_cols = [line.strip() for line in f.readlines()]
    
    print(f"Model loaded from {model_path}")
    print(f"Loaded {len(feature_cols)} feature columns")
    
    return model, feature_cols


if __name__ == "__main__":
    # For testing, load some sample data
    import os
    
    # Check if sample data exists
    sample_dir = "data/sample"
    if not os.path.exists(sample_dir):
        print("Sample data not found. Generating sample data...")
        from src.data_generation.main import generate_sample_data
        data = generate_sample_data()
    else:
        # Load sample data
        print("Loading sample data...")
        interactions_df = pd.read_csv(os.path.join(sample_dir, "interactions_train.csv"))
        venues_df = pd.read_csv(os.path.join(sample_dir, "venues.csv"))
        users_df = pd.read_csv(os.path.join(sample_dir, "users_processed.csv"))
        seasonal_df = pd.read_csv(os.path.join(sample_dir, "seasonal.csv"))
        weather_df = pd.read_csv(os.path.join(sample_dir, "weather.csv"))
        
        # Convert date columns to datetime
        if "date" in weather_df.columns:
            weather_df["date"] = pd.to_datetime(weather_df["date"])
        
        data = {
            "interactions_train": interactions_df,
            "venues": venues_df,
            "users_processed": users_df,
            "seasonal": seasonal_df,
            "weather": weather_df
        }
    
    # Prepare features
    print("Preparing features...")
    features_df = prepare_features(
        data["interactions_train"],
        data["venues"],
        data["users_processed"],
        data["seasonal"],
        data["weather"]
    )
    
    # Train model
    print("Training ranking model...")
    model, feature_cols = train_ranking_model(features_df)
    
    # Save model
    save_model(model, feature_cols)
    
    # Test loading model
    loaded_model, loaded_feature_cols = load_model()
    
    # Test prediction
    print("Testing prediction...")
    results_df = predict_rankings(loaded_model, features_df, loaded_feature_cols)
    
    print("\nSample predictions:")
    sample_cols = ["session_id", "venue_id", "clicked", "predicted_score", "predicted_rank"]
    print(results_df[sample_cols].head(10))
