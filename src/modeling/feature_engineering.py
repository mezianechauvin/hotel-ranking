"""
Feature engineering module for the hotel ranking system.

This module provides functions to prepare features for the ranking model
from raw data.
"""

import numpy as np
import pandas as pd


def prepare_features(interactions_df, venues_df, users_df, seasonal_df, weather_df=None):
    """
    Prepare features for the ranking model.
    
    Parameters:
    -----------
    interactions_df : pandas.DataFrame
        DataFrame containing interaction data
    venues_df : pandas.DataFrame
        DataFrame containing venue data
    users_df : pandas.DataFrame
        DataFrame containing user data
    seasonal_df : pandas.DataFrame
        DataFrame containing seasonal venue data
    weather_df : pandas.DataFrame, optional
        DataFrame containing weather data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing prepared features
    """
    # Start with interaction data
    features_df = interactions_df.copy()
    
    # Merge venue features
    venue_columns = (
        ["venue_id", "venue_type", "star_rating", "day_pass_price"] 
        + [col for col in venues_df.columns if col.startswith("vibe_")]
        + [col for col in venues_df.columns if col.startswith("venue_type_")]
    )
    features_df = pd.merge(
        features_df,
        venues_df[venue_columns],
        on="venue_id",
        how="left"
    )
    
    # Merge user features (use preprocessed user data if available)
    if "prefers_pool" in users_df.columns:
        # Already preprocessed
        user_cols = ["user_id", "home_city"] + [col for col in users_df.columns 
                                 if col.startswith("prefers_") or 
                                 col in ["price_sensitivity", "max_travel_distance"]]
        features_df = pd.merge(
            features_df,
            users_df[user_cols],
            on="user_id",
            how="left"
        )
    else:
        # Need to preprocess user data
        from src.data_generation.user_data import preprocess_user_data
        processed_users_df = preprocess_user_data(users_df)
        
        user_cols = ["user_id", "home_city"] + [col for col in processed_users_df.columns 
                                 if col.startswith("prefers_") or 
                                 col in ["price_sensitivity", "max_travel_distance"]]
        features_df = pd.merge(
            features_df,
            processed_users_df[user_cols],
            on="user_id",
            how="left"
        )
    
    # Merge seasonal data
    features_df = pd.merge(
        features_df,
        seasonal_df[["venue_id", "season", "demand_factor"]], #seasonal_price
        on=["venue_id", "season"],
        how="left"
    )
    
    # # Add weather data if provided
    # if weather_df is not None:
    #     # Extract date from booking_date
    #     features_df["booking_date_only"] = pd.to_datetime(
    #         features_df["booking_date"]
    #     ).dt.floor("D")
        
    #     # Merge weather data
    #     weather_cols = ["city", "date", "temperature", "is_rainy", "weather_quality"]
    #     features_df = pd.merge(
    #         features_df,
    #         weather_df[weather_cols],
    #         left_on=["home_city", "booking_date_only"],
    #         right_on=["city", "date"],
    #         how="left"
    #     )
        
        # # Drop temporary and redundant columns
        # features_df = features_df.drop(columns=["booking_date_only", "city", "date"])
    
    # Create price features
    features_df["price_ratio"] = features_df["seasonal_price"] / features_df["day_pass_price"]
    features_df["price_sensitivity_effect"] = features_df["seasonal_price"] * features_df["price_sensitivity"] / 100
    
    # Create distance features
    features_df["distance_score"] = 1 - features_df["distance_km"] / features_df["max_travel_distance"]
    features_df["distance_score"] = features_df["distance_score"].clip(0, 1)
    
    # Convert is_weekend to int if it's not already
    features_df["is_weekend"] = features_df["is_weekend"].astype(int)
    
    # Note: distance_bucket and booking_hours_bucket are now created during data generation
    
    # Create seasonal interaction features
    
    # Weather-season interactions
    features_df["cold_day"] = ((features_df["temperature"] < 10) & 
                             (features_df["season"].isin(["Winter", "Fall"]))).astype(int)
    features_df["hot_day"] = ((features_df["temperature"] > 25) & 
                            (features_df["season"].isin(["Summer", "Spring"]))).astype(int)
    
    # Create amenity-weather interactions
    amenities = ["pool", "beach_access", "spa", "gym", "hot_tub"]
    for amenity in amenities:
        # Check if amenity columns exist
        amenity_col = f"{amenity}_available"
        if amenity_col in features_df.columns:
            # Outdoor amenities in good weather
            if amenity in ["pool", "beach_access"]:
                features_df[f"{amenity}_good_weather"] = (
                    features_df[amenity_col] & 
                    (features_df["weather_quality"] == "good")
                ).astype(int)
                
                features_df[f"{amenity}_bad_weather"] = (
                    features_df[amenity_col] & 
                    (features_df["weather_quality"] == "bad")
                ).astype(int)
            
            # Indoor amenities in bad weather
            if amenity in ["spa", "gym"]:
                features_df[f"{amenity}_bad_weather"] = (
                    features_df[amenity_col] & 
                    (features_df["weather_quality"] == "bad")
                ).astype(int)
    
    # # Create vibe-user preference match features
    # vibe_options = ["family_friendly", "serene", "luxe", "trendy"]
    # for vibe in vibe_options:
    #     vibe_pref_col = f"prefers_{vibe}"
    #     if vibe_pref_col in features_df.columns:
    #         features_df[f"{vibe}_match"] = (
    #             (features_df["vibe"] == vibe) & 
    #             (features_df[vibe_pref_col] == 1)
    #         ).astype(int)
    
    # Note: Categorical features are now one-hot encoded during data generation
    
    # Fill missing values
    # features_df = features_df.fillna(0)
    
    return features_df


def select_features(features_df, include_user_features=True):
    """
    Select relevant features for the ranking model.
    
    Parameters:
    -----------
    features_df : pandas.DataFrame
        DataFrame containing all features
    include_user_features : bool, optional
        Whether to include user-specific features
        
    Returns:
    --------
    list
        List of selected feature column names
    """
    # Base features that are always included
    base_features = [
        # Venue features
        "star_rating", "day_pass_price", "avg_rating", "review_count",
        
        # Seasonal features
        "seasonal_price", "demand_factor",
        
        # Context features
        "is_weekend", "distance_km", "distance_score",
        "temperature", "is_rainy",
        "hours_until_booking",
        
        # Derived features
        "price_ratio", "price_sensitivity_effect", "amenity_match_count",
        "cold_day", "hot_day"
    ]
    
    # Add amenity-weather interactions if they exist
    amenity_weather_features = [col for col in features_df.columns if 
                              any(col.startswith(f"{amenity}_{weather}") 
                                 for amenity in ["pool", "beach_access", "spa", "gym", "hot_tub"]
                                 for weather in ["good_weather", "bad_weather"])]
    
    # Add vibe match features if they exist
    vibe_match_features = [col for col in features_df.columns if 
                         any(col.startswith(f"{vibe}_match") 
                            for vibe in ["family_friendly", "serene", "luxe", "trendy"])]
    
    # Add one-hot encoded categorical features that were created during data generation
    categorical_features = [col for col in features_df.columns if 
                          any(col.startswith(f"{category}_") 
                             for category in ["venue_type", "vibe", "season", "day_of_week", 
                                            "time_slot", "weather_quality", "distance_bucket", 
                                            "booking_hours_bucket"])]
    
    # User features if requested
    user_features = []
    if include_user_features:
        user_features = [col for col in features_df.columns if 
                       col.startswith("prefers_") or 
                       col in ["price_sensitivity", "max_travel_distance"]]
    
    # Combine all features
    all_features = (base_features + 
                   amenity_weather_features + 
                   vibe_match_features + 
                   categorical_features + 
                   user_features)
    
    # Filter to only include columns that exist in the DataFrame
    # and exclude position features
    position_features = ["position", "is_top_position", "is_top_3", "position_score"]
    selected_features = [col for col in all_features 
                       if col in features_df.columns and col not in position_features]
    
    return selected_features


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
        interactions_df = pd.read_csv(os.path.join(sample_dir, "interactions.csv"))
        venues_df = pd.read_csv(os.path.join(sample_dir, "venues.csv"))
        users_df = pd.read_csv(os.path.join(sample_dir, "users_processed.csv"))
        seasonal_df = pd.read_csv(os.path.join(sample_dir, "seasonal.csv"))
        weather_df = pd.read_csv(os.path.join(sample_dir, "weather.csv"))
        
        # Convert date columns to datetime
        if "date" in weather_df.columns:
            weather_df["date"] = pd.to_datetime(weather_df["date"])
        
        data = {
            "interactions": interactions_df,
            "venues": venues_df,
            "users_processed": users_df,
            "seasonal": seasonal_df,
            "weather": weather_df
        }
    
    # Prepare features
    print("Preparing features...")
    features_df = prepare_features(
        data["interactions"],
        data["venues"],
        data["users_processed"],
        data["seasonal"],
        data["weather"]
    )
    
    # Select features
    selected_features = select_features(features_df)
    
    print(f"Prepared {len(features_df)} samples with {len(selected_features)} features")
    print("\nSample features:")
    print(features_df[selected_features[:10]].head())
    
    print("\nSelected features:")
    for i, feature in enumerate(selected_features[:30], 1):
        print(f"{i}. {feature}")
    
    if len(selected_features) > 30:
        print(f"... and {len(selected_features) - 30} more features")
