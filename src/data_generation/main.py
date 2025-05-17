"""
Main module for data generation in the hotel ranking system.

This module provides functions to generate all necessary data for the
hotel ranking system and save it to disk.
"""

import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

from src.data_generation.venue_data import generate_venue_data
from src.data_generation.user_data import generate_user_data, preprocess_user_data
from src.data_generation.seasonal_data import generate_seasonal_data
from src.data_generation.weather_data import generate_weather_data
from src.data_generation.interaction_data import generate_interaction_data


def generate_all_data(output_dir="data", n_venues=500, n_users=1000, n_interactions=50000, random_seed=42):
    """
    Generate all necessary data for the hotel ranking system.
    
    Parameters:
    -----------
    output_dir : str, optional
        Directory to save generated data
    n_venues : int, optional
        Number of venues to generate
    n_users : int, optional
        Number of users to generate
    n_interactions : int, optional
        Number of interactions to generate
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary containing all generated DataFrames
    """
    print(f"Generating data with random seed {random_seed}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate venue data
    print(f"Generating {n_venues} venues...")
    venues_df = generate_venue_data(n_venues=n_venues, random_seed=random_seed)
    venues_df.to_csv(os.path.join(output_dir, "venues.csv"), index=False)
    print(f"Generated {len(venues_df)} venues")
    
    # Generate user data
    print(f"Generating {n_users} users...")
    users_df = generate_user_data(n_users=n_users, random_seed=random_seed)
    users_df.to_csv(os.path.join(output_dir, "users.csv"), index=False)
    
    # Preprocess user data for modeling
    processed_users_df = preprocess_user_data(users_df)
    processed_users_df.to_csv(os.path.join(output_dir, "users_processed.csv"), index=False)
    print(f"Generated {len(users_df)} users")
    
    # Generate seasonal data
    print("Generating seasonal data...")
    seasonal_df = generate_seasonal_data(venues_df, random_seed=random_seed)
    seasonal_df.to_csv(os.path.join(output_dir, "seasonal.csv"), index=False)
    print(f"Generated {len(seasonal_df)} seasonal records")
    
    # Generate weather data
    print("Generating weather data...")
    cities = venues_df["city"].unique()
    current_year = datetime.now().year
    weather_df = generate_weather_data(
        cities, 
        start_date=f"{current_year}-01-01",
        end_date=f"{current_year}-12-31",
        random_seed=random_seed
    )
    weather_df.to_csv(os.path.join(output_dir, "weather.csv"), index=False)
    print(f"Generated {len(weather_df)} weather records")
    
    # Generate interaction data
    print(f"Generating {n_interactions} interactions...")
    interactions_df = generate_interaction_data(
        venues_df, users_df, seasonal_df, weather_df,
        n_interactions=n_interactions, random_seed=random_seed
    )
    interactions_df.to_csv(os.path.join(output_dir, "interactions.csv"), index=False)
    print(f"Generated {len(interactions_df)} interactions")
    
    # Create train/test split for interactions
    print("Creating train/test split...")
    # Sort by timestamp to ensure temporal ordering
    interactions_df = interactions_df.sort_values("timestamp")
    
    # Use 80% for training, 20% for testing
    train_size = int(len(interactions_df) * 0.8)
    train_df = interactions_df.iloc[:train_size]
    test_df = interactions_df.iloc[train_size:]
    
    train_df.to_csv(os.path.join(output_dir, "interactions_train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "interactions_test.csv"), index=False)
    print(f"Created train set with {len(train_df)} interactions and test set with {len(test_df)} interactions")
    
    # Return all generated data
    return {
        "venues": venues_df,
        "users": users_df,
        "users_processed": processed_users_df,
        "seasonal": seasonal_df,
        "weather": weather_df,
        "interactions": interactions_df,
        "interactions_train": train_df,
        "interactions_test": test_df
    }


def generate_sample_data(output_dir="data/sample", random_seed=42):
    """
    Generate a small sample dataset for testing and development.
    
    Parameters:
    -----------
    output_dir : str, optional
        Directory to save generated data
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary containing all generated DataFrames
    """
    return generate_all_data(
        output_dir=output_dir,
        n_venues=50,
        n_users=100,
        n_interactions=1000,
        random_seed=random_seed
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data for hotel ranking system")
    parser.add_argument("--output-dir", type=str, default="data", help="Directory to save generated data")
    parser.add_argument("--n-venues", type=int, default=500, help="Number of venues to generate")
    parser.add_argument("--n-users", type=int, default=1000, help="Number of users to generate")
    parser.add_argument("--n-interactions", type=int, default=50000, help="Number of interactions to generate")
    parser.add_argument("--sample", action="store_true", help="Generate a small sample dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    if args.sample:
        print("Generating sample dataset...")
        generate_sample_data(output_dir=args.output_dir, random_seed=args.seed)
    else:
        print("Generating full dataset...")
        generate_all_data(
            output_dir=args.output_dir,
            n_venues=args.n_venues,
            n_users=args.n_users,
            n_interactions=args.n_interactions,
            random_seed=args.seed
        )
