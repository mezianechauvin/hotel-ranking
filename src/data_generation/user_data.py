"""
User data generation module for the hotel ranking system.

This module generates synthetic user data with various attributes
including location, preferences, and behavior patterns.
"""

import numpy as np
import pandas as pd


def generate_user_data(n_users=1000, cities=None, random_seed=None):
    """
    Generate synthetic user data for hotel amenity day-access.
    
    Parameters:
    -----------
    n_users : int, optional
        Number of users to generate
    cities : dict, optional
        Dictionary of cities with their coordinates
        If None, default cities will be used
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing generated user data
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if cities is None:
        # Define major cities with their coordinates
        cities = {
            "New York": {"lat": 40.7128, "lng": -74.0060},
            "Los Angeles": {"lat": 34.0522, "lng": -118.2437},
            "Chicago": {"lat": 41.8781, "lng": -87.6298},
            "Miami": {"lat": 25.7617, "lng": -80.1918},
            "Las Vegas": {"lat": 36.1699, "lng": -115.1398},
            "San Francisco": {"lat": 37.7749, "lng": -122.4194},
            "Orlando": {"lat": 28.5383, "lng": -81.3792},
            "Boston": {"lat": 42.3601, "lng": -71.0589},
            "New Orleans": {"lat": 29.9511, "lng": -90.0715},
            "Seattle": {"lat": 47.6062, "lng": -122.3321}
        }
    
    users = []
    
    for user_id in range(1, n_users + 1):
        # Assign user to a home city
        home_city = np.random.choice(list(cities.keys()))
        city_coords = cities[home_city]
        
        # Generate coordinates near the city center for user's location
        # Users are typically within ~10 miles of city center
        lat_offset = np.random.uniform(-0.15, 0.15)  # ~10 miles in latitude
        lng_offset = np.random.uniform(-0.15, 0.15)  # ~10 miles in longitude
        
        latitude = city_coords["lat"] + lat_offset
        longitude = city_coords["lng"] + lng_offset
        
        # Generate user preferences
        preferred_vibes = np.random.choice(
            ["family_friendly", "serene", "luxe", "trendy"],
            size=np.random.randint(1, 3),  # Users typically prefer 1-2 vibes
            replace=False
        )
        
        # Preferred amenities - more focused on specific amenities for day access
        amenity_options = ["pool", "beach_access", "spa", "gym", "hot_tub", "food_service", "bar"]
        n_preferred = np.random.randint(1, 4)  # Users typically have 1-3 primary amenities they're interested in
        preferred_amenities = np.random.choice(amenity_options, size=n_preferred, replace=False)
        
        # Price sensitivity (higher = more price sensitive)
        price_sensitivity = np.random.uniform(0.3, 1.0)
        
        # Preferred time slots
        time_slot_probs = {
            "morning": np.random.uniform(0, 1),
            "afternoon": np.random.uniform(0, 1),
            "evening": np.random.uniform(0, 1)
        }
        # Normalize to sum to 1
        total_prob = sum(time_slot_probs.values())
        time_slot_probs = {k: v/total_prob for k, v in time_slot_probs.items()}
        
        # Preferred day of week (weekday vs weekend)
        weekend_preference = np.random.uniform(0, 1)  # Higher = prefers weekends
        
        # Usage frequency (visits per month)
        usage_frequency = np.random.choice([1, 2, 4, 8, 12], p=[0.3, 0.3, 0.2, 0.1, 0.1])
        
        # Typical booking window (hours in advance)
        booking_window = np.random.choice([2, 6, 12, 24, 48], p=[0.2, 0.3, 0.3, 0.15, 0.05])
        
        # Max travel distance willing to go (in km)
        max_travel_distance = np.random.choice([5, 10, 15, 25, 50], p=[0.2, 0.3, 0.3, 0.15, 0.05])
        
        # Weather sensitivity (higher = more likely to book outdoor amenities in good weather)
        weather_sensitivity = np.random.uniform(0.3, 1.0)
        
        # Demographics (for potential segmentation)
        age = np.random.randint(18, 70)
        gender = np.random.choice(["M", "F", "Other"], p=[0.48, 0.48, 0.04])
        
        # Income level (affects price sensitivity and luxury preferences)
        income_level = np.random.choice(["low", "medium", "high", "very_high"], 
                                      p=[0.2, 0.5, 0.2, 0.1])
        
        # Adjust preferences based on demographics and income
        if income_level in ["high", "very_high"]:
            # Higher income users are less price sensitive
            price_sensitivity *= 0.7
            
            # More likely to prefer luxury amenities
            if "luxe" not in preferred_vibes and np.random.random() < 0.4:
                preferred_vibes = np.append(preferred_vibes, "luxe")
        
        if age > 50:
            # Older users more likely to prefer serene environments
            if "serene" not in preferred_vibes and np.random.random() < 0.5:
                preferred_vibes = np.append(preferred_vibes, "serene")
        elif age < 30:
            # Younger users more likely to prefer trendy environments
            if "trendy" not in preferred_vibes and np.random.random() < 0.5:
                preferred_vibes = np.append(preferred_vibes, "trendy")
        
        user = {
            "user_id": user_id,
            "home_city": home_city,
            "latitude": latitude,
            "longitude": longitude,
            "preferred_vibes": list(preferred_vibes),
            "preferred_amenities": list(preferred_amenities),
            "price_sensitivity": price_sensitivity,
            "time_slot_probs": time_slot_probs,
            "weekend_preference": weekend_preference,
            "usage_frequency": usage_frequency,
            "booking_window": booking_window,
            "max_travel_distance": max_travel_distance,
            "weather_sensitivity": weather_sensitivity,
            "age": age,
            "gender": gender,
            "income_level": income_level
        }
        
        users.append(user)
    
    return pd.DataFrame(users)


def preprocess_user_data(users_df):
    """
    Preprocess user data for modeling by converting complex structures
    to simple features.
    
    Parameters:
    -----------
    users_df : pandas.DataFrame
        DataFrame containing raw user data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with preprocessed user features
    """
    # Create a copy to avoid modifying the original
    processed_df = users_df.copy()
    
    # Extract preferred time slots
    for slot in ["morning", "afternoon", "evening"]:
        processed_df[f"prefers_{slot}"] = processed_df["time_slot_probs"].apply(
            lambda x: x.get(slot, 0) if isinstance(x, dict) else 0
        )
    
    # Extract preferred amenities as binary features
    amenity_options = ["pool", "beach_access", "spa", "gym", "hot_tub", "food_service", "bar"]
    for amenity in amenity_options:
        processed_df[f"prefers_{amenity}"] = processed_df["preferred_amenities"].apply(
            lambda x: 1 if amenity in x else 0
        )
    
    # Extract preferred vibes as binary features
    vibe_options = ["family_friendly", "serene", "luxe", "trendy"]
    for vibe in vibe_options:
        processed_df[f"prefers_{vibe}"] = processed_df["preferred_vibes"].apply(
            lambda x: 1 if vibe in x else 0
        )
    
    # Drop complex columns
    processed_df = processed_df.drop(columns=["time_slot_probs", "preferred_amenities", "preferred_vibes"])
    
    return processed_df


if __name__ == "__main__":
    # Generate sample data when run directly
    users_df = generate_user_data(n_users=100, random_seed=42)
    print(f"Generated {len(users_df)} users")
    print(users_df.head())
    
    # Display some statistics
    print("\nCity distribution:")
    print(users_df['home_city'].value_counts())
    
    print("\nPreferred amenities distribution:")
    amenity_counts = {}
    for amenities in users_df['preferred_amenities']:
        for amenity in amenities:
            if amenity in amenity_counts:
                amenity_counts[amenity] += 1
            else:
                amenity_counts[amenity] = 1
    
    for amenity, count in sorted(amenity_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{amenity}: {count} users ({count/len(users_df):.1%})")
    
    # Test preprocessing
    processed_df = preprocess_user_data(users_df)
    print("\nPreprocessed user data:")
    print(processed_df.head())
