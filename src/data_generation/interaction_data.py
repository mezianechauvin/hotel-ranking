"""
Interaction data generation module for the hotel ranking system.

This module generates synthetic user-venue interaction data with
click-through behavior based on user preferences, venue attributes,
and contextual factors like weather and seasonality.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import OneHotEncoder

from src.utils.distance import calculate_distance


def generate_interaction_data(venues_df, users_df, seasonal_df, weather_df=None, 
                             n_interactions=50000, random_seed=None):
    """
    Generate synthetic user-venue interaction data.
    
    Parameters:
    -----------
    venues_df : pandas.DataFrame
        DataFrame containing venue data
    users_df : pandas.DataFrame
        DataFrame containing user data
    seasonal_df : pandas.DataFrame
        DataFrame containing seasonal venue data
    weather_df : pandas.DataFrame, optional
        DataFrame containing weather data
    n_interactions : int, optional
        Number of interactions to generate
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing generated interaction data
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    interactions = []
    
    # Create timestamps over a 1-year period with hourly granularity
    start_date = pd.Timestamp('2024-01-01')
    end_date = pd.Timestamp('2024-12-31')
    
    # Generate simple weather data if not provided
    if weather_df is None:
        from src.data_generation.weather_data import generate_weather_data
        
        # Generate daily weather for each city
        cities = venues_df["city"].unique()
        weather_df = generate_weather_data(
            cities, 
            start_date=start_date,
            end_date=end_date,
            random_seed=random_seed
        )
    
    # Track user history for sequential interactions
    user_click_history = {user_id: [] for user_id in users_df["user_id"]}
    
    # Create search sessions
    n_sessions = n_interactions // 8  # Average 8 impressions per session
    
    session_id = 1
    for _ in range(n_sessions):
        # Select a random user
        user = users_df.sample(1).iloc[0]
        user_id = user["user_id"]
        
        # Generate search timestamp
        random_days = np.random.randint(0, (end_date - start_date).days)
        random_hours = np.random.randint(0, 24)
        search_timestamp = start_date + pd.Timedelta(days=random_days, hours=random_hours)
        
        # Determine season based on month
        month = search_timestamp.month
        if month in [12, 1, 2]:
            season = "Winter"
        elif month in [3, 4, 5]:
            season = "Spring"
        elif month in [6, 7, 8]:
            season = "Summer"
        else:
            season = "Fall"
        
        # Determine day of week and if weekend
        day_of_week = search_timestamp.day_name()
        is_weekend = day_of_week in ["Saturday", "Sunday"]
        
        # Adjust search probability based on weekend preference
        if is_weekend and user["weekend_preference"] < 0.3:
            # User doesn't prefer weekends but this is a weekend - reduce probability
            if np.random.random() < 0.7:  # 70% chance to skip
                continue
        elif not is_weekend and user["weekend_preference"] > 0.7:
            # User prefers weekends but this is a weekday - reduce probability
            if np.random.random() < 0.5:  # 50% chance to skip
                continue
        
        # Determine time slot being searched for
        time_slot_probs = user["time_slot_probs"]
        hour = search_timestamp.hour
        
        if hour < 12:
            likely_slot = "morning"
        elif hour < 17:
            likely_slot = "afternoon"
        else:
            likely_slot = "evening"
        
        # 80% chance user searches for the time slot matching current hour
        # 20% chance they search for a different time slot
        if np.random.random() < 0.8:
            search_time_slot = likely_slot
        else:
            other_slots = [s for s in ["morning", "afternoon", "evening"] if s != likely_slot]
            search_time_slot = np.random.choice(other_slots)
        
        # Determine booking date
        # For day access, booking window is much shorter
        booking_window_hours = user["booking_window"]
        hours_ahead = np.random.randint(0, booking_window_hours * 2)  # Some variation
        booking_date = search_timestamp + pd.Timedelta(hours=hours_ahead)
        
        # Get weather for booking date and user's city
        booking_day = booking_date.floor('D')  # Floor to day
        weather = weather_df[(weather_df["city"] == user["home_city"]) & 
                           (weather_df["date"] == booking_day)]
        
        if len(weather) > 0:
            weather_quality = weather.iloc[0]["weather_quality"]
            temperature = weather.iloc[0]["temperature"]
            is_rainy = weather.iloc[0]["is_rainy"]
        else:
            # Default if no weather data
            weather_quality = "moderate"
            temperature = 20
            is_rainy = False
        
        # Filter venues by city - users only looking in their home city
        local_venues = venues_df[venues_df["city"] == user["home_city"]]
        
        # If no venues in this city, skip
        if len(local_venues) == 0:
            continue
        
        # Get seasonal data for these venues
        venue_ids = local_venues["venue_id"].tolist()
        seasonal_venues = seasonal_df[(seasonal_df["venue_id"].isin(venue_ids)) & 
                                    (seasonal_df["season"] == season)]
        
        # Merge seasonal data with venue data
        local_venues_with_season = pd.merge(
            local_venues,
            seasonal_venues,
            on="venue_id",
            how="inner"
        )
        
        # Filter by seasonal availability
        # Only show venues where at least one of the user's preferred amenities is available this season
        available_venues = []
        for _, venue in local_venues_with_season.iterrows():
            # Check if user's preferred amenities are available this season
            has_available_amenities = False
            for amenity in user["preferred_amenities"]:
                availability_col = f"{amenity}_available"
                if availability_col in venue and venue[availability_col]:
                    has_available_amenities = True
                    break
            
            if has_available_amenities:
                available_venues.append(venue)
        
        # If no venues with available amenities, skip
        if len(available_venues) == 0:
            continue
        
        # Filter by time slot availability
        available_venues = [v for v in available_venues if search_time_slot in v["time_slots"]]
        
        # If no venues available for this time slot, skip
        if len(available_venues) == 0:
            continue
        
        # Calculate distance to each venue
        venue_distances = []
        for venue in available_venues:
            distance = calculate_distance(
                user["latitude"], user["longitude"],
                venue["latitude"], venue["longitude"]
            )
            venue_distances.append((venue, distance))
        
        # Filter by max travel distance
        max_distance = user["max_travel_distance"]
        nearby_venues = [(v, d) for v, d in venue_distances if d <= max_distance]
        
        # If no venues within travel distance, skip
        if len(nearby_venues) == 0:
            continue
        
        # Score venues based on user preferences, context, and seasonality
        venue_scores = []
        
        for venue, distance in nearby_venues:
            score = 0
            
            # Distance score (closer is better)
            distance_score = 1 - (distance / max_distance)
            score += distance_score * 3  # High weight for distance
            
            # Amenity match with seasonal adjustments
            for amenity in user["preferred_amenities"]:
                availability_col = f"{amenity}_available"
                adjustment_col = f"{amenity}_adjustment"
                
                if venue.get(amenity, False) and venue.get(availability_col, False):
                    score += 2  # Base score for having the amenity
                    
                    # Apply seasonal adjustment
                    if adjustment_col in venue:
                        score += venue[adjustment_col]
                    
                    # Check for premium features
                    feature_col = f"{amenity}_features"
                    if feature_col in venue and isinstance(venue[feature_col], list):
                        premium_features = ["luxury", "premium", "infinity", "full_service", "private"]
                        for feature in premium_features:
                            if any(feature in str(f).lower() for f in venue[feature_col]):
                                score += 0.5
                        
                        # Apply seasonal penalties for outdoor features in winter
                        if season == "Winter" and amenity == "pool":
                            outdoor_features = ["outdoor", "rooftop"]
                            if any(f in outdoor_features for f in venue[feature_col]):
                                score -= 2  # Major penalty for outdoor pools in winter
            
            # Weather considerations
            if weather_quality == "good" and temperature > 20:
                # Good weather boosts outdoor amenities
                if venue.get("pool", False) and venue.get("pool_available", False):
                    pool_features = venue.get("pool_features", [])
                    if isinstance(pool_features, list) and any(f in ["outdoor", "rooftop"] for f in pool_features):
                        score += 2
                
                if venue.get("beach_access", False) and venue.get("beach_access_available", False):
                    score += 2
            elif weather_quality == "bad" or temperature < 10:
                # Bad weather boosts indoor amenities
                if venue.get("spa", False) and venue.get("spa_available", False):
                    score += 1.5
                
                if venue.get("gym", False) and venue.get("gym_available", False):
                    score += 1
                
                # Indoor pools become more attractive in bad weather
                if venue.get("pool", False) and venue.get("pool_available", False):
                    pool_features = venue.get("pool_features", [])
                    if isinstance(pool_features, list) and any(f in ["indoor", "heated"] for f in pool_features):
                        score += 1.5
            
            # Price sensitivity (use seasonal price)
            price_penalty = venue["seasonal_price"] * user["price_sensitivity"] / 100
            score -= price_penalty
            
            # Vibe match
            if venue["vibe"] in user["preferred_vibes"]:
                score += 1
            
            # Rating quality
            score += venue["avg_rating"] - 3
            
            venue_scores.append((venue, score))
        
        # Sort by score and take top venues
        venue_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Generate impressions for this session
        n_impressions = min(len(venue_scores), np.random.randint(3, 13))
        session_venues = venue_scores[:n_impressions]
        
        # Generate impressions and clicks
        for position, (venue, base_score) in enumerate(session_venues, 1):
            # Position bias (higher positions get more clicks)
            position_factor = max(0.2, 1 - (position - 1) * 0.15)
            
            # Base click probability from score (normalize to 0-1 range)
            normalized_score = min(1, max(0, base_score / 10))
            base_click_prob = 0.05 + normalized_score * 0.3  # Range from 0.05 to 0.35
            
            # User history factor
            history_factor = 1.0
            if user_id in user_click_history and len(user_click_history[user_id]) > 0:
                # Check if user has clicked similar venues before
                user_clicks = user_click_history[user_id]
                
                # Check if user has clicked venues with same amenities
                for amenity in user["preferred_amenities"]:
                    if venue.get(amenity, False):
                        clicked_with_amenity = sum(1 for v in user_clicks if v.get(amenity, False))
                        if clicked_with_amenity > 0:
                            history_factor *= 1.1
            
            # Final click probability
            click_prob = base_click_prob * position_factor * history_factor
            
            # Cap probability at 0.95
            click_prob = min(0.95, click_prob)
            
            # Determine if clicked
            clicked = np.random.random() < click_prob
            
            # Format booking date
            booking_date_str = booking_date.strftime('%Y-%m-%d %H:%M')
            
            # Create interaction record
            interaction = {
                "session_id": session_id,
                "user_id": user_id,
                "venue_id": venue["venue_id"],
                "position": position,
                "timestamp": search_timestamp,
                "clicked": clicked,
                "season": season,
                "day_of_week": day_of_week,
                "is_weekend": is_weekend,
                "time_slot": search_time_slot,
                "booking_date": booking_date_str,
                "hours_until_booking": hours_ahead,
                "distance_km": round(distance, 2),
                "weather_quality": weather_quality,
                "temperature": temperature,
                "is_rainy": is_rainy,
                "seasonal_price": venue["seasonal_price"],
                "vibe_match": 1 if venue["vibe"] in user["preferred_vibes"] else 0,
                "amenity_match_count": sum(1 for amenity in user["preferred_amenities"] 
                                         if venue.get(amenity, False) and 
                                         venue.get(f"{amenity}_available", False))
            }
            
            interactions.append(interaction)
            
            # Update user history if clicked
            if clicked:
                user_click_history[user_id].append(venue.to_dict())
        
        session_id += 1
    
    # Create DataFrame from interactions list
    interactions_df = pd.DataFrame(interactions)
    
    # Create distance buckets
    interactions_df["distance_bucket"] = pd.cut(
        interactions_df["distance_km"],
        bins=[0, 1, 2, 5, 10, 20, 50],
        labels=["0-1km", "1-2km", "2-5km", "5-10km", "10-20km", "20-50km"]
    )
    
    # Create booking hours buckets
    interactions_df["booking_hours_bucket"] = pd.cut(
        interactions_df["hours_until_booking"],
        bins=[0, 2, 6, 12, 24, 48, 100],
        labels=["0-2h", "2-6h", "6-12h", "12-24h", "24-48h", "48h+"]
    )
    
    # One-hot encode categorical fields
    categorical_cols = ["day_of_week", "time_slot", "weather_quality", "distance_bucket", "booking_hours_bucket"]
    encoded_dfs = []
    
    for col in categorical_cols:
        if col in interactions_df.columns:
            encoder = OneHotEncoder(sparse=False, drop='first')
            encoded = encoder.fit_transform(interactions_df[[col]])
            encoded_cols = [f"{col}_{cat}" for cat in encoder.categories_[0][1:]]
            encoded_df = pd.DataFrame(encoded, columns=encoded_cols)
            encoded_dfs.append(encoded_df)
    
    # Concatenate all encoded features with original DataFrame
    if encoded_dfs:
        interactions_df = pd.concat([interactions_df] + encoded_dfs, axis=1)
    
    return interactions_df


def get_interaction_stats(interactions_df):
    """
    Calculate statistics about interaction data.
    
    Parameters:
    -----------
    interactions_df : pandas.DataFrame
        DataFrame containing interaction data
        
    Returns:
    --------
    dict
        Dictionary containing interaction statistics
    """
    stats = {}
    
    # Overall CTR
    stats["overall_ctr"] = interactions_df["clicked"].mean()
    
    # CTR by position
    stats["ctr_by_position"] = interactions_df.groupby("position")["clicked"].mean().to_dict()
    
    # CTR by season
    stats["ctr_by_season"] = interactions_df.groupby("season")["clicked"].mean().to_dict()
    
    # CTR by day of week
    stats["ctr_by_day"] = interactions_df.groupby("day_of_week")["clicked"].mean().to_dict()
    
    # CTR by time slot
    stats["ctr_by_time_slot"] = interactions_df.groupby("time_slot")["clicked"].mean().to_dict()
    
    # CTR by weather quality
    stats["ctr_by_weather"] = interactions_df.groupby("weather_quality")["clicked"].mean().to_dict()
    
    # CTR by vibe match
    stats["ctr_by_vibe_match"] = interactions_df.groupby("vibe_match")["clicked"].mean().to_dict()
    
    # CTR by amenity match count
    stats["ctr_by_amenity_matches"] = interactions_df.groupby("amenity_match_count")["clicked"].mean().to_dict()
    
    # CTR by distance buckets
    interactions_df["distance_bucket"] = pd.cut(
        interactions_df["distance_km"],
        bins=[0, 1, 2, 5, 10, 20, 50],
        labels=["0-1km", "1-2km", "2-5km", "5-10km", "10-20km", "20-50km"]
    )
    stats["ctr_by_distance"] = interactions_df.groupby("distance_bucket")["clicked"].mean().to_dict()
    
    return stats


if __name__ == "__main__":
    # For testing, generate some sample data
    from src.data_generation.venue_data import generate_venue_data
    from src.data_generation.user_data import generate_user_data
    from src.data_generation.seasonal_data import generate_seasonal_data
    from src.data_generation.weather_data import generate_weather_data
    
    # Set random seed for reproducibility
    random_seed = 42
    
    # Generate sample venue data
    venues_df = generate_venue_data(n_venues=50, random_seed=random_seed)
    print(f"Generated {len(venues_df)} venues")
    
    # Generate sample user data
    users_df = generate_user_data(n_users=100, random_seed=random_seed)
    print(f"Generated {len(users_df)} users")
    
    # Generate seasonal data
    seasonal_df = generate_seasonal_data(venues_df, random_seed=random_seed)
    print(f"Generated {len(seasonal_df)} seasonal records")
    
    # Generate weather data
    cities = venues_df["city"].unique()
    weather_df = generate_weather_data(cities, random_seed=random_seed)
    print(f"Generated {len(weather_df)} weather records")
    
    # Generate interaction data
    interactions_df = generate_interaction_data(
        venues_df, users_df, seasonal_df, weather_df,
        n_interactions=1000, random_seed=random_seed
    )
    print(f"Generated {len(interactions_df)} interactions")
    
    # Display some statistics
    stats = get_interaction_stats(interactions_df)
    
    print("\nOverall CTR:", stats["overall_ctr"])
    
    print("\nCTR by position:")
    for position, ctr in sorted(stats["ctr_by_position"].items()):
        print(f"  Position {position}: {ctr:.1%}")
    
    print("\nCTR by season:")
    for season, ctr in stats["ctr_by_season"].items():
        print(f"  {season}: {ctr:.1%}")
    
    print("\nCTR by weather quality:")
    for weather, ctr in stats["ctr_by_weather"].items():
        print(f"  {weather}: {ctr:.1%}")
    
    print("\nCTR by distance:")
    for distance, ctr in stats["ctr_by_distance"].items():
        print(f"  {distance}: {ctr:.1%}")
