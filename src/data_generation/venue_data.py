"""
Venue data generation module for the hotel ranking system.

This module generates synthetic venue data with various attributes
including location, amenities, and pricing.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def generate_venue_data(n_venues=500, random_seed=None):
    """
    Generate synthetic venue data for hotel amenity day-access.
    
    Parameters:
    -----------
    n_venues : int, optional
        Number of venues to generate
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing generated venue data
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    venues = []
    
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
    
    # Define venue types with their characteristics
    venue_types = [
        {"name": "Ultra Luxury", "weight": 0.1, "attributes": {
            "star_min": 4.5, "star_max": 5,
            "day_pass_min": 80, "day_pass_max": 300,
            "vibe_probs": {"serene": 0.4, "luxe": 0.5, "trendy": 0.1},
            "amenity_quality": "premium"
        }},
        {"name": "Luxury", "weight": 0.3, "attributes": {
            "star_min": 4, "star_max": 4.5,
            "day_pass_min": 50, "day_pass_max": 150,
            "vibe_probs": {"family_friendly": 0.2, "serene": 0.3, "luxe": 0.3, "trendy": 0.2},
            "amenity_quality": "high"
        }},
        {"name": "Upscale", "weight": 0.4, "attributes": {
            "star_min": 3.5, "star_max": 4,
            "day_pass_min": 30, "day_pass_max": 80,
            "vibe_probs": {"family_friendly": 0.4, "serene": 0.2, "trendy": 0.4},
            "amenity_quality": "good"
        }},
        {"name": "Select", "weight": 0.2, "attributes": {
            "star_min": 3, "star_max": 3.5,
            "day_pass_min": 20, "day_pass_max": 50,
            "vibe_probs": {"family_friendly": 0.6, "trendy": 0.4},
            "amenity_quality": "standard"
        }}
    ]
    
    # Calculate how many venues to generate from each type
    type_counts = [int(t["weight"] * n_venues) for t in venue_types]
    # Adjust to ensure we get exactly n_venues
    type_counts[-1] += n_venues - sum(type_counts)
    
    # Define amenity quality levels
    amenity_quality_levels = {
        "premium": {
            "pool": {"prob": 0.95, "features": ["infinity", "rooftop", "heated", "adults_only", "cabanas"]},
            "beach_access": {"prob": 0.3, "features": ["private", "cabanas", "service"]},
            "spa": {"prob": 0.9, "features": ["full_service", "luxury_treatments", "sauna", "steam_room"]},
            "gym": {"prob": 0.95, "features": ["luxury", "personal_training", "classes", "equipment"]},
            "hot_tub": {"prob": 0.8, "features": ["premium"]},
            "food_service": {"prob": 0.9, "features": ["fine_dining", "poolside_service"]},
            "bar": {"prob": 0.9, "features": ["craft_cocktails", "premium_spirits"]}
        },
        "high": {
            "pool": {"prob": 0.9, "features": ["rooftop", "heated", "family_friendly"]},
            "beach_access": {"prob": 0.2, "features": ["semi_private", "loungers"]},
            "spa": {"prob": 0.8, "features": ["treatments", "sauna"]},
            "gym": {"prob": 0.9, "features": ["modern", "classes"]},
            "hot_tub": {"prob": 0.7, "features": ["standard"]},
            "food_service": {"prob": 0.8, "features": ["casual_dining", "poolside_service"]},
            "bar": {"prob": 0.8, "features": ["cocktails", "beer_wine"]}
        },
        "good": {
            "pool": {"prob": 0.85, "features": ["outdoor", "indoor"]},
            "beach_access": {"prob": 0.15, "features": ["public_access"]},
            "spa": {"prob": 0.6, "features": ["basic_treatments"]},
            "gym": {"prob": 0.8, "features": ["standard"]},
            "hot_tub": {"prob": 0.5, "features": ["standard"]},
            "food_service": {"prob": 0.7, "features": ["cafe"]},
            "bar": {"prob": 0.6, "features": ["basic"]}
        },
        "standard": {
            "pool": {"prob": 0.8, "features": ["standard"]},
            "beach_access": {"prob": 0.1, "features": ["nearby"]},
            "spa": {"prob": 0.3, "features": ["limited"]},
            "gym": {"prob": 0.7, "features": ["basic"]},
            "hot_tub": {"prob": 0.3, "features": ["basic"]},
            "food_service": {"prob": 0.5, "features": ["snacks"]},
            "bar": {"prob": 0.4, "features": ["limited"]}
        }
    }
    
    venue_id = 1
    for i, venue_type in enumerate(venue_types):
        for _ in range(type_counts[i]):
            # Select a city for this venue
            city_name = np.random.choice(list(cities.keys()))
            city_coords = cities[city_name]
            
            # Generate coordinates near the city center
            # Add some random offset (up to ~5 miles in each direction)
            lat_offset = np.random.uniform(-0.07, 0.07)
            lng_offset = np.random.uniform(-0.07, 0.07)
            
            # Adjust offset based on venue type - luxury venues more likely in prime locations
            if venue_type["name"] in ["Ultra Luxury", "Luxury"]:
                # More likely to be in prime areas (downtown, waterfront)
                lat_offset *= 0.5
                lng_offset *= 0.5
            
            # Calculate final coordinates
            latitude = city_coords["lat"] + lat_offset
            longitude = city_coords["lng"] + lng_offset
            
            # Create base venue from type
            star_rating = np.random.uniform(
                venue_type["attributes"]["star_min"],
                venue_type["attributes"]["star_max"]
            )
            
            # Day pass price
            day_pass_price = np.random.uniform(
                venue_type["attributes"]["day_pass_min"],
                venue_type["attributes"]["day_pass_max"]
            )
            
            # Select vibe based on venue type
            vibe_probs = venue_type["attributes"]["vibe_probs"]
            vibe = np.random.choice(
                list(vibe_probs.keys()),
                p=list(vibe_probs.values())
            )
            
            # Generate amenities based on quality level
            amenity_quality = venue_type["attributes"]["amenity_quality"]
            quality_level = amenity_quality_levels[amenity_quality]
            
            amenities = {}
            amenity_details = {}
            
            for amenity, details in quality_level.items():
                # Determine if venue has this amenity
                has_amenity = np.random.random() < details["prob"]
                amenities[amenity] = has_amenity
                
                if has_amenity:
                    # Select features for this amenity
                    n_features = min(len(details["features"]), 
                                    np.random.randint(1, len(details["features"]) + 1))
                    selected_features = np.random.choice(
                        details["features"], size=n_features, replace=False
                    )
                    amenity_details[amenity + "_features"] = list(selected_features)
            
            # Special case for beach access - only available in coastal cities
            coastal_cities = ["Miami", "Los Angeles", "San Francisco", "Boston", "Seattle", "New Orleans"]
            if city_name not in coastal_cities:
                amenities["beach_access"] = False
                if "beach_access_features" in amenity_details:
                    del amenity_details["beach_access_features"]
            
            # Time slots available
            time_slots = []
            # Morning slots (more common for fitness)
            if amenities["gym"] and np.random.random() < 0.8:
                time_slots.append("morning")
            # Afternoon slots (common for pools)
            if amenities["pool"] and np.random.random() < 0.9:
                time_slots.append("afternoon")
            # Evening slots (less common, especially for outdoor amenities)
            if np.random.random() < 0.6:
                time_slots.append("evening")
            
            # If no slots selected, add at least one
            if not time_slots:
                time_slots.append(np.random.choice(["morning", "afternoon", "evening"]))
            
            # Capacity limits
            if venue_type["name"] in ["Ultra Luxury", "Luxury"]:
                capacity = np.random.randint(20, 100)
            else:
                capacity = np.random.randint(50, 200)
            
            # Historical performance metrics
            avg_rating = min(5.0, max(1.0, star_rating + np.random.normal(0, 0.3)))
            review_count = int(np.random.gamma(shape=5, scale=20 * star_rating))
            
            venue = {
                "venue_id": venue_id,
                "name": f"Hotel {venue_id}",
                "venue_type": venue_type["name"],
                "star_rating": round(star_rating, 1),
                "day_pass_price": round(day_pass_price, 2),
                "city": city_name,
                "latitude": latitude,
                "longitude": longitude,
                "vibe": vibe,
                "avg_rating": round(avg_rating, 1),
                "review_count": review_count,
                "time_slots": time_slots,
                "capacity": capacity,
                **amenities,  # Unpack amenities as individual columns
                **amenity_details  # Unpack amenity details
            }
            
            venues.append(venue)
            venue_id += 1
    
    # Create DataFrame from venues list
    venues_df = pd.DataFrame(venues)
    
    # One-hot encode categorical fields
    # Venue type
    venue_type_encoder = OneHotEncoder(sparse=False, drop='first')
    venue_type_encoded = venue_type_encoder.fit_transform(venues_df[['venue_type']])
    venue_type_cols = [f"venue_type_{cat}" for cat in venue_type_encoder.categories_[0][1:]]
    venue_type_df = pd.DataFrame(venue_type_encoded, columns=venue_type_cols)
    
    # Vibe
    vibe_encoder = OneHotEncoder(sparse=False, drop='first')
    vibe_encoded = vibe_encoder.fit_transform(venues_df[['vibe']])
    vibe_cols = [f"vibe_{cat}" for cat in vibe_encoder.categories_[0][1:]]
    vibe_df = pd.DataFrame(vibe_encoded, columns=vibe_cols)
    
    # Concatenate encoded features with original DataFrame
    venues_df = pd.concat([venues_df, venue_type_df, vibe_df], axis=1)
    
    return venues_df


if __name__ == "__main__":
    # Generate sample data when run directly
    venues_df = generate_venue_data(n_venues=100, random_seed=42)
    print(f"Generated {len(venues_df)} venues")
    print(venues_df.head())
    
    # Display some statistics
    print("\nVenue type distribution:")
    print(venues_df['venue_type'].value_counts())
    
    print("\nAmenity availability:")
    for amenity in ['pool', 'spa', 'gym', 'beach_access', 'hot_tub', 'food_service', 'bar']:
        if amenity in venues_df.columns:
            print(f"{amenity}: {venues_df[amenity].mean():.1%}")
