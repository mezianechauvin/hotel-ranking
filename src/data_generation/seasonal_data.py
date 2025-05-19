"""
Seasonal data generation module for the hotel ranking system.

This module generates seasonal availability and pricing data for venues
based on their characteristics and the season.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def generate_seasonal_data(venues_df, random_seed=None):
    """
    Generate seasonal availability and pricing data for venues.
    
    Parameters:
    -----------
    venues_df : pandas.DataFrame
        DataFrame containing venue data
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing seasonal data for each venue
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    seasons = ["Winter", "Spring", "Summer", "Fall"]
    seasonal_data = []
    
    for _, venue in venues_df.iterrows():
        venue_id = venue["venue_id"]
        
        for season in seasons:
            # Determine which amenities are available in this season
            amenity_availability = {}
            amenity_adjustment = {}
            
            # Pool availability and adjustments by season
            if venue.get("pool", False):
                pool_features = venue.get("pool_features", [])
                
                if isinstance(pool_features, list) and any(f in ["outdoor", "rooftop"] for f in pool_features):
                    # Outdoor/rooftop pools often closed in winter
                    if season == "Winter":
                        amenity_availability["pool"] = np.random.random() < 0.2  # 20% chance open in winter
                        amenity_adjustment["pool"] = -0.8  # Major penalty if open but cold
                    elif season in ["Fall", "Spring"]:
                        amenity_availability["pool"] = np.random.random() < 0.7  # 70% chance open in shoulder seasons
                        amenity_adjustment["pool"] = -0.3  # Minor penalty in shoulder seasons
                    else:  # Summer
                        amenity_availability["pool"] = True
                        amenity_adjustment["pool"] = 0.5  # Bonus in summer
                elif isinstance(pool_features, list) and any(f in ["indoor", "heated"] for f in pool_features):
                    # Indoor/heated pools typically open year-round
                    amenity_availability["pool"] = True
                    if season == "Winter":
                        amenity_adjustment["pool"] = 0.3  # Bonus for indoor pools in winter
                    else:
                        amenity_adjustment["pool"] = 0
                else:
                    # Standard pools
                    if season == "Winter":
                        amenity_availability["pool"] = np.random.random() < 0.5  # 50% chance open in winter
                        amenity_adjustment["pool"] = -0.5
                    else:
                        amenity_availability["pool"] = True
                        amenity_adjustment["pool"] = 0.2 if season == "Summer" else 0
            else:
                amenity_availability["pool"] = False
                amenity_adjustment["pool"] = 0
            
            # Beach access seasonal adjustments
            if venue.get("beach_access", False):
                if season == "Winter":
                    amenity_availability["beach_access"] = np.random.random() < 0.7  # Beaches often still accessible
                    amenity_adjustment["beach_access"] = -0.7  # But not appealing
                elif season in ["Fall", "Spring"]:
                    amenity_availability["beach_access"] = True
                    amenity_adjustment["beach_access"] = -0.2  # Slightly less appealing
                else:  # Summer
                    amenity_availability["beach_access"] = True
                    amenity_adjustment["beach_access"] = 0.6  # Major bonus in summer
            else:
                amenity_availability["beach_access"] = False
                amenity_adjustment["beach_access"] = 0
            
            # Spa seasonal adjustments (more appealing in winter)
            if venue.get("spa", False):
                amenity_availability["spa"] = True  # Spas typically open year-round
                if season == "Winter":
                    amenity_adjustment["spa"] = 0.4  # More appealing in winter
                elif season == "Summer":
                    amenity_adjustment["spa"] = -0.1  # Slightly less appealing in summer
                else:
                    amenity_adjustment["spa"] = 0.1
            else:
                amenity_availability["spa"] = False
                amenity_adjustment["spa"] = 0
            
            # Gym seasonal adjustments (more consistent year-round)
            if venue.get("gym", False):
                amenity_availability["gym"] = True  # Gyms open year-round
                if season == "Winter":
                    amenity_adjustment["gym"] = 0.2  # Slightly more appealing in winter
                else:
                    amenity_adjustment["gym"] = 0
            else:
                amenity_availability["gym"] = False
                amenity_adjustment["gym"] = 0
            
            # Hot tub seasonal adjustments
            if venue.get("hot_tub", False):
                amenity_availability["hot_tub"] = True  # Usually available year-round
                if season == "Winter":
                    amenity_adjustment["hot_tub"] = 0.5  # Much more appealing in winter
                elif season == "Summer":
                    amenity_adjustment["hot_tub"] = -0.3  # Less appealing in summer
                else:
                    amenity_adjustment["hot_tub"] = 0.1
            else:
                amenity_availability["hot_tub"] = False
                amenity_adjustment["hot_tub"] = 0
            
            # Food service and bar (generally available year-round)
            for amenity in ["food_service", "bar"]:
                if venue.get(amenity, False):
                    amenity_availability[amenity] = True
                    amenity_adjustment[amenity] = 0  # No seasonal adjustment
                else:
                    amenity_availability[amenity] = False
                    amenity_adjustment[amenity] = 0
            
            # Seasonal pricing adjustments
            base_price = venue["day_pass_price"]
            if season == "Summer":
                # Peak season - higher prices
                seasonal_price = base_price * np.random.uniform(1.1, 1.3)
            elif season == "Winter":
                # Off season - lower prices
                seasonal_price = base_price * np.random.uniform(0.7, 0.9)
            else:
                # Shoulder seasons - slight adjustments
                seasonal_price = base_price * np.random.uniform(0.9, 1.1)
            
            # Seasonal demand factor
            if season == "Summer":
                demand_factor = np.random.uniform(0.8, 1.0)  # High demand
            elif season == "Winter":
                demand_factor = np.random.uniform(0.3, 0.6)  # Lower demand
            else:
                demand_factor = np.random.uniform(0.5, 0.8)  # Moderate demand
            
            # Location-specific seasonal adjustments
            location_type = venue.get("location_type", "")
            city = venue.get("city", "")
            
            # Beach locations more popular in summer
            if location_type == "Beach" or city in ["Miami", "Los Angeles"]:
                if season == "Summer":
                    demand_factor *= 1.2
                    seasonal_price *= 1.1
                elif season == "Winter":
                    demand_factor *= 0.8
            
            # Ski/mountain locations more popular in winter
            if "Mountain" in location_type or city in ["Denver", "Salt Lake City"]:
                if season == "Winter":
                    demand_factor *= 1.3
                    seasonal_price *= 1.1
                elif season == "Summer":
                    demand_factor *= 0.9
            
            # Create seasonal record
            seasonal_record = {
                "venue_id": venue_id,
                "season": season,
                "seasonal_price": round(seasonal_price, 2),
                "demand_factor": round(demand_factor, 2),
                **{f"{amenity}_available": avail for amenity, avail in amenity_availability.items()},
                **{f"{amenity}_adjustment": adj for amenity, adj in amenity_adjustment.items()}
            }
            
            seasonal_data.append(seasonal_record)
    
    # Create DataFrame from seasonal data list
    seasonal_df = pd.DataFrame(seasonal_data)
    
    # One-hot encode season
    season_encoder = OneHotEncoder(sparse=False, drop='first')
    season_encoded = season_encoder.fit_transform(seasonal_df[['season']])
    season_cols = [f"season_{cat}" for cat in season_encoder.categories_[0][1:]]
    season_df = pd.DataFrame(season_encoded, columns=season_cols)
    
    # Concatenate encoded features with original DataFrame
    seasonal_df = pd.concat([seasonal_df, season_df], axis=1)
    
    return seasonal_df


def get_seasonal_availability_stats(seasonal_df):
    """
    Calculate statistics about seasonal availability of amenities.
    
    Parameters:
    -----------
    seasonal_df : pandas.DataFrame
        DataFrame containing seasonal data
        
    Returns:
    --------
    dict
        Dictionary containing availability statistics by season and amenity
    """
    stats = {}
    
    # Get list of amenities
    amenities = [col.replace("_available", "") for col in seasonal_df.columns 
                if col.endswith("_available")]
    
    # Calculate availability percentage by season and amenity
    for season in seasonal_df["season"].unique():
        season_data = seasonal_df[seasonal_df["season"] == season]
        stats[season] = {}
        
        for amenity in amenities:
            avail_col = f"{amenity}_available"
            if avail_col in season_data.columns:
                availability = season_data[avail_col].mean()
                stats[season][amenity] = availability
    
    return stats


if __name__ == "__main__":
    # For testing, generate some venue data
    from venue_data import generate_venue_data
    
    # Generate sample venue data
    venues_df = generate_venue_data(n_venues=50, random_seed=42)
    print(f"Generated {len(venues_df)} venues")
    
    # Generate seasonal data
    seasonal_df = generate_seasonal_data(venues_df, random_seed=42)
    print(f"Generated {len(seasonal_df)} seasonal records")
    
    # Display some statistics
    stats = get_seasonal_availability_stats(seasonal_df)
    print("\nSeasonal availability statistics:")
    for season, amenities in stats.items():
        print(f"\n{season}:")
        for amenity, availability in amenities.items():
            print(f"  {amenity}: {availability:.1%}")
    
    # Display price variations
    print("\nSeasonal price variations:")
    seasonal_price_stats = seasonal_df.groupby("season")["seasonal_price"].agg(["mean", "min", "max"])
    print(seasonal_price_stats)
