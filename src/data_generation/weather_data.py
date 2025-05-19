"""
Weather data generation module for the hotel ranking system.

This module generates synthetic weather data for different cities
across a specified time period.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import OneHotEncoder


def generate_weather_data(cities, start_date=None, end_date=None, random_seed=None):
    """
    Generate synthetic weather data for a list of cities over a time period.
    
    Parameters:
    -----------
    cities : list or dict
        List of city names or dictionary of cities with coordinates
    start_date : str or datetime, optional
        Start date for weather data (default: Jan 1 of current year)
    end_date : str or datetime, optional
        End date for weather data (default: Dec 31 of current year)
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing daily weather data for each city
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Process city input
    if isinstance(cities, dict):
        city_names = list(cities.keys())
    else:
        city_names = cities
    
    # Set default date range if not provided
    current_year = datetime.now().year
    if start_date is None:
        start_date = f"{current_year}-01-01"
    if end_date is None:
        end_date = f"{current_year}-12-31"
    
    # Convert string dates to datetime if needed
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    weather_data = []
    
    for city in city_names:
        # Set base temperature patterns by city
        # These are rough approximations of climate patterns
        if city in ["Miami", "Orlando", "New Orleans"]:
            # Hot, humid cities
            base_temp_range = (15, 32)  # Celsius
            temp_variation = 5
            precip_prob_range = (0.2, 0.5)
        elif city in ["Los Angeles", "San Diego", "San Francisco"]:
            # Mild coastal cities
            base_temp_range = (10, 25)
            temp_variation = 3
            precip_prob_range = (0.1, 0.3)
        elif city in ["New York", "Boston", "Chicago"]:
            # Northern cities with distinct seasons
            base_temp_range = (-10, 30)
            temp_variation = 8
            precip_prob_range = (0.2, 0.4)
        elif city in ["Las Vegas", "Phoenix"]:
            # Desert cities
            base_temp_range = (5, 40)
            temp_variation = 7
            precip_prob_range = (0.05, 0.15)
        elif city in ["Seattle", "Portland"]:
            # Pacific Northwest
            base_temp_range = (2, 25)
            temp_variation = 4
            precip_prob_range = (0.3, 0.6)
        else:
            # Default
            base_temp_range = (0, 30)
            temp_variation = 6
            precip_prob_range = (0.2, 0.4)
        
        for date in dates:
            # Calculate day of year (0-365)
            day_of_year = date.dayofyear
            
            # Calculate seasonal temperature pattern
            # Using sine wave with 365 day period
            # Northern hemisphere: hottest around day 200 (July), coldest around day 20 (January)
            season_factor = np.sin((day_of_year - 20) * 2 * np.pi / 365)
            
            # Base temperature from seasonal pattern
            temp_range = base_temp_range[1] - base_temp_range[0]
            base_temp = base_temp_range[0] + temp_range * (season_factor * 0.5 + 0.5)
            
            # Add random variation
            temperature = base_temp + np.random.normal(0, temp_variation)
            
            # Precipitation probability varies by season
            precip_range = precip_prob_range[1] - precip_prob_range[0]
            precip_prob = precip_prob_range[0] + precip_range * (season_factor * 0.3 + 0.5)
            
            # Determine if it's rainy
            is_rainy = np.random.random() < precip_prob
            
            # Determine weather quality category
            if temperature > 18 and not is_rainy:
                weather_quality = "good"
            elif temperature < 10 or is_rainy:
                weather_quality = "bad"
            else:
                weather_quality = "moderate"
            
            # Create weather record
            weather_record = {
                "city": city,
                "date": date,
                "temperature": round(temperature, 1),
                "is_rainy": is_rainy,
                "weather_quality": weather_quality,
                "season": get_season_from_date(date)
            }
            
            weather_data.append(weather_record)
    
    # Create DataFrame from weather data list
    weather_df = pd.DataFrame(weather_data)
    
    # One-hot encode weather quality
    if len(weather_df) > 0:
        weather_quality_encoder = OneHotEncoder(sparse=False, drop='first')
        weather_quality_encoded = weather_quality_encoder.fit_transform(weather_df[['weather_quality']])
        weather_quality_cols = [f"weather_quality_{cat}" for cat in weather_quality_encoder.categories_[0][1:]]
        weather_quality_df = pd.DataFrame(weather_quality_encoded, columns=weather_quality_cols)
        
        # Concatenate encoded features with original DataFrame
        weather_df = pd.concat([weather_df, weather_quality_df], axis=1)
    
    return weather_df


def get_season_from_date(date):
    """
    Determine season from date (Northern Hemisphere).
    
    Parameters:
    -----------
    date : datetime
        Date to determine season for
        
    Returns:
    --------
    str
        Season name ('Winter', 'Spring', 'Summer', or 'Fall')
    """
    # Get month and day
    month = date.month
    
    # Determine season (Northern Hemisphere)
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:  # month in [9, 10, 11]
        return "Fall"


def get_weather_stats(weather_df):
    """
    Calculate statistics about weather data.
    
    Parameters:
    -----------
    weather_df : pandas.DataFrame
        DataFrame containing weather data
        
    Returns:
    --------
    dict
        Dictionary containing weather statistics by city and season
    """
    stats = {}
    
    # Group by city and season
    grouped = weather_df.groupby(['city', 'season'])
    
    # Calculate statistics
    for (city, season), group in grouped:
        if city not in stats:
            stats[city] = {}
        
        stats[city][season] = {
            'avg_temp': group['temperature'].mean(),
            'min_temp': group['temperature'].min(),
            'max_temp': group['temperature'].max(),
            'rainy_days_pct': group['is_rainy'].mean() * 100,
            'good_weather_pct': (group['weather_quality'] == 'good').mean() * 100
        }
    
    return stats


if __name__ == "__main__":
    # Define cities
    cities = [
        "New York", "Los Angeles", "Chicago", "Miami", "Las Vegas",
        "San Francisco", "Orlando", "Boston", "New Orleans", "Seattle"
    ]
    
    # Generate weather data for current year
    current_year = datetime.now().year
    weather_df = generate_weather_data(
        cities, 
        start_date=f"{current_year}-01-01",
        end_date=f"{current_year}-12-31",
        random_seed=42
    )
    
    print(f"Generated {len(weather_df)} weather records")
    print(weather_df.head())
    
    # Display some statistics
    stats = get_weather_stats(weather_df)
    
    # Print statistics for a few cities
    for city in ["Miami", "New York", "San Francisco"]:
        print(f"\nWeather statistics for {city}:")
        for season, season_stats in stats[city].items():
            print(f"  {season}:")
            print(f"    Average temperature: {season_stats['avg_temp']:.1f}°C")
            print(f"    Temperature range: {season_stats['min_temp']:.1f}°C to {season_stats['max_temp']:.1f}°C")
            print(f"    Rainy days: {season_stats['rainy_days_pct']:.1f}%")
            print(f"    Good weather days: {season_stats['good_weather_pct']:.1f}%")
