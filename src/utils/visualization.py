"""
Visualization utilities for the hotel ranking system.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_venue_locations(venues_df, users_df=None, city=None, figsize=(12, 10)):
    """
    Plot venue locations on a map with optional user locations.
    
    Parameters:
    -----------
    venues_df : pandas.DataFrame
        DataFrame containing venue data with latitude and longitude columns
    users_df : pandas.DataFrame, optional
        DataFrame containing user data with latitude and longitude columns
    city : str, optional
        City name to filter venues and users
    figsize : tuple, optional
        Figure size as (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Filter by city if specified
    if city:
        venues_plot = venues_df[venues_df['city'] == city]
        if users_df is not None:
            users_plot = users_df[users_df['home_city'] == city]
    else:
        venues_plot = venues_df
        users_plot = users_df
    
    # Plot venues
    sns.scatterplot(
        x='longitude', 
        y='latitude', 
        data=venues_plot,
        hue='venue_type',
        size='day_pass_price',
        sizes=(20, 200),
        alpha=0.7,
        palette='viridis',
        label='Venues'
    )
    
    # Plot users if provided
    if users_df is not None:
        sns.scatterplot(
            x='longitude',
            y='latitude',
            data=users_plot,
            color='red',
            alpha=0.3,
            label='Users'
        )
    
    plt.title(f'Venue Locations{f" in {city}" if city else ""}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    
    return plt.gca()


def plot_seasonal_availability(seasonal_df, amenity='pool'):
    """
    Plot seasonal availability of a specific amenity.
    
    Parameters:
    -----------
    seasonal_df : pandas.DataFrame
        DataFrame containing seasonal data
    amenity : str, optional
        Amenity to plot availability for
    """
    # Calculate availability percentage by season
    availability_col = f"{amenity}_available"
    if availability_col not in seasonal_df.columns:
        raise ValueError(f"Column '{availability_col}' not found in seasonal data")
    
    seasonal_availability = seasonal_df.groupby('season')[availability_col].mean()
    
    plt.figure(figsize=(10, 6))
    ax = seasonal_availability.plot(kind='bar', color='skyblue')
    
    plt.title(f'Seasonal Availability of {amenity.capitalize()}')
    plt.xlabel('Season')
    plt.ylabel('Availability Percentage')
    plt.ylim(0, 1)
    
    # Add percentage labels on bars
    for i, v in enumerate(seasonal_availability):
        ax.text(i, v + 0.02, f'{v:.1%}', ha='center')
    
    return plt.gca()


def plot_feature_importance(model, feature_names, top_n=20, figsize=(12, 8)):
    """
    Plot feature importance from an XGBoost model.
    
    Parameters:
    -----------
    model : xgboost.Booster
        Trained XGBoost model
    feature_names : list
        List of feature names
    top_n : int, optional
        Number of top features to display
    figsize : tuple, optional
        Figure size as (width, height)
    """
    # Get feature importance
    importance = model.get_score(importance_type='gain')
    
    # Convert to DataFrame for easier plotting
    importance_df = pd.DataFrame({
        'Feature': list(importance.keys()),
        'Importance': list(importance.values())
    })
    
    # Sort by importance and take top N
    importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
    
    plt.figure(figsize=figsize)
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title(f'Top {top_n} Feature Importance')
    plt.tight_layout()
    
    return plt.gca()


def plot_click_distribution_by_factor(interactions_df, factor_col, title=None):
    """
    Plot click-through rate distribution by a categorical factor.
    
    Parameters:
    -----------
    interactions_df : pandas.DataFrame
        DataFrame containing interaction data
    factor_col : str
        Column name of the categorical factor
    title : str, optional
        Plot title
    """
    # Calculate CTR by factor
    ctr_by_factor = interactions_df.groupby(factor_col)['clicked'].mean().reset_index()
    ctr_by_factor.columns = [factor_col, 'CTR']
    
    # Sort by CTR
    ctr_by_factor = ctr_by_factor.sort_values('CTR', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=factor_col, y='CTR', data=ctr_by_factor)
    
    plt.title(title or f'Click-Through Rate by {factor_col}')
    plt.ylabel('Click-Through Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return plt.gca()
