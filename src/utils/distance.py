"""
Distance calculation utilities for the hotel ranking system.
"""

import numpy as np


def calculate_distance(lat1, lng1, lat2, lng2):
    """
    Calculate the Haversine distance between two points in kilometers.
    
    Parameters:
    -----------
    lat1 : float
        Latitude of the first point in decimal degrees
    lng1 : float
        Longitude of the first point in decimal degrees
    lat2 : float
        Latitude of the second point in decimal degrees
    lng2 : float
        Longitude of the second point in decimal degrees
        
    Returns:
    --------
    float
        Distance between the points in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    
    return c * r


def calculate_distance_matrix(user_locations, venue_locations):
    """
    Calculate distance matrix between multiple users and venues.
    
    Parameters:
    -----------
    user_locations : list of tuples
        List of (latitude, longitude) tuples for users
    venue_locations : list of tuples
        List of (latitude, longitude) tuples for venues
        
    Returns:
    --------
    numpy.ndarray
        2D array where element [i,j] is the distance between 
        user_locations[i] and venue_locations[j]
    """
    n_users = len(user_locations)
    n_venues = len(venue_locations)
    
    distance_matrix = np.zeros((n_users, n_venues))
    
    for i, (user_lat, user_lng) in enumerate(user_locations):
        for j, (venue_lat, venue_lng) in enumerate(venue_locations):
            distance_matrix[i, j] = calculate_distance(
                user_lat, user_lng, venue_lat, venue_lng
            )
    
    return distance_matrix
