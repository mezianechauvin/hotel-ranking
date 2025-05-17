# Hotel Ranking System

A machine learning system for ranking hotel amenity day-access options based on user preferences, location, seasonality, and other contextual factors.

## Project Overview

This project implements a recommendation system for a business that offers day access to luxury hotel amenities. The system ranks hotels based on:

- User preferences and history
- Hotel amenities and their seasonal availability
- Location proximity
- Weather and seasonal factors
- Time slot availability
- Price sensitivity

## Project Structure

```
hotel-ranking/
├── pyproject.toml       # Project dependencies and metadata
├── README.md           # Project documentation
├── run_pipeline.py     # End-to-end pipeline script
├── hotel_ranking_demo.ipynb  # Demo notebook
├── src/                # Source code
│   ├── data_generation/  # Data generation modules
│   │   ├── __init__.py
│   │   ├── venue_data.py       # Generate venue/hotel data
│   │   ├── user_data.py        # Generate user data
│   │   ├── seasonal_data.py    # Generate seasonal availability data
│   │   ├── weather_data.py     # Generate weather data
│   │   ├── interaction_data.py # Generate user-venue interactions
│   │   └── main.py            # Main data generation script
│   ├── modeling/        # ML modeling modules
│   │   ├── __init__.py
│   │   ├── feature_engineering.py  # Feature preparation
│   │   ├── model.py             # XGBoost model implementation
│   │   ├── evaluation.py        # Model evaluation utilities
│   │   └── train.py            # Model training script
│   └── utils/           # Utility functions
│       ├── __init__.py
│       ├── distance.py          # Distance calculation
│       └── visualization.py     # Data visualization utilities
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PDM package manager (or pip)

### Installation

1. Clone the repository
2. Install dependencies:

```bash
# Using PDM
pdm install

# Using pip
pip install -r requirements.txt
```

## Usage

### Generate Data

To generate synthetic data for the hotel ranking system:

```bash
# Generate full dataset
python -m src.data_generation.main --output-dir data

# Generate sample dataset
python -m src.data_generation.main --sample --output-dir data/sample
```

### Train Model

To train a ranking model using the generated data:

```bash
# Train using full dataset
python -m src.modeling.train --data-dir data

# Train using sample dataset
python -m src.modeling.train --data-dir data/sample --sample
```

### Run End-to-End Pipeline

To run the entire pipeline from data generation to model training and evaluation:

```bash
# Run with default settings
python run_pipeline.py

# Run with sample data
python run_pipeline.py --sample

# Skip data generation step
python run_pipeline.py --skip-data-gen
```

### Interactive Demo

For an interactive demonstration of the hotel ranking system, open the Jupyter notebook:

```bash
jupyter notebook hotel_ranking_demo.ipynb
```

## Model Details

The hotel ranking system uses an XGBoost model to rank venues based on various features:

- **User features**: Preferences for amenities, vibes, time slots, price sensitivity
- **Venue features**: Star rating, amenities, vibe, location, pricing
- **Contextual features**: Season, weather, time of day, day of week
- **Interaction features**: Distance, amenity match, vibe match

The model is trained on historical interaction data, where each interaction represents a user being shown a venue and either clicking on it or not.

## Evaluation

The model is evaluated using several metrics:

- **AUC**: Area under the ROC curve
- **Average Precision**: Area under the precision-recall curve
- **NDCG@k**: Normalized Discounted Cumulative Gain at k
- **CTR@k**: Click-through rate for the top k recommendations

## License

This project is licensed under the MIT License - see the LICENSE file for details.
