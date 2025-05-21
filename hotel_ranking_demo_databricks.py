# Databricks notebook source
# MAGIC %md
# MAGIC # Hotel Ranking System Demo
# MAGIC
# MAGIC This notebook demonstrates the hotel ranking system for day-access amenities. It shows how to:
# MAGIC
# MAGIC 1. Generate synthetic data
# MAGIC 2. Explore the data
# MAGIC 3. Train a ranking model
# MAGIC 4. Evaluate the model
# MAGIC 5. Use the model for ranking venues
# MAGIC 6. Perform monthly evaluation

# COMMAND ----------

# MAGIC %pip install pdm
# MAGIC %pip install -r requirements.txt

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup
# MAGIC
# MAGIC First, let's import the necessary modules and set up the environment.

# COMMAND ----------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
plt.style.use('seaborn-whitegrid')
sns.set_palette('viridis')

# Set random seed for reproducibility
np.random.seed(42)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Generate Synthetic Data
# MAGIC
# MAGIC We'll generate a dataset spanning 3 years, with the latest year used for monthly evaluation.

# COMMAND ----------

# Check if data already exists
data_dir = "data"

if not os.path.exists(data_dir):
    print("Generating data...")
    from src.data_generation.main import generate_all_data
    data = generate_all_data()
else:
    print("Data already exists. Loading...")
    # Load common data
    venues_df = pd.read_csv(os.path.join(data_dir, "venues.csv"))
    users_df = pd.read_csv(os.path.join(data_dir, "users.csv"))
    users_processed_df = pd.read_csv(os.path.join(data_dir, "users_processed.csv"))
    seasonal_df = pd.read_csv(os.path.join(data_dir, "seasonal.csv"))
    weather_df = pd.read_csv(os.path.join(data_dir, "weather.csv"))
    interactions_df = pd.read_csv(os.path.join(data_dir, "interactions.csv"))
    
    # Load base training data (first 2 years)
    base_train_df = pd.read_csv(os.path.join(data_dir, "interactions_base_train.csv"))
    
    # Find all monthly evaluation files
    eval_files = [f for f in os.listdir(data_dir) if f.startswith("interactions_eval_") and f.endswith(".csv")]
    eval_months = [f.replace("interactions_eval_", "").replace(".csv", "") for f in eval_files]
    eval_months.sort()
    
    print(f"Found {len(eval_months)} monthly evaluation sets: {eval_months}")
    
    # Load the first month's evaluation data for demonstration
    if eval_months:
        first_month = eval_months[0]
        eval_df = pd.read_csv(os.path.join(data_dir, f"interactions_eval_{first_month}.csv"))
        train_df = pd.read_csv(os.path.join(data_dir, f"interactions_train_{first_month}.csv"))
    else:
        # Fallback to base training data
        eval_df = None
        train_df = base_train_df
    
    # Convert date columns to datetime
    weather_df["date"] = pd.to_datetime(weather_df["date"])
    
    data = {
        "venues": venues_df,
        "users": users_df,
        "users_processed": users_processed_df,
        "seasonal": seasonal_df,
        "weather": weather_df,
        "interactions": interactions_df,
        "interactions_base_train": base_train_df,
        "interactions_train": train_df,
        "interactions_eval": eval_df,
        "eval_months": eval_months
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Explore the Data
# MAGIC
# MAGIC Let's explore the generated data to understand its structure and characteristics.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Venue Data
# MAGIC
# MAGIC First, let's look at the venue data.

# COMMAND ----------

# Display venue data
print(f"Number of venues: {len(data['venues'])}")
display(data['venues'].head())

# COMMAND ----------

# Venue type distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=data['venues'], x='venue_type')
plt.title('Venue Type Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
display(plt.gcf())

# COMMAND ----------

# Amenity availability
amenities = ['pool', 'beach_access', 'spa', 'gym', 'hot_tub', 'food_service', 'bar']
amenity_availability = {amenity: data['venues'][amenity].mean() for amenity in amenities if amenity in data['venues'].columns}

plt.figure(figsize=(12, 6))
sns.barplot(x=list(amenity_availability.keys()), y=list(amenity_availability.values()))
plt.title('Amenity Availability')
plt.ylabel('Percentage of Venues')
plt.xticks(rotation=45)
plt.tight_layout()
display(plt.gcf())

# COMMAND ----------

# Price distribution by venue type
plt.figure(figsize=(12, 6))
sns.boxplot(data=data['venues'], x='venue_type', y='day_pass_price')
plt.title('Day Pass Price by Venue Type')
plt.xticks(rotation=45)
plt.tight_layout()
display(plt.gcf())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Seasonal Data
# MAGIC
# MAGIC Let's examine how amenity availability and pricing change by season.

# COMMAND ----------

# Display seasonal data
print(f"Number of seasonal records: {len(data['seasonal'])}")
display(data['seasonal'].head())

# COMMAND ----------

# Pool availability by season
if 'pool_available' in data['seasonal'].columns:
    pool_availability = data['seasonal'].groupby('season')['pool_available'].mean()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=pool_availability.index, y=pool_availability.values)
    plt.title('Pool Availability by Season')
    plt.ylabel('Percentage of Venues')
    plt.ylim(0, 1)
    plt.tight_layout()
    display(plt.gcf())

# COMMAND ----------

# Seasonal price variation
seasonal_price = data['seasonal'].groupby(['season', 'venue_id'])['seasonal_price'].mean().reset_index()
base_price = data['venues'][['venue_id', 'day_pass_price']]
price_comparison = seasonal_price.merge(base_price, on='venue_id')
price_comparison['price_ratio'] = price_comparison['seasonal_price'] / price_comparison['day_pass_price']

plt.figure(figsize=(10, 6))
sns.boxplot(data=price_comparison, x='season', y='price_ratio')
plt.axhline(y=1, color='r', linestyle='--')
plt.title('Seasonal Price Ratio (Seasonal Price / Base Price)')
plt.ylabel('Price Ratio')
plt.tight_layout()
display(plt.gcf())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 User Interaction Data
# MAGIC
# MAGIC Now let's look at the user interaction data.

# COMMAND ----------

# Display interaction data
print(f"Number of interactions: {len(data['interactions'])}")
display(data['interactions'].head())

# COMMAND ----------

# Overall click-through rate
ctr = data['interactions']['clicked'].mean()
print(f"Overall Click-Through Rate: {ctr:.2%}")

# COMMAND ----------

# CTR by position
ctr_by_position = data['interactions'].groupby('position')['clicked'].mean()

plt.figure(figsize=(12, 6))
sns.barplot(x=ctr_by_position.index, y=ctr_by_position.values)
plt.title('Click-Through Rate by Position')
plt.xlabel('Position')
plt.ylabel('Click-Through Rate')
plt.tight_layout()
display(plt.gcf())

# COMMAND ----------

# CTR by season
ctr_by_season = data['interactions'].groupby('season')['clicked'].mean()

plt.figure(figsize=(10, 6))
sns.barplot(x=ctr_by_season.index, y=ctr_by_season.values)
plt.title('Click-Through Rate by Season')
plt.ylabel('Click-Through Rate')
plt.tight_layout()
display(plt.gcf())

# COMMAND ----------

# CTR by year and month
if 'year' in data['interactions'].columns and 'month' in data['interactions'].columns:
    # Create year_month column if it doesn't exist
    if 'year_month' not in data['interactions'].columns:
        data['interactions']['year_month'] = data['interactions']['year'].astype(str) + '_' + data['interactions']['month'].astype(str).str.zfill(2)
    
    ctr_by_month = data['interactions'].groupby('year_month')['clicked'].mean().reset_index()
    
    plt.figure(figsize=(14, 6))
    sns.barplot(data=ctr_by_month, x='year_month', y='clicked')
    plt.title('Click-Through Rate by Year-Month')
    plt.xlabel('Year-Month')
    plt.ylabel('Click-Through Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    display(plt.gcf())

# COMMAND ----------

# CTR by distance
# Note: distance_bucket is now created during data generation
if 'distance_bucket' not in data['interactions'].columns:
    data['interactions']['distance_bucket'] = pd.cut(
        data['interactions']['distance_km'],
        bins=[0, 1, 2, 5, 10, 20, 50],
        labels=['0-1km', '1-2km', '2-5km', '5-10km', '10-20km', '20-50km']
    )

ctr_by_distance = data['interactions'].groupby('distance_bucket')['clicked'].mean()

plt.figure(figsize=(12, 6))
sns.barplot(x=ctr_by_distance.index, y=ctr_by_distance.values)
plt.title('Click-Through Rate by Distance')
plt.ylabel('Click-Through Rate')
plt.tight_layout()
display(plt.gcf())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.4 Temporal Data Analysis
# MAGIC
# MAGIC Let's analyze how the data is distributed across the 3-year period.

# COMMAND ----------

# Check if year and month columns exist
if 'year' in data['interactions'].columns and 'month' in data['interactions'].columns:
    # Count interactions by year
    year_counts = data['interactions']['year'].value_counts().sort_index()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=year_counts.index, y=year_counts.values)
    plt.title('Interactions by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Interactions')
    plt.tight_layout()
    display(plt.gcf())
    
    # Count interactions by year and month
    if 'year_month' not in data['interactions'].columns:
        data['interactions']['year_month'] = data['interactions']['year'].astype(str) + '_' + data['interactions']['month'].astype(str).str.zfill(2)
    
    month_counts = data['interactions']['year_month'].value_counts().sort_index()
    
    plt.figure(figsize=(14, 6))
    sns.barplot(x=month_counts.index, y=month_counts.values)
    plt.title('Interactions by Year-Month')
    plt.xlabel('Year-Month')
    plt.ylabel('Number of Interactions')
    plt.xticks(rotation=45)
    plt.tight_layout()
    display(plt.gcf())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Train a Ranking Model
# MAGIC
# MAGIC Now let's train a ranking model using the generated data.

# COMMAND ----------

from src.modeling.feature_engineering import prepare_features
from src.modeling.model import train_ranking_model, save_model

# Prepare features for training
print("Preparing features...")
train_features_df = prepare_features(
    data['interactions_train'],
    data['venues'],
    data['users_processed'],
    data['seasonal'],
    data['weather']
)

# Train model
print("Training model...")
model, feature_cols = train_ranking_model(train_features_df)

# Save model
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
save_model(model, feature_cols, model_dir)

# COMMAND ----------

pd.set_option('display.max_rows', 100)
display(train_features_df.dtypes)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Evaluate the Model
# MAGIC
# MAGIC Let's evaluate the trained model on the evaluation data.

# COMMAND ----------

from src.modeling.evaluation import evaluate_model, generate_evaluation_report

# Check if we have evaluation data
if data['interactions_eval'] is not None:
    eval_df = data['interactions_eval']
else:
    # Fallback to using a portion of the training data
    print("No evaluation data found. Using a portion of training data for evaluation.")
    train_size = int(len(data['interactions_train']) * 0.8)
    eval_df = data['interactions_train'].iloc[train_size:]

# Evaluate model
print("Evaluating model...")
metrics, results_df = evaluate_model(
    model,
    eval_df,
    data['venues'],
    data['users_processed'],
    data['seasonal'],
    data['weather'],
    feature_cols
)

# Print metrics
print("\nEvaluation Metrics:")
print(f"AUC: {metrics['auc']:.4f}")
print(f"Average Precision: {metrics['average_precision']:.4f}")
print(f"NDCG@5: {metrics['ndcg@5']:.4f}")
print(f"NDCG@10: {metrics['ndcg@10']:.4f}")
print(f"CTR@1: {metrics['ctr@1']:.4f}")
print(f"CTR@5: {metrics['ctr@5']:.4f}")

# COMMAND ----------

# Plot precision-recall curve
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, _ = precision_recall_curve(results_df['clicked'], results_df['predicted_score'])
ap = average_precision_score(results_df['clicked'], results_df['predicted_score'])

plt.figure(figsize=(10, 6))
plt.plot(recall, precision, lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve (AP = {ap:.4f})')
plt.grid(True)
display(plt.gcf())

# COMMAND ----------

# Plot score distribution by clicked status
plt.figure(figsize=(10, 6))
sns.histplot(
    data=results_df,
    x='predicted_score',
    hue='clicked',
    bins=30,
    alpha=0.6
)
plt.title('Distribution of Predicted Scores by Clicked Status')
plt.xlabel('Predicted Score')
plt.ylabel('Count')
display(plt.gcf())

# COMMAND ----------

# Feature importance
importance = model.get_score(importance_type='gain')
importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

# Plot top 20 features
top_features = importance[:20]
feature_names = [f[0] for f in top_features]
feature_scores = [f[1] for f in top_features]

plt.figure(figsize=(12, 8))
sns.barplot(x=feature_scores, y=feature_names)
plt.title('Top 20 Features by Importance')
plt.xlabel('Importance Score')
plt.tight_layout()
display(plt.gcf())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Use the Model for Ranking Venues
# MAGIC
# MAGIC Now let's use the trained model to rank venues for a specific user and context.

# COMMAND ----------

from src.modeling.model import predict_rankings

# Select a random session from the evaluation data
if data['interactions_eval'] is not None:
    session_id = data['interactions_eval']['session_id'].sample(1).iloc[0]
    session_data = data['interactions_eval'][data['interactions_eval']['session_id'] == session_id]
else:
    # Fallback to using training data
    session_id = data['interactions_train']['session_id'].sample(1).iloc[0]
    session_data = data['interactions_train'][data['interactions_train']['session_id'] == session_id]

print(f"Selected session ID: {session_id}")
print(f"Number of venues in session: {len(session_data)}")

# Get user ID and context information
user_id = session_data['user_id'].iloc[0]
season = session_data['season'].iloc[0]
time_slot = session_data['time_slot'].iloc[0]

print(f"User ID: {user_id}")
print(f"Season: {season}")
print(f"Time slot: {time_slot}")

# COMMAND ----------

# Prepare features for the session
session_features = prepare_features(
    session_data,
    data['venues'],
    data['users_processed'],
    data['seasonal'],
    data['weather']
)

# Predict rankings
ranked_venues = predict_rankings(model, session_features, feature_cols)

# Sort by predicted score
ranked_venues = ranked_venues.sort_values('predicted_score', ascending=False)

# Display ranked venues
display_cols = ['venue_id', 'clicked', 'predicted_score', 'predicted_rank',
               'venue_type', 'star_rating', 'seasonal_price', 'vibe',
               'distance_km', 'time_slot']

display(ranked_venues[display_cols])

# COMMAND ----------

# Compare model ranking with original position
comparison = ranked_venues[['venue_id', 'position', 'predicted_rank', 'clicked', 'predicted_score']]
comparison = comparison.sort_values('position')

plt.figure(figsize=(12, 6))
plt.scatter(comparison['position'], comparison['predicted_rank'], 
           c=comparison['clicked'].map({True: 'green', False: 'red'}),
           s=100, alpha=0.7)

plt.xlabel('Original Position')
plt.ylabel('Predicted Rank')
plt.title('Original Position vs. Predicted Rank')
plt.grid(True)

# Add diagonal line for reference
max_val = max(comparison['position'].max(), comparison['predicted_rank'].max())
plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Clicked'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Not Clicked')
]
plt.legend(handles=legend_elements)

plt.tight_layout()
display(plt.gcf())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Monthly Evaluation
# MAGIC
# MAGIC Now let's demonstrate the monthly evaluation approach, where we train and evaluate models for each month of the evaluation year.

# COMMAND ----------

from src.modeling.train import train_and_evaluate_all_months

# Check if we have monthly evaluation data
if 'eval_months' in data and data['eval_months']:
    print(f"Found {len(data['eval_months'])} monthly evaluation sets: {data['eval_months']}")
    
    # We'll demonstrate the monthly evaluation process by training and evaluating for each month
    # Note: This can be time-consuming, so we'll just show the process for a few months
    sample_months = data['eval_months'] if len(data['eval_months']) > 3 else data['eval_months']
    
    # Store results for each month
    monthly_results = {}
    
    for year_month in sample_months:
        print(f"\n=== Processing month: {year_month} ===\n")
        
        # Load training and evaluation data for this month
        train_df = pd.read_csv(os.path.join(data_dir, f"interactions_train_{year_month}.csv"))
        eval_df = pd.read_csv(os.path.join(data_dir, f"interactions_eval_{year_month}.csv"))
        
        print(f"Training data size: {len(train_df)}")
        print(f"Evaluation data size: {len(eval_df)}")
        
        # Prepare features for training
        print("Preparing features...")
        train_features_df = prepare_features(
            train_df,
            data['venues'],
            data['users_processed'],
            data['seasonal'],
            data['weather']
        )
        
        # Train model
        print("Training model...")
        model, feature_cols = train_ranking_model(train_features_df)
        
        # Evaluate model
        print("Evaluating model...")
        metrics, results_df = evaluate_model(
            model,
            eval_df,
            data['venues'],
            data['users_processed'],
            data['seasonal'],
            data['weather'],
            feature_cols
        )
        
        # Store results
        monthly_results[year_month] = {
            'model': model,
            'feature_cols': feature_cols,
            'metrics': metrics,
            'results_df': results_df
        }
        
        # Print metrics
        print(f"\nEvaluation Metrics for {year_month}:")
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"Average Precision: {metrics['average_precision']:.4f}")
        print(f"NDCG@5: {metrics['ndcg@5']:.4f}")
        print(f"NDCG@10: {metrics['ndcg@10']:.4f}")
        print(f"CTR@1: {metrics['ctr@1']:.4f}")
        print(f"CTR@5: {metrics['ctr@5']:.4f}")
else:
    print("No monthly evaluation data found. Run the pipeline with --monthly-eval flag to generate it.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.1 Compare Performance Across Months
# MAGIC
# MAGIC Let's compare the model performance across different months.

# COMMAND ----------

# Check if we have monthly results
if 'monthly_results' in locals() and monthly_results:
    # Extract metrics for each month
    months = list(monthly_results.keys())
    auc_values = [monthly_results[m]['metrics']['auc'] for m in months]
    ndcg5_values = [monthly_results[m]['metrics']['ndcg@5'] for m in months]
    ctr1_values = [monthly_results[m]['metrics']['ctr@1'] for m in months]
    ctr5_values = [monthly_results[m]['metrics']['ctr@5'] for m in months]
    
    # Create summary table
    summary_df = pd.DataFrame({
        'Month': months,
        'AUC': auc_values,
        'NDCG@5': ndcg5_values,
        'CTR@1': ctr1_values,
        'CTR@5': ctr5_values
    })
    
    # Display summary table
    display(summary_df)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # AUC plot
    axes[0, 0].plot(months, auc_values, marker='o')
    axes[0, 0].set_title('AUC by Month')
    axes[0, 0].set_xlabel('Month')
    axes[0, 0].set_ylabel('AUC')
    axes[0, 0].grid(True)
    
    # NDCG@5 plot
    axes[0, 1].plot(months, ndcg5_values, marker='o', color='orange')
    axes[0, 1].set_title('NDCG@5 by Month')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('NDCG@5')
    axes[0, 1].grid(True)
    
    # CTR@1 plot
    axes[1, 0].plot(months, ctr1_values, marker='o', color='green')
    axes[1, 0].set_title('CTR@1 by Month')
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].set_ylabel('CTR@1')
    axes[1, 0].grid(True)
    
    # CTR@5 plot
    axes[1, 1].plot(months, ctr5_values, marker='o', color='red')
    axes[1, 1].set_title('CTR@5 by Month')
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('CTR@5')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    display(plt.gcf())
else:
    print("No monthly results available.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.2 Feature Importance Across Months
# MAGIC
# MAGIC Let's compare how feature importance changes across different months.

# COMMAND ----------

# Check if we have monthly results
if 'monthly_results' in locals() and monthly_results:
    # Get top 10 features for each month
    top_features_by_month = {}
    
    for month, results in monthly_results.items():
        model = results['model']
        importance = model.get_score(importance_type='gain')
        importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        top_features_by_month[month] = importance
    
    # Plot top features for each month
    fig, axes = plt.subplots(len(top_features_by_month), 1, figsize=(12, 5*len(top_features_by_month)))
    
    for i, (month, features) in enumerate(top_features_by_month.items()):
        feature_names = [f[0] for f in features]
        feature_scores = [f[1] for f in features]
        
        ax = axes[i] if len(top_features_by_month) > 1 else axes
        sns.barplot(x=feature_scores, y=feature_names, ax=ax)
        ax.set_title(f'Top 10 Features by Importance - {month}')
        ax.set_xlabel('Importance Score')
    
    plt.tight_layout()
    display(plt.gcf())
else:
    print("No monthly results available.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Visualize Venue Locations
# MAGIC
# MAGIC Let's visualize the venue locations for a specific city.

# COMMAND ----------

from src.utils.visualization import plot_venue_locations

# Get user's city
user_city = data['users'][data['users']['user_id'] == user_id]['home_city'].iloc[0]
print(f"User's city: {user_city}")

# Plot venue locations
plot_venue_locations(data['venues'], data['users'], city=user_city)
plt.title(f'Venue Locations in {user_city}')
display(plt.gcf())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Seasonal Availability Analysis
# MAGIC
# MAGIC Let's analyze how amenity availability changes by season.

# COMMAND ----------

from src.utils.visualization import plot_seasonal_availability

# Plot seasonal availability for different amenities
amenities = ['pool', 'beach_access', 'spa', 'gym', 'hot_tub']

fig, axes = plt.subplots(len(amenities), 1, figsize=(12, 4*len(amenities)))

for i, amenity in enumerate(amenities):
    try:
        plot_seasonal_availability(data['seasonal'], amenity=amenity, ax=axes[i])
    except (KeyError, ValueError) as e:
        axes[i].text(0.5, 0.5, f"No data available for {amenity}", 
                    horizontalalignment='center', verticalalignment='center')
        axes[i].set_title(f'Seasonal Availability of {amenity.capitalize()}')

plt.tight_layout()
display(plt.gcf())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this notebook, we've demonstrated the hotel ranking system for day-access amenities. We've shown how to:
# MAGIC
# MAGIC 1. Generate synthetic data spanning 3 years
# MAGIC 2. Explore the data to understand its characteristics
# MAGIC 3. Train a ranking model using XGBoost
# MAGIC 4. Evaluate the model's performance
# MAGIC 5. Use the model to rank venues for a specific user and context
# MAGIC 6. Perform monthly evaluation to track model performance over time
# MAGIC 7. Visualize venue locations and seasonal availability
# MAGIC
# MAGIC The model takes into account various factors including:
# MAGIC - User preferences and history
# MAGIC - Venue attributes and amenities
# MAGIC - Seasonal availability and pricing
# MAGIC - Weather conditions
# MAGIC - Location proximity
# MAGIC - Time slot availability
# MAGIC
# MAGIC This approach provides a personalized ranking of venues based on the specific context and user preferences, and the monthly evaluation approach allows us to track how model performance changes over time.

# COMMAND ----------


