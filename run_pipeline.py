#!/usr/bin/env python
"""
Pipeline script for the hotel ranking system.

This script runs the entire pipeline from data generation to model training and evaluation.
"""

import os
import argparse
import time
from datetime import datetime

def run_data_generation(args):
    """Run data generation step."""
    print("\n=== STEP 1: DATA GENERATION ===\n")
    
    if args.sample:
        print("Generating sample data...")
        cmd = f"python -m src.data_generation.main --sample --output-dir {args.data_dir} --seed {args.seed}"
    else:
        print("Generating full data...")
        cmd = (f"python -m src.data_generation.main --output-dir {args.data_dir} "
               f"--n-venues {args.n_venues} --n-users {args.n_users} "
               f"--n-interactions {args.n_interactions} --seed {args.seed}")
    
    print(f"Executing: {cmd}")
    os.system(cmd)


def run_model_training(args):
    """Run model training step."""
    print("\n=== STEP 2: MODEL TRAINING AND EVALUATION ===\n")
    
    data_dir = args.data_dir
    if args.sample:
        data_dir = os.path.join(args.data_dir, "sample")
    
    if args.monthly_eval:
        # Train and evaluate for each month
        print("\n=== Running Monthly Evaluation ===\n")
        cmd = (f"python -m src.modeling.train --data-dir {data_dir} "
               f"--model-dir {args.model_dir} --report-dir {args.report_dir} "
               f"--model-type {args.model_type} --seed {args.seed} --monthly-eval")
    else:
        # Standard training and evaluation
        cmd = (f"python -m src.modeling.train --data-dir {data_dir} "
               f"--model-dir {args.model_dir} --report-dir {args.report_dir} "
               f"--model-type {args.model_type} --seed {args.seed}")
    
    print(f"Executing: {cmd}")
    os.system(cmd)


def main():
    parser = argparse.ArgumentParser(description="Run the hotel ranking system pipeline")
    
    # Data generation arguments
    parser.add_argument("--data-dir", type=str, default="data", help="Directory to save generated data")
    parser.add_argument("--n-venues", type=int, default=500, help="Number of venues to generate")
    parser.add_argument("--n-users", type=int, default=1000, help="Number of users to generate")
    parser.add_argument("--n-interactions", type=int, default=50000, help="Number of interactions to generate")
    
    # Model training arguments
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to save the model")
    parser.add_argument("--report-dir", type=str, default="reports", help="Directory to save the evaluation report")
    parser.add_argument("--model-type", type=str, default="pointwise", choices=["pointwise", "pairwise"], 
                      help="Type of model to train")
    parser.add_argument("--monthly-eval", action="store_true", 
                      help="Run monthly evaluation (train and evaluate for each month)")
    
    # General arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--sample", action="store_true", help="Generate and use sample data")
    parser.add_argument("--skip-data-gen", action="store_true", help="Skip data generation step")
    
    args = parser.parse_args()
    
    # Record start time
    start_time = time.time()
    
    # Create directories if they don't exist
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)
    
    # Run pipeline steps
    if not args.skip_data_gen:
        run_data_generation(args)
    else:
        print("\n=== SKIPPING DATA GENERATION ===\n")
    
    run_model_training(args)
    
    # Print execution time
    execution_time = time.time() - start_time
    print(f"\nPipeline completed in {execution_time:.2f} seconds")


if __name__ == "__main__":
    main()
