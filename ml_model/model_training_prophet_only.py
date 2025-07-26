#!/usr/bin/env python3
"""
Prophet-Only Model Training Module
This version only trains Prophet models, avoiding pmdarima dependency.
"""

import pandas as pd
import os
import joblib

# Import required modules with error handling
try:
    from prophet import Prophet
except ImportError as e:
    print(f"Warning: Could not import Prophet: {e}")
    print("Please install Prophet: pip install prophet")

# Import functions from other modules for individual testing
try:
    from data_ingestion import get_historical_sales_data
    from feature_engineering import preprocess_data
except ImportError as e:
    print(f"Warning: Could not import data modules: {e}")

# Define a directory to save the trained models
MODEL_SAVE_DIR = "trained_models"

# Ensure the model save directory exists
try:
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    print(f"Model save directory '{MODEL_SAVE_DIR}' ensured.")
except Exception as e:
    print(f"Error creating model save directory: {e}")

def train_prophet_model(data: pd.DataFrame, pincode: str, item: str):
    """
    Trains a Prophet model for a specific (pincode, item) combination.

    Args:
        data (pd.DataFrame): Preprocessed sales data. Must contain 'ds' (datetime),
                             'y' (sales quantity), 'pincode', and 'item' columns.
        pincode (str): The pincode for which to train the model.
        item (str): The item for which to train the model.

    Returns:
        Prophet: A trained Prophet model instance. Returns None if no data is available
                 for the given pincode-item combination.
    """
    print(f"\n--- Training Prophet model for Pincode: {pincode}, Item: {item} ---")

    try:
        # Filter data for the specific combination
        filtered_data = data[(data['pincode'] == pincode) & (data['item'] == item)].copy()

        if filtered_data.empty:
            print(f"No data found for Pincode: {pincode}, Item: {item}. Skipping Prophet training.")
            return None

        # Validate required columns
        required_columns = ['ds', 'y']
        missing_columns = [col for col in required_columns if col not in filtered_data.columns]
        if missing_columns:
            print(f"Error: Missing required columns for Prophet: {missing_columns}")
            return None

        # Prophet requires 'ds' and 'y' columns
        prophet_df = filtered_data[['ds', 'y']].copy()

        # Check for sufficient data
        if len(prophet_df) < 10:
            print(f"Not enough data points ({len(prophet_df)}) for Prophet training. Need at least 10.")
            return None

        # Initialize the Prophet model
        model = Prophet(
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05,
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True
        )

        model.fit(prophet_df)
        print(f"Prophet model trained successfully for Pincode: {pincode}, Item: {item}.")

        # Save the trained model to disk
        model_filename = os.path.join(MODEL_SAVE_DIR, f'prophet_model_{pincode}_{item}.joblib')
        joblib.dump(model, model_filename)
        print(f"Prophet model saved to {model_filename}")
        return model
    except Exception as e:
        print(f"Error training Prophet model for {pincode}, {item}: {e}")
        return None

def test_prophet_training():
    """
    Test function to train Prophet models for all available combinations.
    """
    print("="*60)
    print("TESTING PROPHET MODEL TRAINING")
    print("="*60)

    try:
        # Step 1: Get raw data
        print("Step 1: Ingesting raw data...")
        raw_sales_df = get_historical_sales_data("2024-01-01", "2024-02-28")

        if raw_sales_df.empty:
            print("❌ No raw data available. Cannot proceed with training.")
            return False

        # Step 2: Preprocess the raw data
        print("\nStep 2: Preprocessing raw data...")
        preprocessed_sales_df = preprocess_data(raw_sales_df)

        if preprocessed_sales_df.empty:
            print("❌ Preprocessed data is empty. Cannot proceed with training.")
            return False

        # Step 3: Identify unique (pincode, item) combinations
        unique_combinations = preprocessed_sales_df[['pincode', 'item']].drop_duplicates()
        print(f"\nStep 3: Found {len(unique_combinations)} unique pincode-item combinations for training.")

        trained_models_count = 0
        failed_models_count = 0

        # Step 4: Train Prophet models for each combination
        for index, row in unique_combinations.iterrows():
            pincode = row['pincode']
            item = row['item']

            prophet_model = train_prophet_model(preprocessed_sales_df, pincode, item)
            if prophet_model:
                trained_models_count += 1
            else:
                failed_models_count += 1

        print("\n" + "="*60)
        print("PROPHET TRAINING SUMMARY")
        print("="*60)
        print(f"✅ Successfully trained Prophet models: {trained_models_count}")
        print(f"❌ Failed Prophet models: {failed_models_count}")
        print(f"📁 Models saved to: {os.path.abspath(MODEL_SAVE_DIR)}")

        if trained_models_count > 0:
            print("\n🎉 Prophet training completed successfully!")
            return True
        else:
            print("\n⚠️  No Prophet models were trained successfully.")
            return False

    except Exception as e:
        print(f"❌ Error during Prophet training: {e}")
        return False

if __name__ == "__main__":
    test_prophet_training() 