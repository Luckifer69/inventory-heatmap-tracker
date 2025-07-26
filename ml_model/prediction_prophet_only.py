#!/usr/bin/env python3
"""
Prophet-Only Prediction Module
This version only uses Prophet models, avoiding pmdarima dependency.
"""

import pandas as pd
import joblib
import os
from datetime import date, timedelta

# Import required modules with error handling
try:
    from prophet import Prophet
except ImportError as e:
    print(f"Warning: Could not import Prophet: {e}")

# Define the directory where trained models are saved
MODEL_SAVE_DIR = "trained_models"

def load_prophet_model(pincode: str, item: str):
    """
    Loads a trained Prophet model from disk.

    Args:
        pincode (str): The pincode associated with the model.
        item (str): The item associated with the model.

    Returns:
        Prophet: The loaded trained Prophet model instance, or None if not found.
    """
    model_filename = os.path.join(MODEL_SAVE_DIR, f'prophet_model_{pincode}_{item}.joblib')
    print(f"Attempting to load Prophet model from: {model_filename}")

    if not os.path.exists(model_filename):
        print(f"Error: Prophet model file not found at {model_filename}. Please ensure models are trained.")
        return None
    
    try:
        model = joblib.load(model_filename)
        print(f"Prophet model '{model_filename}' loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading Prophet model '{model_filename}': {e}")
        return None

def predict_next_day_demand_prophet(pincode: str, item: str) -> dict:
    """
    Predicts the next day's demand for a given pincode and item using Prophet model.

    Args:
        pincode (str): The pincode for which to predict demand.
        item (str): The item for which to predict demand.

    Returns:
        dict: A dictionary containing prediction results and status.
    """
    print(f"\n--- Predicting next-day demand for Pincode: {pincode}, Item: {item} using Prophet model ---")

    # Validate inputs
    if not pincode or not item:
        return {
            "pincode": pincode,
            "item": item,
            "predicted_demand": 0,
            "restock_quantity": 0,
            "status": "Invalid input: pincode and item are required."
        }

    # Load the trained Prophet model
    model = load_prophet_model(pincode, item)

    if model is None:
        return {
            "pincode": pincode,
            "item": item,
            "predicted_demand": 0,
            "restock_quantity": 0,
            "status": f"Prophet model not found or loaded for {pincode}, {item}."
        }

    # Determine the date for which to make the prediction (tomorrow)
    tomorrow = date.today() + timedelta(days=1)
    predicted_demand = 0

    try:
        # Prophet requires a DataFrame with a 'ds' column for future dates
        future_df = pd.DataFrame({'ds': [pd.to_datetime(tomorrow)]})

        forecast = model.predict(future_df)
        # The 'yhat' column contains the predicted values.
        predicted_demand = max(0, round(forecast['yhat'].iloc[0]))

    except Exception as e:
        print(f"Error during Prophet prediction for {pincode}, {item}: {e}")
        return {
            "pincode": pincode,
            "item": item,
            "predicted_demand": 0,
            "restock_quantity": 0,
            "status": f"Prophet prediction failed: {e}"
        }

    # Calculate restock quantity
    restock_buffer = 5
    restock_quantity = max(0, predicted_demand + restock_buffer)

    print(f"Prophet prediction successful: Pincode: {pincode}, Item: {item}")
    print(f"  Predicted Demand: {predicted_demand} units")
    print(f"  Suggested Restock Quantity: {restock_quantity} units")

    return {
        "pincode": pincode,
        "item": item,
        "predicted_demand": predicted_demand,
        "restock_quantity": restock_quantity,
        "status": "Success"
    }

def test_prophet_predictions():
    """
    Test function to make predictions using trained Prophet models.
    """
    print("="*60)
    print("TESTING PROPHET PREDICTIONS")
    print("="*60)

    # Test cases
    test_cases = [
        ("110037", "Milk"),
        ("400092", "Bread"),
        ("400053", "Apples"),
        ("999999", "FakeItem")  # This should fail
    ]

    successful_predictions = 0
    failed_predictions = 0

    for pincode, item in test_cases:
        print(f"\n--- Test Case: {pincode}, {item} ---")
        
        result = predict_next_day_demand_prophet(pincode, item)
        
        if result["status"] == "Success":
            successful_predictions += 1
            print(f"✅ SUCCESS: Predicted {result['predicted_demand']} units, Restock {result['restock_quantity']} units")
        else:
            failed_predictions += 1
            print(f"❌ FAILED: {result['status']}")

    print("\n" + "="*60)
    print("PROPHET PREDICTION SUMMARY")
    print("="*60)
    print(f"✅ Successful predictions: {successful_predictions}")
    print(f"❌ Failed predictions: {failed_predictions}")
    
    if successful_predictions > 0:
        print("\n🎉 Prophet predictions working correctly!")
        return True
    else:
        print("\n⚠️  No successful predictions. Make sure to train Prophet models first.")
        return False

if __name__ == "__main__":
    test_prophet_predictions() 