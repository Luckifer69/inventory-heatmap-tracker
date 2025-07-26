import pandas as pd
import joblib # For loading trained models
import os
from datetime import date, timedelta
from prophet import Prophet # Required for type hinting and potentially model loading
import pmdarima as pm # Required for type hinting and potentially model loading

# Define the directory where trained models are saved
MODEL_SAVE_DIR = "trained_models"

def load_model(pincode: str, item: str, model_type: str = 'prophet'):
    """
    Loads a trained forecasting model from disk.

    Args:
        pincode (str): The pincode associated with the model.
        item (str): The item associated with the model.
        model_type (str): The type of model to load ('prophet' or 'arima').

    Returns:
        Union[Prophet, pmdarima.ARIMA, None]: The loaded trained model instance,
                                              or None if the model file is not found
                                              or an error occurs during loading.
    """
    model_filename = os.path.join(MODEL_SAVE_DIR, f'{model_type}_model_{pincode}_{item}.joblib')
    print(f"Attempting to load model from: {model_filename}")

    if not os.path.exists(model_filename):
        print(f"Error: Model file not found at {model_filename}. Please ensure models are trained.")
        return None
    try:
        model = joblib.load(model_filename)
        print(f"Model '{model_filename}' loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model '{model_filename}': {e}")
        return None

def predict_next_day_demand(pincode: str, item: str, model_type: str = 'prophet') -> dict:
    """
    Predicts the next day's demand for a given pincode and item using a specified model type.

    Args:
        pincode (str): The pincode for which to predict demand.
        item (str): The item for which to predict demand.
        model_type (str): The type of model to use for prediction ('prophet' or 'arima').
                          Defaults to 'prophet'.

    Returns:
        dict: A dictionary containing:
              - 'pincode': The input pincode.
              - 'item': The input item.
              - 'predicted_demand': The forecasted sales quantity for the next day (integer).
              - 'restock_quantity': The suggested quantity to restock (integer).
              - 'status': A message indicating success or failure/error.
    """
    print(f"\n--- Predicting next-day demand for Pincode: {pincode}, Item: {item} using {model_type} model ---")

    # Load the appropriate trained model
    model = load_model(pincode, item, model_type)

    if model is None:
        return {
            "pincode": pincode,
            "item": item,
            "predicted_demand": 0,
            "restock_quantity": 0,
            "status": f"Model not found or loaded for {pincode}, {item} ({model_type})."
        }

    # Determine the date for which to make the prediction (tomorrow)
    tomorrow = date.today() + timedelta(days=1)
    predicted_demand = 0

    try:
        if model_type == 'prophet':
            # Prophet requires a DataFrame with a 'ds' column for future dates
            future_df = pd.DataFrame({'ds': [pd.to_datetime(tomorrow)]})

            # If your Prophet model was trained with additional regressors (e.g., 'is_weekend'),
            # you must add those features to the 'future_df' as well.
            # Example:
            # future_df['is_weekend'] = (future_df['ds'].dt.dayofweek >= 5).astype(int)

            forecast = model.predict(future_df)
            # The 'yhat' column contains the predicted values.
            # Use max(0, ...) to ensure demand is not negative, and round to nearest integer.
            predicted_demand = max(0, round(forecast['yhat'].iloc[0]))

        elif model_type == 'arima':
            # ARIMA models typically predict a specified number of steps ahead.
            # Here, we predict 1 step (for tomorrow).
            # If your ARIMA model used exogenous variables (exog), you'd pass them here.
            # For auto_arima, it's often simpler if no exog are used for prediction.
            n_periods = 1
            forecast_values, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)
            predicted_demand = max(0, round(forecast_values[0])) # Get the first (and only) prediction

        else:
            return {
                "pincode": pincode,
                "item": item,
                "predicted_demand": 0,
                "restock_quantity": 0,
                "status": f"Unsupported model type: {model_type}. Must be 'prophet' or 'arima'."
            }

    except Exception as e:
        print(f"Error during prediction for {pincode}, {item} ({model_type}): {e}")
        return {
            "pincode": pincode,
            "item": item,
            "predicted_demand": 0,
            "restock_quantity": 0,
            "status": f"Prediction failed: {e}"
        }

    # --- Logic for calculating restock quantity ---
    # This is a simplified rule. In a real system, this would be more sophisticated,
    # considering current inventory, safety stock levels, lead times, and business rules.
    # The PDF example suggests "Restock 40 units by 9 AM tomorrow" based on a ~35 units/day trend.
    # Let's assume a simple buffer or a direct mapping for now.
    # For instance, if predicted demand is X, restock X + a buffer, or a multiple.
    restock_buffer = 5 # A small buffer to account for variability
    restock_quantity = max(0, predicted_demand + restock_buffer)

    print(f"Prediction successful: Pincode: {pincode}, Item: {item}")
    print(f"  Predicted Demand: {predicted_demand} units")
    print(f"  Suggested Restock Quantity: {restock_quantity} units")

    return {
        "pincode": pincode,
        "item": item,
        "predicted_demand": predicted_demand,
        "restock_quantity": restock_quantity,
        "status": "Success"
    }

# This block allows you to test the script individually.
# When you run `python prediction.py` from your terminal,
# the code within this block will execute.
if __name__ == "__main__":
    print("--- Running prediction.py individually ---")

    # Ensure you have run `model_training.py` at least once
    # to create the 'trained_models' directory and save models.

    # --- Hardcoded Inputs for Individual Testing ---
    test_pincode_1 = '110037'
    test_item_1 = 'Milk'
    test_model_type_1 = 'prophet'

    test_pincode_2 = '400092'
    test_item_2 = 'Bread'
    test_model_type_2 = 'arima'

    test_pincode_3 = '999999' # A pincode that likely won't have a trained model
    test_item_3 = 'NonExistentItem'
    test_model_type_3 = 'prophet'

    # Test Case 1: Predict using Prophet model
    print("\n--- Test Case 1: Prophet Prediction ---")
    result_1 = predict_next_day_demand(test_pincode_1, test_item_1, test_model_type_1)
    print(f"Result 1: {result_1}")

    # Test Case 2: Predict using ARIMA model
    print("\n--- Test Case 2: ARIMA Prediction ---")
    result_2 = predict_next_day_demand(test_pincode_2, test_item_2, test_model_type_2)
    print(f"Result 2: {result_2}")

    # Test Case 3: Predict for a non-existent model (should show an error status)
    print("\n--- Test Case 3: Non-existent Model Prediction ---")
    result_3 = predict_next_day_demand(test_pincode_3, test_item_3, test_model_type_3)
    print(f"Result 3: {result_3}")

    print("\nIndividual testing of prediction.py complete.")
