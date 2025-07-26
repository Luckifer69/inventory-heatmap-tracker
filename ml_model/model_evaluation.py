import pandas as pd
import joblib # For loading trained models
import os
from datetime import date, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Import functions from other modules for individual testing and pipeline integration
try:
    from data_ingestion import get_historical_sales_data
    from feature_engineering import preprocess_data
    from prediction import load_model # To load the trained models
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    # Fallback function if prediction module is not available
    def load_model(pincode, item, model_type):
        return None

# Define the directory where trained models are saved
MODEL_SAVE_DIR = "trained_models"

def evaluate_model_performance(pincode: str, item: str, model_type: str = 'prophet',
                                 evaluation_data: pd.DataFrame = None) -> dict:
    """
    Evaluates the performance of a trained model for a specific (pincode, item)
    combination by comparing its predictions against actual historical sales data.

    This function performs a simplified evaluation. For a robust system,
    consider implementing a more rigorous backtesting strategy (e.g., walk-forward validation)
    especially for ARIMA models.

    Args:
        pincode (str): The pincode of the model to evaluate.
        item (str): The item of the model to evaluate.
        model_type (str): The type of model to evaluate ('prophet' or 'arima').
        evaluation_data (pd.DataFrame): A DataFrame containing preprocessed historical
                                        sales data for the evaluation period.
                                        It must have 'ds', 'y', 'pincode', and 'item' columns.

    Returns:
        dict: A dictionary containing evaluation metrics (MAE, RMSE, MAPE) and a status message.
              Returns an error status if the model cannot be loaded or data is insufficient.
    """
    print(f"\n--- Evaluating {model_type} model for Pincode: {pincode}, Item: {item} ---")

    # Validate inputs
    if not pincode or not item:
        return {"status": "Invalid input: pincode and item are required."}

    if model_type not in ['prophet', 'arima']:
        return {"status": f"Unsupported model type: {model_type}. Must be 'prophet' or 'arima'."}

    # Load the trained model
    model = load_model(pincode, item, model_type)

    if model is None:
        return {"status": f"Model not found or loaded for {pincode}, {item} ({model_type}), skipping evaluation."}

    if evaluation_data is None or evaluation_data.empty:
        print("No evaluation data provided. Skipping evaluation.")
        return {"status": "No historical evaluation data provided."}

    # Validate evaluation data columns
    required_columns = ['ds', 'y', 'pincode', 'item']
    missing_columns = [col for col in required_columns if col not in evaluation_data.columns]
    if missing_columns:
        return {"status": f"Missing required columns in evaluation data: {missing_columns}"}

    # Filter the evaluation data for the specific pincode and item
    try:
        eval_series_data = evaluation_data[
            (evaluation_data['pincode'] == pincode) &
            (evaluation_data['item'] == item)
        ].sort_values('ds').copy()

        if eval_series_data.empty:
            print(f"No evaluation data found for Pincode: {pincode}, Item: {item}. Skipping.")
            return {"status": "No specific evaluation data for this combination."}

        actual_values = eval_series_data['y'].values
        predictions = []

        if model_type == 'prophet':
            # For Prophet, create a future DataFrame covering the evaluation period
            # This 'future_df' should contain the 'ds' (date) values from your evaluation data.
            future_df = eval_series_data[['ds']].copy()
            # If regressors were used during training, they must be included here too
            # Example: future_df['is_weekend'] = eval_series_data['is_weekend']
            forecast = model.predict(future_df)
            predictions = forecast['yhat'].values

        elif model_type == 'arima':
            # For ARIMA, we'll use a simplified approach for evaluation
            # In a production system, you might want to implement walk-forward validation
            try:
                # Try to predict the same number of periods as the evaluation data
                predictions = model.predict(n_periods=len(actual_values))
            except Exception as arima_pred_e:
                print(f"Warning: ARIMA direct prediction failed during evaluation ({arima_pred_e}). "
                      "This might indicate the model needs a different evaluation strategy (e.g., walk-forward). "
                      "Falling back to zero predictions.")
                predictions = np.zeros_like(actual_values) # Fallback

    except Exception as e:
        print(f"Error during prediction for evaluation of {pincode}, {item} ({model_type}): {e}")
        return {"status": f"Prediction for evaluation failed: {e}"}

    # Ensure predictions and actual_values have the same length for metric calculation
    min_len = min(len(actual_values), len(predictions))
    if min_len == 0:
        print("No overlapping data for actual vs. predictions after alignment.")
        return {"status": "No data for comparison after alignment."}

    actual_values = actual_values[:min_len]
    predictions = predictions[:min_len]

    # Calculate evaluation metrics
    try:
        mae = mean_absolute_error(actual_values, predictions)
        rmse = np.sqrt(mean_squared_error(actual_values, predictions))

        # MAPE (Mean Absolute Percentage Error): Handle division by zero
        # Add a small epsilon to avoid division by zero for actual_values that are 0
        mape = np.mean(np.abs((actual_values - predictions) / (actual_values + 1e-8))) * 100
        # If all actual values are zero, MAPE becomes undefined or infinite.
        if np.all(actual_values == 0):
            mape = float('inf') # Or handle as a specific case

        metrics = {
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "MAPE": round(mape, 2),
            "status": "Evaluation complete"
        }
        print(f"Evaluation metrics for {pincode}, {item} ({model_type}): {metrics}")
        return metrics
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {"status": f"Error calculating evaluation metrics: {e}"}

# This block allows you to test the script individually.
# When you run `python model_evaluation.py` from your terminal,
# the code within this block will execute.
if __name__ == "__main__":
    print("--- Running model_evaluation.py individually ---")

    # Ensure you have run `model_training.py` first to train and save models.
    # Otherwise, `load_model` will fail.

    try:
        # --- Hardcoded Inputs for Individual Testing ---
        # 1. Get a larger range of raw data that includes both training and evaluation periods
        print("Step 1: Ingesting raw data for a longer period (e.g., 4 months)...")
        full_raw_sales_df = get_historical_sales_data("2024-01-01", "2024-04-30")

        if full_raw_sales_df.empty:
            print("No raw data available. Cannot proceed with evaluation.")
        else:
            # 2. Preprocess the full raw data
            print("\nStep 2: Preprocessing raw data...")
            preprocessed_full_sales_df = preprocess_data(full_raw_sales_df)

            if preprocessed_full_sales_df.empty:
                print("Preprocessed data is empty. Cannot proceed with evaluation.")
            else:
                # Define the evaluation period (e.g., the last month of data)
                # This assumes models were trained on data *before* this period.
                evaluation_start_date = pd.to_datetime("2024-04-01")
                evaluation_data_subset = preprocessed_full_sales_df[
                    preprocessed_full_sales_df['ds'] >= evaluation_start_date
                ].copy()

                if evaluation_data_subset.empty:
                    print(f"No evaluation data found after {evaluation_start_date}. Please adjust date ranges.")
                else:
                    print(f"\nStep 3: Preparing evaluation data from {evaluation_start_date} to {evaluation_data_subset['ds'].max().strftime('%Y-%m-%d')}.")
                    print(f"Evaluation data shape: {evaluation_data_subset.shape}")

                    # Select a specific pincode and item for evaluation
                    test_pincode = '110037'
                    test_item = 'Milk'

                    # Test Case 1: Evaluate Prophet model
                    print(f"\n--- Test Case 1: Evaluating Prophet for {test_pincode}, {test_item} ---")
                    prophet_metrics = evaluate_model_performance(
                        test_pincode, test_item, model_type='prophet',
                        evaluation_data=evaluation_data_subset
                    )
                    print(f"Prophet Evaluation Metrics: {prophet_metrics}")

                    # Test Case 2: Evaluate ARIMA model
                    print(f"\n--- Test Case 2: Evaluating ARIMA for {test_pincode}, {test_item} ---")
                    arima_metrics = evaluate_model_performance(
                        test_pincode, test_item, model_type='arima',
                        evaluation_data=evaluation_data_subset
                    )
                    print(f"ARIMA Evaluation Metrics: {arima_metrics}")

                    # Test Case 3: Evaluate for a non-existent model (should return error status)
                    print(f"\n--- Test Case 3: Evaluating for a non-existent model ---")
                    non_existent_metrics = evaluate_model_performance(
                        '999999', 'FakeItem', model_type='prophet',
                        evaluation_data=evaluation_data_subset
                    )
                    print(f"Non-existent Model Evaluation Status: {non_existent_metrics}")

                    # Test Case 4: Test with invalid inputs
                    print(f"\n--- Test Case 4: Testing with invalid inputs ---")
                    invalid_metrics = evaluate_model_performance(
                        "", "", model_type='prophet',
                        evaluation_data=evaluation_data_subset
                    )
                    print(f"Invalid Inputs Evaluation Status: {invalid_metrics}")

    except Exception as e:
        print(f"Error during individual testing: {e}")

    print("\nIndividual testing of model_evaluation.py complete.")
